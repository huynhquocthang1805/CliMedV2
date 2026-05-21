"""
rag_optimizer.py — CliMedV2
============================

Lớp tối ưu sau hybrid retrieval, biến chunks thô thành "evidence cards" ngắn,
có rerank theo embedding, lọc theo độ liên quan với câu hỏi, và khử trùng lặp
bằng MMR.

Mục tiêu: ép bot trả lời TRỌNG TÂM, không lan man, không bịa liên kết giữa
các chunk không liên quan.

Pipeline:
    user_query
       │
       ▼
    expand_query (synonym y khoa VN/EN)         ← MedicalQueryExpander
       │
       ▼
    hybrid_retrieve(BM25+FAISS+RRF)             ← rag_engine.HybridRetriever
       │       (candidate_k = 20)
       ▼
    bi_encoder_rerank                           ← rerank_chunks()
       │       (cosine sim query↔chunk, drop < rel_threshold)
       ▼
    sentence_level_extract                      ← extract_relevant_sentences()
       │       (mỗi chunk còn 2-4 câu liên quan nhất)
       ▼
    mmr_select                                  ← mmr_dedupe()
       │       (loại chunk trùng ý)
       ▼
    evidence_cards (≤3 thẻ, mỗi thẻ ≤350 ký tự)

Toàn bộ pipeline KHÔNG cần dependency mới — dùng lại embedding backend của
rag_engine (Ollama / sentence-transformers) đã có.

Cách dùng từ chatbot_engine:
    from rag_optimizer import build_evidence_cards
    cards = build_evidence_cards(query, top_n=3)
    block = format_evidence_block(cards)        # text ready-to-inject vào prompt
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# CONFIG — knobs cho việc trim/threshold
# ===========================================================================

CFG = {
    # Số candidate kéo về từ hybrid retriever (trước rerank). To hơn = recall tốt hơn nhưng chậm.
    "candidate_k": 20,
    # Sau rerank theo embedding, giữ tối đa bao nhiêu chunk.
    "after_rerank_k": 6,
    # Ngưỡng cosine similarity query↔chunk. Chunk < ngưỡng này bị loại.
    # 0.35 phù hợp với multilingual-e5-small / nomic-embed-text trên VN.
    "rel_threshold": 0.35,
    # Số câu giữ lại trong mỗi chunk sau sentence-level extract.
    "max_sents_per_chunk": 4,
    # Ngưỡng cosine cho sentence-level. Câu < ngưỡng này bị bỏ.
    "sent_threshold": 0.30,
    # MMR lambda — 0 = chỉ diversity, 1 = chỉ relevance. 0.7 = thiên relevance.
    "mmr_lambda": 0.7,
    # Số evidence card tối đa trả về cho LLM (mỗi card 1 chunk đã trim).
    "final_top_n": 3,
    # Ký tự tối đa mỗi card (sau khi trim) — guardrail chống nhồi prompt.
    "max_chars_per_card": 350,
}


# ===========================================================================
# MEDICAL SYNONYM EXPANSION (cheap, no LLM call)
# ===========================================================================

# Curated từ graph_seed.yaml + Huong-dan-chan-doan-va-dieu-tri.pdf
# Format: key (chuẩn) → list (các biến thể có thể có trong query VN không dấu, viết tắt, EN)
_MEDICAL_SYNONYMS = {
    # ---- Chỉ số xét nghiệm ----
    "tiểu cầu": ["tieu cau", "plt", "platelet", "platelets"],
    "hematocrit": ["hct", "ht", "ti le hct", "tỉ lệ hct"],
    "bạch cầu": ["bach cau", "wbc", "leukocyte"],
    "men gan": ["ast", "alt", "sgot", "sgpt", "transaminase"],
    "albumin": ["alb", "đạm máu"],
    "creatinin": ["creat", "creatinine"],

    # ---- Phân độ ----
    "sốt xuất huyết": ["sxhd", "sxh", "dengue", "df", "dhf"],
    "dấu hiệu cảnh báo": ["warning sign", "canh bao", "cảnh báo"],
    "sốc": ["soc", "shock", "dss", "dengue shock"],
    "xuất huyết nặng": ["xh nặng", "xuất huyết tiêu hóa", "xhth", "severe bleeding"],

    # ---- Triệu chứng / dấu hiệu ----
    "đau bụng": ["dau bung", "abdominal pain", "đau vùng gan"],
    "nôn": ["non", "non oi", "nôn ói", "vomit", "vomiting"],
    "gan to": ["gan lon", "hepatomegaly", "to gan"],
    "tay chân lạnh": ["chi lanh", "cold extremities", "ngoại biên lạnh"],
    "tràn dịch": ["tran dich", "effusion", "tdmp", "tdmb"],
    "lừ đừ": ["lu du", "vật vã", "lethargy", "rối loạn ý thức", "li bì"],

    # ---- Thuốc ----
    "paracetamol": ["pcm", "acetaminophen", "hạ sốt"],
    "aspirin": ["asa", "acetylsalicylic"],
    "ibuprofen": ["ibu", "nsaid", "nsaids"],
    "analgin": ["metamizole", "dipyrone"],

    # ---- Điều trị ----
    "bù dịch": ["bu dich", "fluid resuscitation", "truyền dịch", "ringer", "lactat", "nacl 0.9"],
    "cao phân tử": ["cpt", "colloid", "dextran", "haes", "hydroxyethyl"],

    # ---- Giai đoạn bệnh ----
    "ngày bệnh": ["ngay benh", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "fever day"],
    "giai đoạn nguy hiểm": ["critical phase", "n4-n6", "hết sốt"],
    "giai đoạn hồi phục": ["recovery", "tái hấp thu"],
}

# Reverse index: variant → canonical (lowercase, không dấu)
_REVERSE_SYN: dict = {}
for canon, variants in _MEDICAL_SYNONYMS.items():
    for v in variants + [canon]:
        _REVERSE_SYN[v.lower()] = canon


def _strip_diacritics(s: str) -> str:
    """Bỏ dấu tiếng Việt để match từ viết không dấu."""
    nf = unicodedata.normalize("NFD", s)
    return "".join(c for c in nf if not unicodedata.combining(c))


def expand_query(query: str, max_expansions: int = 6) -> str:
    """
    Mở rộng query bằng synonym y khoa. KHÔNG gọi LLM — tra dict.

    Ví dụ:
        "plt thấp có nguy hiểm không"
        → "plt thấp có nguy hiểm không tiểu cầu platelet"

    Trả về query GỐC + các canonical từ matched (không lặp).
    """
    if not query:
        return query

    q_low = query.lower()
    q_low_nodia = _strip_diacritics(q_low)
    added: List[str] = []

    for variant, canon in _REVERSE_SYN.items():
        if len(added) >= max_expansions:
            break
        # Match bằng word boundary trên cả bản có dấu và không dấu
        v_nodia = _strip_diacritics(variant)
        pattern = rf"\b{re.escape(v_nodia)}\b"
        if re.search(pattern, q_low_nodia) and canon not in q_low and canon not in added:
            added.append(canon)
            # Cũng thêm các synonym khác cùng nhóm để boost BM25
            other_variants = [v for v in _MEDICAL_SYNONYMS.get(canon, []) if v != variant]
            added.extend(other_variants[:2])

    if not added:
        return query
    expansion = " ".join(added[:max_expansions])
    expanded = f"{query} {expansion}"
    logger.debug("Query expanded: %r → %r", query, expanded)
    return expanded


# ===========================================================================
# EvidenceCard — đơn vị grounding gửi vào LLM
# ===========================================================================

@dataclass
class EvidenceCard:
    """1 thẻ bằng chứng đã trim, có citation."""
    text: str                # nội dung đã trim ở mức sentence
    citation: str            # "Huong-dan... trang 220, mục: 27.2.3.1"
    relevance: float = 0.0   # cosine sim với query (0-1)
    source_chunk_id: str = ""

    def to_block(self) -> str:
        return f"[Nguồn: {self.citation}] (rel={self.relevance:.2f})\n{self.text}"


# ===========================================================================
# Sentence splitter — VN-aware
# ===========================================================================

# Pattern tách câu: . ! ? ; theo sau là space + chữ cái / xuống dòng.
# Tránh tách ở số thập phân ("3.5") và viết tắt phổ biến ("vd.", "tr.", "Bs.").
_SENT_SPLIT_RE = re.compile(
    r"(?<=[\.!?;])\s+(?=[A-ZĐÊÔƠƯÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ])",
    re.UNICODE,
)


def split_sentences(text: str) -> List[str]:
    """Tách câu đơn giản, đủ dùng cho text guideline đã chunk."""
    text = (text or "").strip()
    if not text:
        return []
    # Trước split: gộp xuống dòng bullet thành câu
    text = re.sub(r"\n+\s*-\s*", ". ", text)
    text = re.sub(r"\n+", " ", text)
    parts = _SENT_SPLIT_RE.split(text)
    # Lọc rác (câu < 8 ký tự thường là số trang / heading rỗng)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


# ===========================================================================
# Rerank: bi-encoder (dùng lại embedder của rag_engine)
# ===========================================================================

def rerank_chunks(query_emb: np.ndarray,
                  chunks: List,             # List[DocumentChunk]
                  chunk_embs: np.ndarray,
                  threshold: float = CFG["rel_threshold"],
                  top_k: int = CFG["after_rerank_k"],
                  ) -> List[Tuple[object, float]]:
    """
    Rerank chunks theo cosine sim với query (cả 2 đã L2-normalized →
    inner product = cosine).

    Trả về list (chunk, score) đã sort giảm dần, đã filter < threshold.
    """
    if chunk_embs.size == 0 or len(chunks) == 0:
        return []
    sims = chunk_embs @ query_emb.reshape(-1)   # (n_chunks,)
    order = np.argsort(-sims)
    out: List[Tuple[object, float]] = []
    for idx in order:
        s = float(sims[idx])
        if s < threshold:
            break
        out.append((chunks[idx], s))
        if len(out) >= top_k:
            break
    return out


# ===========================================================================
# Sentence-level extraction: trim chunk còn 2-4 câu liên quan nhất
# ===========================================================================

def extract_relevant_sentences(query_emb: np.ndarray,
                               chunk_text: str,
                               embedder,                       # EmbeddingBackend
                               max_sents: int = CFG["max_sents_per_chunk"],
                               threshold: float = CFG["sent_threshold"],
                               max_chars: int = CFG["max_chars_per_card"],
                               ) -> Tuple[str, float]:
    """
    Từ 1 chunk dài, giữ lại 2-4 câu liên quan nhất tới query (giữ thứ tự gốc
    để giữ flow logic).

    Trả về (trimmed_text, avg_relevance).
    """
    sents = split_sentences(chunk_text)
    if not sents:
        return chunk_text[:max_chars], 0.0
    if len(sents) <= 2:
        # Chunk ngắn — pass nguyên, không cắt thêm
        return " ".join(sents)[:max_chars], 1.0

    # Embed câu (batch 1 lần)
    sent_embs = embedder.embed(sents, is_query=False)
    sims = sent_embs @ query_emb.reshape(-1)   # (n_sents,)

    # Lọc theo threshold, sau đó lấy top-k theo điểm
    keep_idx = [i for i, s in enumerate(sims) if s >= threshold]
    if not keep_idx:
        # Không có câu nào đạt — fallback giữ 2 câu top
        keep_idx = list(np.argsort(-sims)[:2])
    # Trong số đạt threshold, lấy top max_sents theo điểm
    keep_idx = sorted(keep_idx, key=lambda i: -sims[i])[:max_sents]
    # SẮP THEO THỨ TỰ GỐC để giữ flow câu chuyện
    keep_idx = sorted(keep_idx)

    selected = [sents[i] for i in keep_idx]
    avg = float(np.mean([sims[i] for i in keep_idx]))
    text = " ".join(selected)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text, avg


# ===========================================================================
# MMR (Maximal Marginal Relevance) — khử trùng ý giữa các chunk
# ===========================================================================

def mmr_select(query_emb: np.ndarray,
               candidates: List[Tuple[object, float, np.ndarray]],
               top_n: int = CFG["final_top_n"],
               lam: float = CFG["mmr_lambda"],
               ) -> List[Tuple[object, float]]:
    """
    MMR: score(d) = λ * sim(d, q) − (1−λ) * max sim(d, d_selected)

    candidates: list (chunk, relevance_to_query, chunk_embedding)
    Trả về list (chunk, mmr_score) sau khi chọn top_n.
    """
    if not candidates:
        return []
    selected: List[int] = []
    selected_embs: List[np.ndarray] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_n:
        best_i = None
        best_score = -1e9
        for i in remaining:
            chunk, rel, emb = candidates[i]
            if not selected_embs:
                redundancy = 0.0
            else:
                # max cosine sim với cluster đã chọn
                sims_to_sel = np.stack(selected_embs) @ emb.reshape(-1)
                redundancy = float(np.max(sims_to_sel))
            mmr = lam * rel - (1.0 - lam) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        selected_embs.append(candidates[best_i][2])
        remaining.remove(best_i)

    return [(candidates[i][0], candidates[i][1]) for i in selected]


# ===========================================================================
# Orchestrator — gọi từ chatbot_engine
# ===========================================================================

def build_evidence_cards(query: str,
                         top_n: int = CFG["final_top_n"],
                         expand: bool = True,
                         ) -> List[EvidenceCard]:
    """
    Pipeline đầy đủ: query → expand → hybrid retrieve → rerank → sentence trim
    → MMR → evidence cards.

    Trả về list ≤ top_n EvidenceCard. Có thể trả 0 nếu không có chunk nào đạt
    threshold liên quan (lúc đó LLM sẽ trả lời từ kiến thức nền — đúng hành vi
    medical: không bịa khi không có nguồn).
    """
    if not query or not query.strip():
        return []

    try:
        from rag_engine import get_rag_engine
        rag = get_rag_engine()
    except Exception as e:
        logger.warning("RAG engine không sẵn sàng: %s", e)
        return []

    embedder = rag.embedder

    # 1. Expand query (cheap)
    q_expanded = expand_query(query) if expand else query

    # 2. Embed query 1 lần dùng cho cả rerank + sentence trim
    q_emb = embedder.embed([q_expanded], is_query=True)[0]

    # 3. Hybrid retrieve nhiều candidate
    try:
        candidates = rag.retrieve(q_expanded,
                                  top_k=CFG["candidate_k"],
                                  candidate_k=CFG["candidate_k"] + 10)
    except Exception as e:
        logger.warning("Hybrid retrieve fail: %s", e)
        return []
    if not candidates:
        return []

    # 4. Lấy chunk embeddings từ index để rerank (đã có sẵn trong rag._embeddings)
    #    Map chunk → index bằng id() ko ổn — dùng chunk_id để tra
    chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(rag.chunks)}
    cand_embs = []
    for c in candidates:
        idx = chunk_id_to_idx.get(c.chunk_id)
        if idx is not None and rag._embeddings is not None:
            cand_embs.append(rag._embeddings[idx])
        else:
            cand_embs.append(None)
    # Lọc candidate không có emb
    pairs = [(c, e) for c, e in zip(candidates, cand_embs) if e is not None]
    if not pairs:
        return []
    cand_chunks = [p[0] for p in pairs]
    cand_emb_arr = np.stack([p[1] for p in pairs])

    # 5. Rerank theo cosine sim
    reranked = rerank_chunks(q_emb, cand_chunks, cand_emb_arr,
                             threshold=CFG["rel_threshold"],
                             top_k=CFG["after_rerank_k"])
    if not reranked:
        logger.info("Rerank: không có chunk nào đạt threshold %.2f cho query: %r",
                    CFG["rel_threshold"], query[:80])
        return []

    # 6. Sentence-level trim + chuẩn bị cho MMR
    mmr_candidates: List[Tuple[object, float, np.ndarray]] = []
    trimmed_map = {}   # chunk_id → trimmed text
    for chunk, rel in reranked:
        trimmed, avg_sent = extract_relevant_sentences(
            q_emb, chunk.text, embedder,
            max_sents=CFG["max_sents_per_chunk"],
            threshold=CFG["sent_threshold"],
            max_chars=CFG["max_chars_per_card"],
        )
        trimmed_map[chunk.chunk_id] = (trimmed, max(rel, avg_sent))
        # Lấy embedding chunk gốc cho MMR
        idx = chunk_id_to_idx[chunk.chunk_id]
        mmr_candidates.append((chunk, rel, rag._embeddings[idx]))

    # 7. MMR
    final = mmr_select(q_emb, mmr_candidates, top_n=top_n, lam=CFG["mmr_lambda"])

    # 8. Build EvidenceCard
    cards: List[EvidenceCard] = []
    for chunk, mmr_score in final:
        trimmed, rel = trimmed_map[chunk.chunk_id]
        cards.append(EvidenceCard(
            text=trimmed,
            citation=chunk.citation,
            relevance=rel,
            source_chunk_id=chunk.chunk_id,
        ))
    logger.info("Evidence cards: %d (từ %d candidates, query=%r)",
                len(cards), len(candidates), query[:60])
    return cards


def format_evidence_block(cards: List[EvidenceCard]) -> str:
    """
    Format list cards thành 1 text block sẵn để inject vào system prompt.
    Có header nhấn mạnh "chỉ dùng nếu liên quan" để hạn chế bot ngoài lề.
    """
    if not cards:
        return ""
    lines = [
        "## BẰNG CHỨNG TRÍCH TỪ HƯỚNG DẪN ĐIỀU TRỊ (Bộ Y tế VN)",
        "*(Chỉ dùng những thẻ thật sự trả lời câu hỏi. Mỗi thẻ đã được lọc theo độ liên quan; thẻ relevance < 0.5 chỉ tham khảo phụ.)*",
        "",
    ]
    for i, card in enumerate(cards, 1):
        lines.append(f"### Thẻ {i} — {card.citation}  (rel={card.relevance:.2f})")
        lines.append(card.text)
        lines.append("")
    return "\n".join(lines).rstrip()


# ===========================================================================
# CLI test
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) < 2:
        print("Cách dùng: python rag_optimizer.py 'câu hỏi'")
        sys.exit(1)
    q = sys.argv[1]
    print(f"=== Query: {q}")
    print(f"=== Expanded: {expand_query(q)}")
    cards = build_evidence_cards(q, top_n=3)
    print(f"\n=== {len(cards)} evidence cards ===\n")
    for i, c in enumerate(cards, 1):
        print(f"--- Card {i} (rel={c.relevance:.2f}) ---")
        print(f"Citation: {c.citation}")
        print(c.text)
        print()