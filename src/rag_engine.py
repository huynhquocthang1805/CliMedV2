"""
rag_engine.py — CliMedV2
========================

RAG engine cho PDF guideline điều trị — dùng cho chatbot Qwen.

Kiến trúc:
    PDF ──► SectionAwareChunker ──► [DocumentChunk]
                                         │
                            ┌────────────┴───────────┐
                            ▼                        ▼
                       BM25 (lexical)         Embedding + FAISS (semantic)
                            │                        │
                            └─── RRF fusion ─────────┘
                                       │
                               top-k chunks ──► LLM context

Cài đặt:
    pip install rank-bm25 faiss-cpu pdfplumber
    # Embedding backend (chọn 1):
    ollama pull nomic-embed-text         # khuyên dùng (đã có Ollama cho Qwen)
    # hoặc:
    pip install sentence-transformers     # fallback

Cách dùng:
    from rag_engine import get_rag_engine
    rag = get_rag_engine()
    chunks = rag.retrieve("tiểu cầu giảm dấu hiệu cảnh báo", top_k=5)
    for c in chunks:
        print(c.section_path, c.text[:200])
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
CACHE_DIR = KNOWLEDGE_DIR / ".cache"
CACHE_VERSION = "v1"  # bump khi thay chunking strategy → invalidate cache

DEFAULT_OLLAMA_HOST = os.getenv("CLIMED_OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_EMBED_MODEL = os.getenv("CLIMED_EMBED_MODEL", "nomic-embed-text")
DEFAULT_ST_MODEL = os.getenv("CLIMED_ST_MODEL", "intfloat/multilingual-e5-small")


# ===========================================================================
# DocumentChunk — đơn vị retrieval
# ===========================================================================

@dataclass
class DocumentChunk:
    """1 chunk văn bản kèm metadata để LLM trích nguồn."""
    chunk_id: str
    text: str
    source_file: str
    page: int
    section_path: List[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """Chuỗi trích nguồn ngắn cho LLM."""
        sec = " > ".join(self.section_path) if self.section_path else "—"
        return f"{self.source_file} (trang {self.page}, mục: {sec})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# Section-aware chunking
# ===========================================================================

# Patterns nhận diện heading
_HEADING_PATTERNS = [
    # "8.2.1." hoặc "1.1.1.1." (numbered heading)
    re.compile(r"^(\d+(?:\.\d+){0,4}\.?)\s+(.{3,80})$"),
    # "Phụ lục 23:" / "Phần A:" / "Chương 5:"
    re.compile(r"^(Phụ lục|Phần|Chương|PHỤ LỤC|PHẦN|CHƯƠNG)\s+[\dIVXivx]+\s*[:.]?\s*(.{3,80})$"),
    # ALL CAPS heading (>= 5 chars, all uppercase + space, không số)
    re.compile(r"^([A-ZĐÊÔƠƯÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ\s]{5,80})$"),
]

# Header/footer noise đặc trưng của guideline này (lặp mỗi trang)
_NOISE_LINES = {
    "TRUYỀN NHIỄM",
    "Truyền nhiễm",
}

# Footer page number đứng riêng dòng
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")


def _normalize_unicode(text: str) -> str:
    """NFKC normalize, fix Unicode kỳ lạ trong PDF cũ."""
    return unicodedata.normalize("NFKC", text or "")


def _detect_heading(line: str) -> Optional[Tuple[int, str]]:
    """
    Detect 1 dòng có phải heading không. Trả về (level, title) hoặc None.

    Level:
      1 = "Phần/Chương/Phụ lục" (top)
      2 = ALL CAPS heading
      3+ = "1." (3), "1.1." (4), "1.1.1." (5)...
    """
    line = line.strip()
    if not line or len(line) > 100:
        return None

    # Numbered: "8.2.1. Điều trị cụ thể"
    m = _HEADING_PATTERNS[0].match(line)
    if m:
        num = m.group(1).rstrip(".")
        depth = num.count(".") + 1  # "8" → depth 1, "8.2" → 2, "8.2.1" → 3
        return (2 + depth, line)  # offset để < ALL CAPS heading

    # Phần/Chương/Phụ lục
    m = _HEADING_PATTERNS[1].match(line)
    if m:
        return (1, line)

    # ALL CAPS (chú ý: dễ false positive với các viết tắt y khoa)
    m = _HEADING_PATTERNS[2].match(line)
    if m and not any(c.isdigit() for c in line):
        # heuristic: heading thật thường có >= 2 từ
        if len(line.split()) >= 2:
            return (2, line)

    return None


def _is_noise(line: str) -> bool:
    """Bỏ header lặp + page number đứng riêng dòng."""
    s = line.strip()
    if s in _NOISE_LINES:
        return True
    if _PAGE_NUMBER_RE.match(s):
        return True
    return False


class SectionAwareChunker:
    """
    Chunker giữ context section.

    Mỗi chunk có `section_path` = stack heading từ rễ tới hiện tại.
    Chunk size soft 800 chars, break tại dấu câu gần nhất, max 1500.
    """

    def __init__(self, target_size: int = 800, max_size: int = 1500):
        self.target_size = target_size
        self.max_size = max_size

    def chunk_pages(
        self,
        pages: List[Tuple[int, str]],
        source_file: str,
        min_chunk_chars: int = 150,
    ) -> List[DocumentChunk]:
        """
        pages: List[(page_number, page_text)]
        min_chunk_chars: bỏ chunk ngắn hơn ngưỡng này (lọc heading rỗng).
        Trả về list chunks tuần tự theo thứ tự đọc.
        """
        # heading_stack[level] = title; cao hơn level hiện tại sẽ bị clear khi gặp heading mới
        heading_stack: Dict[int, str] = {}
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        # Buffer text + page tracking cho chunk đang build
        buf_lines: List[str] = []
        buf_start_page: Optional[int] = None
        buf_section_path: List[str] = []

        def buf_size() -> int:
            return sum(len(l) for l in buf_lines) + max(0, len(buf_lines) - 1)

        def flush():
            nonlocal chunk_idx, buf_lines, buf_start_page
            if not buf_lines:
                return
            text = "\n".join(buf_lines).strip()
            # Capture page TRƯỚC khi reset (BUG cũ: reset rồi mới đọc)
            page_for_chunk = buf_start_page or 1
            buf_lines = []
            buf_start_page = None
            if not text or len(text) < min_chunk_chars:
                # chunk quá ngắn (chỉ heading rỗng, hoặc trang trắng) → bỏ
                return
            chunk_idx += 1
            chunks.append(DocumentChunk(
                chunk_id=f"{Path(source_file).stem}_c{chunk_idx:05d}",
                text=text,
                source_file=source_file,
                page=page_for_chunk,
                section_path=list(buf_section_path),
            ))

        for page_num, page_text in pages:
            page_text = _normalize_unicode(page_text)
            for raw_line in page_text.split("\n"):
                line = raw_line.strip()
                if not line or _is_noise(line):
                    continue

                # Detect heading
                heading = _detect_heading(line)
                if heading is not None:
                    level, title = heading

                    # CHỈ flush khi buffer đã đủ "ý nghĩa" (>= 50% target).
                    if buf_size() >= self.target_size // 2:
                        flush()

                    # Heading top-level (Chương/Phần/Phụ lục/ALL CAPS heading)
                    # → reset hoàn toàn stack để section_path không bị "rò"
                    # qua các chương khác.
                    if level <= 2:
                        heading_stack.clear()
                    else:
                        keys_to_remove = [k for k in heading_stack if k >= level]
                        for k in keys_to_remove:
                            del heading_stack[k]
                    heading_stack[level] = title

                    # Cap section_path = 3 heading gần nhất (deepest)
                    sorted_keys = sorted(heading_stack.keys())
                    buf_section_path = [
                        heading_stack[k] for k in sorted_keys[-3:]
                    ]
                    if buf_start_page is None:
                        buf_start_page = page_num
                    buf_lines.append(line)
                    continue

                # Regular line — add vào buffer
                if buf_start_page is None:
                    buf_start_page = page_num
                buf_lines.append(line)

                # Soft size limit: flush nếu vượt target VÀ kết thúc câu
                cur = buf_size()
                if cur >= self.max_size:
                    flush()
                elif cur >= self.target_size and line.endswith(
                        (".", "!", "?", ":", ";")):
                    flush()

        flush()
        return chunks

    def chunk_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        """Đọc PDF và chunk. Dùng pdfplumber (đã có trong project)."""
        import pdfplumber
        pages: List[Tuple[int, str]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                pages.append((i, txt))
        return self.chunk_pages(pages, source_file=pdf_path.name)


# ===========================================================================
# Embedding backends
# ===========================================================================

class EmbeddingBackend(ABC):
    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Trả về np.ndarray shape (len(texts), dim). L2-normalized."""
        ...


class OllamaEmbedding(EmbeddingBackend):
    """Embed qua Ollama API (recommended — đã có cho Qwen)."""

    name = "ollama"

    def __init__(self,
                 model: str = DEFAULT_OLLAMA_EMBED_MODEL,
                 host: str = DEFAULT_OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")
        self._dim_cached: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim_cached is None:
            # Probe để biết dim
            v = self.embed(["probe"])[0]
            self._dim_cached = len(v)
        return self._dim_cached

    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=2.0)
            if r.status_code != 200:
                return False
            tags = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in t for t in tags):
                logger.warning("Ollama đang chạy nhưng chưa pull '%s'. "
                               "Chạy: ollama pull %s",
                               self.model, self.model)
                return False
            return True
        except Exception:
            return False

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        import requests
        vectors: List[List[float]] = []
        for txt in texts:
            r = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": txt},
                timeout=60.0,
            )
            r.raise_for_status()
            vectors.append(r.json()["embedding"])
        arr = np.array(vectors, dtype=np.float32)
        # L2 normalize để dùng cosine = inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms


class SentenceTransformerEmbedding(EmbeddingBackend):
    """Fallback nếu user không có Ollama."""

    name = "sentence-transformers"

    def __init__(self, model: str = DEFAULT_ST_MODEL):
        self.model_name = model
        self._model = None
        # multilingual-e5 yêu cầu prefix "query: " và "passage: "
        self._needs_e5_prefix = "e5" in model.lower()

    def is_available(self) -> bool:
        try:
            import sentence_transformers  # noqa
            return True
        except ImportError:
            return False

    @property
    def dim(self) -> int:
        self._load()
        return int(self._model.get_sentence_embedding_dimension())

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        self._load()
        if self._needs_e5_prefix:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        arr = self._model.encode(
            texts, normalize_embeddings=True,
            batch_size=32, show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)


def get_embedding_backend(prefer: str = "auto") -> EmbeddingBackend:
    """auto: ưu tiên Ollama, fallback sentence-transformers."""
    if prefer == "ollama":
        b = OllamaEmbedding()
        if not b.is_available():
            raise RuntimeError(
                f"Ollama embedding chưa sẵn sàng. "
                f"Chạy: ollama pull {DEFAULT_OLLAMA_EMBED_MODEL}"
            )
        return b
    if prefer in ("st", "sentence-transformers"):
        b = SentenceTransformerEmbedding()
        if not b.is_available():
            raise RuntimeError("sentence-transformers chưa cài.")
        return b
    if prefer == "auto":
        ollama = OllamaEmbedding()
        if ollama.is_available():
            logger.info("Embedding backend: Ollama (%s)", ollama.model)
            return ollama
        st = SentenceTransformerEmbedding()
        if st.is_available():
            logger.info("Embedding backend: sentence-transformers (%s)",
                        st.model_name)
            return st
        raise RuntimeError(
            "Không có embedding backend nào.\n"
            "Lựa chọn:\n"
            f"  A) ollama pull {DEFAULT_OLLAMA_EMBED_MODEL}\n"
            "  B) pip install sentence-transformers"
        )
    raise ValueError(f"Unknown backend: {prefer!r}")


# ===========================================================================
# Hybrid retriever
# ===========================================================================

def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize đơn giản cho BM25 (whitespace + lowercase + alphanum)."""
    text = _normalize_unicode(text).lower()
    # Giữ chữ Việt + số, bỏ punctuation
    return re.findall(r"\w+", text, flags=re.UNICODE)


class HybridRetriever:
    """
    BM25 + Dense (FAISS) với Reciprocal Rank Fusion.

    RRF công thức: score(d) = Σ 1 / (k + rank_i(d))
    với rank_i là rank của d theo retriever i. Default k=60 (paper ban đầu).
    """

    def __init__(self,
                 embedder: EmbeddingBackend,
                 chunks: List[DocumentChunk]):
        self.embedder = embedder
        self.chunks = chunks
        self._bm25 = None
        self._faiss = None
        self._embeddings: Optional[np.ndarray] = None

    # ----- Build / load -----

    def build_index(self) -> None:
        """Build cả BM25 và FAISS index từ chunks."""
        from rank_bm25 import BM25Okapi
        import faiss

        if not self.chunks:
            raise ValueError("No chunks to index.")

        logger.info("Tokenize %d chunks cho BM25...", len(self.chunks))
        tokenized = [_tokenize_for_bm25(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

        logger.info("Embedding %d chunks (có thể mất vài phút lần đầu)...",
                    len(self.chunks))
        # Batch để không gọi API quá nhiều lần
        embs: List[np.ndarray] = []
        BATCH = 64
        for i in range(0, len(self.chunks), BATCH):
            batch = [c.text for c in self.chunks[i:i + BATCH]]
            embs.append(self.embedder.embed(batch, is_query=False))
            if (i // BATCH) % 10 == 0:
                logger.info("  ... %d/%d", i, len(self.chunks))
        self._embeddings = np.vstack(embs).astype(np.float32)

        # FAISS inner-product index (L2 normalized → IP = cosine)
        d = self._embeddings.shape[1]
        self._faiss = faiss.IndexFlatIP(d)
        self._faiss.add(self._embeddings)
        logger.info("Indexed: BM25 + FAISS (dim=%d)", d)

    def save_to_disk(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Pickle BM25 (rank-bm25 object là pickle-able)
        with open(cache_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)
        # Save embeddings + chunks
        np.save(cache_dir / "embeddings.npy", self._embeddings)
        with open(cache_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        meta = {
            "version": CACHE_VERSION,
            "embedder": self.embedder.name,
            "embedder_model": getattr(self.embedder, "model",
                                      getattr(self.embedder, "model_name", "?")),
            "n_chunks": len(self.chunks),
            "dim": self._embeddings.shape[1],
        }
        with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("Cache saved to %s", cache_dir)

    @classmethod
    def load_from_disk(cls,
                       cache_dir: Path,
                       embedder: EmbeddingBackend,
                       expected_n_chunks: Optional[int] = None
                       ) -> Optional["HybridRetriever"]:
        """Load nếu cache valid + match. Trả về None nếu cần rebuild."""
        try:
            meta_file = cache_dir / "meta.json"
            if not meta_file.exists():
                return None
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("version") != CACHE_VERSION:
                logger.info("Cache version mismatch — rebuild")
                return None
            if meta.get("embedder") != embedder.name:
                logger.info("Embedder backend changed — rebuild")
                return None
            if expected_n_chunks and meta.get("n_chunks") != expected_n_chunks:
                logger.info("Chunk count changed — rebuild")
                return None

            # Load chunks
            chunks: List[DocumentChunk] = []
            with open(cache_dir / "chunks.jsonl", encoding="utf-8") as f:
                for line in f:
                    chunks.append(DocumentChunk(**json.loads(line)))

            inst = cls(embedder=embedder, chunks=chunks)
            with open(cache_dir / "bm25.pkl", "rb") as f:
                inst._bm25 = pickle.load(f)
            inst._embeddings = np.load(cache_dir / "embeddings.npy")

            import faiss
            inst._faiss = faiss.IndexFlatIP(inst._embeddings.shape[1])
            inst._faiss.add(inst._embeddings)

            logger.info("Cache loaded: %d chunks", len(chunks))
            return inst
        except Exception as e:
            logger.warning("Cache load failed: %s — rebuild", e)
            return None

    # ----- Retrieve -----

    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 candidate_k: int = 30,
                 rrf_k: int = 60) -> List[DocumentChunk]:
        """
        Hybrid retrieve với RRF fusion.

        candidate_k: lấy nhiêu candidate từ mỗi retriever trước khi fuse.
        """
        if self._bm25 is None or self._faiss is None:
            raise RuntimeError("Index chưa build. Gọi build_index() trước.")

        # 1. BM25
        bm25_scores = self._bm25.get_scores(_tokenize_for_bm25(query))
        bm25_top = np.argsort(-bm25_scores)[:candidate_k]

        # 2. Dense
        q_emb = self.embedder.embed([query], is_query=True)
        D, I = self._faiss.search(q_emb, candidate_k)
        dense_top = I[0]

        # 3. RRF fusion
        scores: Dict[int, float] = {}
        for rank, idx in enumerate(bm25_top):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank)
        for rank, idx in enumerate(dense_top):
            if idx == -1:  # FAISS empty slot
                continue
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank)

        # 4. Sort và trả top-k
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [self.chunks[idx] for idx, _ in ranked]


# ===========================================================================
# Factory: build hoặc load RAG cho thư mục knowledge
# ===========================================================================

def build_or_load_rag(force_rebuild: bool = False,
                      embed_backend: str = "auto") -> HybridRetriever:
    """
    Entry chính:
      1. Init embedding backend (Ollama/ST)
      2. Thử load cache
      3. Nếu fail → chunk PDF + build index + save cache
    """
    embedder = get_embedding_backend(embed_backend)

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(f"Thư mục {KNOWLEDGE_DIR} không tồn tại.")

    # Liệt kê PDF/text trong knowledge dir
    pdf_files = sorted([f for f in KNOWLEDGE_DIR.iterdir()
                        if f.suffix.lower() in (".pdf", ".txt")])
    if not pdf_files:
        raise FileNotFoundError(
            f"Không có PDF nào trong {KNOWLEDGE_DIR}"
        )

    # Chunk hết để biết n_chunks expected (cho cache validation)
    chunker = SectionAwareChunker()
    all_chunks: List[DocumentChunk] = []
    for f in pdf_files:
        if f.suffix.lower() == ".pdf":
            logger.info("Chunking PDF: %s", f.name)
            all_chunks.extend(chunker.chunk_pdf(f))
        else:  # txt
            text = f.read_text(encoding="utf-8", errors="ignore")
            all_chunks.extend(chunker.chunk_pages(
                [(1, text)], source_file=f.name))
    logger.info("Total chunks: %d", len(all_chunks))

    # Try cache
    if not force_rebuild:
        cached = HybridRetriever.load_from_disk(
            CACHE_DIR, embedder=embedder,
            expected_n_chunks=len(all_chunks),
        )
        if cached is not None:
            return cached

    # Build fresh
    retriever = HybridRetriever(embedder=embedder, chunks=all_chunks)
    retriever.build_index()
    retriever.save_to_disk(CACHE_DIR)
    return retriever


# Module-level singleton (lazy)
_RAG_SINGLETON: Optional[HybridRetriever] = None


def get_rag_engine(force_rebuild: bool = False,
                   embed_backend: str = "auto") -> HybridRetriever:
    """Singleton accessor — chỉ build 1 lần per process."""
    global _RAG_SINGLETON
    if _RAG_SINGLETON is None or force_rebuild:
        _RAG_SINGLETON = build_or_load_rag(
            force_rebuild=force_rebuild,
            embed_backend=embed_backend,
        )
    return _RAG_SINGLETON


def retrieve_for_chat(query: str, top_k: int = 5) -> List[str]:
    """
    Convenience function cho chatbot_engine: retrieve và format thành list
    string ready-to-inject vào prompt.
    """
    try:
        rag = get_rag_engine()
    except Exception as e:
        logger.warning("RAG unavailable: %s", e)
        return []

    chunks = rag.retrieve(query, top_k=top_k)
    snippets: List[str] = []
    for c in chunks:
        sec = " > ".join(c.section_path) if c.section_path else ""
        header = f"[Nguồn: {c.source_file}, trang {c.page}"
        if sec:
            header += f", mục: {sec}"
        header += "]"
        snippets.append(f"{header}\n{c.text}")
    return snippets


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 2:
        print("Cách dùng:")
        print("  python rag_engine.py build              # build/load index")
        print("  python rag_engine.py query 'câu hỏi'    # test retrieval")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "build":
        rag = get_rag_engine(force_rebuild=True)
        print(f"✅ Đã build {len(rag.chunks)} chunks")
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Thiếu query")
            sys.exit(1)
        q = sys.argv[2]
        rag = get_rag_engine()
        results = rag.retrieve(q, top_k=5)
        print(f"\n=== Top {len(results)} kết quả cho: '{q}' ===\n")
        for i, c in enumerate(results, 1):
            sec = " > ".join(c.section_path) if c.section_path else "—"
            print(f"--- [{i}] {c.source_file} trang {c.page} ---")
            print(f"Section: {sec}")
            print(c.text[:400])
            print()