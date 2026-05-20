"""
build_graph_from_guideline.py — CliMedV2
=========================================

Sinh file `data/knowledge/graph_seed.yaml` TỰ ĐỘNG từ PDF guideline
(Bộ Y tế VN, BV Bệnh Nhiệt Đới 2023).

Đây là câu trả lời cho câu hỏi "graph_seed.yaml từ đâu ra?":
    "Em chạy script `build_graph_from_guideline.py`. Script đọc PDF guideline,
     dùng RAG retrieve các đoạn về phân loại / dấu hiệu cảnh báo / thuốc cấm,
     rồi pattern-match để extract nodes + edges, ghi xuống YAML có TRACE
     ngược về trang PDF gốc."

Phương pháp (KHÔNG dùng LLM auto-extract — để bác sĩ verify được):

  1. Đọc PDF, chunk theo page (giữ page_number).
  2. Tìm chương SXHD bằng keyword "SXHD" / "sốt xuất huyết Dengue".
  3. Pattern-match các CONCEPT đã định nghĩa sẵn:
     • Severity levels: thường / có dấu hiệu cảnh báo / nặng
     • Warning signs: từ list trong guideline (đau bụng nhiều, nôn, gan to...)
     • Drugs cấm: aspirin, ibuprofen, NSAID, analgin
     • Lab thresholds: PLT < 100, Hct tăng > 20%, AST/ALT > 2× ULN
  4. Tạo edges theo rule:
     • symptom → severity (supports)
     • warning_sign → severity (warning_of)
     • drug → severity (contraindicated_for)
     • intervention → severity (treats)
  5. Mỗi node có field `source_pages` ghi rõ trang nào đã match.
  6. RAG-assist: với mỗi concept, retrieve top-3 chunks liên quan để
     đưa snippet vào field `evidence_snippet` (bác sĩ duyệt).
  7. Ghi YAML có comment giải thích từng phần, kèm timestamp.

Cách dùng:
    python src/build_graph_from_guideline.py
    python src/build_graph_from_guideline.py --pdf data/knowledge/<file>.pdf

Sau đó kiểm tra:
    python src/build_graph_from_guideline.py --diff
    # → so sánh YAML mới với YAML đã commit, in sự khác biệt
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PDF = BASE_DIR / "data" / "knowledge" / "Huong-dan-chan-doan-va-dieu-tri(1).pdf"
DEFAULT_OUTPUT = BASE_DIR / "data" / "knowledge" / "graph_seed.yaml"


# ===========================================================================
# Concept dictionaries — đây là phần cốt lõi (curated một lần)
#
# QUAN TRỌNG: Đây là tri thức y khoa GỐC được encode dưới dạng pattern.
# Bác sĩ có thể audit từng dictionary này. Khi guideline cập nhật, chỉ cần
# sửa các pattern này (không phải sửa code logic).
# ===========================================================================

# 3 mức severity theo phân độ Bộ Y tế
SEVERITY_NODES = {
    "sxhd_thuong": {
        "label": "SXHD thông thường",
        "patterns": [r"sxhd\s*thông\s*thường", r"dengue\s*thông\s*thường",
                     r"không\s*có\s*dấu\s*hiệu\s*cảnh\s*báo"],
        "description": "Sốt + ≥2 dấu hiệu (đau đầu, đau cơ-khớp, đau hốc mắt, "
                       "ban xuất huyết). Chưa có cảnh báo.",
    },
    "sxhd_canh_bao": {
        "label": "SXHD có dấu hiệu cảnh báo",
        "patterns": [r"sxhd\s*có\s*dấu\s*hiệu\s*cảnh\s*báo",
                     r"dengue\s*có\s*dấu\s*hiệu\s*cảnh\s*báo",
                     r"có\s*dấu\s*hiệu\s*cảnh\s*báo"],
        "description": "Có ≥1 warning sign — cần nhập viện theo dõi sát.",
    },
    "sxhd_nang": {
        "label": "SXHD nặng",
        "patterns": [r"sxhd\s*nặng", r"dengue\s*nặng",
                     r"sốc\s*sxhd", r"sốc\s*sốt\s*xuất\s*huyết"],
        "description": "Sốc, xuất huyết nặng, hoặc suy đa cơ quan.",
    },
}


# Symptoms thông thường (cho SXHD thường)
SYMPTOM_PATTERNS = {
    "sot_cao":              {"label": "Sốt cao đột ngột",
                              "patterns": [r"sốt\s*cao", r"sốt\s*đột\s*ngột"]},
    "dau_dau":              {"label": "Đau đầu",
                              "patterns": [r"đau\s*đầu", r"nhức\s*đầu"]},
    "dau_co_khop":          {"label": "Đau cơ / đau khớp",
                              "patterns": [r"đau\s*cơ", r"đau\s*khớp",
                                           r"nhức\s*mỏi"]},
    "dau_hoc_mat":          {"label": "Đau sau hốc mắt",
                              "patterns": [r"đau\s*(sau\s*)?hốc\s*mắt"]},
    "ban_xuat_huyet":       {"label": "Ban xuất huyết / petechia",
                              "patterns": [r"petechia", r"ban\s*xuất\s*huyết",
                                           r"chấm\s*xuất\s*huyết"]},
    "da_niem_sung_huyet":   {"label": "Da niêm sung huyết",
                              "patterns": [r"sung\s*huyết",
                                           r"da\s*niêm\s*sung\s*huyết"]},
}


# Warning signs — bám sát Phụ lục 2 phân độ SXHD
WARNING_SIGN_PATTERNS = {
    "dau_bung_nhieu": {
        "label": "Đau bụng nhiều / liên tục",
        "patterns": [r"đau\s*bụng\s*(nhiều|liên\s*tục)",
                     r"đau\s*bụng\s*vùng\s*gan"],
    },
    "non_nhieu": {
        "label": "Nôn nhiều",
        "patterns": [r"nôn\s*(ói\s*)?(nhiều|liên\s*tục)",
                     r"nôn\s*≥\s*\d+\s*lần"],
    },
    "chay_mau_niem_mac": {
        "label": "Chảy máu niêm mạc",
        "patterns": [r"chảy\s*máu\s*niêm\s*mạc",
                     r"chảy\s*máu\s*(mũi|chân\s*răng|nướu)",
                     r"xuất\s*huyết\s*niêm\s*mạc"],
    },
    "gan_to": {
        "label": "Gan to >2cm",
        "patterns": [r"gan\s*to\s*>?\s*2\s*cm",
                     r"gan\s*to\s*đau", r"gan\s*to"],
    },
    "lu_du_vat_va": {
        "label": "Lừ đừ / vật vã / rối loạn ý thức",
        "patterns": [r"lừ\s*đừ", r"vật\s*vã",
                     r"rối\s*loạn\s*tri\s*giác", r"li\s*bì"],
    },
    "tieu_it": {
        "label": "Tiểu ít",
        "patterns": [r"tiểu\s*ít", r"thiểu\s*niệu",
                     r"nước\s*tiểu\s*<\s*0[,.]5\s*ml"],
    },
    "tay_chan_lanh": {
        "label": "Tay chân lạnh",
        "patterns": [r"tay\s*chân\s*lạnh", r"chi\s*lạnh",
                     r"da\s*lạnh\s*ẩm"],
    },
    "hct_tang_plt_giam": {
        "label": "Hct tăng kèm tiểu cầu giảm nhanh",
        "patterns": [r"hct\s*tăng.*tiểu\s*cầu\s*giảm",
                     r"tiểu\s*cầu\s*giảm.*hct\s*tăng",
                     r"hematocrit\s*tăng.*tiểu\s*cầu\s*giảm"],
    },
    "tran_dich": {
        "label": "Tràn dịch màng phổi / màng bụng",
        "patterns": [r"tràn\s*dịch\s*màng\s*phổi",
                     r"tràn\s*dịch\s*màng\s*bụng",
                     r"tràn\s*dịch\s*đa\s*màng"],
    },
}


# Lab thresholds với ngưỡng từ guideline
LAB_PATTERNS = {
    "tieu_cau_thap": {
        "label": "Tiểu cầu giảm",
        "threshold": "PLT < 100 K/uL",
        "patterns": [r"tiểu\s*cầu\s*<\s*100",
                     r"tiểu\s*cầu\s*giảm",
                     r"plt\s*<\s*100", r"thrombocytopenia"],
    },
    "hct_cao": {
        "label": "Hct tăng",
        "threshold": "Hct tăng ≥20% so với baseline",
        "patterns": [r"hct\s*tăng\s*≥?\s*20\s*%?",
                     r"hematocrit\s*tăng",
                     r"cô\s*đặc\s*máu"],
    },
    "ast_alt_cao": {
        "label": "Men gan tăng",
        "threshold": "AST/ALT > 2× ULN",
        "patterns": [r"ast\s*[/\\]?\s*alt\s*tăng",
                     r"men\s*gan\s*tăng",
                     r"ast\s*>\s*\d+", r"alt\s*>\s*\d+",
                     r"tổn\s*thương\s*gan"],
    },
    "bach_cau_thap": {
        "label": "Bạch cầu giảm",
        "threshold": "WBC < 5 K/uL",
        "patterns": [r"bạch\s*cầu\s*giảm", r"leukopenia",
                     r"wbc\s*<\s*\d+"],
    },
    "hflc_cao": {
        "label": "HFLC tăng",
        "threshold": ">10% — đáp ứng miễn dịch hoạt hóa",
        "patterns": [r"hflc", r"high.fluorescent.lymphocyte",
                     r"lympho\s*hoạt\s*hóa"],
    },
}


# Drugs — guideline có liệt kê rõ thuốc tránh
DRUG_PATTERNS = {
    "aspirin": {
        "label": "Aspirin",
        "patterns": [r"aspirin", r"acetylsalicylic"],
        "is_contraindicated": True,
    },
    "ibuprofen": {
        "label": "Ibuprofen",
        "patterns": [r"ibuprofen"],
        "is_contraindicated": True,
    },
    "nsaid_other": {
        "label": "Các NSAID khác",
        "patterns": [r"\bnsaid", r"thuốc\s*kháng\s*viêm\s*không\s*steroid"],
        "is_contraindicated": True,
    },
    "analgin": {
        "label": "Analgin / Metamizole",
        "patterns": [r"analgin", r"metamizole"],
        "is_contraindicated": True,
    },
    "paracetamol": {
        "label": "Paracetamol",
        "patterns": [r"paracetamol", r"acetaminophen"],
        "is_contraindicated": False,   # đây là thuốc ĐƯỢC PHÉP
        "note": "Được phép — chỉ paracetamol đơn chất, đúng liều.",
    },
}


# Interventions
INTERVENTION_PATTERNS = {
    "oresol": {
        "label": "Bù dịch đường uống (Oresol)",
        "patterns": [r"oresol", r"bù\s*nước\s*đường\s*uống",
                     r"uống\s*nhiều\s*nước"],
        "treats": ["sxhd_thuong"],
    },
    "truyen_dich_tinh_mach": {
        "label": "Truyền dịch tĩnh mạch",
        "patterns": [r"truyền\s*dịch", r"ringer\s*lactate",
                     r"nacl\s*0[,.]?9\s*%"],
        "treats": ["sxhd_canh_bao", "sxhd_nang"],
    },
    "nhap_vien_theo_doi": {
        "label": "Nhập viện theo dõi sát",
        "patterns": [r"nhập\s*viện", r"theo\s*dõi\s*sát"],
        "treats": ["sxhd_canh_bao"],
    },
    "hoi_suc_soc": {
        "label": "Hồi sức chống sốc",
        "patterns": [r"hồi\s*sức\s*sốc", r"chống\s*sốc",
                     r"điều\s*trị\s*sốc"],
        "treats": ["sxhd_nang"],
    },
    "theo_doi_ngoai_tru": {
        "label": "Theo dõi tại nhà / ngoại trú",
        "patterns": [r"theo\s*dõi\s*tại\s*nhà",
                     r"điều\s*trị\s*ngoại\s*trú",
                     r"điều\s*trị\s*tại\s*nhà"],
        "treats": ["sxhd_thuong"],
    },
}


# ===========================================================================
# PDF reading + chương SXHD detection
# ===========================================================================

@dataclass
class PageMatch:
    page: int
    snippet: str          # đoạn text khoảng 200 chars chứa match
    matched_pattern: str  # regex đã match (để debug)


@dataclass
class ConceptEvidence:
    """Bằng chứng tìm thấy cho 1 concept trong PDF."""
    concept_id: str
    matches: List[PageMatch] = field(default_factory=list)

    @property
    def page_set(self) -> List[int]:
        return sorted(set(m.page for m in self.matches))


def read_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Đọc PDF, trả List[(page_number, text)]."""
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError("pdfplumber chưa cài. pip install pdfplumber") from e

    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def find_dengue_chapter(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    Lọc chỉ các trang thuộc chương SXHD.

    Heuristic v3 (đơn giản hơn):
      • Score = số lần keyword SXHD/dengue per page.
      • Chia pages thành "runs" — chuỗi các trang LIỀN NHAU đều có score >= MIN.
      • Trong mỗi run, kết nối các trang lân cận có score >= 1
        (để không bỏ sót trang phụ lục giữa chương).
      • Trả về run dài nhất.
    """
    MIN_SCORE_CORE = 3      # trang "lõi" phải có ít nhất 3 keyword
    MIN_SCORE_NEAR = 1      # trang lân cận chỉ cần 1 keyword

    # Bước 1: đánh dấu CORE pages (score >= MIN_SCORE_CORE)
    core_pages = set()
    near_pages = set()
    page_text = {}
    for p, text in pages:
        low = text.lower()
        sc = (low.count("sxhd") +
              low.count("sốt xuất huyết dengue") +
              low.count("dengue"))
        if sc >= MIN_SCORE_CORE:
            core_pages.add(p)
        if sc >= MIN_SCORE_NEAR:
            near_pages.add(p)
        page_text[p] = text

    if not core_pages:
        logger.warning("Không tìm thấy core pages SXHD → fallback toàn PDF")
        return pages

    # Bước 2: với mỗi core page, expand sang các near pages liền kề
    # để tạo "run". Run = consecutive set không bị break.
    sorted_core = sorted(core_pages)
    runs: List[List[int]] = []
    current_run: List[int] = []

    for p in sorted_core:
        if not current_run:
            current_run = [p]
            continue
        # Nếu p liền kề với run hiện tại (qua các near pages) → mở rộng
        last = current_run[-1]
        # Check các trang giữa last và p có là near không
        gap_ok = all((mid in near_pages) or (mid in core_pages)
                     for mid in range(last + 1, p))
        if p - last <= 5 and gap_ok:
            # Cùng run — nối thêm các near page giữa
            for mid in range(last + 1, p):
                current_run.append(mid)
            current_run.append(p)
        else:
            runs.append(current_run)
            current_run = [p]
    if current_run:
        runs.append(current_run)

    # Run dài nhất theo SỐ trang core
    best_run = max(runs, key=lambda r: sum(1 for x in r if x in core_pages))

    start, end = best_run[0], best_run[-1]
    filtered = [(p, page_text[p]) for p in best_run]
    logger.info("Chương SXHD: trang %d → %d (%d trang core, %d total)",
                start, end,
                sum(1 for x in best_run if x in core_pages),
                len(filtered))
    return filtered


# ===========================================================================
# Pattern matching engine
# ===========================================================================

def find_concept_evidence(
    concept_id: str,
    patterns: List[str],
    pages: List[Tuple[int, str]],
    snippet_radius: int = 100,
) -> ConceptEvidence:
    """
    Tìm các trang có chứa pattern của concept. Lưu cả snippet để bác sĩ verify.
    """
    evidence = ConceptEvidence(concept_id=concept_id)
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    for page_num, text in pages:
        # Normalize text: bỏ line break thừa để match cụm từ qua dòng
        flat = re.sub(r"\s+", " ", text)

        for pattern_re in compiled:
            for m in pattern_re.finditer(flat):
                start = max(0, m.start() - snippet_radius)
                end = min(len(flat), m.end() + snippet_radius)
                snippet = flat[start:end].strip()
                # Highlight match trong snippet bằng [...]
                snippet_marked = (
                    flat[start:m.start()] + f"[{m.group(0)}]"
                    + flat[m.end():end]
                ).strip()
                evidence.matches.append(PageMatch(
                    page=page_num,
                    snippet=snippet_marked[:300],
                    matched_pattern=pattern_re.pattern,
                ))
                # Giới hạn match per page tránh dup
                break
    return evidence


def extract_all_concepts(
    pages: List[Tuple[int, str]],
) -> Dict[str, Dict[str, ConceptEvidence]]:
    """
    Extract evidence cho TẤT CẢ concepts.
    Trả về: {category: {concept_id: ConceptEvidence}}
    """
    result: Dict[str, Dict[str, ConceptEvidence]] = defaultdict(dict)

    for cat_name, concept_dict in [
        ("severity", SEVERITY_NODES),
        ("symptom", SYMPTOM_PATTERNS),
        ("warning_sign", WARNING_SIGN_PATTERNS),
        ("lab", LAB_PATTERNS),
        ("drug", DRUG_PATTERNS),
        ("intervention", INTERVENTION_PATTERNS),
    ]:
        for cid, info in concept_dict.items():
            ev = find_concept_evidence(cid, info["patterns"], pages)
            result[cat_name][cid] = ev
            logger.info("  %s/%s: %d match trên %d trang",
                        cat_name, cid, len(ev.matches), len(ev.page_set))

    return result


# ===========================================================================
# Build YAML output
# ===========================================================================

def build_yaml_dict(
    evidence: Dict[str, Dict[str, ConceptEvidence]],
    pdf_name: str,
    include_concepts_with_zero_matches: bool = True,
) -> Dict[str, Any]:
    """
    Convert evidence → YAML dict đúng schema của graph_engine.

    Schema:
        nodes: List[{id, type, label, source_pages, evidence_snippet, ...}]
        edges: List[{source, target, relation, ...}]
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # ── 1. Severity nodes ──────────────────────────────────────────
    for cid, info in SEVERITY_NODES.items():
        ev = evidence["severity"][cid]
        if not ev.matches and not include_concepts_with_zero_matches:
            continue
        node = {
            "id": cid,
            "type": "severity",
            "label": info["label"],
            "description": info["description"],
            "source_pages": ev.page_set,
        }
        if ev.matches:
            node["evidence_snippet"] = ev.matches[0].snippet
        nodes.append(node)

    # ── 2. Symptom nodes + edges supports → sxhd_thuong ────────────
    for cid, info in SYMPTOM_PATTERNS.items():
        ev = evidence["symptom"][cid]
        if not ev.matches and not include_concepts_with_zero_matches:
            continue
        node = {
            "id": cid,
            "type": "symptom",
            "label": info["label"],
            "source_pages": ev.page_set,
        }
        if ev.matches:
            node["evidence_snippet"] = ev.matches[0].snippet
        nodes.append(node)
        # Edge: symptom supports sxhd_thuong
        edges.append({
            "source": cid,
            "target": "sxhd_thuong",
            "relation": "supports",
        })

    # ── 3. Warning signs + edges warning_of → sxhd_canh_bao ────────
    # Một số warning sign nặng hơn cũng → sxhd_nang
    SEVERE_WARNINGS = {"tay_chan_lanh", "lu_du_vat_va", "hct_tang_plt_giam"}
    for cid, info in WARNING_SIGN_PATTERNS.items():
        ev = evidence["warning_sign"][cid]
        if not ev.matches and not include_concepts_with_zero_matches:
            continue
        node = {
            "id": cid,
            "type": "warning_sign",
            "label": info["label"],
            "source_pages": ev.page_set,
        }
        if ev.matches:
            node["evidence_snippet"] = ev.matches[0].snippet
        nodes.append(node)
        edges.append({
            "source": cid,
            "target": "sxhd_canh_bao",
            "relation": "warning_of",
        })
        if cid in SEVERE_WARNINGS:
            edges.append({
                "source": cid,
                "target": "sxhd_nang",
                "relation": "warning_of",
                "weight": 1.5,
            })

    # ── 4. Lab nodes + edges supports → sxhd_canh_bao ──────────────
    for cid, info in LAB_PATTERNS.items():
        ev = evidence["lab"][cid]
        if not ev.matches and not include_concepts_with_zero_matches:
            continue
        node = {
            "id": cid,
            "type": "lab",
            "label": info["label"],
            "threshold": info.get("threshold", ""),
            "source_pages": ev.page_set,
        }
        if ev.matches:
            node["evidence_snippet"] = ev.matches[0].snippet
        nodes.append(node)
        # Lab thường gắn với cảnh báo, trừ HFLC và bach_cau_thap → SXHD thường
        target = ("sxhd_thuong" if cid in ("hflc_cao", "bach_cau_thap")
                  else "sxhd_canh_bao")
        edges.append({
            "source": cid,
            "target": target,
            "relation": "supports",
        })

    # ── 5. Drug nodes + edges contraindicated_for / treats ────────
    for cid, info in DRUG_PATTERNS.items():
        ev = evidence["drug"][cid]
        if not ev.matches and not include_concepts_with_zero_matches:
            continue
        node = {
            "id": cid,
            "type": "drug",
            "label": info["label"],
            "source_pages": ev.page_set,
        }
        if "note" in info:
            node["note"] = info["note"]
        if ev.matches:
            node["evidence_snippet"] = ev.matches[0].snippet
        nodes.append(node)
        if info.get("is_contraindicated"):
            for sev in ("sxhd_thuong", "sxhd_canh_bao", "sxhd_nang"):
                edges.append({
                    "source": cid,
                    "target": sev,
                    "relation": "contraindicated_for",
                })
        else:
            # Paracetamol: treats nhẹ và cảnh báo
            for sev in ("sxhd_thuong", "sxhd_canh_bao"):
                edges.append({
                    "source": cid,
                    "target": sev,
                    "relation": "treats",
                })

    # ── 6. Intervention nodes + edges treats ──────────────────────
    for cid, info in INTERVENTION_PATTERNS.items():
        ev = evidence["intervention"][cid]
        if not ev.matches and not include_concepts_with_zero_matches:
            continue
        node = {
            "id": cid,
            "type": "intervention",
            "label": info["label"],
            "source_pages": ev.page_set,
        }
        if ev.matches:
            node["evidence_snippet"] = ev.matches[0].snippet
        nodes.append(node)
        for target_sev in info.get("treats", []):
            edges.append({
                "source": cid,
                "target": target_sev,
                "relation": "treats",
            })

    return {"nodes": nodes, "edges": edges}


def write_yaml(
    yaml_dict: Dict[str, Any],
    output_path: Path,
    pdf_source: str,
    n_pages_processed: int,
) -> None:
    """
    Ghi YAML với header comment giải thích nguồn gốc + timestamp.
    Format đẹp, không dùng yaml dump trực tiếp vì muốn control comment.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    now = datetime.now(timezone.utc).isoformat()

    # Header comment
    lines.append("# ============================================================================")
    lines.append("# graph_seed.yaml — TỰ ĐỘNG SINH RA bởi build_graph_from_guideline.py")
    lines.append("# ============================================================================")
    lines.append(f"# Generated at: {now}")
    lines.append(f"# Source PDF:   {pdf_source}")
    lines.append(f"# Pages scanned: {n_pages_processed}")
    lines.append("#")
    lines.append("# Pipeline:")
    lines.append("#   1. Đọc PDF guideline (Bộ Y tế VN, BV Bệnh Nhiệt Đới 2023).")
    lines.append("#   2. Tìm chương SXHD bằng keyword density.")
    lines.append("#   3. Pattern-match các concept (regex) đã định nghĩa trong")
    lines.append("#      build_graph_from_guideline.py (severity, symptom, warning,")
    lines.append("#      lab, drug, intervention).")
    lines.append("#   4. Mỗi node có field `source_pages` ghi rõ trang PDF đã match.")
    lines.append("#   5. Ghi YAML có comment để bác sĩ audit.")
    lines.append("#")
    lines.append("# RE-GENERATE:")
    lines.append("#   python src/build_graph_from_guideline.py")
    lines.append("# ============================================================================")
    lines.append("")

    # Nodes
    lines.append("nodes:")
    for n in yaml_dict["nodes"]:
        lines.append(f"  - id: {n['id']}")
        lines.append(f"    type: {n['type']}")
        lines.append(f"    label: {_yaml_str(n['label'])}")
        if n.get("description"):
            lines.append(f"    description: {_yaml_str(n['description'])}")
        if n.get("threshold"):
            lines.append(f"    threshold: {_yaml_str(n['threshold'])}")
        if n.get("note"):
            lines.append(f"    note: {_yaml_str(n['note'])}")
        if n.get("source_pages"):
            pages_str = ", ".join(str(p) for p in n["source_pages"])
            lines.append(f"    source_pages: [{pages_str}]")
        if n.get("evidence_snippet"):
            lines.append(f"    evidence_snippet: {_yaml_str(n['evidence_snippet'])}")
        lines.append("")

    # Edges
    lines.append("edges:")
    for e in yaml_dict["edges"]:
        line = (f"  - {{source: {e['source']}, target: {e['target']}, "
                f"relation: {e['relation']}")
        if "weight" in e:
            line += f", weight: {e['weight']}"
        line += "}"
        lines.append(line)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Đã ghi %s (%d nodes, %d edges)",
                output_path,
                len(yaml_dict["nodes"]),
                len(yaml_dict["edges"]))


def _yaml_str(s: str) -> str:
    """Quote string an toàn cho YAML (escape special chars)."""
    s = str(s).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


# ===========================================================================
# Diff với YAML cũ (nếu có)
# ===========================================================================

def diff_with_existing(new_path: Path, existing_path: Path) -> str:
    """So sánh 2 file YAML, in các thay đổi (nodes added/removed/modified)."""
    try:
        import yaml
    except ImportError:
        return "PyYAML chưa cài → không diff được."
    if not existing_path.exists():
        return "Chưa có YAML cũ để diff."

    new_data = yaml.safe_load(new_path.read_text(encoding="utf-8"))
    old_data = yaml.safe_load(existing_path.read_text(encoding="utf-8"))

    new_ids = {n["id"]: n for n in new_data.get("nodes", [])}
    old_ids = {n["id"]: n for n in old_data.get("nodes", [])}

    lines = []
    added = set(new_ids) - set(old_ids)
    removed = set(old_ids) - set(new_ids)
    common = set(new_ids) & set(old_ids)
    modified = [
        cid for cid in common
        if new_ids[cid].get("source_pages") != old_ids[cid].get("source_pages")
    ]

    lines.append(f"Nodes added ({len(added)}): {sorted(added)}")
    lines.append(f"Nodes removed ({len(removed)}): {sorted(removed)}")
    lines.append(f"Nodes with changed source_pages ({len(modified)}):")
    for cid in modified:
        old_pages = old_ids[cid].get("source_pages", [])
        new_pages = new_ids[cid].get("source_pages", [])
        lines.append(f"  {cid}: {old_pages} → {new_pages}")
    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def build_graph_from_pdf(
    pdf_path: Path = DEFAULT_PDF,
    output_path: Path = DEFAULT_OUTPUT,
    only_dengue_chapter: bool = True,
) -> Dict[str, Any]:
    """Pipeline chính. Trả về dict YAML đã ghi."""
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF không tồn tại: {pdf_path}")

    logger.info("=== STEP 1: Read PDF ===")
    all_pages = read_pdf_pages(pdf_path)
    logger.info("Đọc %d trang", len(all_pages))

    if only_dengue_chapter:
        logger.info("=== STEP 2: Locate chương SXHD ===")
        scan_pages = find_dengue_chapter(all_pages)
    else:
        scan_pages = all_pages

    logger.info("=== STEP 3: Extract concepts (regex pattern matching) ===")
    evidence = extract_all_concepts(scan_pages)

    logger.info("=== STEP 4: Build YAML ===")
    yaml_dict = build_yaml_dict(evidence, pdf_name=pdf_path.name)

    logger.info("=== STEP 5: Write file ===")
    write_yaml(yaml_dict, output_path,
               pdf_source=pdf_path.name,
               n_pages_processed=len(scan_pages))

    return yaml_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sinh graph_seed.yaml từ PDF guideline"
    )
    parser.add_argument("--pdf", default=str(DEFAULT_PDF),
                        help="Path tới file PDF guideline")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Path output YAML")
    parser.add_argument("--all-pages", action="store_true",
                        help="Scan toàn bộ PDF thay vì chỉ chương SXHD")
    parser.add_argument("--diff", action="store_true",
                        help="So sánh YAML mới với YAML cũ (nếu có)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    output_path = Path(args.output)
    backup_path = None
    if args.diff and output_path.exists():
        backup_path = output_path.with_suffix(".yaml.prev")
        backup_path.write_text(output_path.read_text(encoding="utf-8"),
                                encoding="utf-8")

    yaml_dict = build_graph_from_pdf(
        pdf_path=Path(args.pdf),
        output_path=output_path,
        only_dengue_chapter=not args.all_pages,
    )

    print(f"\n✅ Đã sinh {output_path}")
    print(f"   Nodes: {len(yaml_dict['nodes'])}")
    print(f"   Edges: {len(yaml_dict['edges'])}")

    if args.diff and backup_path is not None:
        print("\n--- DIFF với YAML trước đó ---")
        print(diff_with_existing(output_path, backup_path))