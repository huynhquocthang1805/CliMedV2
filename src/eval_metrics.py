"""
eval_metrics.py — CliMedV2
==========================

Comprehensive metrics cho bài toán phân loại SXHD 3 mức:
    1 = thuong   (SXHD thông thường)
    2 = canhbao  (SXHD có dấu hiệu cảnh báo)
    3 = nang     (SXHD nặng)

Khác với accuracy + macro F1 thuần túy trong eval_chatbot.py v1, module này thêm:

  • **Ordinal metrics**: Quadratic Weighted Kappa (QWK), MAE — vì 1 < 2 < 3 là
    ordinal scale.
  • **Clinical safety metrics**: under-triage rate, over-triage rate, critical
    miss rate (nang predicted as thuong) — quan trọng nhất cho bài y tế.
  • **Per-class P/R/F1**: thay vì chỉ macro.
  • **Confidence calibration**: Brier score, ECE.
  • **Response quality**: length, citation rate, grounding rate, format
    compliance.
  • **Latency**: p50/p95/max ms/query.

Dùng:
    from eval_metrics import compute_all_metrics, format_report
    metrics = compute_all_metrics(per_patient_records)
    print(format_report(metrics))
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Label mapping & ordering
# ===========================================================================

# Severity ordinal — quan trọng cho under/over-triage tính toán
SEVERITY_ORDER = {"thuong": 1, "canhbao": 2, "nang": 3}
SEVERITY_VI = {
    "thuong":  "SXHD thông thường",
    "canhbao": "SXHD có dấu hiệu cảnh báo",
    "nang":    "SXHD nặng",
}
CLASSES_ORDERED = ["thuong", "canhbao", "nang"]


def label_to_int(label: Optional[str]) -> Optional[int]:
    if label is None:
        return None
    return SEVERITY_ORDER.get(label)


# ===========================================================================
# Per-patient record schema (input của compute_all_metrics)
# ===========================================================================
# Mỗi record là dict có TỐI THIỂU:
#   - "ground_truth":   str trong {thuong, canhbao, nang}
#   - "bot_label":      str | None
#   - "bot_confidence": float | None
#   - "raw_response":   str
# Có thể có thêm:
#   - "latency_ms":     float
#   - "bot_reasoning":  str
#   - "patient_text":   str (nguyên BN paste — dùng để check grounding)


# ===========================================================================
# 1. CLASSIFICATION METRICS (chính)
# ===========================================================================

def _confusion_matrix(records: List[dict],
                      classes: List[str] = CLASSES_ORDERED
                      ) -> np.ndarray:
    """CM theo thứ tự classes. Rows = true, Cols = pred."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    cls_idx = {c: i for i, c in enumerate(classes)}
    for r in records:
        gt = r.get("ground_truth")
        pd = r.get("bot_label")
        if gt in cls_idx and pd in cls_idx:
            cm[cls_idx[gt], cls_idx[pd]] += 1
    return cm


def _per_class_prf(cm: np.ndarray,
                   classes: List[str]) -> Dict[str, Dict[str, float]]:
    """Tính precision/recall/F1 mỗi class từ CM."""
    out: Dict[str, Dict[str, float]] = {}
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(cm[i, :].sum())
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    return out


def _accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm)) / total if total > 0 else 0.0


def _balanced_accuracy(cm: np.ndarray) -> float:
    """Mean recall across classes. Robust với imbalanced."""
    n = cm.shape[0]
    recalls = []
    for i in range(n):
        support = cm[i, :].sum()
        if support > 0:
            recalls.append(cm[i, i] / support)
    return float(np.mean(recalls)) if recalls else 0.0


def _macro_f1(per_class: Dict[str, Dict[str, float]]) -> float:
    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    return float(np.mean(f1s)) if f1s else 0.0


def _weighted_f1(per_class: Dict[str, Dict[str, float]]) -> float:
    total = sum(v["support"] for v in per_class.values())
    if total == 0:
        return 0.0
    return float(sum(v["f1"] * v["support"] for v in per_class.values()) / total)


def _cohen_kappa(cm: np.ndarray, weights: Optional[str] = None) -> float:
    """
    Cohen's kappa. weights=None: unweighted. weights='quadratic': QWK (ordinal).
    """
    n = cm.shape[0]
    total = cm.sum()
    if total == 0:
        return 0.0
    sum_rows = cm.sum(axis=1)
    sum_cols = cm.sum(axis=0)

    if weights == "quadratic":
        # QWK
        w = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                w[i, j] = ((i - j) ** 2) / ((n - 1) ** 2 if n > 1 else 1)
        exp = np.outer(sum_rows, sum_cols) / total
        num = (w * cm).sum()
        den = (w * exp).sum()
        return 1.0 - (num / den) if den > 0 else 0.0
    else:
        # unweighted
        po = float(np.trace(cm)) / total
        pe = float(np.sum(sum_rows * sum_cols)) / (total ** 2)
        return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0


def _mae_ordinal(records: List[dict]) -> float:
    """Mean Absolute Error trên ordinal scale 1-3. Càng nhỏ càng tốt."""
    diffs = []
    for r in records:
        gt = label_to_int(r.get("ground_truth"))
        pd = label_to_int(r.get("bot_label"))
        if gt is not None and pd is not None:
            diffs.append(abs(gt - pd))
    return float(np.mean(diffs)) if diffs else float("nan")


# ===========================================================================
# 2. CLINICAL SAFETY METRICS (quan trọng nhất cho y tế)
# ===========================================================================

def _safety_metrics(records: List[dict]) -> Dict[str, Any]:
    """
    Phân tích các loại sai phân loại theo HƯỚNG (under vs over) — quan trọng
    cho safety vì hậu quả không đối xứng.

    - under_triage:    bot predict NHẸ HƠN ground truth (NGUY HIỂM)
    - over_triage:     bot predict NẶNG HƠN ground truth (lãng phí nhưng an toàn)
    - exact:           predict đúng
    - critical_miss:   true=nang nhưng bot=thuong (cực nguy hiểm)
    - severe_recall:   recall trên class 'nang' (sensitivity cho ca nặng)
    - severe_npv:      negative predictive value — khi bot nói KHÔNG nặng, xác suất
                       BN thực sự KHÔNG nặng (đáng tin cậy để discharge)
    """
    n_under = n_over = n_exact = n_invalid = 0
    n_critical_miss = 0
    n_total = len(records)

    # Cho severe_recall / severe_npv
    tp_severe = fn_severe = 0   # true=nang
    tn_severe = fp_severe = 0   # true!=nang

    for r in records:
        gt = label_to_int(r.get("ground_truth"))
        pd = label_to_int(r.get("bot_label"))
        if gt is None:
            n_invalid += 1
            continue
        if pd is None:
            n_invalid += 1
            continue
        if pd == gt:
            n_exact += 1
        elif pd < gt:
            n_under += 1
            if gt == 3 and pd == 1:
                n_critical_miss += 1
        else:
            n_over += 1

        gt_severe = (gt == 3)
        pd_severe = (pd == 3)
        if gt_severe and pd_severe:
            tp_severe += 1
        elif gt_severe and not pd_severe:
            fn_severe += 1
        elif not gt_severe and not pd_severe:
            tn_severe += 1
        else:
            fp_severe += 1

    n_classified = n_total - n_invalid
    base = max(n_classified, 1)
    severe_recall = tp_severe / max(tp_severe + fn_severe, 1)
    severe_npv = tn_severe / max(tn_severe + fn_severe, 1)
    severe_precision = tp_severe / max(tp_severe + fp_severe, 1)

    return {
        "n_total": n_total,
        "n_classified": n_classified,
        "n_invalid": n_invalid,
        "exact_match_rate": n_exact / base,
        "under_triage_rate": n_under / base,   # NGUY HIỂM: càng thấp càng tốt
        "over_triage_rate": n_over / base,
        "n_critical_miss": n_critical_miss,
        "critical_miss_rate": n_critical_miss / base,   # CỰC QUAN TRỌNG
        "severe_recall": severe_recall,        # sensitivity ca nặng
        "severe_precision": severe_precision,
        "severe_npv": severe_npv,              # khi bot nói "không nặng" có tin được không
        "severe_tp": tp_severe,
        "severe_fn": fn_severe,
        "severe_fp": fp_severe,
        "severe_tn": tn_severe,
    }


# ===========================================================================
# 3. CONFIDENCE CALIBRATION
# ===========================================================================

def _calibration_metrics(records: List[dict],
                         n_bins: int = 5) -> Dict[str, Any]:
    """
    Brier score + ECE để xem confidence bot báo cáo có "ăn khớp" với accuracy
    thực không. Bot tốt: confidence cao ⇔ đúng nhiều hơn.

    Brier (1 vs all đúng/sai): sum (conf - is_correct)^2 / n
    ECE: |mean_conf - accuracy| trung bình các bin theo confidence.
    """
    confs = []
    correctness = []
    for r in records:
        c = r.get("bot_confidence")
        gt = r.get("ground_truth")
        pd = r.get("bot_label")
        if c is None or gt is None or pd is None:
            continue
        try:
            cf = float(c)
        except (ValueError, TypeError):
            continue
        if not (0.0 <= cf <= 1.0):
            continue
        confs.append(cf)
        correctness.append(1 if pd == gt else 0)

    if not confs:
        return {"brier": float("nan"), "ece": float("nan"),
                "n_with_conf": 0, "bins": []}

    confs_a = np.array(confs)
    correct_a = np.array(correctness, dtype=float)
    brier = float(np.mean((confs_a - correct_a) ** 2))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_info: List[Dict[str, float]] = []
    ece = 0.0
    total = len(confs_a)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confs_a >= lo) & (confs_a <= hi)
        else:
            mask = (confs_a >= lo) & (confs_a < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            bin_info.append({"range": f"{lo:.2f}-{hi:.2f}", "n": 0,
                             "avg_conf": 0.0, "accuracy": 0.0})
            continue
        avg_conf = float(confs_a[mask].mean())
        acc = float(correct_a[mask].mean())
        ece += (n_in / total) * abs(avg_conf - acc)
        bin_info.append({"range": f"{lo:.2f}-{hi:.2f}", "n": n_in,
                         "avg_conf": avg_conf, "accuracy": acc})

    return {"brier": brier, "ece": ece,
            "n_with_conf": len(confs), "bins": bin_info}


# ===========================================================================
# 4. RESPONSE QUALITY METRICS
# ===========================================================================

# Regex: trang số trong response (vd "trang 220", "p.218", "p 215")
_CITATION_RE = re.compile(r"(trang|tr\.|p\.|page)\s*\d{1,4}", re.I)

# Regex: số đi kèm đơn vị y khoa (vd "PLT 30", "Hct 47%", "AST 200")
_LAB_NUM_RE = re.compile(
    r"\b(plt|tiểu cầu|tieu cau|hct|hematocrit|wbc|bạch cầu|"
    r"bach cau|ast|alt|sgot|sgpt|creatinin|alb)\b\s*[:=]?\s*[\d.]+",
    re.I,
)


def _quality_metrics(records: List[dict]) -> Dict[str, Any]:
    """
    Đo chất lượng câu trả lời:
      - n_chars / n_words trung bình & median
      - citation_rate: % có trích trang
      - lab_ground_rate: % chứa số kèm tên xét nghiệm (gợi ý ground BN data)
      - reasoning_present_rate: % có field reasoning trong JSON
      - parse_success_rate: % parse được JSON
    """
    n = len(records)
    if n == 0:
        return {}

    n_chars: List[int] = []
    n_words: List[int] = []
    n_cite = 0
    n_lab_ground = 0
    n_reasoning = 0
    n_parsed = 0

    for r in records:
        raw = (r.get("raw_response") or "").strip()
        n_chars.append(len(raw))
        n_words.append(len(raw.split()))
        if _CITATION_RE.search(raw):
            n_cite += 1
        if _LAB_NUM_RE.search(raw):
            n_lab_ground += 1
        if r.get("bot_reasoning"):
            n_reasoning += 1
        if r.get("bot_label"):
            n_parsed += 1

    def _stat(lst):
        if not lst:
            return {"mean": 0, "median": 0, "min": 0, "max": 0}
        return {
            "mean": float(np.mean(lst)),
            "median": float(np.median(lst)),
            "min": int(np.min(lst)),
            "max": int(np.max(lst)),
        }

    return {
        "chars": _stat(n_chars),
        "words": _stat(n_words),
        "citation_rate": n_cite / n,
        "lab_ground_rate": n_lab_ground / n,
        "reasoning_present_rate": n_reasoning / n,
        "parse_success_rate": n_parsed / n,
    }


# ===========================================================================
# 5. LATENCY METRICS
# ===========================================================================

def _latency_metrics(records: List[dict]) -> Dict[str, Any]:
    lats = [r.get("latency_ms") for r in records
            if r.get("latency_ms") is not None]
    if not lats:
        return {"n": 0}
    lats = sorted(lats)
    return {
        "n": len(lats),
        "p50_ms": float(np.percentile(lats, 50)),
        "p95_ms": float(np.percentile(lats, 95)),
        "mean_ms": float(np.mean(lats)),
        "max_ms": float(np.max(lats)),
        "total_s": float(sum(lats) / 1000),
    }


# ===========================================================================
# Aggregator
# ===========================================================================

@dataclass
class EvalReport:
    """Toàn bộ metrics tổng hợp."""
    # Classification
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    cohen_kappa: float = 0.0
    qwk: float = 0.0
    mae_ordinal: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    classes: List[str] = field(default_factory=lambda: list(CLASSES_ORDERED))

    # Safety
    safety: Dict[str, Any] = field(default_factory=dict)

    # Calibration
    calibration: Dict[str, Any] = field(default_factory=dict)

    # Quality
    quality: Dict[str, Any] = field(default_factory=dict)

    # Latency
    latency: Dict[str, Any] = field(default_factory=dict)

    # Meta
    n_total: int = 0
    n_valid: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_all_metrics(records: List[dict]) -> EvalReport:
    """
    Tính toàn bộ metrics từ list per-patient records.

    records: list các dict, mỗi dict cho 1 BN, có ít nhất:
        ground_truth (str), bot_label (str|None), bot_confidence (float|None),
        raw_response (str). Optional: latency_ms, bot_reasoning, patient_text.
    """
    report = EvalReport()
    report.n_total = len(records)
    report.label_distribution = dict(Counter(
        r.get("ground_truth", "?") for r in records
    ))

    valid_records = [r for r in records if r.get("bot_label") is not None]
    report.n_valid = len(valid_records)

    if not valid_records:
        logger.warning("Không có response hợp lệ nào. Trả về report rỗng.")
        return report

    # Classification
    cm = _confusion_matrix(valid_records, CLASSES_ORDERED)
    report.confusion_matrix = cm.tolist()
    report.per_class = _per_class_prf(cm, CLASSES_ORDERED)
    report.accuracy = _accuracy(cm)
    report.balanced_accuracy = _balanced_accuracy(cm)
    report.macro_f1 = _macro_f1(report.per_class)
    report.weighted_f1 = _weighted_f1(report.per_class)
    report.cohen_kappa = _cohen_kappa(cm)
    report.qwk = _cohen_kappa(cm, weights="quadratic")
    report.mae_ordinal = _mae_ordinal(valid_records)

    # Safety
    report.safety = _safety_metrics(records)
    # Calibration
    report.calibration = _calibration_metrics(records)
    # Quality
    report.quality = _quality_metrics(records)
    # Latency
    report.latency = _latency_metrics(records)

    return report


# ===========================================================================
# Report formatter (Markdown)
# ===========================================================================

def _pct(x: float) -> str:
    return f"{x*100:.1f}%" if x is not None and not (isinstance(x, float) and (x != x)) else "—"


def format_report(report: EvalReport, title: str = "Đánh giá Chatbot SXHD") -> str:
    """Render báo cáo Markdown."""
    lines: List[str] = [f"# {title}", ""]
    lines.append(f"**Tổng số BN test:** {report.n_total}  ·  "
                 f"**Parse hợp lệ:** {report.n_valid} ({_pct(report.n_valid/max(report.n_total,1))})")
    lines.append("")
    lines.append("**Phân phối nhãn ground truth:**  "
                 + ", ".join(f"`{k}`: {v}" for k, v in report.label_distribution.items()))
    lines.append("")

    # --- A. Classification ---
    lines.append("## 1. Phân loại tổng quan")
    lines.append("")
    lines.append("| Metric | Giá trị | Diễn giải |")
    lines.append("|---|---|---|")
    lines.append(f"| Accuracy | {_pct(report.accuracy)} | % nhãn đúng |")
    lines.append(f"| Balanced accuracy | {_pct(report.balanced_accuracy)} | Mean recall, fair với imbalanced |")
    lines.append(f"| Macro F1 | {report.macro_f1:.3f} | Trung bình F1 các class |")
    lines.append(f"| Weighted F1 | {report.weighted_f1:.3f} | F1 weighted theo support |")
    lines.append(f"| Cohen's Kappa | {report.cohen_kappa:.3f} | Loại bỏ may rủi (0=ngẫu nhiên, 1=hoàn hảo) |")
    lines.append(f"| **QWK** (ordinal) | {report.qwk:.3f} | Phạt nặng lỗi xa (thuong↔nang nặng hơn thuong↔canhbao) |")
    lines.append(f"| MAE ordinal | {report.mae_ordinal:.2f} | Khoảng cách trung bình (1=lệch 1 mức) |")
    lines.append("")

    # --- B. Per-class ---
    lines.append("## 2. Per-class metrics")
    lines.append("")
    lines.append("| Class | VN | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|---|")
    for cls in CLASSES_ORDERED:
        pc = report.per_class.get(cls, {})
        vi = SEVERITY_VI[cls]
        lines.append(f"| `{cls}` | {vi} | {_pct(pc.get('precision', 0))} "
                     f"| {_pct(pc.get('recall', 0))} | {pc.get('f1', 0):.3f} "
                     f"| {pc.get('support', 0)} |")
    lines.append("")

    # --- C. Confusion matrix ---
    lines.append("## 3. Confusion matrix")
    lines.append("")
    lines.append("Rows = true, Cols = predicted")
    lines.append("")
    header = "| true \\ pred | " + " | ".join(f"`{c}`" for c in CLASSES_ORDERED) + " |"
    sep = "|---" * (len(CLASSES_ORDERED) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for i, cls in enumerate(CLASSES_ORDERED):
        row = report.confusion_matrix[i] if i < len(report.confusion_matrix) else [0] * len(CLASSES_ORDERED)
        lines.append(f"| `{cls}` | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    # --- D. Clinical Safety ---
    s = report.safety
    if s:
        lines.append("## 4. ⚕️ Clinical safety metrics  *(quan trọng nhất)*")
        lines.append("")
        lines.append("| Metric | Giá trị | Ý nghĩa |")
        lines.append("|---|---|---|")
        lines.append(f"| Exact match | {_pct(s.get('exact_match_rate', 0))} | Đoán đúng mức độ |")
        lines.append(f"| **Under-triage** ⚠️ | {_pct(s.get('under_triage_rate', 0))} | Bot dự đoán NHẸ HƠN ground truth — **NGUY HIỂM** (BN có thể bị cho về nhà) |")
        lines.append(f"| Over-triage | {_pct(s.get('over_triage_rate', 0))} | Bot dự đoán NẶNG HƠN — lãng phí nhưng an toàn |")
        lines.append(f"| **Critical miss** 🚨 | {_pct(s.get('critical_miss_rate', 0))} ({s.get('n_critical_miss', 0)} ca) | True=`nang` nhưng bot=`thuong` — **cực kỳ nguy hiểm** |")
        lines.append(f"| Severe recall (sensitivity) | {_pct(s.get('severe_recall', 0))} | % ca `nang` được bot phát hiện đúng |")
        lines.append(f"| Severe precision | {_pct(s.get('severe_precision', 0))} | Khi bot nói `nang`, đúng tỉ lệ bao nhiêu |")
        lines.append(f"| Severe NPV | {_pct(s.get('severe_npv', 0))} | Khi bot nói KHÔNG `nang`, độ tin cậy |")
        lines.append("")
        lines.append("> **Ngưỡng deploy được tối thiểu cho ứng dụng triage:** "
                     "critical_miss = 0, under-triage ≤ 10%, severe_recall ≥ 90%. "
                     "Nếu chưa đạt → bot chỉ nên dùng **hỗ trợ tham khảo**, không thay thế bác sĩ triage.")
        lines.append("")

    # --- E. Calibration ---
    c = report.calibration
    if c and c.get("n_with_conf", 0) > 0:
        lines.append("## 5. Confidence calibration")
        lines.append("")
        lines.append(f"- **Brier score:** {c['brier']:.3f}  *(0 = hoàn hảo, 0.25 = tệ nhất với binary)*")
        lines.append(f"- **ECE (Expected Calibration Error):** {_pct(c['ece'])}  *(càng thấp càng tốt; <5% là tốt)*")
        lines.append(f"- n có confidence: {c['n_with_conf']}")
        lines.append("")
        lines.append("| Confidence range | N | Avg conf | Accuracy thực | Gap |")
        lines.append("|---|---|---|---|---|")
        for b in c.get("bins", []):
            if b["n"] > 0:
                gap = b['avg_conf'] - b['accuracy']
                lines.append(f"| {b['range']} | {b['n']} | {b['avg_conf']:.2f} "
                             f"| {b['accuracy']:.2f} | {gap:+.2f} |")
        lines.append("")
        lines.append("> Bot calibrate tốt ⇔ Gap luôn nhỏ. Gap dương = bot tự tin quá mức.")
        lines.append("")

    # --- F. Quality ---
    q = report.quality
    if q:
        lines.append("## 6. Chất lượng câu trả lời")
        lines.append("")
        lines.append("| Metric | Giá trị |")
        lines.append("|---|---|")
        lines.append(f"| Số ký tự (mean / median / max) | {q['chars']['mean']:.0f} / {q['chars']['median']:.0f} / {q['chars']['max']} |")
        lines.append(f"| Số từ (mean / median / max) | {q['words']['mean']:.0f} / {q['words']['median']:.0f} / {q['words']['max']} |")
        lines.append(f"| Citation rate (có trích trang) | {_pct(q['citation_rate'])} |")
        lines.append(f"| Lab grounding rate (có trích số XN) | {_pct(q['lab_ground_rate'])} |")
        lines.append(f"| Reasoning present rate | {_pct(q['reasoning_present_rate'])} |")
        lines.append(f"| JSON parse success | {_pct(q['parse_success_rate'])} |")
        lines.append("")

    # --- G. Latency ---
    l = report.latency
    if l.get("n", 0) > 0:
        lines.append("## 7. Latency")
        lines.append("")
        lines.append(f"- p50: {l['p50_ms']:.0f} ms · p95: {l['p95_ms']:.0f} ms · "
                     f"max: {l['max_ms']:.0f} ms · mean: {l['mean_ms']:.0f} ms")
        lines.append(f"- Tổng runtime: {l['total_s']:.1f}s cho {l['n']} BN  "
                     f"(≈ {l['mean_ms']/1000:.1f}s/BN)")
        lines.append("")

    return "\n".join(lines)


def export_per_patient_csv(records: List[dict], out_path: str) -> None:
    """Dump tất cả per-patient records ra CSV."""
    import csv
    if not records:
        return
    keys = sorted(set().union(*(r.keys() for r in records)))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            r_copy = {k: r.get(k, "") for k in keys}
            # Truncate raw_response để CSV không quá lớn
            if "raw_response" in r_copy and isinstance(r_copy["raw_response"], str):
                r_copy["raw_response"] = r_copy["raw_response"][:1000]
            w.writerow(r_copy)


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    # Demo với fake records
    fake = [
        {"ground_truth": "nang",    "bot_label": "nang",    "bot_confidence": 0.9,
         "raw_response": "PLT 30 K/uL ở N5 (trang 220)... cần nhập ICU.",
         "bot_reasoning": "PLT giảm sâu", "latency_ms": 1200},
        {"ground_truth": "canhbao", "bot_label": "canhbao", "bot_confidence": 0.7,
         "raw_response": "Có warning sign (trang 218): đau bụng + Hct tăng.",
         "bot_reasoning": "warning sign", "latency_ms": 950},
        {"ground_truth": "nang",    "bot_label": "canhbao", "bot_confidence": 0.6,  # UNDER-TRIAGE
         "raw_response": "PLT 41 K/uL...", "bot_reasoning": "PLT thấp",
         "latency_ms": 1100},
        {"ground_truth": "canhbao", "bot_label": "thuong",  "bot_confidence": 0.5,  # UNDER-TRIAGE
         "raw_response": "Triệu chứng nhẹ.", "bot_reasoning": "ổn",
         "latency_ms": 800},
        {"ground_truth": "nang",    "bot_label": "thuong",  "bot_confidence": 0.4,  # CRITICAL MISS
         "raw_response": "Bệnh nhân khá ổn.", "bot_reasoning": "—",
         "latency_ms": 900},
        {"ground_truth": "thuong",  "bot_label": "canhbao", "bot_confidence": 0.6,  # OVER-TRIAGE
         "raw_response": "Có dấu hiệu nhẹ (trang 215).", "bot_reasoning": "—",
         "latency_ms": 1000},
        {"ground_truth": "canhbao", "bot_label": "canhbao", "bot_confidence": 0.85,
         "raw_response": "PLT 50, đau bụng nhiều (trang 220).", "bot_reasoning": "—",
         "latency_ms": 1050},
        {"ground_truth": "canhbao", "bot_label": None, "bot_confidence": None,
         "raw_response": "ill-formed JSON", "bot_reasoning": None,
         "latency_ms": 700},   # parse fail
    ]
    rep = compute_all_metrics(fake)
    print(format_report(rep, title="Demo eval — 8 BN giả lập"))