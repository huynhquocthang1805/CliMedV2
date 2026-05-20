"""
eval_chatbot.py — CliMedV2
==========================

Module đánh giá Chatbot Qwen trên test holdout (20%).

Có 2 chức năng chính:

1. **Export test set** thành 2 dạng:
   • CSV đầy đủ (cho audit)
   • Text dễ đọc cho từng BN — copy-paste vào chatbox để hỏi bot
     và check thủ công.

2. **Auto-evaluator**: chạy bot trên cả test set, parse JSON output,
   so sánh với ground truth → tính accuracy / F1 / confusion matrix.

PROMPT STRUCTURE:
   Bot được yêu cầu output JSON DUY NHẤT có format:
       {"label": "thuong" | "canhbao" | "nang",
        "confidence": 0.0-1.0,
        "reasoning": "..."}
   → exact-match grading.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Label mapping (consistent với feature_engineering.build_target)
# ===========================================================================

# Binary mode (target_mode='binary_warning'):
#   y=0 → thường (lớp 2 gốc)
#   y=1 → cảnh báo + nặng (lớp 1+3 gốc)
#
# Multiclass mode:
#   y=0 → benhcanhcuoicunglagi=1 (mức nhẹ nhất theo CSV)
#   y=1 → benhcanhcuoicunglagi=2 (mức trung bình)
#   y=2 → benhcanhcuoicunglagi=3 (mức nặng nhất)

BINARY_LABEL_MAP = {0: "thuong", 1: "canhbao_or_nang"}
MULTICLASS_LABEL_MAP = {0: "thuong", 1: "canhbao", 2: "nang"}

LABEL_VI = {
    "thuong":   "SXHD thông thường (Mức 1)",
    "canhbao":  "SXHD có dấu hiệu cảnh báo (Mức 2)",
    "nang":     "SXHD nặng (Mức 3)",
    "canhbao_or_nang": "SXHD cảnh báo hoặc nặng",
}


# ===========================================================================
# Render 1 BN ra text dễ đọc
# ===========================================================================

# Map các symptom column → mô tả tiếng Việt
_SYMPTOM_VI = {
    "NVnonoi":       "nôn ói",
    "non oi":       "nôn ói",
    "nonoi":       "nôn ói",
    "nôn ói":       "nôn ói",
    "nonramau": "nôn ra máu",
    "NVtieuchay":    "tiêu chảy",
    "tieu chay": "tiêu chảy",
    "tieuchay": "tiêu chảy",
    "tiêu chảy": "tiêu chảy",
    "NVdaubung":     "đau bụng",
    "daubung":     "đau bụng",
    "dau bung":     "đau bụng",
    "đau bụng":     "đau bụng",
    "NVdauco":       "đau cơ",
    "dauco":       "đau cơ",
    "dau co":       "đau cơ",
    "đau cơ":       "đau cơ",
    "NVdaudau":      "đau đầu",
    "dau dau":      "đau đầu",
    "daudau":      "đau đầu",
    "đau đầu":      "đau đầu",
    "NVdausauhocmat": "đau sau hốc mắt",
    "dau sau hoc mat": "đau sau hốc mắt",
    "đau sau hốc mắt": "đau sau hốc mắt",
    "NVho":          "ho",
    "ho":          "ho",
    "NVxuathuyet":   "xuất huyết",
    "xuất huyết":   "xuất huyết",
    "xuat huyet":   "xuất huyết",
    "xuathuyet":   "xuất huyết",
    "NVganto":       "gan to",
    "Petechia":      "petechia",
    "tiencansxh":    "tiền căn SXH",
}

# Demo cols
_DEMO_VI = {
    "gioitinh":   ("Giới tính", lambda v: "nam" if v == 1 else "nữ"),
    "Tuoi":       ("Tuổi", lambda v: f"{int(v)}"),
    "Cannang":    ("Cân nặng (kg)", lambda v: f"{float(v):.1f}"),
    "Chieucao":   ("Chiều cao (cm)", lambda v: f"{float(v):.1f}"),
}


def _format_series(row: pd.Series, prefix: str, label: str,
                   unit: str = "",
                   focus_days: Optional[List[int]] = None) -> str:
    """
    Format chuỗi N1..N10 thành 'N3=14.2, N4=18.5, ...'.

    Nếu `focus_days` được truyền, CHỈ render các ngày trong list này.
    Mặc định None = render tất cả N1-N10 (giữ behavior cũ).
    """
    days_to_show = focus_days if focus_days else list(range(1, 11))

    cols = []
    for i in days_to_show:
        for variant in (f"{prefix}N{i}", f"{prefix.lower()}N{i}"):
            if variant in row.index:
                cols.append((i, variant))
                break

    if not cols:
        return ""

    parts = []
    for day, col in cols:
        val = row[col]
        if pd.isna(val):
            parts.append(f"N{day}=—")
        else:
            try:
                parts.append(f"N{day}={float(val):.1f}")
            except (ValueError, TypeError):
                parts.append(f"N{day}={val}")
    line = f"  {label}{(' ' + unit) if unit else ''}: " + ", ".join(parts)
    return line


def _trend_summary(row: pd.Series, prefix: str,
                   focus_days: List[int]) -> Optional[str]:
    """
    Phân tích xu hướng cho 1 chỉ số trong focus_days.
    Trả về câu mô tả ngắn (vd: "PLT giảm sâu N4 26→N6 10").
    None nếu không đủ dữ liệu.
    """
    values: List[Tuple[int, float]] = []
    for d in focus_days:
        for variant in (f"{prefix}N{d}", f"{prefix.lower()}N{d}"):
            if variant in row.index:
                v = row[variant]
                if pd.notna(v):
                    try:
                        values.append((d, float(v)))
                    except (ValueError, TypeError):
                        pass
                break

    if len(values) < 2:
        return None

    first_day, first_val = values[0]
    last_day, last_val = values[-1]
    min_day, min_val = min(values, key=lambda x: x[1])
    max_day, max_val = max(values, key=lambda x: x[1])

    delta = last_val - first_val
    direction = ("giảm" if delta < 0 else "tăng" if delta > 0 else "ổn định")
    pct = (abs(delta) / first_val * 100) if first_val else 0
    return (f"{prefix.upper()} {direction} N{first_day} {first_val:.1f}"
            f"→N{last_day} {last_val:.1f} "
            f"(Δ={delta:+.1f}, {pct:.0f}%; "
            f"min N{min_day}={min_val:.1f}, max N{max_day}={max_val:.1f})")


def render_patient_card(row: pd.Series, patient_id: int,
                        ground_truth_label: Optional[str] = None,
                        focus_days: Optional[List[int]] = None,
                        ) -> str:
    """
    Render 1 BN thành đoạn text dễ đọc.

    Args:
        focus_days: list ngày cần render (vd [3,4,5,6,7]).
                    None = render tất cả N1-N10.

    Format:
        --- Bệnh nhân #ID (test set) ---
        Demographic: nam, 24 tuổi, 60kg
        Triệu chứng: đau bụng, nôn, ...
        Lab theo ngày (N3-N7):
          Bạch cầu: N3=4.2, N4=4.5, N5=3.8, ...
          Tiểu cầu: ...
        Lab one-shot: AST=85, ALT=42, ...
    """
    lines = [f"--- Bệnh nhân #{patient_id} (test holdout) ---"]

    # Demographic
    demo_parts = []
    for col, (label, fmt) in _DEMO_VI.items():
        if col in row.index and pd.notna(row[col]):
            try:
                demo_parts.append(f"{label}: {fmt(row[col])}")
            except Exception:
                pass
    if demo_parts:
        lines.append("Thông tin: " + " | ".join(demo_parts))

    # Days of fever at admission
    if "ngaybenhlucnhapvien" in row.index and pd.notna(row["ngaybenhlucnhapvien"]):
        lines.append(f"Ngày bệnh lúc nhập viện: {row['ngaybenhlucnhapvien']}")

    # Symptoms (chỉ liệt kê các symptom có giá trị positive)
    pos_symptoms = []
    for col, vi in _SYMPTOM_VI.items():
        if col in row.index and pd.notna(row[col]):
            v = row[col]
            try:
                if float(v) > 0:
                    pos_symptoms.append(vi)
            except (ValueError, TypeError):
                pass
    if pos_symptoms:
        lines.append("Triệu chứng (+): " + ", ".join(pos_symptoms))

    # Lab series — focus_days nếu có
    lab_lines = []
    for prefix, label, unit in [
        ("bachcau",      "Bạch cầu",  "(K/uL)"),
        ("tieucau",      "Tiểu cầu",  "(K/uL)"),
        ("Hct",          "Hct",        "(%)"),
        ("HFLC",         "HFLC",       "(%)"),
        ("phantramHFLC", "%HFLC",      ""),
    ]:
        line = _format_series(row, prefix, label, unit, focus_days=focus_days)
        if line:
            lab_lines.append(line)

    if lab_lines:
        if focus_days:
            day_label = (f"N{min(focus_days)}-N{max(focus_days)}"
                         if len(focus_days) > 1 else f"N{focus_days[0]}")
            lines.append(f"Lab theo ngày ({day_label} — giai đoạn giữa):")
            # Note thêm nếu có ngày nguy hiểm N4-N6
            danger_days = [d for d in focus_days if d in (3, 4, 5, 6, 7)]
            if danger_days:
                lines.append(
                    f"   Trong đó N{danger_days[0]}-N{danger_days[-1]} "
                    f"là giai đoạn nguy hiểm (có thể sốc nặng).")
        else:
            lines.append("Lab theo ngày:")
        lines.extend(lab_lines)

        # Trend summary cho các chỉ số chính
        if focus_days and len(focus_days) >= 2:
            trends = []
            for prefix in ("tieucau", "Hct", "bachcau"):
                ts = _trend_summary(row, prefix, focus_days)
                if ts:
                    trends.append(f"  • {ts}")
            if trends:
                lines.append("Xu hướng (trong focus window):")
                lines.extend(trends)

    # Lab one-shot
    one_shot = []
    for col in ["AST", "ALT", "Creatinin", "Albumin", "Bilirubin",
                "TQphantram", "Fibrinogen", "TCKgiay", "Troponin"]:
        if col in row.index and pd.notna(row[col]):
            try:
                one_shot.append(f"{col}={float(row[col]):.1f}")
            except (ValueError, TypeError):
                one_shot.append(f"{col}={row[col]}")
    if one_shot:
        lines.append("Lab khác: " + ", ".join(one_shot))

    if ground_truth_label is not None:
        lines.append(f"  [Đáp án thật: {ground_truth_label}]")

    return "\n".join(lines)


# ===========================================================================
# Export test set ra file
# ===========================================================================

def export_test_set_for_chatbox(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_spec: Any,
    output_dir: str,
    include_ground_truth_in_text: bool = False,
    focus_days: Optional[List[int]] = None,
) -> Dict[str, Path]:
    """
    Export 20% test holdout thành 3 file:

    1. {output_dir}/test_holdout.csv — full data + ground truth
    2. {output_dir}/test_for_chatbox.txt — mỗi BN 1 paragraph dễ đọc
    3. {output_dir}/test_summary.json — list ground truth labels

    Args:
        focus_days: nếu set (vd [3,4,5,6,7]), text card chỉ render
                    các ngày này — gọn hơn cho bác sĩ paste vào chatbox.
        include_ground_truth_in_text: nếu False (mặc định), text file KHÔNG
            chứa label thật → blind test cho bác sĩ.

    Returns: dict các path đã tạo.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_classes = target_spec.n_classes if target_spec else int(y_test.nunique())
    label_map = (BINARY_LABEL_MAP if n_classes == 2
                 else MULTICLASS_LABEL_MAP)

    # 1. CSV
    csv_path = output_path / "test_holdout.csv"
    out_df = X_test.copy()
    out_df["_ground_truth"] = y_test.values
    out_df["_ground_truth_label"] = y_test.map(label_map).values
    out_df.to_csv(csv_path, index=False)

    # 2. Text dễ đọc (với focus_days)
    txt_path = output_path / "test_for_chatbox.txt"
    blocks = []
    focus_label = (f"N{min(focus_days)}-N{max(focus_days)}"
                   if focus_days else "N1-N10")
    blocks.append(
        "=" * 70 + "\n"
        "TEST HOLDOUT — copy từng BN paste vào chatbox để test bot\n"
        f"Tổng số BN: {len(X_test)} | "
        f"Class distribution: {dict(y_test.map(label_map).value_counts())}\n"
        f"Focus days: {focus_label}\n"
        + "=" * 70 + "\n"
    )
    for i, (idx, row) in enumerate(X_test.iterrows()):
        gt_label = label_map.get(int(y_test.iloc[i]), "?")
        gt_for_text = gt_label if include_ground_truth_in_text else None
        blocks.append(render_patient_card(
            row, patient_id=i,
            ground_truth_label=gt_for_text,
            focus_days=focus_days,
        ))
        blocks.append("")
    txt_path.write_text("\n".join(blocks), encoding="utf-8")

    # 3. JSON summary
    json_path = output_path / "test_summary.json"
    summary = {
        "n_test": len(X_test),
        "n_classes": n_classes,
        "label_map": {str(k): v for k, v in label_map.items()},
        "ground_truth_int": y_test.tolist(),
        "ground_truth_label": [label_map[int(v)] for v in y_test],
        "class_distribution": {
            v: int(c) for v, c in
            y_test.map(label_map).value_counts().items()
        },
        "focus_days": (sorted(focus_days) if focus_days
                        else list(range(1, 11))),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    logger.info("Exported test set:")
    logger.info("  CSV:  %s (%d rows)", csv_path, len(X_test))
    logger.info("  TXT:  %s (focus=%s)", txt_path, focus_label)
    logger.info("  JSON: %s", json_path)
    return {"csv": csv_path, "txt": txt_path, "json": json_path}


# ===========================================================================
# Structured prompts cho bot
# ===========================================================================

# Mode 1 (CŨ): chỉ JSON khô — nhanh, ít token
STRUCTURED_INSTRUCTION = """Hãy phân loại bệnh nhân SAU đây vào MỘT trong 3 nhóm:
- "thuong":  SXHD thông thường (sốt, đau cơ, không có dấu hiệu cảnh báo)
- "canhbao": SXHD có dấu hiệu cảnh báo (PLT giảm, đau bụng nhiều, nôn nhiều, gan to, men gan tăng...)
- "nang":    SXHD nặng (sốc, xuất huyết nặng, suy đa cơ quan)

QUY TẮC OUTPUT BẮT BUỘC:
1. Trả về DUY NHẤT 1 JSON object hợp lệ, KHÔNG có text trước hoặc sau.
2. JSON có cấu trúc:
{
  "label": "thuong" | "canhbao" | "nang",
  "confidence": <số thực 0.0 đến 1.0>,
  "reasoning": "<giải thích ngắn gọn, tối đa 2 câu>"
}
3. Nếu binary mode (chỉ có 2 nhóm): dùng "thuong" hoặc "canhbao_or_nang".
4. KHÔNG markdown code block, KHÔNG comment, KHÔNG disclaimer.

DỮ LIỆU BỆNH NHÂN:
"""

# Mode 2 (MỚI): script-style — bot diễn giải như bác sĩ rồi mới kết luận JSON.
# Eval vẫn parse JSON ở cuối → grading exact-match như cũ, NHƯNG bác sĩ
# có thêm reasoning trace để verify quyết định.
SCRIPT_INSTRUCTION = """Bạn là bác sĩ điều trị SXHD. Hãy đọc thông tin BN và viết
BÁO CÁO LÂM SÀNG NGẮN theo định dạng cố định bên dưới.

📋 ĐỊNH DẠNG OUTPUT BẮT BUỘC (theo đúng thứ tự):

📋 BÁO CÁO LÂM SÀNG — Bệnh nhân #<ID>

**Tóm tắt:** 1 câu về tuổi, giới, ngày bệnh, triệu chứng nổi bật.

**Diễn biến chỉ số:** 2-4 gạch đầu dòng. Mỗi dòng nói về 1 chỉ số,
nêu xu hướng có ý nghĩa lâm sàng (vd "Tiểu cầu giảm sâu N4 26→N6 10 K/uL").
Đặc biệt nhấn mạnh giai đoạn N4-N6 (nguy hiểm) nếu có.

**Đánh giá:** 1-2 câu giải thích pattern lâm sàng. Nói rõ có
warning sign / dấu hiệu nặng nào.

**Hướng xử trí cần bác sĩ cân nhắc:** 1-2 câu, KHÔNG kê đơn cụ thể,
chỉ nói chung (theo dõi, nhập viện, bù dịch...).

**Kết luận (JSON cuối cùng — dòng RIÊNG, KHÔNG markdown):**
{"label": "thuong"|"canhbao"|"nang", "confidence": 0.0-1.0, "reasoning": "<tối đa 1 câu>"}

QUY TẮC:
- Phân loại 3 nhóm: thuong / canhbao / nang.
- Nếu binary mode: dùng thuong / canhbao_or_nang.
- KHÔNG bịa số liệu không có trong dữ liệu BN.
- KHÔNG kê liều thuốc cụ thể.
- JSON cuối phải hợp lệ, ở DÒNG RIÊNG, không có markdown ```.

DỮ LIỆU BỆNH NHÂN:
"""


def build_eval_prompt(patient_text: str,
                      mode: str = "multiclass",
                      style: str = "json") -> str:
    """
    Build prompt cho eval.

    Args:
        style: 'json'   = chỉ JSON khô (nhanh, fewer tokens).
               'script' = bác sĩ-style với báo cáo lâm sàng + JSON cuối.
                          Hữu ích cho UX khi paste vào chatbox và xem reasoning.
    """
    instruction = (SCRIPT_INSTRUCTION if style == "script"
                   else STRUCTURED_INSTRUCTION)
    return instruction + "\n" + patient_text


# ===========================================================================
# Parse bot output (tolerant của prose ngoài JSON)
# ===========================================================================

# Regex bắt JSON object đầu tiên (cân đối braces nông)
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class BotPrediction:
    """Output của 1 lần bot trả lời 1 BN."""
    raw_response: str
    parsed: Optional[Dict[str, Any]] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    parse_error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.label is not None


def parse_bot_response(response: str) -> BotPrediction:
    """
    Parse response text → BotPrediction.

    Tolerant: chấp nhận:
      - JSON sạch
      - markdown ```json``` block
      - prose lẫn JSON (script mode)
      - tiếng Việt có dấu (cảnh báo → canhbao)

    Strategy:
      1. Strip markdown fences.
      2. Tìm JSON object CUỐI CÙNG (script mode đặt ở cuối).
      3. Fallback: tìm JSON đầu tiên.
      4. Fallback cuối: parse cả response như JSON.
    """
    pred = BotPrediction(raw_response=response)
    text = response.strip()

    # 1. Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    # 2. Tìm TẤT CẢ json blocks (regex chỉ bắt được object đơn cấp)
    matches = list(_JSON_BLOCK_RE.finditer(text))

    parsed: Optional[Dict[str, Any]] = None
    parse_errors: List[str] = []

    # 2a. Ưu tiên parse JSON CUỐI CÙNG (script mode đặt JSON ở cuối)
    for m in reversed(matches):
        try:
            candidate = json.loads(m.group(0))
            if isinstance(candidate, dict) and "label" in candidate:
                parsed = candidate
                break
        except json.JSONDecodeError as e:
            parse_errors.append(str(e))
            continue

    # 2b. Fallback: thử parse toàn bộ text
    if parsed is None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            parse_errors.append(str(e))

    if parsed is None:
        pred.parse_error = "; ".join(parse_errors[:2]) or "No valid JSON found"
        return pred

    pred.parsed = parsed
    label = parsed.get("label")
    if label:
        # Normalize: strip, lowercase, map các biến thể tiếng Việt
        label_norm = str(label).strip().lower()
        label_map = {
            "thuong":           "thuong",
            "thường":           "thuong",
            "sxh":              "thuong",
            "thông thường":     "thuong",
            "binh thường":      "thuong",
            "bình thường":      "thuong",
            "canhbao":          "canhbao",
            "cảnh báo":         "canhbao",
            "warning":          "canhbao",
            "sxhcanhbao":       "canhbao",
            "tieuhuyetsacto":   "canhbao",
            "canhbao_or_nang":  "canhbao_or_nang",
            "canh bao or nang": "canhbao_or_nang",
            "nang":             "nang",
            "nặng":             "nang",
            "soc sxh":          "nang",
            "soxsxh":          "nang",
            "socsxh":           "nang",
            "thexuathuyet":     "nang",
            "tonthuonganthan":  "nang",
            "socsxhnang":     "nang",
            "soc,gan":         "nang",
            "soc,xhthtren":     "nang",
            "soctaisoc1lan":    "nang",
            "shh":              "nang",
            "gan":           "nang",
            "tonthuongtang":   "nang",
            "xuất huyết":       "nang",
            "xuathuyet":       "nang",
            "severe":           "nang",
        }
        pred.label = label_map.get(label_norm, label_norm)

    try:
        pred.confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        pred.confidence = None

    reasoning = parsed.get("reasoning")
    if reasoning:
        pred.reasoning = str(reasoning).strip()

    return pred


# ===========================================================================
# Auto-evaluator: chạy bot trên cả test set
# ===========================================================================

@dataclass
class EvalResult:
    """Kết quả đánh giá toàn bộ test set."""
    n_total: int = 0
    n_valid: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    f1_macro: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    per_patient: List[Dict[str, Any]] = field(default_factory=list)
    label_classes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Chatbot Eval Result ===",
            f"  Total patients:  {self.n_total}",
            f"  Valid responses: {self.n_valid} ({self.n_valid/max(self.n_total,1):.1%})",
            f"  Correct:         {self.n_correct} ({self.accuracy:.1%})",
            f"  F1 (macro):      {self.f1_macro:.3f}",
        ]
        if self.confusion_matrix:
            lines.append("  Confusion matrix:")
            n = len(self.confusion_matrix)
            header = "             " + " ".join(f"pred={c[:8]:>10s}"
                                                  for c in self.label_classes)
            lines.append(header)
            for i, row in enumerate(self.confusion_matrix):
                cls = self.label_classes[i][:8]
                vals = " ".join(f"{v:>10d}" for v in row)
                lines.append(f"  true={cls:>8s}   {vals}")
        return "\n".join(lines)


def evaluate_chatbot_on_test(
    chat_fn,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_spec: Any,
    label_classes: Optional[List[str]] = None,
    on_progress=None,
    style: str = "json",
    focus_days: Optional[List[int]] = None,
) -> EvalResult:
    """
    Đánh giá bot tự động trên cả test set.

    Args:
        chat_fn: callable(prompt: str) -> str
        X_test, y_test: từ meta của prepare_features.
        target_spec: TargetSpec.
        label_classes: list label string theo thứ tự để tính CM.
        on_progress: callback(idx, total, prediction) cho UI.
        style: 'json' | 'script' — quyết định prompt template.
        focus_days: nếu set, render BN chỉ với các ngày này.

    Returns: EvalResult.
    """
    from sklearn.metrics import (
        confusion_matrix as sk_cm, f1_score, accuracy_score
    )

    n_classes = target_spec.n_classes if target_spec else int(y_test.nunique())
    label_map = (BINARY_LABEL_MAP if n_classes == 2
                 else MULTICLASS_LABEL_MAP)
    if label_classes is None:
        label_classes = list(label_map.values())

    result = EvalResult(n_total=len(X_test), label_classes=label_classes)

    y_true_list: List[str] = []
    y_pred_list: List[str] = []

    for i, (idx, row) in enumerate(X_test.iterrows()):
        gt_int = int(y_test.iloc[i])
        gt_label = label_map[gt_int]
        patient_text = render_patient_card(
            row, patient_id=i, focus_days=focus_days)
        prompt = build_eval_prompt(patient_text, mode="multiclass", style=style)

        try:
            response = chat_fn(prompt)
        except Exception as e:
            logger.warning("Bot failed on patient %d: %s", i, e)
            response = ""

        pred = parse_bot_response(response)

        record = {
            "patient_id": i,
            "ground_truth": gt_label,
            "bot_label": pred.label,
            "bot_confidence": pred.confidence,
            "bot_reasoning": pred.reasoning,
            "is_valid": pred.is_valid,
            "is_correct": pred.label == gt_label if pred.is_valid else False,
            "raw_response": pred.raw_response[:1000],
            "parse_error": pred.parse_error,
        }
        result.per_patient.append(record)

        if pred.is_valid:
            result.n_valid += 1
            y_true_list.append(gt_label)
            y_pred_list.append(pred.label)
            if record["is_correct"]:
                result.n_correct += 1

        if on_progress is not None:
            try:
                on_progress(i + 1, len(X_test), record)
            except Exception:
                pass

    if result.n_valid > 0:
        result.accuracy = accuracy_score(y_true_list, y_pred_list)
        result.f1_macro = f1_score(y_true_list, y_pred_list,
                                   labels=label_classes,
                                   average="macro", zero_division=0)
        cm = sk_cm(y_true_list, y_pred_list, labels=label_classes)
        result.confusion_matrix = cm.tolist()

    return result


# ===========================================================================
# CLI test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from feature_engineering import prepare_features

    df = pd.read_csv("data/NghiencuuHFLC.csv")
    X, y, meta = prepare_features(df, target_mode="multiclass")

    paths = export_test_set_for_chatbox(
        meta["X_test_raw"], meta["y_test"],
        target_spec=meta["target_spec"],
        output_dir="data/eval_test_set",
    )
    print("\nĐã xuất:")
    for k, p in paths.items():
        print(f"  {k}: {p}")

    # Demo: render 1 BN
    print("\n=== Sample patient card (BN đầu tiên) ===")
    row = meta["X_test_raw"].iloc[0]
    print(render_patient_card(row, patient_id=0))

    # Demo: parse response giả lập
    print("\n=== Demo parse_bot_response ===")
    fake = '''```json
    {"label": "canhbao", "confidence": 0.85, "reasoning": "PLT giảm và đau bụng nhiều"}
    ```'''
    pred = parse_bot_response(fake)
    print(f"label={pred.label}, conf={pred.confidence}, valid={pred.is_valid}")