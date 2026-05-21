"""
mini_ai_bot.py — CliMedV2
==========================

Mini AI Triage Bot cho SXHD: 1 wrapper sạch, đầu vào là dữ liệu BN, đầu ra là
TriageResult có cấu trúc — phân loại 3 mức (thuong/canhbao/nang) kèm
confidence, reasoning, evidence, và **khuyến nghị hành động lâm sàng**.

Dùng được trong 4 ngữ cảnh:
  1. **Streamlit chatbot tab** — đã có sẵn qua chatbot_engine.
  2. **Batch eval** — `bot.triage_batch(test_df)` cho run_eval.py.
  3. **CLI demo** — `python mini_ai_bot.py "<bệnh nhân text>"`.
  4. **API/library** — `from mini_ai_bot import TriageBot`.

Kiến trúc:
    TriageBot(backend, use_rag, use_graph)
       │
       ├── _build_prompt(BN) ──► (system, user) messages
       │       sử dụng SYSTEM_PROMPT_TRIAGE: ép output JSON cuối cố định
       │
       ├── _call_llm(messages) ──► raw response
       │       sử dụng llm_qwen.OllamaClient / HFClient
       │
       ├── _parse_response(raw) ──► severity_level + confidence + reasoning
       │       sử dụng parse_bot_response của eval_chatbot
       │
       └── _enrich_with_evidence_and_action(...) ──► TriageResult cuối cùng
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

SEVERITY_LEVEL_OF = {"thuong": 1, "canhbao": 2, "nang": 3}
LEVEL_TO_LABEL = {1: "thuong", 2: "canhbao", 3: "nang"}
SEVERITY_VI = {
    "thuong":  "SXHD thông thường",
    "canhbao": "SXHD có dấu hiệu cảnh báo",
    "nang":    "SXHD nặng",
}

# Khuyến nghị hành động lâm sàng — theo hướng dẫn Bộ Y tế VN 2023 (general)
RECOMMENDED_ACTION = {
    "thuong": (
        "Theo dõi ngoại trú. Tái khám hằng ngày. Uống nhiều nước/oresol, "
        "paracetamol khi sốt (theo cân nặng, KHÔNG dùng aspirin/ibuprofen). "
        "Đến BV ngay nếu có 1 trong: đau bụng nhiều, nôn nhiều, vật vã/lừ đừ, "
        "chảy máu niêm mạc, tay chân lạnh, hết sốt mà mệt hơn."
    ),
    "canhbao": (
        "**Nhập viện theo dõi sát.** Kiểm tra Hct/PLT mỗi 6-12h (đặc biệt N4-N6). "
        "Bù dịch theo phác đồ. Theo dõi sinh hiệu thường xuyên. "
        "Bác sĩ cần đánh giá khả năng chuyển độ nặng."
    ),
    "nang": (
        "**Hồi sức ngay tại đơn vị có khả năng cấp cứu** (HSCC/ICU). "
        "Theo dõi sinh hiệu + Hct mỗi 1-2h. Bù dịch khẩn cấp theo phác đồ sốc "
        "(crystalloid → cao phân tử nếu chỉ định). Hội chẩn chuyên khoa "
        "Truyền nhiễm/Hồi sức. Chuẩn bị máu/chế phẩm máu nếu xuất huyết."
    ),
}


# ===========================================================================
# Prompt cho triage — JSON BẮT BUỘC, có evidence + recommendation tham khảo
# ===========================================================================

TRIAGE_SYSTEM_PROMPT = """Bạn là **trợ lý phân loại SXHD** của CliMed. Nhiệm vụ: đọc thông tin bệnh nhân và phân loại theo hướng dẫn Bộ Y tế VN.

# 3 NHÓM PHÂN LOẠI
- **thuong** (mức 1): SXHD thông thường — sốt + ≥2 dấu hiệu (đau đầu/cơ/khớp/hốc mắt, ban xuất huyết, sung huyết), CHƯA có dấu hiệu cảnh báo.
- **canhbao** (mức 2): có ≥1 dấu hiệu cảnh báo — đau bụng nhiều/liên tục, nôn nhiều, gan to >2cm, xuất huyết niêm mạc, lừ đừ/vật vã, **Hct tăng kèm PLT giảm nhanh**, tràn dịch lâm sàng.
- **nang** (mức 3): sốc giảm thể tích (tay chân lạnh, mạch nhanh, HA tụt), **xuất huyết nặng** (tiêu hóa/nội tạng), hoặc **suy đa cơ quan** (gan/thận/tim/thần kinh).

# QUY TẮC PHÂN LOẠI (theo độ ưu tiên)
1. Có dấu hiệu sốc / xuất huyết nặng / suy tạng → `nang`.
2. Có dấu hiệu cảnh báo (kể cả chỉ 1) → `canhbao`.
3. Chỉ có triệu chứng thông thường, không có cảnh báo → `thuong`.
4. Nếu **không chắc**, ưu tiên mức nặng hơn (an toàn). KHÔNG bao giờ down-grade vô căn cứ.

# OUTPUT BẮT BUỘC

Bạn PHẢI output đúng định dạng dưới, KHÔNG markdown code block, KHÔNG text ngoài:

```
[Lý do]: <1-2 câu, trích dẫn con số XN cụ thể (ví dụ "PLT N5 = 41 K/uL")>
[Bằng chứng]: <liệt kê 2-4 bằng chứng từ BN, mỗi cái 1 dòng bắt đầu bằng "-">
[Cờ đỏ]: <list dấu hiệu nặng nếu có, nếu không có để "(không)">

{"label": "thuong" | "canhbao" | "nang", "confidence": 0.0-1.0, "reasoning": "<≤2 câu>"}
```

JSON ở dòng CUỐI CÙNG, riêng 1 dòng. CẤM thêm text sau JSON.
"""


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class TriageResult:
    """Kết quả phân loại 1 BN."""
    # Phân loại
    severity_level: Optional[int] = None        # 1, 2, 3 (None nếu parse fail)
    severity_label: Optional[str] = None        # "thuong" | "canhbao" | "nang"
    severity_text_vi: Optional[str] = None

    # Độ tin cậy
    confidence: Optional[float] = None          # 0.0-1.0
    reasoning: Optional[str] = None             # ≤2 câu giải thích từ LLM

    # Bằng chứng có cấu trúc (parsed từ response)
    evidence_lines: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)

    # Khuyến nghị hành động (deterministic theo severity)
    recommended_action: Optional[str] = None

    # Metadata kỹ thuật
    raw_response: str = ""
    latency_ms: float = 0.0
    parsed_successfully: bool = False
    parse_error: Optional[str] = None
    patient_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def short_summary(self) -> str:
        """1 dòng cho log/CLI."""
        if not self.parsed_successfully:
            return f"BN {self.patient_id}: parse FAIL ({self.parse_error})"
        return (f"BN {self.patient_id}: {self.severity_label} "
                f"(mức {self.severity_level}, conf={self.confidence:.2f}) "
                f"- {self.latency_ms:.0f}ms")


# ===========================================================================
# Parser cho format mới (có Lý do/Bằng chứng/Cờ đỏ + JSON cuối)
# ===========================================================================

_LYDO_RE = re.compile(r"\[Lý do\]:?\s*(.+?)(?=\[Bằng chứng\]|\[Cờ đỏ\]|\{|$)",
                      re.IGNORECASE | re.DOTALL)
_BANGCHUNG_RE = re.compile(r"\[Bằng chứng\]:?\s*(.+?)(?=\[Cờ đỏ\]|\{|$)",
                           re.IGNORECASE | re.DOTALL)
_CODO_RE = re.compile(r"\[Cờ đỏ\]:?\s*(.+?)(?=\{|$)",
                      re.IGNORECASE | re.DOTALL)
_JSON_RE = re.compile(r"\{[^{}]*\"label\"[^{}]*\}", re.DOTALL)


def _parse_triage_response(raw: str) -> Dict[str, Any]:
    """
    Parse output của bot:
      [Lý do]: ...
      [Bằng chứng]: - ... - ...
      [Cờ đỏ]: ...
      {"label": ..., "confidence": ..., "reasoning": ...}

    Trả về dict có các trường + parse_error nếu fail JSON.
    """
    out: Dict[str, Any] = {
        "lydo": None,
        "evidence_lines": [],
        "red_flags": [],
        "label": None,
        "confidence": None,
        "reasoning": None,
        "parse_error": None,
    }

    # Lý do
    m = _LYDO_RE.search(raw)
    if m:
        out["lydo"] = m.group(1).strip()

    # Bằng chứng (list bullet)
    m = _BANGCHUNG_RE.search(raw)
    if m:
        ev = m.group(1).strip()
        # Split theo dòng "-" bullet hoặc xuống dòng
        ev_lines: List[str] = []
        for line in re.split(r"\n+|(?<=\.)\s+(?=-)", ev):
            line = line.strip().lstrip("-").strip()
            if line and len(line) >= 4:
                ev_lines.append(line)
        out["evidence_lines"] = ev_lines[:6]

    # Cờ đỏ
    m = _CODO_RE.search(raw)
    if m:
        flags_text = m.group(1).strip()
        if flags_text.lower().startswith("(không") or flags_text.lower() == "không":
            out["red_flags"] = []
        else:
            flags = [f.strip().lstrip("-").strip()
                     for f in re.split(r"\n+|,(?=\s)", flags_text)
                     if f.strip() and len(f.strip()) >= 3]
            out["red_flags"] = flags[:6]

    # JSON cuối
    json_matches = list(_JSON_RE.finditer(raw))
    if not json_matches:
        out["parse_error"] = "Không tìm thấy JSON object trong response"
        return out
    last_json_str = json_matches[-1].group(0)
    try:
        # Vietnamese accent in JSON value sometimes uses smart quotes — normalize
        cleaned = (last_json_str
                   .replace('"', '"').replace('"', '"')
                   .replace(''', "'").replace(''', "'"))
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        out["parse_error"] = f"JSON decode error: {e}"
        return out

    label = parsed.get("label")
    if isinstance(label, str):
        # Normalize VN biến thể
        ln = label.strip().lower()
        label_map = {
            "thuong": "thuong", "thường": "thuong", "thông thường": "thuong",
            "canhbao": "canhbao", "cảnh báo": "canhbao", "canh bao": "canhbao",
            "nang": "nang", "nặng": "nang", "severe": "nang",
        }
        out["label"] = label_map.get(ln, ln)

    try:
        out["confidence"] = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        out["confidence"] = None

    r = parsed.get("reasoning")
    if r:
        out["reasoning"] = str(r).strip()
    return out


# ===========================================================================
# Patient renderer — wrap eval_chatbot.render_patient_card với fallback
# ===========================================================================

def _render_patient_input(patient: Union[pd.Series, dict, str],
                         patient_id: Optional[str] = None,
                         focus_days: Optional[List[int]] = None) -> str:
    """
    Chuẩn hóa đầu vào về text BN. Chấp nhận:
      • pd.Series — 1 hàng từ DataFrame
      • dict — đã có sẵn các field
      • str — đã render text từ trước, trả luôn
    """
    if isinstance(patient, str):
        return patient
    if isinstance(patient, dict):
        patient = pd.Series(patient)
    if isinstance(patient, pd.Series):
        try:
            from eval_chatbot import render_patient_card
            return render_patient_card(patient,
                                       patient_id=patient_id or "?",
                                       focus_days=focus_days or [3, 4, 5, 6, 7])
        except Exception as e:
            logger.warning("render_patient_card fail, fallback đơn giản: %s", e)
            # Fallback nhẹ: dump các field non-null
            lines = [f"--- Bệnh nhân #{patient_id or '?'} ---"]
            for k, v in patient.items():
                if pd.notna(v):
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)
    raise TypeError(f"patient phải là str/dict/Series, nhận: {type(patient)}")


# ===========================================================================
# TriageBot
# ===========================================================================

class TriageBot:
    """
    Mini AI bot phân loại SXHD 3 mức. Dùng LLM (Qwen via Ollama hoặc HF) +
    optional RAG/Graph để bổ sung context guideline.

    Ví dụ:
        bot = TriageBot()                       # auto backend
        result = bot.triage(patient_row)
        print(result.severity_label, result.recommended_action)
    """

    def __init__(self,
                 backend: str = "auto",
                 model: Optional[str] = None,
                 use_rag: bool = True,
                 use_graph: bool = False,        # graph cần BN context đầy đủ
                 temperature: float = 0.2,
                 max_tokens: int = 500,
                 evidence_top_k: int = 3,
                 ):
        from llm_qwen import get_llm_client
        self.llm = get_llm_client(backend=backend, model=model)
        self.use_rag = use_rag
        self.use_graph = use_graph
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.evidence_top_k = evidence_top_k
        logger.info("TriageBot khởi tạo với backend=%s, use_rag=%s, use_graph=%s",
                    self.llm.name, use_rag, use_graph)

    # ----- Internal helpers -----

    def _build_messages(self, patient_text: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
        ]
        # RAG: lấy evidence cards về phân loại SXHD chung
        if self.use_rag:
            try:
                from rag_optimizer import build_evidence_cards, format_evidence_block
                # Query cố định + có thể thêm 1 từ khóa từ patient text nếu cần
                rag_query = "phân loại SXHD thông thường cảnh báo nặng tiêu chí"
                cards = build_evidence_cards(rag_query, top_n=self.evidence_top_k)
                if cards:
                    messages.append({
                        "role": "system",
                        "content": format_evidence_block(cards),
                    })
            except Exception as e:
                logger.debug("RAG skip: %s", e)

        messages.append({
            "role": "user",
            "content": f"Phân loại bệnh nhân sau:\n\n{patient_text}",
        })
        return messages

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Non-streaming call. Đo latency."""
        try:
            return self.llm.chat(messages,
                                 temperature=self.temperature,
                                 max_tokens=self.max_tokens)
        except Exception as e:
            logger.exception("LLM call fail")
            raise

    # ----- Public API -----

    def triage(self,
               patient: Union[pd.Series, dict, str],
               patient_id: Optional[str] = None,
               focus_days: Optional[List[int]] = None,
               ) -> TriageResult:
        """
        Phân loại 1 BN. Trả về TriageResult đầy đủ.

        Lỗi LLM không raise — đóng gói vào TriageResult.parsed_successfully=False.
        """
        patient_text = _render_patient_input(patient, patient_id, focus_days)
        messages = self._build_messages(patient_text)

        t0 = time.time()
        try:
            raw = self._call_llm(messages)
        except Exception as e:
            return TriageResult(
                raw_response="",
                latency_ms=(time.time() - t0) * 1000,
                parsed_successfully=False,
                parse_error=f"LLM call fail: {e}",
                patient_id=str(patient_id) if patient_id is not None else None,
            )
        latency = (time.time() - t0) * 1000

        # Parse
        parsed = _parse_triage_response(raw)
        if parsed["parse_error"] or not parsed["label"]:
            return TriageResult(
                raw_response=raw,
                latency_ms=latency,
                parsed_successfully=False,
                parse_error=parsed["parse_error"] or "Không có label trong JSON",
                patient_id=str(patient_id) if patient_id is not None else None,
            )

        label = parsed["label"]
        if label not in SEVERITY_LEVEL_OF:
            return TriageResult(
                raw_response=raw,
                latency_ms=latency,
                parsed_successfully=False,
                parse_error=f"Label '{label}' không thuộc {{thuong, canhbao, nang}}",
                patient_id=str(patient_id) if patient_id is not None else None,
            )

        return TriageResult(
            severity_level=SEVERITY_LEVEL_OF[label],
            severity_label=label,
            severity_text_vi=SEVERITY_VI[label],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            evidence_lines=parsed["evidence_lines"],
            red_flags=parsed["red_flags"],
            recommended_action=RECOMMENDED_ACTION[label],
            raw_response=raw,
            latency_ms=latency,
            parsed_successfully=True,
            patient_id=str(patient_id) if patient_id is not None else None,
        )

    def triage_batch(self,
                     df: pd.DataFrame,
                     ground_truth_col: Optional[str] = "_ground_truth_label",
                     focus_days: Optional[List[int]] = None,
                     on_progress: Optional[Callable[[int, int, TriageResult], None]] = None,
                     max_patients: Optional[int] = None,
                     ) -> List[Dict[str, Any]]:
        """
        Phân loại nhiều BN. Trả về list dict đúng schema mà
        `eval_metrics.compute_all_metrics` cần.

        df: DataFrame mỗi hàng 1 BN. Nếu có cột `ground_truth_col`, sẽ kèm GT
            vào output để eval.
        """
        records: List[Dict[str, Any]] = []
        total = len(df) if max_patients is None else min(len(df), max_patients)
        for i, (idx, row) in enumerate(df.iterrows()):
            if max_patients is not None and i >= max_patients:
                break
            gt = (row.get(ground_truth_col)
                  if ground_truth_col and ground_truth_col in row.index else None)
            result = self.triage(row, patient_id=str(i), focus_days=focus_days)
            rec = {
                "patient_id": i,
                "ground_truth": gt,
                "bot_label": result.severity_label,
                "bot_confidence": result.confidence,
                "bot_reasoning": result.reasoning,
                "is_valid": result.parsed_successfully,
                "raw_response": result.raw_response,
                "latency_ms": result.latency_ms,
                "evidence_lines": result.evidence_lines,
                "red_flags": result.red_flags,
                "recommended_action": result.recommended_action,
                "parse_error": result.parse_error,
            }
            records.append(rec)
            if on_progress is not None:
                try:
                    on_progress(i + 1, total, result)
                except Exception:
                    pass
        return records


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) < 2:
        print("Cách dùng:")
        print("  python mini_ai_bot.py 'paste BN text vào đây'")
        print("  python mini_ai_bot.py @path/to/patient.txt")
        print("  python mini_ai_bot.py --batch data/eval_test_set/test_holdout.csv")
        sys.exit(1)

    arg = sys.argv[1]
    bot = TriageBot()

    if arg == "--batch":
        if len(sys.argv) < 3:
            print("Cần đường dẫn CSV")
            sys.exit(1)
        df = pd.read_csv(sys.argv[2])
        max_n = int(sys.argv[3]) if len(sys.argv) > 3 else None
        records = bot.triage_batch(df, max_patients=max_n,
                                   on_progress=lambda i, n, r: print(
                                       f"  [{i}/{n}] {r.short_summary()}"))
        # Quick metrics
        try:
            from eval_metrics import compute_all_metrics, format_report
            rep = compute_all_metrics(records)
            print("\n" + format_report(rep))
        except Exception as e:
            print(f"Metrics fail: {e}")
        sys.exit(0)

    if arg.startswith("@"):
        with open(arg[1:], encoding="utf-8") as f:
            patient_text = f.read()
    else:
        patient_text = arg

    result = bot.triage(patient_text, patient_id="cli")
    print("\n=== KẾT QUẢ ===")
    print(f"Phân loại:    {result.severity_label} (mức {result.severity_level}) - {result.severity_text_vi}")
    print(f"Confidence:   {result.confidence}")
    print(f"Reasoning:    {result.reasoning}")
    print(f"Bằng chứng:")
    for ev in result.evidence_lines:
        print(f"  - {ev}")
    if result.red_flags:
        print(f"🚨 Cờ đỏ:")
        for f in result.red_flags:
            print(f"  - {f}")
    print(f"\n💊 Khuyến nghị:\n  {result.recommended_action}")
    print(f"\nLatency: {result.latency_ms:.0f}ms")