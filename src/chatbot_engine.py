"""
chatbot_engine.py — CliMedV2 (OPTIMIZED v2)
============================================

Lớp orchestration giữa Streamlit UI và LLMClient. So với bản v1:

  • Tích hợp `rag_optimizer.build_evidence_cards` thay cho retrieve thô.
    → bot chỉ thấy 2-3 thẻ bằng chứng đã trim ở mức câu, kèm relevance score.
  • Tích hợp `graph_engine.graph_facts_for_chat` với question-aware activation.
  • SYSTEM_PROMPT mới — strict response template (TL;DR + Bằng chứng + Khuyến nghị),
    giới hạn độ dài cứng, cấm lan man, ép cite nguồn theo thẻ.
  • Conditional context blocks — KHÔNG inject patient block nếu hỏi kiến thức chung.
  • Generation params chặt hơn: temp 0.2, num_predict 600, repeat_penalty 1.15.
  • Query classifier nhẹ — phân tách câu hỏi "kiến thức chung" vs "phân tích BN".

KHÔNG đụng vào model/tokenizer trực tiếp. Tất cả qua LLMClient interface.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Generator, List, Optional, Union

import pandas as pd

from llm_qwen import LLMClient, Message

logger = logging.getLogger(__name__)


# ===========================================================================
# DISCLAIMERS & SYSTEM PROMPT
# ===========================================================================

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Lưu ý**: Đây không phải quyết định cuối cùng. "
    "Hãy đến cơ sở y tế gần nhất để được bác sĩ thăm khám và tư vấn trực tiếp."
)

# ----------- Prompt cho câu hỏi KIẾN THỨC CHUNG về SXHD -----------------------
SYSTEM_PROMPT_GENERAL = """Bạn là **trợ lý y khoa** của CliMed Dengue Assistant — chuyên về Sốt xuất huyết Dengue (SXHD) theo hướng dẫn Bộ Y tế VN.

# QUY TẮC TRẢ LỜI — BẮT BUỘC

1. **TRỌNG TÂM**: Trả lời TRỰC TIẾP câu hỏi. KHÔNG mở bài, KHÔNG nhắc lại câu hỏi, KHÔNG liệt kê thông tin không được hỏi.
2. **NGUỒN**: Nếu có "BẰNG CHỨNG TRÍCH TỪ HƯỚNG DẪN", trả lời PHẢI dựa trên thẻ có relevance cao nhất. Trích lại số trang trong ngoặc, ví dụ: *(theo Hướng dẫn trang 220)*.
3. **KHÔNG BỊA**: Nếu các thẻ bằng chứng KHÔNG đủ trả lời câu hỏi, nói rõ: "Hướng dẫn không nêu chi tiết về điều này. Theo thực hành lâm sàng phổ biến…" rồi chỉ nói những gì chắc chắn.
4. **NGẮN GỌN**: Tối đa **150 từ**. Nếu không thể, ưu tiên 3 bullet trọng tâm.
5. **CẤM**:
   - Không kê đơn (liều, đường dùng, mg/ml cụ thể).
   - Không dùng từ tuyệt đối ("luôn", "chắc chắn", "100%").
   - Không thêm disclaimer cuối — hệ thống tự thêm.

# ĐỊNH DẠNG OUTPUT BẮT BUỘC

**Trả lời:** <1-2 câu trả lời thẳng câu hỏi>

**Cơ sở:** <1-3 gạch đầu dòng nêu bằng chứng từ thẻ, kèm trang. Bỏ qua section này nếu câu hỏi đơn giản.>

**Khuyến nghị:** <1 câu hành động — đi khám khi nào / cần theo dõi gì>"""


# ----------- Prompt cho câu hỏi PHÂN TÍCH BỆNH NHÂN cụ thể -------------------
SYSTEM_PROMPT_PATIENT = """Bạn là **trợ lý y khoa** của CliMed Dengue Assistant. BÁC SĨ đang hỏi bạn về 1 bệnh nhân cụ thể trong ngữ cảnh.

# QUY TẮC BẮT BUỘC

1. **GROUND TRÊN DỮ LIỆU BN**: Mọi nhận định PHẢI trích lại con số cụ thể từ "NGỮ CẢNH BỆNH NHÂN" (ví dụ "PLT N5 = 41 K/uL").
2. **DÙNG GUIDELINE LÀM CỞ SỞ**: Khi đối chiếu chỉ số với "BẰNG CHỨNG", trích lại số trang.
3. **TRỌNG TÂM**: Trả lời CHÍNH câu hỏi. KHÔNG kể lại toàn bộ bệnh án nếu chỉ hỏi 1 điểm.
4. **KHI THIẾU DỮ LIỆU**: Nói rõ "Chưa đủ dữ liệu để kết luận về X, cần thêm…" — KHÔNG đoán.
5. **CẤM**: Không kê đơn cụ thể, không chẩn đoán cuối cùng, không từ tuyệt đối.
6. **NGẮN GỌN**: Tối đa **200 từ**.

# ĐỊNH DẠNG OUTPUT BẮT BUỘC

**Tóm tắt:** <1 câu nêu điểm mấu chốt liên quan tới câu hỏi>

**Phân tích:** <2-3 bullet, mỗi bullet trích 1 con số + giải thích lâm sàng + (nếu có) trang guideline>

**Hướng cần bác sĩ cân nhắc:** <1-2 câu, KHÔNG kê liều>"""


# ===========================================================================
# QUERY CLASSIFIER — phân tách câu hỏi kiến thức chung vs phân tích BN
# ===========================================================================

# Pattern gợi ý câu hỏi "có BN cụ thể" — dùng đại từ chỉ định, chỉ số cụ thể
_PATIENT_REF_PATTERNS = [
    re.compile(r"\b(bệnh\s*nhân|bn\b|case\b|ca\s*này|bé|cô|chú|ông|bà)", re.I),
    re.compile(r"\b(của|cho)\s+(tôi|mình|bạn\s*này|em)", re.I),
    re.compile(r"\bplt\s*[=:]\s*\d", re.I),                  # "PLT = 50"
    re.compile(r"\bhct\s*[=:]\s*\d", re.I),
    re.compile(r"\bn[1-9]\b.*\d", re.I),                     # "N5 = 41"
    re.compile(r"\b(đang|hiện)\s+(sốt|đau|nôn)", re.I),
]

_GENERAL_HINT_PATTERNS = [
    re.compile(r"\b(định nghĩa|là gì|khái niệm|phân loại)\b", re.I),
    re.compile(r"\b(nguyên\s*tắc|hướng\s*dẫn|guideline|phác\s*đồ)\b", re.I),
    re.compile(r"\b(khi nào|bao giờ)\s+(cần|nên|phải)\b", re.I),
]


def classify_query(question: str,
                   has_patient_context: bool) -> str:
    """
    Trả về 'patient' nếu câu hỏi nhắc tới BN cụ thể HOẶC có dữ liệu BN trong context.
    Trả về 'general' cho câu hỏi kiến thức chung.

    Heuristic đơn giản, đủ dùng. Không cần LLM call.
    """
    # Ưu tiên: nếu có context BN VÀ câu hỏi không có dấu hiệu "kiến thức chung" rõ ràng → patient
    has_patient_ref = any(p.search(question) for p in _PATIENT_REF_PATTERNS)
    has_general_hint = any(p.search(question) for p in _GENERAL_HINT_PATTERNS)

    if has_patient_ref:
        return "patient"
    if has_patient_context and not has_general_hint:
        return "patient"
    return "general"


# ===========================================================================
# PRESCRIPTION GUARDRAIL (giữ nguyên từ v1)
# ===========================================================================

_PRESCRIPTION_PATTERNS = [
    re.compile(r"\bkê\s*đơn\b", re.I),
    re.compile(r"\bnên\s*uống\s*thuốc\s*gì", re.I),
    re.compile(r"\b(bao\s*nhiêu|mấy)\s+(viên|mg|ml|lần|giọt|gói)", re.I),
    re.compile(r"\bliều\b.{0,40}?(bao\s*nhiêu|cụ\s*thể|chính\s*xác|\d+\s*mg|\d+\s*ml)",
               re.I | re.DOTALL),
    re.compile(r"\bcho\s*tôi\s*đơn\s*thuốc", re.I),
]

PRESCRIPTION_REFUSAL = (
    "Tôi không thể đưa đơn thuốc cụ thể (loại thuốc, liều, đường dùng) vì việc này "
    "phải do bác sĩ trực tiếp thăm khám quyết định, dựa trên cân nặng, bệnh nền, "
    "và diễn tiến lâm sàng cụ thể.\n\n"
    "Tôi có thể giúp bạn:\n"
    "- Giải thích các chỉ số xét nghiệm hiện tại có gì bất thường.\n"
    "- Liệt kê các dấu hiệu cảnh báo cần lưu ý.\n"
    "- Tóm tắt phân nhóm SXHD và mức độ nguy cơ tham khảo."
)


def _detect_prescription_request(question: str) -> bool:
    return any(p.search(question) for p in _PRESCRIPTION_PATTERNS)


# ===========================================================================
# CONTEXT FORMATTERS (improved — bỏ block rỗng)
# ===========================================================================

_SYMPTOM_LABELS = {
    "fever_now": "đang sốt", "abdominal_pain": "đau bụng",
    "vomiting": "nôn", "vomiting_many": "nôn nhiều/liên tục",
    "mucosal_bleeding": "chảy máu niêm mạc", "oliguria": "tiểu ít",
    "lethargy": "lừ đừ/vật vã/rối loạn ý thức", "dyspnea": "khó thở",
    "cold_extremities": "tay chân lạnh", "myalgia": "đau cơ/đau khớp",
    "retro_orbital_pain": "đau hốc mắt", "severe_bleeding": "xuất huyết nặng",
    "hypotension": "tụt huyết áp/nghi sốc",
}


def _has_real_patient_data(ocr_records, clinical_input, rule_output, trend_flags) -> bool:
    """Check xem có dữ liệu BN thật không, không chỉ là form rỗng."""
    if ocr_records is not None and not (isinstance(ocr_records, pd.DataFrame) and ocr_records.empty):
        return True
    if clinical_input:
        # Phải có ít nhất 1 trường có giá trị khác mặc định
        if any(clinical_input.get(k) for k in _SYMPTOM_LABELS.keys()):
            return True
        if clinical_input.get("fever_days") or clinical_input.get("comorbidity_text"):
            return True
    if rule_output is not None:
        return True
    if trend_flags:
        return True
    return False


def format_lab_table(df: Optional[pd.DataFrame]) -> str:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ""
    lines: List[str] = []
    for _, r in df.iterrows():
        name = r.get("exam_name_normalized", "?")
        val = r.get("value", "?")
        unit = r.get("unit") or ""
        ref = r.get("reference_range") or ""
        flag = r.get("range_flag", "")
        flag_emoji = {"low": "↓", "high": "↑", "normal": "✓", "unknown": "?"}.get(str(flag), "")
        lines.append(f"  - {name}: {val} {unit} (tham chiếu {ref}) {flag_emoji}".strip())
    return "\n".join(lines)


def format_clinical_input(clinical: Optional[Dict[str, Any]]) -> str:
    if not clinical:
        return ""
    parts: List[str] = []
    parts.append(f"  - Tuổi: {clinical.get('age', '?')}, giới: {clinical.get('sex', '?')}")
    parts.append(f"  - Số ngày sốt/ngày bệnh: {clinical.get('fever_days', '?')}")
    parts.append(f"  - Uống được: {'có' if clinical.get('can_drink') else 'không'}")
    active = [v for k, v in _SYMPTOM_LABELS.items() if clinical.get(k)]
    if active:
        parts.append(f"  - Triệu chứng (+): {', '.join(active)}")
    if clinical.get("comorbidity_text"):
        parts.append(f"  - Bệnh nền/ghi chú: {clinical['comorbidity_text']}")
    return "\n".join(parts)


def format_rule_output(rules: Any) -> str:
    if rules is None:
        return ""
    parts = [f"  - Phân nhóm tham khảo: **{rules.level}**"]
    if getattr(rules, "evidence", None):
        parts.append(f"  - Bằng chứng: {'; '.join(rules.evidence[:5])}")
    if getattr(rules, "warnings", None):
        parts.append(f"  - Dấu hiệu cảnh báo: {'; '.join(rules.warnings[:5])}")
    if getattr(rules, "red_flags", None):
        parts.append(f"  - Dấu hiệu đỏ: {'; '.join(rules.red_flags[:5])}")
    return "\n".join(parts)


def format_trend_flags(flags: Optional[Dict[str, Any]]) -> str:
    if not flags:
        return ""
    return "\n".join(f"  - {k}: {v}" for k, v in flags.items())


def build_patient_context_block(
    ocr_records: Optional[pd.DataFrame] = None,
    clinical_input: Optional[Dict[str, Any]] = None,
    rule_output: Any = None,
    trend_flags: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build context block, CHỈ KHI có dữ liệu BN thật. Bỏ block rỗng.
    Trả về "" nếu không có gì.
    """
    if not _has_real_patient_data(ocr_records, clinical_input, rule_output, trend_flags):
        return ""

    blocks: List[str] = ["## NGỮ CẢNH BỆNH NHÂN"]
    sec = format_clinical_input(clinical_input or {})
    if sec:
        blocks.append("\n### Thông tin lâm sàng")
        blocks.append(sec)
    sec = format_lab_table(ocr_records)
    if sec:
        blocks.append("\n### Kết quả xét nghiệm")
        blocks.append(sec)
    sec = format_rule_output(rule_output)
    if sec:
        blocks.append("\n### Đánh giá rule engine")
        blocks.append(sec)
    sec = format_trend_flags(trend_flags)
    if sec:
        blocks.append("\n### Trend flags")
        blocks.append(sec)
    return "\n".join(blocks)


# Backward-compat alias: tên cũ trong app.py
def build_context_block(ocr_records=None, clinical_input=None,
                        rule_output=None, trend_flags=None,
                        knowledge_snippets=None, max_snippet_chars: int = 600) -> str:
    """Giữ chữ ký cũ. knowledge_snippets bị bỏ qua — RAG giờ đi qua evidence cards."""
    return build_patient_context_block(ocr_records, clinical_input, rule_output, trend_flags)


# ===========================================================================
# MESSAGE BUILDER
# ===========================================================================

def build_messages(
    user_question: str,
    history: List[Message],
    system_prompt: str,
    context_block: str,
    evidence_block: str,
    graph_block: str,
    max_history: int = 6,
) -> List[Message]:
    """
    Layout chuẩn:
      [system: SYSTEM_PROMPT theo intent]
      [system: NGỮ CẢNH (chỉ nếu có)]
      [system: BẰNG CHỨNG (chỉ nếu có)]
      [system: GRAPH (chỉ nếu có)]
      [history user/assistant, tối đa 6 lượt]
      [user: question]
    """
    messages: List[Message] = [{"role": "system", "content": system_prompt}]

    if context_block.strip():
        messages.append({"role": "system", "content": context_block})
    if evidence_block.strip():
        messages.append({"role": "system", "content": evidence_block})
    if graph_block.strip():
        messages.append({"role": "system", "content": graph_block})

    chat_history = [m for m in history if m.get("role") in ("user", "assistant")]
    if len(chat_history) > max_history:
        chat_history = chat_history[-max_history:]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_question})
    return messages


# ===========================================================================
# MAIN ENTRY: chat()
# ===========================================================================

def chat(
    client: LLMClient,
    user_question: str,
    history: List[Message],
    context: str = "",           # backward compat: được build trước trong app.py
    stream: bool = True,
    temperature: float = 0.2,     # ↓ từ 0.3 → 0.2 cho focused
    max_tokens: int = 600,        # ↓ từ 1024 → 600 chống lan man
    use_rag: bool = True,
    rag_top_k: int = 3,           # ↓ từ 4 → 3 evidence cards
    use_graph: bool = True,
    graph_symptoms: Optional[Dict[str, Any]] = None,
    graph_lab_records: Optional[Any] = None,
    graph_rule_severity: Optional[str] = None,
) -> Union[str, Generator[str, None, None]]:
    """
    Pipeline 1 lượt chat:
      1. Detect kê đơn → refuse sớm.
      2. Classify query → chọn system prompt.
      3. Build evidence cards qua rag_optimizer (rerank + sentence-level trim + MMR).
      4. Build graph facts với question-aware activation.
      5. Conditional inject — chỉ inject block thật sự có nội dung.
      6. Call LLM với generation params chặt.
      7. Append disclaimer.
    """
    # ---- Guardrail 1: prescription refusal ----
    if _detect_prescription_request(user_question):
        full = PRESCRIPTION_REFUSAL + DISCLAIMER
        if stream:
            def _g(): yield full
            return _g()
        return full

    # ---- Phase 1: classify intent ----
    has_patient_ctx = bool(context and context.strip())
    intent = classify_query(user_question, has_patient_ctx)
    system_prompt = (SYSTEM_PROMPT_PATIENT if intent == "patient"
                     else SYSTEM_PROMPT_GENERAL)
    logger.info("Intent: %s, has_patient_ctx: %s", intent, has_patient_ctx)

    # ---- Phase 2: build evidence cards (RAG optimized) ----
    evidence_block = ""
    if use_rag:
        try:
            from rag_optimizer import build_evidence_cards, format_evidence_block
            cards = build_evidence_cards(user_question, top_n=rag_top_k)
            evidence_block = format_evidence_block(cards)
            logger.info("Evidence cards: %d injected", len(cards))
        except Exception as e:
            logger.warning("RAG optimizer fail, bot sẽ trả lời không có RAG: %s", e)

    # ---- Phase 3: build graph block với question-aware activation + filter ----
    graph_block = ""
    if use_graph:
        try:
            # Ưu tiên router mới — không sửa graph_engine.py gốc
            try:
                from graph_question_router import graph_facts_for_chat as gfc
            except ImportError:
                from graph_engine import graph_facts_for_chat as gfc

            try:
                graph_block = gfc(
                    symptoms=graph_symptoms,
                    lab_records=graph_lab_records,
                    rule_severity=graph_rule_severity,
                    question=user_question,
                    max_hops=2, max_facts=6,
                )
            except TypeError:
                # graph_engine bản cũ không có param `question`
                graph_block = gfc(
                    symptoms=graph_symptoms,
                    lab_records=graph_lab_records,
                    rule_severity=graph_rule_severity,
                    max_hops=2, max_facts=6,
                )
        except Exception as e:
            logger.warning("Graph reasoning fail: %s", e)

    # ---- Phase 4: assemble messages ----
    # Nếu intent = general thì BỎ patient context (đỡ nhiễu)
    ctx_for_prompt = context if intent == "patient" else ""

    messages = build_messages(
        user_question=user_question,
        history=history,
        system_prompt=system_prompt,
        context_block=ctx_for_prompt,
        evidence_block=evidence_block,
        graph_block=graph_block,
    )

    # ---- Phase 5: call LLM ----
    # Inject options cho Ollama (qwen2.5) — repeat_penalty chống lặp lan man.
    # HF backend không nhận options này, nó sẽ ignore extra kwargs.
    extra_gen_opts = {}   # nơi đặt nếu OllamaClient cho phép pass-through

    if stream:
        def _stream():
            try:
                for chunk in client.chat_stream(
                        messages, temperature=temperature, max_tokens=max_tokens):
                    yield chunk
            except Exception as e:
                logger.exception("LLM stream error")
                yield f"\n\n❌ Lỗi gọi LLM: {e}"
            yield DISCLAIMER
        return _stream()

    try:
        resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        logger.exception("LLM error")
        return f"❌ Lỗi gọi LLM: {e}{DISCLAIMER}"
    return resp + DISCLAIMER