"""
chatbot_engine.py — CliMedV2
============================

Lớp orchestration giữa Streamlit UI và `llm_qwen.LLMClient`.

Trách nhiệm:
  • Build context block từ session state (OCR records, clinical input, rule output, trend flags, RAG snippets).
  • Build danh sách messages chuẩn để gửi LLM.
  • Append disclaimer y khoa bắt buộc vào mọi câu trả lời.
  • Detect các câu hỏi "kê đơn thuốc cụ thể" → từ chối, redirect sang bác sĩ.

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
# Disclaimers & system prompt
# ===========================================================================

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Lưu ý**: Đây không phải quyết định cuối cùng. "
    "Hãy đến cơ sở y tế gần nhất để được bác sĩ thăm khám và tư vấn trực tiếp."
)

SYSTEM_PROMPT = """Bạn là **trợ lý y khoa nội bộ** của ứng dụng CliMed Dengue Assistant — chuyên hỗ trợ phân tích dữ liệu bệnh nhân sốt xuất huyết Dengue (SXHD).

# QUY TẮC BẮT BUỘC
1. Trả lời bằng **tiếng Việt**, ngắn gọn, có cấu trúc rõ.
2. CHỈ dựa trên dữ liệu trong "NGỮ CẢNH BỆNH NHÂN" và "TRI THỨC NỀN" — KHÔNG bịa thông tin, KHÔNG đoán mò.
3. Khi nhắc đến chỉ số xét nghiệm, dẫn lại con số CỤ THỂ từ ngữ cảnh (ví dụ: "PLT của bệnh nhân là 80 K/uL").
4. Khi không đủ dữ liệu để trả lời, nói thẳng: "Chưa đủ dữ liệu để kết luận, cần thêm…".
5. KHÔNG kê đơn thuốc cụ thể (liều, đường dùng), KHÔNG đưa chẩn đoán cuối cùng. Chỉ gợi ý hướng cần bác sĩ cân nhắc.
6. Phong cách: cẩn trọng, lâm sàng, có dẫn chứng. Không dùng từ tuyệt đối ("chắc chắn", "100%").

# KIẾN THỨC NỀN VỀ SXHD (theo guideline Bộ Y tế Việt Nam)

**Phân nhóm:**
- **SXHD thông thường**: sốt + ≥2 dấu hiệu (đau đầu, đau cơ-khớp, đau hốc mắt, ban xuất huyết, da niêm sung huyết).
- **SXHD có dấu hiệu cảnh báo (warning signs)**: đau bụng nhiều/liên tục, nôn nhiều, gan to >2cm, xuất huyết niêm mạc, lừ đừ/vật vã, HCT tăng kèm tiểu cầu giảm nhanh, tràn dịch lâm sàng.
- **SXHD nặng**: sốc giảm thể tích, xuất huyết nặng (tiêu hóa/nội tạng), suy đa cơ quan (gan/thận/tim/thần kinh).

**Diễn biến điển hình theo ngày sốt:**
- N1-N3 (sốt cao): ít biến chứng, theo dõi triệu chứng.
- N4-N6 (giai đoạn nguy hiểm): hết sốt → có thể vào sốc; cần theo dõi sát HCT, PLT, dấu cảnh báo.
- N7+ (hồi phục): tái hấp thu dịch, có thể quá tải dịch.

**Thuốc cần TRÁNH** (gây xuất huyết / tổn thương gan):
- Aspirin, ibuprofen, các NSAID khác
- Analgin (metamizole)
**Hạ sốt được phép**: Paracetamol (theo cân nặng, không quá liều khuyến cáo).

# ĐỊNH DẠNG TRẢ LỜI
- Mở đầu: tóm tắt 1 câu cô đọng nhất.
- Phân tích: 2-4 ý chính, mỗi ý 1-2 câu.
- Khuyến nghị: hành động cần làm (theo dõi gì, đi khám khi nào).
- KHÔNG cần thêm disclaimer (hệ thống tự thêm)."""

# Pattern phát hiện câu hỏi "yêu cầu kê đơn cụ thể" → từ chối
_PRESCRIPTION_PATTERNS = [
    re.compile(r"\bkê\s*đơn\b", re.I),
    re.compile(r"\bnên\s*uống\s*thuốc\s*gì", re.I),
    # "(bao nhiêu | mấy) (viên | mg | ml | lần)"
    re.compile(r"\b(bao\s*nhiêu|mấy)\s+(viên|mg|ml|lần|giọt|gói)", re.I),
    # "liều" + (gì đó) + (bao nhiêu | cụ thể | chính xác | mg | mL)
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
    """Phát hiện câu hỏi yêu cầu kê đơn cụ thể."""
    return any(p.search(question) for p in _PRESCRIPTION_PATTERNS)


# ===========================================================================
# Context formatters — biến session_state thành text gọn cho prompt
# ===========================================================================

def format_lab_table(df: Optional[pd.DataFrame]) -> str:
    """Format bảng OCR thành bullet list ngắn."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return "  (chưa có dữ liệu xét nghiệm OCR)"

    lines: List[str] = []
    for _, r in df.iterrows():
        name = r.get("exam_name_normalized", "?")
        val = r.get("value", "?")
        unit = r.get("unit") or ""
        ref = r.get("reference_range") or ""
        flag = r.get("range_flag", "")
        flag_emoji = {"low": "↓", "high": "↑", "normal": "✓",
                      "unknown": "?"}.get(str(flag), "")
        lines.append(f"  - {name}: {val} {unit} "
                     f"(tham chiếu {ref}) {flag_emoji}".strip())
    return "\n".join(lines)


_SYMPTOM_LABELS = {
    "fever_now": "đang sốt",
    "abdominal_pain": "đau bụng",
    "vomiting": "nôn",
    "vomiting_many": "nôn nhiều/liên tục",
    "mucosal_bleeding": "chảy máu niêm mạc",
    "oliguria": "tiểu ít",
    "lethargy": "lừ đừ/vật vã/rối loạn ý thức",
    "dyspnea": "khó thở",
    "cold_extremities": "tay chân lạnh",
    "myalgia": "đau cơ/đau khớp",
    "retro_orbital_pain": "đau hốc mắt",
    "severe_bleeding": "xuất huyết nặng",
    "hypotension": "tụt huyết áp/nghi sốc",
}


def format_clinical_input(clinical: Optional[Dict[str, Any]]) -> str:
    """Format clinical input thành bullet list."""
    if not clinical:
        return "  (chưa có thông tin lâm sàng)"

    parts: List[str] = []
    parts.append(f"  - Tuổi: {clinical.get('age', '?')}, "
                 f"giới: {clinical.get('sex', '?')}")
    parts.append(f"  - Số ngày sốt/ngày bệnh: {clinical.get('fever_days', '?')}")
    parts.append(f"  - Uống được: "
                 f"{'có' if clinical.get('can_drink') else 'không'}")

    active_symptoms = [v for k, v in _SYMPTOM_LABELS.items()
                       if clinical.get(k)]
    if active_symptoms:
        parts.append(f"  - Triệu chứng (+): {', '.join(active_symptoms)}")

    if clinical.get("comorbidity_text"):
        parts.append(f"  - Bệnh nền/ghi chú: {clinical['comorbidity_text']}")
    return "\n".join(parts)


def format_rule_output(rules: Any) -> str:
    """Format kết quả rule engine."""
    if rules is None:
        return "  (chưa chạy rule engine)"

    parts = [f"  - Phân nhóm tham khảo: **{rules.level}**"]
    if getattr(rules, "evidence", None):
        parts.append(f"  - Bằng chứng: {'; '.join(rules.evidence[:5])}")
    if getattr(rules, "warnings", None):
        parts.append(f"  - Dấu hiệu cảnh báo: {'; '.join(rules.warnings[:5])}")
    if getattr(rules, "red_flags", None):
        parts.append(f"  - Dấu hiệu đỏ: {'; '.join(rules.red_flags[:5])}")
    return "\n".join(parts)


def format_trend_flags(flags: Optional[Dict[str, Any]]) -> str:
    """Format trend flags."""
    if not flags:
        return "  (chưa có)"
    return "\n".join(f"  - {k}: {v}" for k, v in flags.items())


def build_context_block(
    ocr_records: Optional[pd.DataFrame] = None,
    clinical_input: Optional[Dict[str, Any]] = None,
    rule_output: Any = None,
    trend_flags: Optional[Dict[str, Any]] = None,
    knowledge_snippets: Optional[List[str]] = None,
    max_snippet_chars: int = 600,
) -> str:
    """
    Build full context string để inject vào prompt.

    Layout:
      ## NGỮ CẢNH BỆNH NHÂN
        ### Thông tin lâm sàng
        ### Kết quả xét nghiệm
        ### Đánh giá rule engine
        ### Trend flags
      ## TRI THỨC NỀN (nếu có RAG snippets)
    """
    blocks: List[str] = []

    blocks.append("## NGỮ CẢNH BỆNH NHÂN")

    blocks.append("\n### Thông tin lâm sàng")
    blocks.append(format_clinical_input(clinical_input or {}))

    blocks.append("\n### Kết quả xét nghiệm (từ OCR / nhập tay)")
    blocks.append(format_lab_table(ocr_records))

    blocks.append("\n### Đánh giá rule engine")
    blocks.append(format_rule_output(rule_output))

    blocks.append("\n### Trend flags (xu hướng theo ngày)")
    blocks.append(format_trend_flags(trend_flags))

    if knowledge_snippets:
        blocks.append("\n## TRI THỨC NỀN (trích từ guideline điều trị)")
        for i, snip in enumerate(knowledge_snippets, 1):
            cleaned = (snip or "").strip()[:max_snippet_chars]
            blocks.append(f"\n### Đoạn {i}\n{cleaned}")

    return "\n".join(blocks)


# ===========================================================================
# Message builder
# ===========================================================================

def build_messages(
    user_question: str,
    history: List[Message],
    context: str,
    max_history: int = 8,
) -> List[Message]:
    """
    Ráp message stack chuẩn cho LLM.

    Layout:
      [system: SYSTEM_PROMPT]
      [system: context block]
      [history user/assistant... — limited to max_history turns]
      [user: current question]
    """
    messages: List[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"DỮ LIỆU HIỆN TẠI:\n\n{context}"},
    ]

    # Thêm history (chỉ user/assistant, bỏ system cũ)
    chat_history = [m for m in history
                    if m.get("role") in ("user", "assistant")]
    # Cắt history nếu quá dài (giữ lượt mới nhất)
    if len(chat_history) > max_history:
        chat_history = chat_history[-max_history:]
    messages.extend(chat_history)

    messages.append({"role": "user", "content": user_question})
    return messages


# ===========================================================================
# Main entry: chat()
# ===========================================================================

def chat(
    client: LLMClient,
    user_question: str,
    history: List[Message],
    context: str,
    stream: bool = True,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    use_rag: bool = False,
    rag_top_k: int = 4,
    use_graph: bool = False,
    graph_symptoms: Optional[Dict[str, Any]] = None,
    graph_lab_records: Optional[Any] = None,
    graph_rule_severity: Optional[str] = None,
) -> Union[str, Generator[str, None, None]]:
    """
    Chạy 1 lượt chat. Tự động:
      - Detect câu hỏi kê đơn → từ chối ngay (không gọi LLM).
      - (Nếu use_rag=True) Truy xuất top-k snippets từ guideline PDF
        và prepend vào context.
      - (Nếu use_graph=True) Activate knowledge graph từ context BN,
        traverse multi-hop, format facts có path → inject vào context.
      - Append disclaimer ở cuối câu trả lời.

    Args:
        use_rag: bật RAG retrieval từ guideline PDF.
        rag_top_k: số chunk lấy về.
        use_graph: bật reasoning từ knowledge graph SXHD.
        graph_symptoms: dict symptoms (từ st.session_state['clinical_input']).
        graph_lab_records: DataFrame OCR lab records.
        graph_rule_severity: severity từ rule engine (free text).

    Trả về:
      - stream=False: str
      - stream=True:  generator yield từng chunk, kết thúc bằng disclaimer.
    """
    # Guardrail 1: từ chối kê đơn (phát hiện sớm, tiết kiệm token)
    if _detect_prescription_request(user_question):
        full = PRESCRIPTION_REFUSAL + DISCLAIMER
        if stream:
            def _refusal_gen():
                yield full
            return _refusal_gen()
        return full

    # Guardrail 2: inject RAG snippets nếu được bật
    rag_block = ""
    if use_rag:
        try:
            from rag_engine import retrieve_for_chat
            snippets = retrieve_for_chat(user_question, top_k=rag_top_k)
            if snippets:
                rag_block = ("\n\n## TRI THỨC NỀN (trích từ guideline điều trị)\n\n"
                             + "\n\n---\n\n".join(snippets))
                logger.info("RAG: injected %d snippets", len(snippets))
        except Exception as e:
            logger.warning("RAG retrieval failed, bot vẫn trả lời: %s", e)

    # Guardrail 3: inject graph facts nếu được bật
    graph_block = ""
    if use_graph:
        try:
            from graph_engine import graph_facts_for_chat
            graph_block_text = graph_facts_for_chat(
                symptoms=graph_symptoms,
                lab_records=graph_lab_records,
                rule_severity=graph_rule_severity,
                max_hops=2, max_facts=10,
            )
            if graph_block_text:
                graph_block = "\n\n" + graph_block_text
                logger.info("Graph: injected facts block (%d chars)",
                            len(graph_block_text))
        except Exception as e:
            logger.warning("Graph reasoning failed: %s", e)

    full_context = context + rag_block + graph_block

    # Build messages và gọi LLM
    messages = build_messages(user_question, history, full_context)

    if stream:
        def _stream_with_disclaimer():
            try:
                for chunk in client.chat_stream(
                        messages, temperature=temperature, max_tokens=max_tokens):
                    yield chunk
            except Exception as e:
                logger.exception("LLM streaming error")
                yield f"\n\n❌ Lỗi gọi LLM: {e}"
            yield DISCLAIMER
        return _stream_with_disclaimer()

    try:
        response = client.chat(
            messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        logger.exception("LLM error")
        return f"❌ Lỗi gọi LLM: {e}{DISCLAIMER}"
    return response + DISCLAIMER