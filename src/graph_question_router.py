"""
graph_question_router.py — CliMedV2
====================================

Wrapper quanh `graph_engine.graph_facts_for_chat` để bổ sung:

  1. **Question-aware activation**: Activate thêm node trong KG dựa trên
     KEYWORDS xuất hiện trong câu hỏi user — không chỉ dựa vào triệu chứng/lab
     của bệnh nhân.
     → User hỏi "aspirin có dùng được không?" vẫn activate node `aspirin`
        kể cả khi BN không có triệu chứng nào nhập trong form.

  2. **Question-aware fact filtering**: Sau khi traverse, score mỗi fact theo
     độ trùng từ khóa với câu hỏi → drop facts không liên quan.
     → Câu hỏi về thuốc thì không hiển thị fact về triệu chứng không liên quan.

Đây là wrapper KHÔNG sửa graph_engine.py gốc. An toàn cho rollback.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(c))


def _normalize_for_match(s: str) -> str:
    """Lowercase + bỏ dấu để match từ khóa với label node."""
    return _strip_diacritics((s or "").lower())


def _activate_from_question(graph, question: str) -> List[str]:
    """
    Tìm node nào trong graph có label trùng khớp từ trong câu hỏi.

    Match strategy: label (đã bỏ dấu, lowercase) là substring của câu hỏi
    (đã bỏ dấu, lowercase). Yêu cầu label >= 3 ký tự để tránh false-positive.

    Trả về list node_id activated từ question.
    """
    if not question:
        return []
    q_norm = _normalize_for_match(question)
    activated: List[str] = []
    for nid, data in graph.G.nodes(data=True):
        label = data.get("label", "")
        if len(label) < 3:
            continue
        label_norm = _normalize_for_match(label)
        # Match: label xuất hiện như "phrase" trong question
        # Dùng word boundary để tránh "ast" match trong "aspirin"
        if re.search(rf"\b{re.escape(label_norm)}\b", q_norm):
            activated.append(nid)
    if activated:
        logger.debug("Question activated %d extra nodes: %s",
                     len(activated), activated[:5])
    return activated


def _score_fact_by_question(fact, graph, question_tokens: Set[str]) -> float:
    """
    Score 1 fact theo overlap giữa labels trong path và tokens câu hỏi.

    Trả về float 0-1+: cao = liên quan, thấp = không liên quan.
    """
    if not question_tokens:
        return 0.5   # fallback trung tính nếu không có token

    labels_tokens: Set[str] = set()
    for nid in fact.path:
        label = graph.label_of(nid)
        for tok in re.findall(r"\w+", _normalize_for_match(label)):
            if len(tok) >= 3:
                labels_tokens.add(tok)
    if not labels_tokens:
        return 0.0
    overlap = labels_tokens & question_tokens
    # Score = số tokens overlap / tokens trong path (cap 1.0)
    return min(1.0, len(overlap) / max(1, len(labels_tokens))) + fact.score * 0.3


def _tokenize_question(question: str) -> Set[str]:
    """Tokens >= 3 ký tự, normalized."""
    norm = _normalize_for_match(question)
    return {t for t in re.findall(r"\w+", norm) if len(t) >= 3}


def graph_facts_for_chat(
    symptoms: Optional[Dict[str, bool]] = None,
    lab_records: Optional[pd.DataFrame] = None,
    rule_severity: Optional[str] = None,
    question: Optional[str] = None,
    max_hops: int = 2,
    max_facts: int = 6,
) -> str:
    """
    Drop-in replacement cho graph_engine.graph_facts_for_chat — bổ sung
    `question` param để activation và filter bám sát câu hỏi.

    Trả về "" nếu không có gì để inject.
    """
    try:
        from graph_engine import get_graph, format_graph_facts_for_llm
        g = get_graph()
    except Exception as e:
        logger.warning("Graph load failed: %s", e)
        return ""

    # 1. Activate từ context BN (như cũ)
    activated_from_ctx: List[str] = []
    try:
        activated_from_ctx = g.activate_from_context(
            symptoms=symptoms,
            lab_records=lab_records,
            rule_severity=rule_severity,
        )
    except Exception as e:
        logger.warning("activate_from_context fail: %s", e)

    # 2. Activate THÊM từ câu hỏi (cải tiến mới)
    activated_from_q = _activate_from_question(g, question or "")

    # 3. Hợp nhất, giữ thứ tự (q-activated lên trước → traverse từ chúng trước)
    activated = list(dict.fromkeys(activated_from_q + activated_from_ctx))
    if not activated:
        return ""

    # 4. Traverse
    try:
        facts = g.traverse(activated, max_hops=max_hops)
    except Exception as e:
        logger.warning("graph traverse fail: %s", e)
        return ""
    if not facts:
        return ""

    # 5. Question-aware re-score & filter
    q_tokens = _tokenize_question(question or "")
    if q_tokens:
        scored = [(f, _score_fact_by_question(f, g, q_tokens)) for f in facts]
        scored.sort(key=lambda x: -x[1])
        # Drop facts có score < 0.15 (rất ít liên quan)
        scored = [(f, s) for f, s in scored if s >= 0.15]
        facts_filtered = [f for f, _ in scored[:max_facts]]
        if facts_filtered:
            facts = facts_filtered
        else:
            # Nếu filter quá mạnh, giữ top max_facts gốc theo path score
            facts = facts[:max_facts]
    else:
        facts = facts[:max_facts]

    # 6. Format
    try:
        text = format_graph_facts_for_llm(g, activated, facts, max_facts=max_facts)
    except Exception as e:
        logger.warning("format_graph_facts_for_llm fail: %s", e)
        return ""

    if q_tokens and activated_from_q:
        # Thêm 1 dòng nhắc LLM rằng đã activate từ question
        text += (f"\n\n*(Đã kích hoạt {len(activated_from_q)} node từ "
                 f"câu hỏi: {', '.join(activated_from_q[:3])}...)*")
    return text