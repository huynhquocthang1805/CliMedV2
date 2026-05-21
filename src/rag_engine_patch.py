"""
rag_engine_patch.py — CliMedV2
==============================

File nhỏ chỉ để OVERRIDE `rag_engine.retrieve_for_chat` cho bất cứ chỗ nào còn
gọi tên hàm cũ (không nằm trong chatbot_engine v2). chatbot_engine v2 đã gọi
trực tiếp `rag_optimizer.build_evidence_cards`, nên file này chỉ cần thiết nếu
bạn còn module khác (CLI, eval, notebook…) đang import `retrieve_for_chat`.

Cách dùng:
    # Trong app.py, đầu file:
    import rag_engine_patch    # noqa: F401  (chỉ cần import là patched)
"""

import logging

logger = logging.getLogger(__name__)

try:
    import rag_engine
    from rag_optimizer import build_evidence_cards

    _ORIGINAL = rag_engine.retrieve_for_chat

    def retrieve_for_chat(query: str, top_k: int = 3):
        """
        Wrapper override: trả về list[str] (giữ chữ ký cũ) NHƯNG mỗi phần tử
        là 1 evidence card đã trim ở mức câu, có citation.
        """
        try:
            cards = build_evidence_cards(query, top_n=top_k)
        except Exception as e:
            logger.warning("rag_optimizer fail, fallback bản cũ: %s", e)
            return _ORIGINAL(query, top_k=top_k)
        if not cards:
            return []
        out = []
        for c in cards:
            out.append(f"[Nguồn: {c.citation}] (rel={c.relevance:.2f})\n{c.text}")
        return out

    rag_engine.retrieve_for_chat = retrieve_for_chat
    logger.info("rag_engine.retrieve_for_chat đã được patch bởi rag_optimizer.")

except ImportError as e:
    logger.warning("Không patch được rag_engine: %s", e)