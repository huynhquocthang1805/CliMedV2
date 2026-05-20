"""
chat_history.py — CliMedV2
==========================

Quản lý lịch sử chat đa-cuộc-trò-chuyện (multi-conversation), persistent xuống disk.

Đặc điểm:
  • Mỗi conversation = 1 file JSON trong `data/chat_sessions/`.
  • Mỗi file chứa: id, title, created_at, updated_at, messages[].
  • Mỗi message có: role ('user'|'assistant'), content, timestamp.
  • Title tự động sinh từ user message đầu tiên (truncate 60 chars).
  • Có thể: liệt kê, load, append, rename, delete.

API kiểu ChatGPT:
    sm = SessionManager()
    sm.create_session()                    # tạo cuộc mới, trả về session_id
    sm.append_message(sid, "user", "Hi")   # nối tin nhắn
    sm.list_sessions()                     # list tất cả (cho sidebar)
    sm.load_session(sid)                   # load full messages
    sm.delete_session(sid)
    sm.rename_session(sid, "Tên mới")
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[1]
SESSIONS_DIR = BASE_DIR / "data" / "chat_sessions"


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class ChatMessage:
    role: str             # 'user' | 'assistant' | 'system'
    content: str
    timestamp: str        # ISO format

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatSession:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        return cls(
            session_id=data["session_id"],
            title=data.get("title", "Untitled"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=[ChatMessage(**m) for m in data.get("messages", [])],
        )


# ===========================================================================
# SessionManager
# ===========================================================================

class SessionManager:
    """
    Quản lý chat sessions persistent.

    File layout:
        data/chat_sessions/
            <uuid_8>_<slug>.json
            <uuid_8>_<slug>.json
            ...
    """

    def __init__(self, sessions_dir: Path = SESSIONS_DIR):
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ----- Create / load / save -----

    def create_session(self, title: str = "Cuộc trò chuyện mới") -> str:
        """Tạo session mới, trả về session_id."""
        session_id = uuid.uuid4().hex[:8]
        now = _now_iso()
        session = ChatSession(
            session_id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            messages=[],
        )
        self._save(session)
        logger.info("Tạo session %s: %s", session_id, title)
        return session_id

    def load_session(self, session_id: str) -> Optional[ChatSession]:
        path = self._path_for(session_id)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ChatSession.from_dict(data)
        except Exception as e:
            logger.error("Lỗi load session %s: %s", session_id, e)
            return None

    def append_message(self, session_id: str, role: str, content: str) -> bool:
        """Append 1 message, auto-update title nếu là tin user đầu tiên."""
        session = self.load_session(session_id)
        if session is None:
            logger.warning("Session %s không tồn tại", session_id)
            return False

        msg = ChatMessage(role=role, content=content, timestamp=_now_iso())
        session.messages.append(msg)
        session.updated_at = msg.timestamp

        # Auto-rename: nếu title vẫn là default và đây là user msg đầu → đặt theo content
        is_default_title = session.title.startswith("Cuộc trò chuyện")
        if (is_default_title and role == "user"
                and sum(1 for m in session.messages if m.role == "user") == 1):
            session.title = _make_title_from_content(content)

        self._save(session)
        return True

    def list_sessions(self, sort_by: str = "updated") -> List[Dict[str, Any]]:
        """
        List tất cả sessions, mỗi item là metadata (không load messages).

        sort_by: 'updated' (mặc định, mới nhất trước) | 'created' | 'title'
        """
        items = []
        for path in self.dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                items.append({
                    "session_id": data["session_id"],
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "n_messages": len(data.get("messages", [])),
                })
            except Exception as e:
                logger.warning("Skip session file %s: %s", path.name, e)
                continue

        key = {"updated": "updated_at", "created": "created_at",
               "title": "title"}.get(sort_by, "updated_at")
        items.sort(key=lambda x: x.get(key, ""), reverse=(sort_by != "title"))
        return items

    def rename_session(self, session_id: str, new_title: str) -> bool:
        session = self.load_session(session_id)
        if session is None:
            return False
        session.title = (new_title or "Untitled").strip()[:80]
        session.updated_at = _now_iso()
        self._save(session)
        return True

    def delete_session(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        if path is None or not path.exists():
            return False
        path.unlink()
        logger.info("Đã xóa session %s", session_id)
        return True

    # ----- Internal -----

    def _save(self, session: ChatSession) -> None:
        slug = _slug(session.title)
        path = self.dir / f"{session.session_id}_{slug}.json"
        # Nếu title đã đổi, có thể có file cũ với slug khác → cleanup
        for old in self.dir.glob(f"{session.session_id}_*.json"):
            if old != path:
                old.unlink(missing_ok=True)
        path.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _path_for(self, session_id: str) -> Optional[Path]:
        matches = list(self.dir.glob(f"{session_id}_*.json"))
        return matches[0] if matches else None


# ===========================================================================
# Helpers
# ===========================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str, max_len: int = 30) -> str:
    """Slug an toàn cho filename: bỏ dấu, ký tự đặc biệt."""
    # Strip diacritics đơn giản (giữ ASCII + dấu vạch ngang)
    s = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s.strip())
    if not s:
        s = "untitled"
    return s[:max_len]


def _make_title_from_content(content: str, max_len: int = 60) -> str:
    """Tạo title từ message đầu tiên (cắt theo từ, không ngắt giữa)."""
    text = re.sub(r"\s+", " ", content).strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut + "..."


# ===========================================================================
# CLI test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    sm = SessionManager()
    sid = sm.create_session()
    sm.append_message(sid, "user", "Bệnh nhân nữ 16 tuổi, PLT giảm sâu")
    sm.append_message(sid, "assistant",
                       "Đây là dấu hiệu cảnh báo SXHD, cần nhập viện.")

    print(f"\nSessions hiện có ({len(sm.list_sessions())}):")
    for item in sm.list_sessions():
        print(f"  [{item['session_id']}] {item['title']} "
              f"({item['n_messages']} msg, updated {item['updated_at'][:19]})")

    s = sm.load_session(sid)
    print(f"\nLoad lại session {sid}:")
    for m in s.messages:
        print(f"  [{m.role}] {m.content}")