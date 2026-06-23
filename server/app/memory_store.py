"""In-process per-session conversation memory: a recent-turn window plus a
folded text summary for everything older.

Runs entirely in this process's RAM — no database, no disk, no network call.
When a session's message list grows past MAX_RECENT_MESSAGES, the oldest
message is folded into `summary` as a short truncated line instead of being
dropped outright, so multi-turn context survives past the recent window
without keeping an unbounded transcript.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

MAX_RECENT_MESSAGES = 12
MAX_SUMMARY_CHARS = 1500
TRUNCATE_CHARS = 160


def _truncate(text: str, limit: int = TRUNCATE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


@dataclass
class SessionMemory:
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    last_active: float = field(default_factory=time.time)


class MemoryStore:
    """Thread-safe session_id -> SessionMemory map."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionMemory] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if not content:
            return
        with self._lock:
            memory = self._sessions.setdefault(session_id, SessionMemory())
            memory.messages.append({"role": role, "content": content})
            memory.last_active = time.time()
            while len(memory.messages) > MAX_RECENT_MESSAGES:
                oldest = memory.messages.pop(0)
                self._fold_into_summary(memory, oldest)

    def get_recent_messages(self, session_id: str) -> list[dict]:
        with self._lock:
            memory = self._sessions.get(session_id)
            return list(memory.messages) if memory else []

    def get_summary(self, session_id: str) -> str:
        with self._lock:
            memory = self._sessions.get(session_id)
            return memory.summary if memory else ""

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    @staticmethod
    def _fold_into_summary(memory: SessionMemory, message: dict) -> None:
        """Appends one truncated line to the summary, trimming from the front
        (oldest summary content first) if the summary grows past its cap."""
        line = f"{message['role']}: {_truncate(message['content'])}"
        combined = f"{memory.summary}\n{line}".strip() if memory.summary else line
        if len(combined) > MAX_SUMMARY_CHARS:
            combined = combined[-MAX_SUMMARY_CHARS:]
            newline = combined.find("\n")
            if newline != -1:
                combined = combined[newline + 1 :]
        memory.summary = combined


memory_store = MemoryStore()
