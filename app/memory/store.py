"""Conversation memory, keyed by session id.

Design goal: start simple (in-memory) but make it trivial to swap in Redis or
PostgreSQL later. We do that by programming against an abstract `BaseMemoryStore`
interface — the rest of the app only depends on the interface, never on the
concrete implementation.

To switch to Redis: write a `RedisMemoryStore(BaseMemoryStore)` that implements
the same three methods, and change the one line in `app/main.py` that creates
the store. Nothing else has to change.
"""

from abc import ABC, abstractmethod
from collections import defaultdict


class BaseMemoryStore(ABC):
    """Interface every memory backend must implement."""

    @abstractmethod
    def get_messages(self, session_id: str) -> list[dict]:
        """Return the message history for a session (oldest first)."""

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append one message (role = 'user' or 'assistant') to a session."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all messages for a session."""


class InMemoryMemoryStore(BaseMemoryStore):
    """A dict-backed store. Data lives only as long as the process runs.

    Perfect for local development and tests. Not suitable for multi-process or
    multi-server deployments — use Redis/PostgreSQL there.
    """

    def __init__(self) -> None:
        # session_id -> list of {"role": ..., "content": ...}
        self._sessions: dict[str, list[dict]] = defaultdict(list)

    def get_messages(self, session_id: str) -> list[dict]:
        # Return a copy so callers can't mutate our internal state by accident.
        return list(self._sessions[session_id])

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._sessions[session_id].append({"role": role, "content": content})

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
