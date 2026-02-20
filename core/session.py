"""
core/session.py
────────────────
SessionContext holds all state for one conversation.
SessionManager persists it to/from the database between HTTP requests.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from database.db import DatabaseManager


@dataclass
class SessionContext:
    """
    All state for a single conversation session.

    active_agent:  Class name of the agent currently handling multi-turn dialogue
                   (e.g. 'LeaveRequestAgent'). None means fresh routing each turn.
    agent_state:   Free-form dict the active agent uses for slot-filling state.
    last_intent:   Most recently detected intent label.
    last_agent:    Most recently invoked agent (for display/logging).
    """
    session_id:   str
    user_id:      int
    active_agent: Optional[str]      = None
    agent_state:  Dict[str, Any]     = field(default_factory=dict)
    last_intent:  Optional[str]      = None
    last_agent:   Optional[str]      = None
    history:      List[Dict]         = field(default_factory=list)


class SessionManager:
    """Creates and persists SessionContext objects using the database."""

    def __init__(self, db: DatabaseManager, history_size: int = 3) -> None:
        self.db = db
        self.history_size = history_size  # how many recent *exchanges* to load

    def get_or_create(self, session_id: Optional[str], user_id: int) -> SessionContext:
        """
        Load an existing session from DB or create a fresh one.

        Args:
            session_id: Client-provided session UUID (None → create new).
            user_id:    Active user ID from config.

        Returns:
            A populated SessionContext.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # Try to restore state from DB
        saved = self.db.load_session(session_id)

        # Load recent conversation history (last history_size exchanges = pairs)
        recent = self.db.get_recent_messages(
            session_id, limit=self.history_size * 2
        )

        ctx = SessionContext(
            session_id=session_id,
            user_id=user_id,
            active_agent=saved["active_agent"] if saved else None,
            agent_state=saved["agent_state"] if saved else {},
            history=recent,
        )
        return ctx

    def save(self, ctx: SessionContext) -> None:
        """Persist session state back to the database."""
        self.db.save_session(
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            active_agent=ctx.active_agent,
            agent_state=ctx.agent_state,
        )
