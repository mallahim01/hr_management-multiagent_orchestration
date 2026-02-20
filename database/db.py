"""
database/db.py
──────────────
Thin SQLite wrapper providing a context-managed connection and transparent
CRUD helpers. Keeps all SQL visible and readable – no ORM magic.
"""

import sqlite3
import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


class DatabaseManager:
    """Manages a single SQLite database file for the HR demo."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @contextmanager
    def _connect(self):
        """Yield a connection with row_factory set so rows behave like dicts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row          # access columns by name
        conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Generic helpers ─────────────────────────────────────────────────────

    def execute(self, sql: str, params: tuple = ()) -> None:
        """Run a write statement (INSERT / UPDATE / DELETE / CREATE)."""
        with self._connect() as conn:
            conn.execute(sql, params)

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict]:
        """Return a single row as a dict, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> List[Dict]:
        """Return all matching rows as a list of dicts."""
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def fetch_last_id(self, sql: str, params: tuple = ()) -> int:
        """Run an INSERT and return the new row's id."""
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            return cur.lastrowid

    # ── High-level helpers (used by agents) ─────────────────────────────────

    def get_user(self, user_id: int) -> Optional[Dict]:
        return self.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))

    def get_leave_balance(self, user_id: int) -> Optional[Dict]:
        return self.fetch_one(
            "SELECT * FROM leave_balance WHERE user_id = ?", (user_id,)
        )

    def insert_leave_request(
        self,
        user_id: int,
        start_date: str,
        end_date: str,
        reason: str,
        status: str = "Pending",
    ) -> int:
        return self.fetch_last_id(
            """INSERT INTO leave_requests
               (user_id, start_date, end_date, reason, status, created_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))""",
            (user_id, start_date, end_date, reason, status),
        )

    def insert_hr_request(self, user_id: int, request_text: str) -> int:
        return self.fetch_last_id(
            """INSERT INTO hr_requests (user_id, request_text, created_at)
               VALUES (?, ?, datetime('now'))""",
            (user_id, request_text),
        )

    def get_leave_requests(self, user_id: int) -> List[Dict]:
        return self.fetch_all(
            "SELECT * FROM leave_requests WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        )

    def get_hr_requests(self, user_id: int) -> List[Dict]:
        return self.fetch_all(
            "SELECT * FROM hr_requests WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        )

    # ── Conversation history ─────────────────────────────────────────────────

    def save_message(
        self,
        session_id: str,
        user_id: int,
        role: str,
        content: str,
        agent: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> None:
        """Persist a single conversation turn."""
        self.execute(
            """INSERT INTO conversation_history
               (session_id, user_id, role, content, agent, intent, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            (session_id, user_id, role, content, agent, intent),
        )

    def get_recent_messages(
        self, session_id: str, limit: int = 6
    ) -> List[Dict]:
        """Return the most recent `limit` messages (user+assistant pairs)."""
        return self.fetch_all(
            """SELECT role, content, agent, intent, timestamp
               FROM conversation_history
               WHERE session_id = ?
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit),
        )[::-1]  # reverse so oldest is first

    def get_all_messages(self, user_id: int) -> List[Dict]:
        """Return all conversation history for a user (for observability)."""
        return self.fetch_all(
            """SELECT session_id, role, content, agent, intent, timestamp
               FROM conversation_history
               WHERE user_id = ?
               ORDER BY id DESC LIMIT 50""",
            (user_id,),
        )

    # ── Session state ────────────────────────────────────────────────────────

    def save_session(
        self,
        session_id: str,
        user_id: int,
        active_agent: Optional[str],
        agent_state: Dict,
    ) -> None:
        self.execute(
            """INSERT INTO sessions (session_id, user_id, active_agent, agent_state, updated_at)
               VALUES (?, ?, ?, ?, datetime('now'))
               ON CONFLICT(session_id) DO UPDATE SET
                   active_agent  = excluded.active_agent,
                   agent_state   = excluded.agent_state,
                   updated_at    = excluded.updated_at""",
            (session_id, user_id, active_agent, json.dumps(agent_state)),
        )

    def load_session(self, session_id: str) -> Optional[Dict]:
        row = self.fetch_one(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        if row:
            row["agent_state"] = json.loads(row["agent_state"] or "{}")
        return row
