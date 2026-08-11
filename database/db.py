"""
database/db.py
──────────────
Thin SQLite wrapper providing a context-managed connection and transparent
CRUD helpers. Keeps all SQL visible and readable – no ORM magic.
"""

import sqlite3
import json
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence


# Statuses a leave request may hold. Anything else is a programming error.
LEAVE_STATUSES = ("Pending", "Approved", "Rejected", "Cancelled")

# Statuses that occupy the calendar, i.e. that a new request may not overlap.
# Pending is included because every self-service request starts Pending, so
# restricting the check to Approved would let a user double-book freely.
BLOCKING_LEAVE_STATUSES = ("Pending", "Approved")


class RecordValidationError(ValueError):
    """
    Raised when a caller tries to write a record that would be malformed or
    would conflict with existing data.

    Callers are expected to catch this and turn it into a user-facing message;
    it is not an internal assertion.
    """


def _parse_iso_date(value: Any, field: str) -> date:
    """Parse a YYYY-MM-DD string, or raise RecordValidationError naming the field."""
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{field} is required")
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise RecordValidationError(
            f"{field} must be a calendar date in YYYY-MM-DD format, got {value!r}"
        ) from None


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
        # SQLite ignores REFERENCES clauses unless this is enabled per connection,
        # so without it the FKs declared in schema.py are decorative and a request
        # can be written against a user_id that does not exist.
        conn.execute("PRAGMA foreign_keys=ON")
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
        """
        Insert a leave request after checking the row is well-formed.

        Raises:
            RecordValidationError: dates unparseable or reversed, empty reason,
                                   unknown status, or a user_id with no user.
        """
        start, end = self._validated_range(start_date, end_date)
        if not isinstance(reason, str) or not reason.strip():
            raise RecordValidationError("reason is required")
        if status not in LEAVE_STATUSES:
            raise RecordValidationError(
                f"status must be one of {LEAVE_STATUSES}, got {status!r}"
            )

        try:
            return self.fetch_last_id(
                """INSERT INTO leave_requests
                   (user_id, start_date, end_date, reason, status, created_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (user_id, start.isoformat(), end.isoformat(), reason.strip(), status),
            )
        except sqlite3.IntegrityError as e:
            # Almost always the users FK; surface it as a validation failure
            # rather than a raw driver error.
            raise RecordValidationError(
                f"cannot store leave request for user_id {user_id}: {e}"
            ) from e

    def leave_days(self, start_date: str, end_date: str) -> int:
        """Inclusive day count for a leave range. Raises on a malformed range."""
        start, end = self._validated_range(start_date, end_date)
        return (end - start).days + 1

    def get_overlapping_leave_requests(
        self,
        user_id: int,
        start_date: str,
        end_date: str,
        statuses: Sequence[str] = BLOCKING_LEAVE_STATUSES,
    ) -> List[Dict]:
        """
        Return existing requests whose dates intersect [start_date, end_date].

        Dates are stored as ISO strings, so lexicographic comparison is also
        chronological — which only holds because insert_leave_request()
        normalises every value to YYYY-MM-DD before writing it.
        """
        start, end = self._validated_range(start_date, end_date)
        placeholders = ", ".join("?" for _ in statuses)
        return self.fetch_all(
            f"""SELECT * FROM leave_requests
                WHERE user_id = ?
                  AND status IN ({placeholders})
                  AND start_date <= ?
                  AND end_date   >= ?
                ORDER BY start_date""",
            (user_id, *statuses, end.isoformat(), start.isoformat()),
        )

    def submit_leave_request(
        self,
        user_id: int,
        start_date: str,
        end_date: str,
        reason: str,
        status: str = "Pending",
    ) -> int:
        """
        Store a leave request and deduct the days from the balance atomically.

        Doing these as two separate execute() calls opened a window where the
        request existed but the balance had not moved (or vice versa). Both now
        share one transaction, and the balance UPDATE is guarded so it cannot
        drive remaining_leaves negative under a concurrent write.

        Raises:
            RecordValidationError: on a malformed range, or if the balance row
                                   is missing or no longer covers the request.
        """
        start, end = self._validated_range(start_date, end_date)
        if not isinstance(reason, str) or not reason.strip():
            raise RecordValidationError("reason is required")
        if status not in LEAVE_STATUSES:
            raise RecordValidationError(
                f"status must be one of {LEAVE_STATUSES}, got {status!r}"
            )
        days = (end - start).days + 1

        try:
            with self._connect() as conn:
                cur = conn.execute(
                    """INSERT INTO leave_requests
                       (user_id, start_date, end_date, reason, status, created_at)
                       VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                    (user_id, start.isoformat(), end.isoformat(), reason.strip(), status),
                )
                request_id = cur.lastrowid

                updated = conn.execute(
                    """UPDATE leave_balance
                          SET used_leaves      = used_leaves + ?,
                              remaining_leaves = remaining_leaves - ?
                        WHERE user_id = ? AND remaining_leaves >= ?""",
                    (days, days, user_id, days),
                ).rowcount
                if updated != 1:
                    # Rolls back the INSERT above via _connect()'s error path.
                    raise RecordValidationError(
                        f"leave balance for user {user_id} does not cover "
                        f"{days} day(s); nothing was recorded"
                    )
                return request_id
        except sqlite3.IntegrityError as e:
            raise RecordValidationError(
                f"cannot store leave request for user_id {user_id}: {e}"
            ) from e

    def insert_hr_request(self, user_id: int, request_text: str) -> int:
        """
        Insert an HR ticket.

        Raises:
            RecordValidationError: empty request text, or an unknown user_id.
        """
        if not isinstance(request_text, str) or not request_text.strip():
            raise RecordValidationError("request_text is required")
        try:
            return self.fetch_last_id(
                """INSERT INTO hr_requests (user_id, request_text, created_at)
                   VALUES (?, ?, datetime('now'))""",
                (user_id, request_text.strip()),
            )
        except sqlite3.IntegrityError as e:
            raise RecordValidationError(
                f"cannot store HR request for user_id {user_id}: {e}"
            ) from e

    @staticmethod
    def _validated_range(start_date: str, end_date: str):
        """Parse both dates and assert the range runs forwards."""
        start = _parse_iso_date(start_date, "start_date")
        end = _parse_iso_date(end_date, "end_date")
        if end < start:
            raise RecordValidationError(
                f"end_date {end.isoformat()} is before start_date {start.isoformat()}"
            )
        return start, end

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
            try:
                row["agent_state"] = json.loads(row["agent_state"] or "{}")
            except (TypeError, json.JSONDecodeError):
                # A corrupt state blob must not make the session unusable —
                # drop the slot-filling state and let routing start fresh.
                print(
                    f"[DatabaseManager] Corrupt agent_state for session "
                    f"{session_id[:8]}... – resetting it"
                )
                row["agent_state"] = {}
                row["active_agent"] = None
        return row
