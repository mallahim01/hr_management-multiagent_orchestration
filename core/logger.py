"""
core/logger.py
──────────────
Appends timestamped JSON records to data/interactions.log.
Kept intentionally simple – one line per record for easy grep / review.

Two record shapes share the file, distinguished by the "event" field:
  • "interaction"  – one conversation turn (written by main.py / app.py)
  • anything else  – a structured domain event, e.g. a rejected leave request
"""

import json
import os
from datetime import datetime
from typing import Any, Optional


class InteractionLogger:
    """Writes one JSON line per interaction turn to a log file."""

    def __init__(self, log_path: str = "data/interactions.log") -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path

    def log(
        self,
        session_id: str,
        user_id: int,
        user_input: str,
        intent: Optional[str],
        confidence: Optional[float],
        target_agent: Optional[str],
        agent_response: str,
        backend: str,
    ) -> None:
        """Append a single interaction record to the log file."""
        self._write({
            "event": "interaction",
            "session_id": session_id,
            "user_id": user_id,
            "user_input": user_input,
            "intent": intent,
            "confidence": confidence,
            "target_agent": target_agent,
            "agent_response": agent_response[:200],  # truncate long responses
            "backend": backend,
        })

    def log_event(self, event: str, **fields: Any) -> None:
        """
        Append a structured domain event.

        Used for things that are not a conversation turn but still need an
        audit trail – notably a leave request rejected by validation, where the
        user sees a friendly sentence but an operator needs the numbers.

        Args:
            event:  Short snake_case event name, e.g. 'leave_request_rejected'.
            fields: Any JSON-serialisable context for the event.
        """
        self._write({"event": event, **fields})

    # ── Internal ─────────────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        """Append one JSON line, timestamp first."""
        line = {"timestamp": datetime.utcnow().isoformat() + "Z", **record}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
