"""
core/logger.py
──────────────
Appends timestamped JSON interaction records to data/interactions.log.
Kept intentionally simple – one line per interaction for easy grep / review.
"""

import json
import os
from datetime import datetime
from typing import Optional


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
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "user_id": user_id,
            "user_input": user_input,
            "intent": intent,
            "confidence": confidence,
            "target_agent": target_agent,
            "agent_response": agent_response[:200],  # truncate long responses
            "backend": backend,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
