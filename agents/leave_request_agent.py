"""
agents/leave_request_agent.py
──────────────────────────────
Multi-turn slot-filling agent for leave requests.

Flow:
  1. User says "I want leave" / "I'm sick" / "Need a day off"
  2. Agent extracts: start_date, end_date, reason (via LLM)
  3. Missing slots → agent asks for them one at a time
  4. Once all slots filled → shows balance, asks confirmation
  5. User confirms → insert into DB, clear active_agent

State keys (ctx.agent_state):
  start_date, end_date, reason, awaiting_confirmation
"""

import json
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from core.session import SessionContext

EXTRACT_PROMPT = """You are an HR assistant helping a user fill out a leave request form.

Extract the following fields from the user message:
- start_date: The first day of leave (YYYY-MM-DD format). Resolve relative terms like "tomorrow" or
  "Monday" based on today being {today}. Leave null if not mentioned.
- end_date: The last day of leave (YYYY-MM-DD). If only one day is mentioned, start_date = end_date.
  Leave null if not clearly stated.
- reason: Why the user needs leave (sick, personal, vacation, family, etc.). Leave null if not given.

Respond ONLY with valid JSON:
{{"start_date": "..." or null, "end_date": "..." or null, "reason": "..." or null}}
"""


class LeaveRequestAgent(BaseAgent):
    """Collects leave details via multi-turn dialogue, then submits to the database."""

    display_name = "Leave Request Agent"
    colour = "green"

    def handle(self, user_input: str, ctx: SessionContext) -> str:
        state = ctx.agent_state  # persistent between turns

        # ── Step 1: If awaiting_confirmation, handle yes/no ───────────────
        if state.get("awaiting_confirmation"):
            return self._handle_confirmation(user_input, ctx, state)

        # ── Step 2: Try to extract / update slots from current message ─────
        today = self._get_today()
        extracted = self._extract_slots(user_input, ctx, today)

        # Merge newly extracted slots into persistent state
        for key in ("start_date", "end_date", "reason"):
            if extracted.get(key) and not state.get(key):
                state[key] = extracted[key]

        ctx.agent_state = state
        ctx.active_agent = "LeaveRequestAgent"  # stay active till done

        # ── Step 3: Ask for any missing slots ─────────────────────────────
        if not state.get("start_date"):
            return "Sure, I'll help you apply for leave! 📅 What date would your leave start?"

        if not state.get("end_date"):
            return f"Got it – starting {state['start_date']}. When will your leave end? (or type 'same day' for a single day)"

        if not state.get("reason"):
            return "What's the reason for your leave? (e.g. sick, personal, vacation)"

        # ── Step 4: All slots collected – show summary and ask to confirm ──
        balance = self.db.get_leave_balance(ctx.user_id)
        remaining = balance["remaining_leaves"] if balance else "?"

        state["awaiting_confirmation"] = True
        ctx.agent_state = state

        return (
            f"Here's a summary of your leave request:\n\n"
            f"📅 **From:** {state['start_date']}\n"
            f"📅 **To:** {state['end_date']}\n"
            f"📝 **Reason:** {state['reason']}\n\n"
            f"You currently have **{remaining} days** of leave remaining.\n\n"
            f"Shall I go ahead and submit this request? (yes / no)"
        )

    def _handle_confirmation(self, user_input: str, ctx: SessionContext, state: Dict) -> str:
        """Process user's yes/no confirmation."""
        words = set(user_input.lower().split())
        cancel_words = {"no", "n", "cancel", "nope", "nah", "nevermind", "stop", "abort"}
        yes_words = {"yes", "y", "confirm", "sure", "ok", "yep", "yeah", "submit", "go"}

        # Check for cancellation first (takes priority)
        if words & cancel_words or "cancel" in user_input.lower() or "changed my mind" in user_input.lower():
            ctx.active_agent = None
            ctx.agent_state = {}
            return "No problem! Your leave request has been cancelled. Let me know if you need anything else. 😊"

        if words & yes_words:
            leave_id = self.db.insert_leave_request(
                user_id=ctx.user_id,
                start_date=state["start_date"],
                end_date=state["end_date"],
                reason=state["reason"],
            )
            # Clear the slot-filling state
            ctx.active_agent = None
            ctx.agent_state = {}
            return (
                f"✅ Your leave request has been submitted successfully!\n\n"
                f"**Reference ID:** LR-{leave_id:04d}\n"
                f"📅 {state['start_date']} → {state['end_date']}\n"
                f"📝 Reason: {state['reason']}\n"
                f"Status: **Pending** – HR will review and confirm."
            )
        else:
            # User cancelled
            ctx.active_agent = None
            ctx.agent_state = {}
            return "No problem! Your leave request has been cancelled. Let me know if you need anything else. 😊"

    def _extract_slots(self, user_input: str, ctx: SessionContext, today: str) -> Dict:
        """Use LLM to extract leave slots from the user message + history."""
        prompt = EXTRACT_PROMPT.format(today=today)
        messages = self._build_messages(prompt, user_input, ctx)
        try:
            raw = self.llm.chat(messages, json_mode=True, temperature=0.0)
            return json.loads(raw)
        except Exception:
            return {}

    @staticmethod
    def _get_today() -> str:
        from datetime import date
        return date.today().isoformat()
