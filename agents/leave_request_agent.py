"""
agents/leave_request_agent.py
──────────────────────────────
Multi-turn slot-filling agent for leave requests.

Flow:
  1. User says "I want leave" / "I'm sick" / "Need a day off"
  2. Agent extracts: start_date, end_date, reason (via LLM)
  3. Missing slots → agent asks for them one at a time
  4. Once all slots filled → validate, then show balance and ask confirmation
  5. User confirms → re-validate, insert into DB, clear active_agent

Validation (see _validate_request) rejects a request that is malformed, that
exceeds the user's remaining balance, or that overlaps leave already on the
books. It runs twice — before asking for confirmation, and again at submission,
because the balance can move between the two turns. Every rejection is written
to the interaction log as a structured event alongside the friendly reply.

State keys (ctx.agent_state):
  start_date, end_date, reason, awaiting_confirmation
"""

import json
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from core.session import SessionContext
from database.db import RecordValidationError

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

        # ── Step 4: Validate before asking the user to confirm ─────────────
        rejection = self._validate_request(ctx, state)
        if rejection:
            return rejection

        # ── Step 5: All slots collected – show summary and ask to confirm ──
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
            # Re-validate: the balance may have moved since the summary turn.
            rejection = self._validate_request(ctx, state)
            if rejection:
                return rejection

            try:
                days = self.db.leave_days(state["start_date"], state["end_date"])
                leave_id = self.db.submit_leave_request(
                    user_id=ctx.user_id,
                    start_date=state["start_date"],
                    end_date=state["end_date"],
                    reason=state["reason"],
                )
            except RecordValidationError as e:
                # Last line of defence: the database refused the row. Nothing
                # was written, because insert and deduction share a transaction.
                return self._reject(
                    ctx, state,
                    reason_code="storage_rejected",
                    detail=str(e),
                    message=(
                        "❌ I couldn't save that leave request — the details didn't "
                        "pass a final check, so **nothing has been recorded**.\n\n"
                        "Please try again, or contact HR if this keeps happening."
                    ),
                )

            # Clear the slot-filling state
            ctx.active_agent = None
            ctx.agent_state = {}
            return (
                f"✅ Your leave request has been submitted successfully!\n\n"
                f"**Reference ID:** LR-{leave_id:04d}\n"
                f"📅 {state['start_date']} → {state['end_date']} ({days} days deducted)\n"
                f"📝 Reason: {state['reason']}\n"
                f"Status: **Pending** – HR will review and confirm."
            )
        else:
            # User cancelled
            ctx.active_agent = None
            ctx.agent_state = {}
            return "No problem! Your leave request has been cancelled. Let me know if you need anything else. 😊"

    # ── Validation ────────────────────────────────────────────────────────

    def _validate_request(self, ctx: SessionContext, state: Dict) -> Optional[str]:
        """
        Check the collected slots against the database.

        Returns a user-facing rejection message, or None when the request is
        acceptable. Every rejection is also written to the interaction log via
        _reject() so an operator can see the numbers behind the friendly text.
        """
        start_date, end_date = state.get("start_date"), state.get("end_date")

        # (0) Malformed dates — the LLM extracts these, so they are untrusted.
        try:
            days = self.db.leave_days(start_date, end_date)
        except RecordValidationError as e:
            return self._reject(
                ctx, state,
                reason_code="malformed_dates",
                detail=str(e),
                message=(
                    "❌ Those dates don't look right to me "
                    f"({start_date} → {end_date}).\n\n"
                    "Could you give me the start date again, in YYYY-MM-DD form?"
                ),
            )

        # (a) Not enough balance left.
        balance = self.db.get_leave_balance(ctx.user_id)
        if not balance:
            return self._reject(
                ctx, state,
                reason_code="no_balance_record",
                detail=f"no leave_balance row for user_id {ctx.user_id}",
                message=(
                    "❌ I couldn't find a leave balance on your record, so I can't "
                    "submit this request. Please contact HR."
                ),
            )

        remaining = balance["remaining_leaves"]
        if days > remaining:
            return self._reject(
                ctx, state,
                reason_code="insufficient_balance",
                detail=f"requested {days} day(s), {remaining} remaining",
                requested_days=days,
                remaining_leaves=remaining,
                message=(
                    f"❌ I can't submit this request — it's for **{days} day"
                    f"{'s' if days != 1 else ''}**, but you only have "
                    f"**{remaining} day{'s' if remaining != 1 else ''}** remaining.\n\n"
                    "You could shorten the request, or contact HR about unpaid leave. "
                    "What dates would you like instead?"
                ),
            )

        # (b) Overlaps leave already on the books.
        clashes = self.db.get_overlapping_leave_requests(
            ctx.user_id, start_date, end_date
        )
        if clashes:
            clash = clashes[0]
            return self._reject(
                ctx, state,
                reason_code="overlapping_request",
                detail=(
                    f"overlaps request LR-{clash['id']:04d} "
                    f"({clash['start_date']} → {clash['end_date']}, {clash['status']})"
                ),
                conflicting_request_id=clash["id"],
                message=(
                    f"❌ That overlaps leave you've already booked:\n\n"
                    f"📅 **LR-{clash['id']:04d}** — {clash['start_date']} → "
                    f"{clash['end_date']} ({clash['status']})\n"
                    f"📝 {clash['reason']}\n\n"
                    "Pick dates that don't clash, or cancel the existing request first. "
                    "What dates would you like instead?"
                ),
            )

        return None

    def _reject(
        self,
        ctx: SessionContext,
        state: Dict,
        reason_code: str,
        detail: str,
        message: str,
        **extra,
    ) -> str:
        """
        Log a structured rejection and reset the date slots so the user can retry.

        The reason is kept — it is rarely the thing that was wrong — and the
        agent stays active, so the next message re-enters slot filling instead
        of being re-routed by the intent detector.
        """
        self.logger.log_event(
            "leave_request_rejected",
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            reason_code=reason_code,
            detail=detail,
            start_date=state.get("start_date"),
            end_date=state.get("end_date"),
            **extra,
        )
        print(f"  [LeaveRequestAgent] ⛔ Rejected ({reason_code}): {detail}")

        state.pop("start_date", None)
        state.pop("end_date", None)
        state.pop("awaiting_confirmation", None)
        ctx.agent_state = state
        ctx.active_agent = "LeaveRequestAgent"
        return message

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
