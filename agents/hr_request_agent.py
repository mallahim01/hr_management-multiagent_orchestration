"""
agents/hr_request_agent.py
───────────────────────────
Collects free-form HR requests via a multi-turn dialogue, asks for confirmation,
and then stores them to the database.
"""

import json
from agents.base_agent import BaseAgent
from core.session import SessionContext
from database.db import RecordValidationError


EXTRACT_PROMPT = """You are an HR assistant helping a user formulate an HR request.
Extract the user's HR request description from their message.
If they are just asking 'how to submit a request' or greeting, the description might be null.
Respond ONLY with valid JSON: {"request_description": "extracted detail" or null}
"""


class HRRequestAgent(BaseAgent):
    """Collects generic HR requests interactively and confirms receipt."""

    display_name = "HR Request Agent"
    colour = "orange"

    CANCEL_WORDS = {"no", "n", "cancel", "nope", "nah", "nevermind", "stop",
                    "abort", "quit", "exit", "forget"}
    CANCEL_PHRASES = ("cancel", "changed my mind", "never mind", "nevermind",
                      "forget it", "start over", "not now")

    def handle(self, user_input: str, ctx: SessionContext) -> str:
        state = ctx.agent_state

        # This agent holds the session across turns, and while it does the
        # orchestrator skips routing — so the user needs an unconditional way
        # out, not just one while a confirmation is pending.
        if self._is_cancellation(user_input):
            ctx.active_agent = None
            ctx.agent_state = {}
            return ("No problem! I've cancelled that request. "
                    "Let me know if you need anything else.")

        if state.get("awaiting_confirmation"):
            return self._handle_confirmation(user_input, ctx, state)

        extracted = self._extract_request(user_input, ctx)
        if extracted.get("request_description"):
            state["request_description"] = extracted["request_description"]

        ctx.agent_state = state
        ctx.active_agent = "HRRequestAgent"

        if not state.get("request_description"):
            return "I can help you submit an HR request. What would you like to request or ask the HR team about?"

        state["awaiting_confirmation"] = True
        ctx.agent_state = state

        return (
            f"I have recorded your request as follows:\n\n"
            f"📝 **Request:** {state['request_description']}\n\n"
            f"Would you like me to go ahead and submit this to the HR team? (yes / no)"
        )

    @classmethod
    def _is_cancellation(cls, user_input: str) -> bool:
        """True when the user is trying to abandon the HR request flow."""
        lowered = user_input.lower()
        return bool(set(lowered.split()) & cls.CANCEL_WORDS) or \
            any(phrase in lowered for phrase in cls.CANCEL_PHRASES)

    def _handle_confirmation(self, user_input: str, ctx: SessionContext, state: dict) -> str:
        words = set(user_input.lower().split())
        yes_words = {"yes", "y", "confirm", "sure", "ok", "yep", "yeah", "submit", "go"}

        if words & yes_words:
            try:
                request_id = self.db.insert_hr_request(
                    user_id=ctx.user_id,
                    request_text=state["request_description"],
                )
            except RecordValidationError as e:
                self.logger.log_event(
                    "hr_request_rejected",
                    session_id=ctx.session_id,
                    user_id=ctx.user_id,
                    reason_code="storage_rejected",
                    detail=str(e),
                )
                print(f"  [HRRequestAgent] ⛔ Rejected: {e}")
                ctx.active_agent = None
                ctx.agent_state = {}
                return (
                    "❌ I couldn't save that request, so **nothing has been "
                    "recorded**. Please try again, or contact HR directly."
                )
            ctx.active_agent = None
            ctx.agent_state = {}
            return (
                f"✅ Your request has been submitted successfully!\n\n"
                f"📋 **Ticket Reference:** HR-{request_id:04d}\n"
                f"HR will review your request and follow up within 2 business days."
            )
        else:
            ctx.active_agent = None
            ctx.agent_state = {}
            return "I've cancelled the submission. Let me know if you need anything else! 😊"

    def _extract_request(self, user_input: str, ctx: SessionContext) -> dict:
        messages = self._build_messages(EXTRACT_PROMPT, user_input, ctx)
        try:
            raw = self.llm.chat(messages, json_mode=True, temperature=0.0)
            return json.loads(raw)
        except Exception:
            return {}
