"""
agents/hr_request_agent.py
───────────────────────────
Stores free-form HR requests (reimbursements, grievances, etc.) to the database
and acknowledges the user with a ticket-style confirmation.
"""

from agents.base_agent import BaseAgent
from core.session import SessionContext

SYSTEM_PROMPT = """You are an HR Request intake assistant for ACME Corporation.
Acknowledge the employee's request warmly and professionally.
Tell them their request has been logged and HR will follow up within 2 business days.
Keep the response to 3–4 sentences. Do NOT ask follow-up questions."""


class HRRequestAgent(BaseAgent):
    """Logs generic HR requests to the database and confirms receipt."""

    display_name = "HR Request Agent"
    colour = "orange"

    def handle(self, user_input: str, ctx: SessionContext) -> str:
        # Store the request in the database first
        request_id = self.db.insert_hr_request(
            user_id=ctx.user_id,
            request_text=user_input,
        )

        # Generate a warm acknowledgment
        messages = self._build_messages(SYSTEM_PROMPT, user_input, ctx)
        reply = self.llm.chat(messages)

        # Append the ticket reference
        full_reply = f"{reply}\n\n📋 **Ticket Reference:** HR-{request_id:04d}"

        ctx.active_agent = None
        ctx.agent_state = {}
        return full_reply
