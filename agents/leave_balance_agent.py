"""
agents/leave_balance_agent.py
──────────────────────────────
Fetches and returns the active user's leave balance.
Single-turn agent – no slot filling required.
"""

from agents.base_agent import BaseAgent
from core.session import SessionContext


class LeaveBalanceAgent(BaseAgent):
    """Returns current leave balance for the session user."""

    display_name = "Leave Balance Agent"
    colour = "blue"

    def handle(self, user_input: str, ctx: SessionContext) -> str:
        balance = self.db.get_leave_balance(ctx.user_id)
        user = self.db.get_user(ctx.user_id)
        name = user["name"] if user else "Employee"

        if not balance:
            return "I couldn't find your leave balance. Please contact HR."

        total     = balance["total_leaves"]
        used      = balance["used_leaves"]
        remaining = balance["remaining_leaves"]

        # Use LLM to format a natural, friendly response
        system_prompt = (
            "You are a friendly HR assistant. Present the employee's leave balance "
            "in a warm, clear, and concise way. Keep it to 3–4 lines."
        )
        context = (
            f"Employee: {name}\n"
            f"Total annual leave entitlement: {total} days\n"
            f"Leave used so far: {used} days\n"
            f"Leave remaining: {remaining} days"
        )
        messages = self._build_messages(system_prompt, user_input, ctx, context)
        reply = self.llm.chat(messages)

        # Ensure this agent doesn't hold the conversation
        ctx.active_agent = None
        ctx.agent_state = {}
        return reply
