"""
agents/general_assistant_agent.py
───────────────────────────────────
Fallback conversational agent for greetings and off-topic messages.
"""

from agents.base_agent import BaseAgent
from core.session import SessionContext

SYSTEM_PROMPT = """You are a friendly and helpful HR assistant for ACME Corporation.
You handle general conversation, greetings, and questions that don't fit a specific HR category.
Keep your replies concise (2–3 sentences max). If the user seems to need HR help, gently guide
them toward what you can assist with: leave requests, leave balance, company policies, or HR requests."""


class GeneralAssistantAgent(BaseAgent):
    """Handles chitchat and any messages that don't match a specific intent."""

    display_name = "General Assistant"
    colour = "gray"

    def handle(self, user_input: str, ctx: SessionContext) -> str:
        messages = self._build_messages(SYSTEM_PROMPT, user_input, ctx)
        reply = self.llm.chat(messages)
        ctx.active_agent = None
        ctx.agent_state = {}
        return reply
