"""
agents/base_agent.py
─────────────────────
Abstract base class all sub-agents inherit from.
Provides shared LLM access and a helper to build message lists from history.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from core.llm_wrapper import LLMWrapper
from core.logger import InteractionLogger
from core.session import SessionContext
from database.db import DatabaseManager


class BaseAgent(ABC):
    """
    Abstract agent. Each sub-agent implements `handle()` which receives the
    user message and session context, performs its task, and returns a reply.

    Agents may update ctx.active_agent and ctx.agent_state to indicate that
    they need to continue handling the next user turn (slot-filling).
    Setting ctx.active_agent = None signals that the conversation is complete
    and the orchestrator should resume normal intent routing.
    """

    # Human-readable name shown in the UI
    display_name: str = "Agent"
    # Colour tag for the frontend badge
    colour: str = "gray"

    def __init__(
        self,
        llm: LLMWrapper,
        db: DatabaseManager,
        logger: Optional[InteractionLogger] = None,
    ) -> None:
        self.llm = llm
        self.db = db
        # Agents log domain events (e.g. a rejected leave request) through the
        # same logger as conversation turns. Optional so the orchestrators can
        # keep constructing agents as cls(llm, db); the default lands on
        # InteractionLogger's default path, which is also what main.py and
        # app.py use, so events and turns interleave in one file. Pass an
        # explicit logger when you need them somewhere else (the tests do).
        self.logger = logger or InteractionLogger()

    @abstractmethod
    def handle(self, user_input: str, ctx: SessionContext) -> str:
        """
        Process user_input given the current session context.

        Args:
            user_input: The latest message from the user.
            ctx:        Live session context (read and write).

        Returns:
            Agent's reply as a plain string.
        """

    # ── Utility ──────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        system_prompt: str,
        user_input: str,
        ctx: SessionContext,
        extra_context: str = "",
    ) -> List[Dict[str, str]]:
        """
        Build the message list for a chat call, injecting recent history.

        Args:
            system_prompt: Agent's system instruction.
            user_input:    Current user message.
            ctx:           Session context (contains history).
            extra_context: Any additional context to append to the system prompt.

        Returns:
            List of {"role", "content"} dicts ready for the LLM.
        """
        system_content = system_prompt
        if extra_context:
            system_content += f"\n\n{extra_context}"

        messages = [{"role": "system", "content": system_content}]

        # Inject last N conversation turns for continuity
        for msg in ctx.history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_input})
        return messages
