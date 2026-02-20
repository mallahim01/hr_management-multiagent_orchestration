"""
orchestration/native.py
────────────────────────
Pure-Python orchestrator. The default and fully-functional backend.
Uses IntentDetector for routing and instantiates agents directly.
"""

from typing import Dict

from agents import AGENT_REGISTRY
from core.intent_detector import IntentDetector
from core.llm_wrapper import LLMWrapper
from core.session import SessionContext
from database.db import DatabaseManager
from orchestration.base import BaseOrchestrator


class NativeOrchestrator(BaseOrchestrator):
    """
    The reference implementation. Demonstrates intent-based routing
    and agent handoffs entirely in plain Python – no external framework needed.
    """

    backend_name = "native"

    def __init__(self, llm: LLMWrapper, db: DatabaseManager, history_size: int = 3) -> None:
        self.llm = llm
        self.db = db
        self.history_size = history_size
        self.intent_detector = IntentDetector(llm)
        # Cache agent instances (they are stateless; state lives in SessionContext)
        self._agent_cache: Dict = {}

    # ── Interface implementation ─────────────────────────────────────────────

    def route_intent(self, user_input: str, ctx: SessionContext) -> Dict:
        """Use the LLM-based IntentDetector to classify the user message."""
        print(f"  [NativeOrchestrator] Detecting intent for: '{user_input[:60]}...'")
        return self.intent_detector.detect(user_input, ctx.history)

    def invoke_agent(self, agent_name: str, user_input: str, ctx: SessionContext) -> str:
        """Retrieve (or create) the agent and call its handle() method."""
        agent = self._get_agent(agent_name)
        print(f"  [NativeOrchestrator] Invoking {agent_name}")
        return agent.handle(user_input, ctx)

    def handoff_context(self, from_agent: str, to_agent: str, ctx: SessionContext) -> None:
        """Log when the orchestrator hands off from one agent to another."""
        print(
            f"  [NativeOrchestrator] ✦ Handoff: {from_agent} → {to_agent} "
            f"(session: {ctx.session_id[:8]}...)"
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_agent(self, agent_name: str):
        """Return a cached agent instance, creating it on first use."""
        if agent_name not in self._agent_cache:
            agent_cls = AGENT_REGISTRY.get(agent_name)
            if agent_cls is None:
                # Fallback to general if unknown agent name
                agent_cls = AGENT_REGISTRY["GeneralAssistantAgent"]
            self._agent_cache[agent_name] = agent_cls(self.llm, self.db)
        return self._agent_cache[agent_name]
