"""
orchestration/base.py
──────────────────────
Abstract orchestrator interface. All backends must implement these three methods
so the business logic (agents, session) never knows which backend is active.
"""

from abc import ABC, abstractmethod
from typing import Dict

from core import metrics
from core.session import SessionContext


class BaseOrchestrator(ABC):
    """
    Pluggable orchestration interface.

    Every backend exposes:
      • route_intent()    – decide which agent should respond
      • invoke_agent()    – call the chosen agent with the user message
      • handoff_context() – transfer state between agents

    The higher-level method `process()` ties these together and is the only
    method external callers (app.py / main.py) need to call.
    """

    # Name displayed in the UI and logs
    backend_name: str = "base"

    @abstractmethod
    def route_intent(self, user_input: str, ctx: SessionContext) -> Dict:
        """
        Determine which agent should handle this turn.

        Returns a dict with keys:
          intent, confidence, target_agent, reasoning
        """

    @abstractmethod
    def invoke_agent(
        self, agent_name: str, user_input: str, ctx: SessionContext
    ) -> str:
        """
        Instantiate (or retrieve) the named agent and call its handle() method.

        Returns the agent's reply string.
        """

    @abstractmethod
    def handoff_context(
        self, from_agent: str, to_agent: str, ctx: SessionContext
    ) -> None:
        """
        Called when the active agent changes mid-session.
        Use this hook to log, annotate, or transform context on handoff.
        """

    def process(self, user_input: str, ctx: SessionContext) -> Dict:
        """
        Full pipeline: route → (handoff if needed) → invoke → return result.

        Returns a dict suitable for JSON serialisation:
          {
            "reply":        str,
            "intent":       str,
            "confidence":   float,
            "agent":        str,       # display name
            "agent_class":  str,       # class name
            "reasoning":    str,
            "session_id":   str,
            "backend":      str,
          }
        """
        # If mid-slot-filling, skip intent detection and continue with same agent
        if ctx.active_agent:
            intent_result = {
                "intent": ctx.last_intent or "continuation",
                "confidence": 1.0,
                "target_agent": ctx.active_agent,
                "reasoning": f"Continuing slot-filling with {ctx.active_agent}",
            }
        else:
            intent_result = self.route_intent(user_input, ctx)

        target_agent = intent_result["target_agent"]
        previous_agent = ctx.last_agent

        # Notify on agent handoff
        if previous_agent and previous_agent != target_agent:
            self.handoff_context(previous_agent, target_agent, ctx)

        ctx.last_intent = intent_result["intent"]
        ctx.last_agent = target_agent

        # Everything the agent does counts as generation unless it opens a more
        # specific stage of its own — CompanyKnowledgeAgent nests "retrieval"
        # around its Milvus round trip, and the inner stage wins.
        with metrics.stage("generation", progress=False):
            reply = self.invoke_agent(target_agent, user_input, ctx)

        return {
            "reply":       reply,
            "intent":      intent_result["intent"],
            "confidence":  intent_result["confidence"],
            "agent":       self._agent_display_name(target_agent),
            "agent_class": target_agent,
            "reasoning":   intent_result.get("reasoning", ""),
            "session_id":  ctx.session_id,
            "backend":     self.backend_name,
        }

    @staticmethod
    def _agent_display_name(class_name: str) -> str:
        """Convert CamelCase class name to readable label."""
        import re
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", class_name).replace("Agent", "").strip()
