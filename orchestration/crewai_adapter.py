"""
orchestration/crewai_adapter.py
────────────────────────────────
CrewAI-style orchestrator adapter.

In a real implementation you would define CrewAI Agents and Tasks here.
For this demo it showcases WHERE CrewAI integration would live and delegates
to the same native logic so the demo works without installing crewai.
"""

from typing import Dict

from core.session import SessionContext
from orchestration.native import NativeOrchestrator


class CrewAIOrchestrator(NativeOrchestrator):
    """
    CrewAI adapter stub.

    Demonstrates the interface point where you would wire in CrewAI Agents,
    Tasks, and a Crew object. The routing and agent-call logic is identical to
    NativeOrchestrator; the difference is the framework scaffolding around it.

    To activate a real CrewAI implementation:
      1. pip install crewai
      2. Replace the methods below with proper crewai.Agent / crewai.Task / crewai.Crew usage.
    """

    backend_name = "crewai"

    def route_intent(self, user_input: str, ctx: SessionContext) -> Dict:
        print(
            "  [CrewAIOrchestrator] 🚢 CrewAI would define an IntentRouter Agent here.\n"
            "                       Delegating to native intent detection for this demo."
        )
        return super().route_intent(user_input, ctx)

    def invoke_agent(self, agent_name: str, user_input: str, ctx: SessionContext) -> str:
        print(
            f"  [CrewAIOrchestrator] 🚢 CrewAI would assign a Task to the '{agent_name}' crew member.\n"
            f"                       Delegating to native agent invocation for this demo."
        )
        return super().invoke_agent(agent_name, user_input, ctx)

    def handoff_context(self, from_agent: str, to_agent: str, ctx: SessionContext) -> None:
        print(
            f"  [CrewAIOrchestrator] 🚢 CrewAI would pass context via Task output chaining.\n"
            f"                       Handoff: {from_agent} → {to_agent}"
        )
