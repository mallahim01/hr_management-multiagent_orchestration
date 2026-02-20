"""
orchestration/adk_adapter.py
─────────────────────────────
Google ADK-style orchestrator adapter.

ADK (Agent Development Kit) uses a tool-dispatch model where the orchestrator
acts as a coordinator that calls specialised tool-agents. This stub mirrors
that pattern using our existing agents as "tools".
"""

from typing import Dict

from core.session import SessionContext
from orchestration.native import NativeOrchestrator


class ADKOrchestrator(NativeOrchestrator):
    """
    Google ADK-style mock adapter.

    In ADK, agents are registered as tools callable by the root agent.
    The root agent (orchestrator) decides which tool-agent to invoke
    based on the conversation.

    This stub logs the ADK-style dispatch pattern while delegating to native logic.

    To activate a real ADK implementation:
      1. pip install google-adk
      2. Define tool functions wrapping each agent's handle() method.
      3. Register tools with the ADK root agent and invoke via adk.run().
    """

    backend_name = "adk"

    def route_intent(self, user_input: str, ctx: SessionContext) -> Dict:
        print(
            "  [ADKOrchestrator] 🤖 ADK root agent would analyse the message and\n"
            "                    select the appropriate tool-agent to call. Delegating to native."
        )
        return super().route_intent(user_input, ctx)

    def invoke_agent(self, agent_name: str, user_input: str, ctx: SessionContext) -> str:
        print(
            f"  [ADKOrchestrator] 🤖 ADK dispatches tool call → {agent_name}\n"
            f"                    Tool result flows back to root agent. Delegating to native."
        )
        return super().invoke_agent(agent_name, user_input, ctx)

    def handoff_context(self, from_agent: str, to_agent: str, ctx: SessionContext) -> None:
        print(
            f"  [ADKOrchestrator] 🤖 ADK tool switch: {from_agent} → {to_agent}\n"
            f"                    Session state passed via ADK context object."
        )
