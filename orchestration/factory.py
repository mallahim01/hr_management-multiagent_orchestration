"""
orchestration/factory.py
─────────────────────────
Returns the correct orchestrator instance based on the config.yaml setting.
This is the only place where the backend selection logic lives.
"""

from core.llm_wrapper import LLMWrapper
from database.db import DatabaseManager
from orchestration.base import BaseOrchestrator


def get_orchestrator(
    backend: str,
    llm: LLMWrapper,
    db: DatabaseManager,
    history_size: int = 3,
) -> BaseOrchestrator:
    """
    Factory function – swap orchestration engine by changing config.yaml.

    Args:
        backend:      One of 'native', 'crewai', 'langgraph', 'adk'
        llm:          Shared LLMWrapper instance
        db:           Shared DatabaseManager instance
        history_size: Number of recent exchanges to use as context

    Returns:
        Configured orchestrator instance.
    """
    backend = backend.lower().strip()

    if backend == "native":
        from orchestration.native import NativeOrchestrator
        print(f"[Factory] ✅ Using NativeOrchestrator")
        return NativeOrchestrator(llm, db, history_size)

    elif backend == "crewai":
        from orchestration.crewai_adapter import CrewAIOrchestrator
        print(f"[Factory] 🚢 Using CrewAIOrchestrator (stub mode)")
        return CrewAIOrchestrator(llm, db, history_size)

    elif backend == "langgraph":
        from orchestration.langgraph_adapter import LangGraphOrchestrator
        print(f"[Factory] 🔗 Using LangGraphOrchestrator (stub mode)")
        return LangGraphOrchestrator(llm, db, history_size)

    elif backend == "adk":
        from orchestration.adk_adapter import ADKOrchestrator
        print(f"[Factory] 🤖 Using ADKOrchestrator (mock mode)")
        return ADKOrchestrator(llm, db, history_size)

    else:
        print(f"[Factory] ⚠️  Unknown backend '{backend}' – falling back to native")
        from orchestration.native import NativeOrchestrator
        return NativeOrchestrator(llm, db, history_size)
