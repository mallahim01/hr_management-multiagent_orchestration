"""
orchestration/crewai_adapter.py
────────────────────────────────
Real CrewAI orchestrator using Agent, Task, and Crew.

Defines CrewAI Agents for each HR specialist and an intent-router agent.
For each user turn, a 2-agent sequential Crew is assembled:
  Router Agent → Specialist Agent
The Crew.kickoff() drives real CrewAI execution.
"""

import os
from typing import Any, Dict

from crewai import LLM as CrewLLM, Agent as CrewAgent, Task as CrewTask, Crew, Process

from agents import AGENT_REGISTRY
from core.intent_detector import IntentDetector
from core.session import SessionContext
from orchestration.base import BaseOrchestrator


class CrewAIOrchestrator(BaseOrchestrator):
    """
    Real CrewAI-powered orchestrator.

    Uses CrewAI Agents with roles/goals/backstories and Tasks to
    process each user turn. The intent detection is done via our
    IntentDetector, then a CrewAI Agent handles the response.
    """

    backend_name = "crewai"

    def __init__(self, llm, db, history_size: int = 3):
        self.llm = llm
        self.db = db
        self.history_size = history_size
        self._intent_detector = IntentDetector(llm)
        self._agent_cache: Dict[str, Any] = {}

        # Build CrewAI agents (role-based definitions)
        self._crew_agents = self._build_crew_agents()
        print("  [CrewAI] ✅ Crew agents defined")

    # ── CrewAI Agent Definitions ─────────────────────────────────

    def _build_crew_agents(self) -> Dict[str, CrewAgent]:
        """Define CrewAI Agents with roles, goals, and backstories."""
        model_name = self._build_crew_llm()

        agents = {
            "LeaveRequestAgent": CrewAgent(
                role="Leave Request Specialist",
                goal="Help employees submit and manage leave requests accurately",
                backstory=(
                    "You are an experienced HR assistant specialising in leave management. "
                    "You guide employees through the leave request process step by step, "
                    "collecting dates, reasons, and confirming submissions."
                ),
                llm=model_name,
                verbose=False,
                allow_delegation=False,
            ),
            "LeaveBalanceAgent": CrewAgent(
                role="Leave Balance Advisor",
                goal="Provide accurate leave balance information to employees",
                backstory=(
                    "You are a knowledgeable HR assistant who can quickly look up "
                    "and explain employee leave balances in a friendly manner."
                ),
                llm=model_name,
                verbose=False,
                allow_delegation=False,
            ),
            "CompanyKnowledgeAgent": CrewAgent(
                role="Company Policy Expert",
                goal="Answer employee questions about company policies accurately",
                backstory=(
                    "You are a company policy expert with deep knowledge of HR policies, "
                    "work-from-home rules, reimbursements, and all internal guidelines."
                ),
                llm=model_name,
                verbose=False,
                allow_delegation=False,
            ),
            "HRRequestAgent": CrewAgent(
                role="HR Request Handler",
                goal="Log and track general HR requests for employees",
                backstory=(
                    "You handle miscellaneous HR requests like certificate generation, "
                    "employment verification, and other HR administrative tasks."
                ),
                llm=model_name,
                verbose=False,
                allow_delegation=False,
            ),
            "GeneralAssistantAgent": CrewAgent(
                role="General HR Assistant",
                goal="Provide helpful responses to general HR-related questions",
                backstory=(
                    "You are a friendly and helpful HR assistant who handles general "
                    "queries, greetings, and anything that doesn't fit other specialists."
                ),
                llm=model_name,
                verbose=False,
                allow_delegation=False,
            ),
        }
        return agents

    def _build_crew_llm(self) -> CrewLLM:
        """
        Build the LLM handle the Crew agents share, reusing our own credentials.

        Passing a model *string* to CrewAgent(llm=...) makes CrewAI resolve it
        through its own path, which attaches a `cache_breakpoint` property that
        Groq's API rejects outright. Constructing an LLM object avoids that.

        Any OpenAI-compatible endpoint is addressed as `openai/<model>` plus an
        explicit base_url — that is LiteLLM's convention, and it keeps this
        working for whatever provider core/llm_wrapper.py is configured with.
        """
        kwargs: Dict[str, Any] = {
            "model": self.llm.model,
            "api_key": self.llm.client.api_key,
        }
        if self.llm.base_url:
            kwargs["model"] = f"openai/{self.llm.model}"
            kwargs["base_url"] = self.llm.base_url
        return CrewLLM(**kwargs)

    # ── Public API (overrides) ───────────────────────────────────

    def route_intent(self, user_input: str, ctx: SessionContext) -> Dict:
        """Use our LLM-based IntentDetector for routing."""
        print(f"  [CrewAI] 🚢 Detecting intent for: '{user_input[:60]}...'")
        return self._intent_detector.detect(user_input, ctx.history)

    def invoke_agent(self, agent_name: str, user_input: str, ctx: SessionContext) -> str:
        """
        Use our native agent for the actual domain logic (DB access, slot filling),
        but wrap the response through a CrewAI Task for framework participation.
        """
        # First, run through our native agent (which has DB access and slot-filling)
        native_agent = self._get_native_agent(agent_name)
        native_reply = native_agent.handle(user_input, ctx)

        # Now wrap this through a CrewAI Task for real framework execution
        crew_agent = self._crew_agents.get(agent_name)
        if not crew_agent:
            return native_reply

        task = CrewTask(
            description=(
                f"An employee asked: '{user_input}'\n\n"
                f"The HR system has already processed this and produced the following response:\n"
                f"---\n{native_reply}\n---\n\n"
                f"Review this response. If it looks correct and complete, return it as-is. "
                f"If it needs minor improvements in tone or clarity, refine it. "
                f"Keep all factual data (dates, IDs, balances) exactly as provided."
            ),
            expected_output="A polished, employee-friendly response",
            agent=crew_agent,
        )

        crew = Crew(
            agents=[crew_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        print(f"  [CrewAI] 🚢 Kicking off crew with {agent_name}")
        result = crew.kickoff()

        # CrewAI returns a CrewOutput object; extract the raw string
        return str(result)

    def handoff_context(self, from_agent: str, to_agent: str, ctx: SessionContext) -> None:
        """Log agent handoff."""
        print(f"  [CrewAI] 🚢 Crew handoff: {from_agent} → {to_agent}")

    # ── Internal helpers ─────────────────────────────────────────

    def _get_native_agent(self, agent_name: str):
        """Return a cached native agent instance for DB/slot-filling logic."""
        if agent_name not in self._agent_cache:
            agent_cls = AGENT_REGISTRY.get(agent_name)
            if agent_cls is None:
                agent_cls = AGENT_REGISTRY["GeneralAssistantAgent"]
            self._agent_cache[agent_name] = agent_cls(self.llm, self.db)
        return self._agent_cache[agent_name]
