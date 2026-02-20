"""
orchestration/adk_adapter.py
─────────────────────────────
Real Google ADK (Agent Development Kit) orchestrator.

Uses google.adk.agents.Agent with tool functions wrapping each HR agent.
The ADK root agent decides which tool to call based on the conversation.

ALL google.adk imports are deferred to first use (inside methods) because
the Google Cloud SDK dependency chain takes 60-90 seconds to import.
This keeps the Flask server startup instant.
"""

import asyncio
import os
import threading
from typing import Any, Dict

from agents import AGENT_REGISTRY
from core.intent_detector import IntentDetector
from core.session import SessionContext
from orchestration.base import BaseOrchestrator


# ── Shared state for module-level tool functions ─────────────────
_native_agents: Dict[str, Any] = {}
_shared_llm = None
_shared_db = None


def _get_native(name: str):
    """Lazy-init a native agent."""
    global _native_agents, _shared_llm, _shared_db
    if name not in _native_agents:
        cls = AGENT_REGISTRY.get(name, AGENT_REGISTRY["GeneralAssistantAgent"])
        _native_agents[name] = cls(_shared_llm, _shared_db)
    return _native_agents[name]


class ADKOrchestrator(BaseOrchestrator):
    """
    Real Google ADK-powered orchestrator.

    Creates an ADK root Agent with tool functions wrapping each HR agent.
    The Gemini model decides which tool to invoke based on the user message.

    All google.adk imports and runner init are deferred to first process() call
    to keep Flask startup fast.
    """

    backend_name = "adk"

    def __init__(self, llm, db, history_size: int = 3):
        self.llm = llm
        self.db = db
        self.history_size = history_size
        self._intent_detector = IntentDetector(llm)
        self._agent_cache: Dict[str, Any] = {}

        # Set shared references for tool functions
        global _shared_llm, _shared_db
        _shared_llm = llm
        _shared_db = db

        # Lazy-init: runner is created on first process() call
        self._runner = None
        self._adk_agent = None
        self._runner_lock = threading.Lock()
        self._session_counter = 0
        self._adk_ready = False

        print("  [ADK] ✅ Google ADK orchestrator configured (lazy init on first request)")

    # ── Lazy initialisation ──────────────────────────────────────

    def _ensure_runner(self):
        """Lazily import google.adk and create the Agent + Runner on first use."""
        if self._adk_ready:
            return

        with self._runner_lock:
            if self._adk_ready:
                return

            print("  [ADK] ⏳ Initialising Google ADK (first request, may take a moment)...")

            # Deferred imports — these pull in the entire Google Cloud SDK
            from google.adk.agents import Agent as ADKAgent
            from google.adk.runners import InMemoryRunner

            model = os.getenv("ADK_MODEL", "gemini-2.0-flash")

            # Define tool functions inline (they reference module-level _get_native)
            def handle_leave_request(user_message: str) -> dict:
                """Handle employee leave requests. Call when the user wants to apply for
                leave, take time off, or submit a sick day request.

                Args:
                    user_message: The employee's message about their leave request.
                Returns:
                    dict: status and the agent response.
                """
                agent = _get_native("LeaveRequestAgent")
                ctx = SessionContext(session_id="adk_session", user_id=1)
                reply = agent.handle(user_message, ctx)
                return {"status": "success", "response": reply}

            def check_leave_balance(user_message: str) -> dict:
                """Check employee leave balance. Call when the user asks about their
                remaining leave days, annual leave, or leave balance.

                Args:
                    user_message: The employee's question about their leave balance.
                Returns:
                    dict: status and the leave balance information.
                """
                agent = _get_native("LeaveBalanceAgent")
                ctx = SessionContext(session_id="adk_session", user_id=1)
                reply = agent.handle(user_message, ctx)
                return {"status": "success", "response": reply}

            def answer_company_question(user_message: str) -> dict:
                """Answer questions about company policies. Call when the user asks about
                company rules, WFH policy, reimbursements, code of conduct, or HR policies.

                Args:
                    user_message: The employee's question about company policy.
                Returns:
                    dict: status and the policy answer.
                """
                agent = _get_native("CompanyKnowledgeAgent")
                ctx = SessionContext(session_id="adk_session", user_id=1)
                reply = agent.handle(user_message, ctx)
                return {"status": "success", "response": reply}

            def submit_hr_request(user_message: str) -> dict:
                """Submit a general HR request. Call when the user needs HR certificates,
                employment verification, salary slips, or other administrative HR tasks.

                Args:
                    user_message: The employee's HR request description.
                Returns:
                    dict: status and the HR request ticket details.
                """
                agent = _get_native("HRRequestAgent")
                ctx = SessionContext(session_id="adk_session", user_id=1)
                reply = agent.handle(user_message, ctx)
                return {"status": "success", "response": reply}

            def handle_general_query(user_message: str) -> dict:
                """Handle general conversation and HR-related questions. Call for greetings,
                small talk, or general questions that don't fit other categories.

                Args:
                    user_message: The employee's general message or question.
                Returns:
                    dict: status and the assistant's response.
                """
                agent = _get_native("GeneralAssistantAgent")
                ctx = SessionContext(session_id="adk_session", user_id=1)
                reply = agent.handle(user_message, ctx)
                return {"status": "success", "response": reply}

            self._adk_agent = ADKAgent(
                name="hr_orchestrator",
                model=model,
                description="HR multi-agent orchestrator routing employee queries to specialist tools",
                instruction=(
                    "You are an HR assistant orchestrator. When an employee sends a message, "
                    "analyse their intent and call the most appropriate tool function:\n\n"
                    "- handle_leave_request: For leave applications, sick days, time off\n"
                    "- check_leave_balance: For checking remaining leave days\n"
                    "- answer_company_question: For questions about company policies, WFH, reimbursements\n"
                    "- submit_hr_request: For HR certificates, employment verification, admin tasks\n"
                    "- handle_general_query: For greetings, small talk, general questions\n\n"
                    "Always pass the user's exact message to the tool. "
                    "Return the tool's response directly to the user without modification."
                ),
                tools=[
                    handle_leave_request,
                    check_leave_balance,
                    answer_company_question,
                    submit_hr_request,
                    handle_general_query,
                ],
            )

            self._runner = InMemoryRunner(
                agent=self._adk_agent,
                app_name="hr_multiagent",
            )

            self._adk_ready = True
            print("  [ADK] ✅ Google ADK runner ready")

    # ── Public API (overrides) ───────────────────────────────────

    def route_intent(self, user_input: str, ctx: SessionContext) -> Dict:
        """Not used directly — ADK root agent handles routing via tool selection."""
        pass

    def invoke_agent(self, agent_name: str, user_input: str, ctx: SessionContext) -> str:
        """Not used directly — ADK root agent invokes tools internally."""
        pass

    def handoff_context(self, from_agent: str, to_agent: str, ctx: SessionContext) -> None:
        """Handoff managed by ADK tool dispatch."""
        print(f"  [ADK] 🤖 Tool switch: {from_agent} → {to_agent}")

    def process(self, user_input: str, ctx: SessionContext) -> Dict:
        """
        Run the full ADK pipeline.

        Lazily initialises the runner on first call. Sends the user message
        through ADK and collects the final response.
        Falls back to native orchestration if ADK fails.
        """
        # If mid-slot-filling, bypass ADK and use native agent directly
        if ctx.active_agent:
            agent_name = ctx.active_agent  # save before handle() may clear it
            native_agent = self._get_native_agent(agent_name)
            reply = native_agent.handle(user_input, ctx)
            return {
                "reply":       reply,
                "intent":      ctx.last_intent or "continuation",
                "confidence":  1.0,
                "agent":       self._agent_display_name(agent_name),
                "agent_class": agent_name,
                "reasoning":   f"Continuing slot-filling with {agent_name}",
                "session_id":  ctx.session_id,
                "backend":     self.backend_name,
            }

        # Detect intent (for metadata regardless of which path executes)
        intent_result = self._intent_detector.detect(user_input, ctx.history)
        target_agent = intent_result.get("target_agent", "GeneralAssistantAgent")

        # Try ADK execution
        try:
            self._ensure_runner()
            reply = self._run_adk(user_input, ctx.session_id)
        except Exception as e:
            print(f"  [ADK] ⚠️  ADK execution failed: {e}")
            print(f"  [ADK] Falling back to native orchestration")
            native_agent = self._get_native_agent(target_agent)
            reply = native_agent.handle(user_input, ctx)

        ctx.last_intent = intent_result.get("intent", "general")
        ctx.last_agent = target_agent

        return {
            "reply":       reply,
            "intent":      intent_result.get("intent", "general"),
            "confidence":  intent_result.get("confidence", 0.9),
            "agent":       self._agent_display_name(target_agent),
            "agent_class": target_agent,
            "reasoning":   intent_result.get("reasoning", "ADK tool dispatch"),
            "session_id":  ctx.session_id,
            "backend":     self.backend_name,
        }

    def _run_adk(self, user_input: str, session_id: str) -> str:
        """Execute the ADK agent, handling the async/sync bridge."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._run_adk_in_new_loop, user_input, session_id)
            return future.result(timeout=60)

    def _run_adk_in_new_loop(self, user_input: str, session_id: str) -> str:
        """Run the ADK invocation in a fresh asyncio event loop."""
        from google.genai import types as genai_types

        async def _invoke():
            self._session_counter += 1
            sid = f"adk_{session_id}_{self._session_counter}"

            # Create session in the runner's internal session service
            # (InMemoryRunner uses InMemorySessionService with auto_create_session=False)
            await self._runner.session_service.create_session(
                app_name=self._runner.app_name,
                user_id="employee_1",
                session_id=sid,
            )

            content = genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=user_input)],
            )

            final_response = ""
            async for event in self._runner.run_async(
                user_id="employee_1",
                session_id=sid,
                new_message=content,
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text
                    break

            return final_response or "I'm sorry, I couldn't process that request."

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_invoke())
        finally:
            loop.close()

    # ── Internal helpers ─────────────────────────────────────────

    def _get_native_agent(self, agent_name: str):
        """Return a cached native agent instance."""
        if agent_name not in self._agent_cache:
            agent_cls = AGENT_REGISTRY.get(agent_name)
            if agent_cls is None:
                agent_cls = AGENT_REGISTRY["GeneralAssistantAgent"]
            self._agent_cache[agent_name] = agent_cls(self.llm, self.db)
        return self._agent_cache[agent_name]
