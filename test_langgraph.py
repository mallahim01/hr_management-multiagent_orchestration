"""
test_langgraph.py – Deterministic tests for the LangGraph orchestrator.

Runs entirely offline against a fake LLM and a throwaway SQLite file, so it
needs no API key, no network, and produces the same result every run.

Covers:
  1. Graph structure           – nodes, entry point, terminal edges
  2. Base-contract conformance – the three abstract hooks really work
  3. Routing                   – intent → the matching agent node
  4. Slot-fill continuation    – active agent bypasses intent detection
  5. Unknown-agent fallback    – a stale session name cannot crash the graph
  6. Detector-failure fallback – an LLM error degrades to GeneralAssistantAgent
  7. Agent-node containment    – an agent raising cannot abort the graph
  8. Execution trace           – the result reports the nodes actually visited
  9. Native parity             – same input, same routing decision as native

Usage: python test_langgraph.py
"""

import json
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import AGENT_REGISTRY
from core.logger import InteractionLogger
from core.session import SessionContext
from database.db import DatabaseManager
from database.schema import initialize_database
from langgraph.graph import END
from orchestration.langgraph_adapter import FALLBACK_AGENT, LangGraphOrchestrator
from orchestration.native import NativeOrchestrator

TEST_USER_ID = 1


# ── Test doubles ─────────────────────────────────────────────────────────────

class FakeLLM:
    """
    Stand-in for LLMWrapper.

    `intent` is the label chat_json() will classify every message as. Plain
    chat() returns a fixed sentence (or a JSON blob when the caller asked for
    json_mode, which is how the slot-extracting agents call it). Every call is
    recorded so tests can assert on what the graph *did not* call.
    """

    CANNED_REPLY = "Canned assistant reply."

    def __init__(self, intent: str = "general", fail: bool = False) -> None:
        self.intent = intent
        self.fail = fail
        self.json_calls: list = []
        self.chat_calls: list = []
        # Mirror the real wrapper's attributes; the CrewAI adapter reads these.
        self.model = "fake-model"
        self.provider = "fake"

    def chat(self, messages, json_mode=False, temperature=None) -> str:
        self.chat_calls.append(messages)
        if self.fail:
            raise RuntimeError("fake LLM failure")
        if json_mode:
            # Extraction prompts want JSON; return "nothing extracted".
            return json.dumps({"start_date": None, "end_date": None,
                               "reason": None, "request_description": None})
        return self.CANNED_REPLY

    def chat_json(self, messages) -> dict:
        self.json_calls.append(messages)
        if self.fail:
            raise RuntimeError("fake LLM failure")
        return {
            "intent": self.intent,
            "confidence": 0.95,
            "target_agent": "ignored – IntentDetector maps intent → agent",
            "reasoning": "canned classification",
        }


# ── Harness ──────────────────────────────────────────────────────────────────

_results: list = []


def check(label: str, fn) -> None:
    """Run one test, record pass/fail, never abort the rest of the suite."""
    try:
        fn()
        print(f"  PASS  {label}")
        _results.append((label, None))
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        _results.append((label, e))


def make_db() -> DatabaseManager:
    """Create a seeded throwaway database."""
    path = os.path.join(tempfile.mkdtemp(prefix="hr_lg_test_"), "test.db")
    db = DatabaseManager(path)
    initialize_database(db, TEST_USER_ID)
    return db


def make_ctx(**overrides) -> SessionContext:
    ctx = SessionContext(session_id=str(uuid.uuid4()), user_id=TEST_USER_ID)
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


# ── 1. Graph structure ───────────────────────────────────────────────────────

def test_graph_structure() -> None:
    orch = LangGraphOrchestrator(FakeLLM(), make_db())
    graph = orch._compiled.get_graph()
    nodes = set(graph.nodes)

    for agent_name in AGENT_REGISTRY:
        assert agent_name in nodes, f"missing node for {agent_name}"
    assert "detect_intent" in nodes, "missing entry node"

    # Every agent node must terminate the graph.
    terminal = {e.source for e in graph.edges if e.target == END}
    assert set(AGENT_REGISTRY) <= terminal, (
        f"agents not wired to END: {set(AGENT_REGISTRY) - terminal}"
    )

    # Conditional edges must fan out from detect_intent to every agent.
    from_detect = {e.target for e in graph.edges if e.source == "detect_intent"}
    assert set(AGENT_REGISTRY) <= from_detect, (
        f"detect_intent does not reach: {set(AGENT_REGISTRY) - from_detect}"
    )

    # ASCII rendering is offline and must not raise.
    assert "detect_intent" in orch.graph_ascii()


# ── 2. Base-contract conformance ─────────────────────────────────────────────

def test_implements_base_contract() -> None:
    """
    The three abstract hooks must be genuinely callable, not `pass` fakes.
    The factory and any future backend-agnostic caller rely on this.
    """
    orch = LangGraphOrchestrator(FakeLLM(intent="leave_balance"), make_db())
    ctx = make_ctx()

    routed = orch.route_intent("how many days left?", ctx)
    assert isinstance(routed, dict), f"route_intent returned {type(routed)}"
    assert routed["target_agent"] == "LeaveBalanceAgent", routed
    assert set(routed) >= {"intent", "confidence", "target_agent", "reasoning"}

    reply = orch.invoke_agent("LeaveBalanceAgent", "how many days left?", ctx)
    assert isinstance(reply, str) and reply, f"invoke_agent returned {reply!r}"

    # Must not raise.
    orch.handoff_context("LeaveBalanceAgent", "GeneralAssistantAgent", ctx)


# ── 3. Routing ───────────────────────────────────────────────────────────────

def test_routes_by_intent() -> None:
    cases = {
        "leave_balance":    "LeaveBalanceAgent",
        "company_question": "CompanyKnowledgeAgent",
        "hr_request":       "HRRequestAgent",
        "general":          "GeneralAssistantAgent",
        "leave_request":    "LeaveRequestAgent",
    }
    db = make_db()
    for intent, expected_agent in cases.items():
        llm = FakeLLM(intent=intent)
        orch = LangGraphOrchestrator(llm, db)
        result = orch.process("some message", make_ctx())

        assert result["agent_class"] == expected_agent, (
            f"intent {intent!r} routed to {result['agent_class']}, "
            f"expected {expected_agent}"
        )
        assert result["intent"] == intent, result["intent"]
        assert result["backend"] == "langgraph", result["backend"]
        assert result["reply"], "empty reply"
        assert len(llm.json_calls) == 1, (
            f"expected exactly 1 classification call, got {len(llm.json_calls)}"
        )


# ── 4. Slot-fill continuation ────────────────────────────────────────────────

def test_slot_fill_continuation_skips_detection() -> None:
    """
    A session already owned by LeaveRequestAgent must go straight back to it,
    without spending an intent-detection call.
    """
    llm = FakeLLM(intent="general")   # would route elsewhere if detection ran
    orch = LangGraphOrchestrator(llm, make_db())
    ctx = make_ctx(
        active_agent="LeaveRequestAgent",
        last_intent="leave_request",
        agent_state={"start_date": "2026-03-02"},
    )

    result = orch.process("until the 5th", ctx)

    assert result["agent_class"] == "LeaveRequestAgent", result["agent_class"]
    assert result["intent"] == "leave_request", result["intent"]
    assert result["confidence"] == 1.0, result["confidence"]
    assert llm.json_calls == [], "intent detection ran during slot-filling"
    assert "Continuing slot-filling" in result["reasoning"], result["reasoning"]
    # The agent kept ownership and its state survived the graph traversal.
    assert ctx.active_agent == "LeaveRequestAgent", ctx.active_agent
    assert ctx.agent_state.get("start_date") == "2026-03-02", ctx.agent_state


def test_state_mutations_reach_the_caller() -> None:
    """
    The caller's SessionContext must be the object the agents mutate, since
    main.py/app.py persist that same instance after process() returns.
    """
    orch = LangGraphOrchestrator(FakeLLM(intent="leave_request"), make_db())
    ctx = make_ctx()
    assert ctx.active_agent is None

    orch.process("I need time off", ctx)

    assert ctx.active_agent == "LeaveRequestAgent", (
        "agent did not claim the session through the graph"
    )
    assert ctx.last_intent == "leave_request", ctx.last_intent
    assert ctx.last_agent == "LeaveRequestAgent", ctx.last_agent


# ── 5. Unknown-agent fallback ────────────────────────────────────────────────

def test_unknown_active_agent_falls_back() -> None:
    """
    A session row naming an agent that no longer exists must degrade to the
    fallback agent rather than raising out of the conditional edge.
    """
    orch = LangGraphOrchestrator(FakeLLM(intent="general"), make_db())
    ctx = make_ctx(active_agent="RetiredAgentFromAnOldBuild")

    result = orch.process("hello?", ctx)

    assert result["agent_class"] == FALLBACK_AGENT, result["agent_class"]
    assert result["reply"], "fallback produced no reply"


# ── 6. Detector-failure fallback ─────────────────────────────────────────────

def test_detector_failure_falls_back_to_general() -> None:
    """
    IntentDetector swallows LLM errors and returns 'general'. The graph must
    honour that rather than propagating the exception.
    """
    class FailingClassifierLLM(FakeLLM):
        def chat_json(self, messages):
            self.json_calls.append(messages)
            raise RuntimeError("classification unavailable")

    orch = LangGraphOrchestrator(FailingClassifierLLM(), make_db())
    result = orch.process("anything at all", make_ctx())

    assert result["agent_class"] == "GeneralAssistantAgent", result["agent_class"]
    assert result["intent"] == "general", result["intent"]
    assert result["reply"], "no reply produced on detector failure"


# ── 7. Agent-node containment ────────────────────────────────────────────────

def test_agent_exception_is_contained() -> None:
    """
    An agent blowing up must not abort the graph. The user gets an apology,
    the failure is logged, and the wedged session is released.
    """
    log_path = os.path.join(tempfile.mkdtemp(prefix="hr_lg_fail_"), "events.log")
    orch = LangGraphOrchestrator(FakeLLM(intent="leave_balance"), make_db())
    orch.logger = InteractionLogger(log_path)

    class ExplodingAgent:
        def handle(self, user_input, ctx):
            raise RuntimeError("database on fire")

    orch._agent_cache["LeaveBalanceAgent"] = ExplodingAgent()
    ctx = make_ctx(active_agent="LeaveBalanceAgent")

    result = orch.process("how many days left?", ctx)   # must not raise

    assert result["failed"] is True, result
    assert result["reply"], "no reply produced after a node failure"
    assert result["agent_class"] == "LeaveBalanceAgent", result["agent_class"]
    assert ctx.active_agent is None, "failed agent kept hold of the session"

    with open(log_path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert len(events) == 1, events
    assert events[0]["event"] == "agent_node_failed", events[0]
    assert events[0]["node"] == "LeaveBalanceAgent", events[0]
    assert "database on fire" in events[0]["detail"], events[0]


# ── 8. Execution trace ───────────────────────────────────────────────────────

def test_reports_graph_path() -> None:
    """The result carries the ordered nodes the turn actually traversed."""
    orch = LangGraphOrchestrator(FakeLLM(intent="company_question"), make_db())
    result = orch.process("what is the wfh policy?", make_ctx())

    assert result["graph_path"] == ["detect_intent", "CompanyKnowledgeAgent"], (
        result["graph_path"]
    )
    assert orch.last_path == result["graph_path"], orch.last_path

    # A slot-fill continuation still traverses detect_intent, which
    # short-circuits, then the held agent.
    ctx = make_ctx(active_agent="HRRequestAgent", last_intent="hr_request")
    result = orch.process("a payslip copy", ctx)
    assert result["graph_path"] == ["detect_intent", "HRRequestAgent"], (
        result["graph_path"]
    )


# ── 9. Native parity ─────────────────────────────────────────────────────────

def test_matches_native_routing() -> None:
    """
    The whole point of the pluggable backend is that swapping it changes the
    machinery, not the behaviour. Same fake LLM, same routing decision.
    """
    db = make_db()
    for intent in ("leave_balance", "company_question", "hr_request", "general"):
        native = NativeOrchestrator(FakeLLM(intent=intent), db)
        langgraph = LangGraphOrchestrator(FakeLLM(intent=intent), db)

        native_result = native.process("some message", make_ctx())
        langgraph_result = langgraph.process("some message", make_ctx())

        assert native_result["agent_class"] == langgraph_result["agent_class"], (
            f"intent {intent!r}: native → {native_result['agent_class']}, "
            f"langgraph → {langgraph_result['agent_class']}"
        )
        assert native_result["intent"] == langgraph_result["intent"]
        assert native_result["backend"] == "native"
        assert langgraph_result["backend"] == "langgraph"


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== LangGraph Orchestrator – Tests ===\n")

    check("graph structure",                     test_graph_structure)
    check("implements BaseOrchestrator contract", test_implements_base_contract)
    check("routes by detected intent",           test_routes_by_intent)
    check("slot-fill skips intent detection",    test_slot_fill_continuation_skips_detection)
    check("state mutations reach the caller",    test_state_mutations_reach_the_caller)
    check("unknown active agent falls back",     test_unknown_active_agent_falls_back)
    check("detector failure falls back",         test_detector_failure_falls_back_to_general)
    check("agent exception is contained",        test_agent_exception_is_contained)
    check("reports the graph path taken",        test_reports_graph_path)
    check("routing matches native backend",      test_matches_native_routing)

    failures = [label for label, err in _results if err is not None]
    print("\n" + "=" * 50)
    if failures:
        print(f"FAILED: {len(failures)}/{len(_results)} – {failures}")
        sys.exit(1)
    print(f"All {len(_results)} checks PASSED")


if __name__ == "__main__":
    main()
