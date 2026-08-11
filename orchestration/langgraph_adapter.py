"""
orchestration/langgraph_adapter.py
───────────────────────────────────
LangGraph orchestrator built on a compiled StateGraph.

The graph mirrors the native flow exactly:

    detect_intent ──┬─► LeaveRequestAgent     ──┐
                    ├─► LeaveBalanceAgent     ──┤
                    ├─► CompanyKnowledgeAgent ──┼─► END
                    ├─► HRRequestAgent        ──┤
                    └─► GeneralAssistantAgent ──┘

`detect_intent` short-circuits to the active agent when the session is mid
slot-filling; otherwise it calls the shared IntentDetector. A conditional edge
maps the resolved agent name onto the matching node.

No business logic is reimplemented here: the nodes call this class's own
route_intent() / invoke_agent() / handoff_context() hooks, which are the same
BaseOrchestrator contract every other backend implements, and those in turn use
the shared IntentDetector and the agents in AGENT_REGISTRY. Interaction logging
happens at the caller (main.py / app.py) for every backend alike.
"""

import os
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from agents import AGENT_REGISTRY
from core.intent_detector import IntentDetector
from core.llm_wrapper import LLMWrapper
from core.logger import InteractionLogger
from core.session import SessionContext
from database.db import DatabaseManager
from orchestration.base import BaseOrchestrator


# Used when a resolved agent name is not in AGENT_REGISTRY — e.g. a session row
# written by an older build naming an agent that no longer exists.
FALLBACK_AGENT = "GeneralAssistantAgent"


class HRGraphState(TypedDict):
    """
    State that flows through the graph.

    `ctx` is the caller's live SessionContext, passed through by reference
    rather than serialised. That is deliberate: agents mutate ctx.active_agent
    and ctx.agent_state in place to drive slot-filling, and the caller persists
    the same object afterwards via SessionManager.save(). Copying it into the
    graph would silently break multi-turn dialogue.
    """
    user_input:   str
    ctx:          Any            # SessionContext
    intent:       str
    confidence:   float
    target_agent: str
    reasoning:    str
    reply:        str
    failed:       bool           # an agent node raised and was contained


class LangGraphOrchestrator(BaseOrchestrator):
    """
    LangGraph-powered orchestrator.

    Builds a StateGraph with an intent-detection entry node and one node per
    registered agent, routes between them with a conditional edge, and compiles
    the result once at construction time.
    """

    backend_name = "langgraph"

    def __init__(
        self, llm: LLMWrapper, db: DatabaseManager, history_size: int = 3
    ) -> None:
        self.llm = llm
        self.db = db
        self.history_size = history_size
        self.intent_detector = IntentDetector(llm)
        self.logger = InteractionLogger()
        # Agent instances are stateless; conversation state lives in SessionContext.
        self._agent_cache: Dict[str, Any] = {}
        self._graph = self._build_graph()
        self._compiled = self._graph.compile()
        # Nodes visited by the most recent process() call, in order.
        self.last_path: List[str] = []

    # ── Graph construction ───────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """Define the StateGraph: entry node, agent nodes, conditional routing."""
        graph = StateGraph(HRGraphState)

        graph.add_node("detect_intent", self._detect_intent_node)
        for agent_name in AGENT_REGISTRY:
            graph.add_node(agent_name, self._make_agent_node(agent_name))

        graph.set_entry_point("detect_intent")
        graph.add_conditional_edges(
            "detect_intent",
            self._route_to_agent,
            {name: name for name in AGENT_REGISTRY},
        )
        for agent_name in AGENT_REGISTRY:
            graph.add_edge(agent_name, END)

        return graph

    # ── Node functions ───────────────────────────────────────────────────────

    def _detect_intent_node(self, state: HRGraphState) -> Dict:
        """
        Entry node. Mirrors BaseOrchestrator.process(): a session that is mid
        slot-filling stays with its active agent and skips classification
        entirely, which also saves an LLM call on every follow-up turn.
        """
        ctx = state["ctx"]

        if ctx.active_agent:
            return {
                "intent":       ctx.last_intent or "continuation",
                "confidence":   1.0,
                "target_agent": ctx.active_agent,
                "reasoning":    f"Continuing slot-filling with {ctx.active_agent}",
            }

        return self.route_intent(state["user_input"], ctx)

    def _route_to_agent(self, state: HRGraphState) -> str:
        """
        Conditional edge. Returns the name of the agent node to run.

        An unrecognised name routes to the fallback agent instead of raising,
        so a stale session row can never take the whole graph down.
        """
        target = state.get("target_agent") or ""
        if target not in AGENT_REGISTRY:
            print(
                f"  [LangGraph] ⚠️  Unknown target agent '{target}' – "
                f"routing to {FALLBACK_AGENT}"
            )
            return FALLBACK_AGENT
        return target

    def _make_agent_node(self, agent_name: str):
        """Build the node function for one agent."""

        def agent_node(state: HRGraphState) -> Dict:
            ctx = state["ctx"]

            previous = ctx.last_agent
            if previous and previous != agent_name:
                self.handoff_context(previous, agent_name, ctx)

            ctx.last_intent = state["intent"]
            ctx.last_agent = agent_name

            try:
                reply = self.invoke_agent(agent_name, state["user_input"], ctx)
            except Exception as e:
                # Contain the failure at the node. An exception escaping here
                # would abort the whole graph and surface as a 500 with no
                # reply at all; the user gets a plain apology instead and the
                # cause is recorded for whoever is on the hook for it.
                print(f"  [LangGraph] ⛔ Node {agent_name} raised: {e!r}")
                self.logger.log_event(
                    "agent_node_failed",
                    session_id=ctx.session_id,
                    user_id=ctx.user_id,
                    reason_code="agent_exception",
                    detail=f"{type(e).__name__}: {e}",
                    node=agent_name,
                    backend=self.backend_name,
                )
                # Release the session so a wedged slot-fill cannot trap the
                # user in an agent that fails on every turn.
                ctx.active_agent = None
                ctx.agent_state = {}
                return {
                    "reply": (
                        "⚠️ Something went wrong while handling that. Nothing has "
                        "been recorded. Please try again, or contact HR if it "
                        "keeps happening."
                    ),
                    "target_agent": agent_name,
                    "failed": True,
                }

            # Write target_agent back so the result reports the node that
            # actually ran, not the name the router was asked for.
            return {"reply": reply, "target_agent": agent_name, "failed": False}

        return agent_node

    # ── Interface implementation ─────────────────────────────────────────────

    def route_intent(self, user_input: str, ctx: SessionContext) -> Dict:
        """Use the shared LLM-based IntentDetector to classify the user message."""
        print(f"  [LangGraph] 🔗 Detecting intent for: '{user_input[:60]}...'")
        return self.intent_detector.detect(user_input, ctx.history)

    def invoke_agent(self, agent_name: str, user_input: str, ctx: SessionContext) -> str:
        """Retrieve (or create) the agent and call its handle() method."""
        agent = self._get_agent(agent_name)
        print(f"  [LangGraph] 🔗 Invoking node {agent_name}")
        return agent.handle(user_input, ctx)

    def handoff_context(self, from_agent: str, to_agent: str, ctx: SessionContext) -> None:
        """Log the graph edge traversal when the active agent changes."""
        print(
            f"  [LangGraph] ✦ Graph handoff: {from_agent} → {to_agent} "
            f"(session: {ctx.session_id[:8]}...)"
        )

    def process(self, user_input: str, ctx: SessionContext) -> Dict:
        """
        Run one turn through the compiled graph.

        Overrides BaseOrchestrator.process() because the routing decision is
        expressed as a graph edge rather than a straight-line call, but returns
        the identical result dict so callers cannot tell the difference.
        """
        initial_state: HRGraphState = {
            "user_input":   user_input,
            "ctx":          ctx,
            "intent":       "",
            "confidence":   0.0,
            "target_agent": "",
            "reasoning":    "",
            "reply":        "",
            "failed":       False,
        }

        # stream() rather than invoke() so the nodes the turn actually visited
        # can be reported. Same execution either way; this is what makes the
        # routing decision observable instead of inferred from the final state.
        result: Dict[str, Any] = dict(initial_state)
        path: List[str] = []
        for step in self._compiled.stream(initial_state, stream_mode="updates"):
            for node_name, update in step.items():
                path.append(node_name)
                if update:
                    result.update(update)

        self.last_path = path
        target_agent = result["target_agent"]

        return {
            "reply":       result["reply"],
            "intent":      result["intent"],
            "confidence":  result["confidence"],
            "agent":       self._agent_display_name(target_agent),
            "agent_class": target_agent,
            "reasoning":   result.get("reasoning", ""),
            "session_id":  ctx.session_id,
            "backend":     self.backend_name,
            # LangGraph-specific: the ordered node path for this turn.
            "graph_path":  path,
            "failed":      bool(result.get("failed")),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_agent(self, agent_name: str):
        """Return a cached agent instance, creating it on first use."""
        if agent_name not in self._agent_cache:
            agent_cls = AGENT_REGISTRY.get(agent_name, AGENT_REGISTRY[FALLBACK_AGENT])
            self._agent_cache[agent_name] = agent_cls(self.llm, self.db)
        return self._agent_cache[agent_name]

    # ── Graph introspection / export ─────────────────────────────────────────

    def graph_ascii(self) -> str:
        """
        Render the compiled graph as ASCII art.

        Offline and deterministic (via grandalf), unlike the PNG export, so it
        is safe to call in tests and from the CLI.
        """
        return self._compiled.get_graph().draw_ascii()

    def save_graph_image(self, path: str = "data/langgraph_flow.png") -> str:
        """
        Save the compiled graph as a PNG. Returns the path, or an error string.

        Not called automatically: LangGraph's Mermaid renderer posts the graph
        to the mermaid.ink web service, so this needs network access and should
        stay an explicit, opt-in action rather than a constructor side effect.
        """
        try:
            img_data = self._compiled.get_graph().draw_mermaid_png()
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "wb") as f:
                f.write(img_data)
            return path
        except Exception as e:
            return f"Error: {e} (the PNG renderer requires internet access)"
