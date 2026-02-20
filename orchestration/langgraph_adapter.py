"""
orchestration/langgraph_adapter.py
───────────────────────────────────
Real LangGraph orchestrator using StateGraph.

Defines a compiled graph with:
  - An intent_detector node (entry point)
  - One node per agent
  - Conditional edges based on detected intent
  - Graph image export via .get_graph().draw_png()
"""

import os
from typing import Annotated, Any, Dict, TypedDict

from langgraph.graph import StateGraph, END

from agents import AGENT_REGISTRY
from core.intent_detector import IntentDetector
from core.session import SessionContext
from orchestration.base import BaseOrchestrator


# ── Graph state schema ──────────────────────────────────────────

class HRGraphState(TypedDict):
    """State that flows through the graph."""
    user_input: str
    ctx: Any                # SessionContext (not serialised by langgraph)
    intent: str
    confidence: float
    target_agent: str
    reasoning: str
    reply: str


# ── Intent → agent mapping ──────────────────────────────────────

INTENT_TO_AGENT = {
    "leave_request":    "LeaveRequestAgent",
    "leave_balance":    "LeaveBalanceAgent",
    "company_question": "CompanyKnowledgeAgent",
    "hr_request":       "HRRequestAgent",
    "general":          "GeneralAssistantAgent",
}


class LangGraphOrchestrator(BaseOrchestrator):
    """
    Real LangGraph-powered orchestrator.

    Builds a StateGraph with nodes for intent detection and each agent,
    uses conditional edges to route, compiles the graph, and invokes it.
    Also supports exporting the graph as a PNG image.
    """

    backend_name = "langgraph"

    def __init__(self, llm, db, history_size: int = 3):
        self.llm = llm
        self.db = db
        self.history_size = history_size
        self._intent_detector = IntentDetector(llm)
        self._agent_cache: Dict[str, Any] = {}
        self._graph = self._build_graph()
        self._compiled = self._graph.compile()
        # Auto-save graph image on init
        self._save_graph_image()

    # ── Graph construction ───────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """Define the full StateGraph with nodes and edges."""
        graph = StateGraph(HRGraphState)

        # 1. Add the intent detector node
        graph.add_node("detect_intent", self._detect_intent_node)

        # 2. Add one node per agent
        for agent_name in AGENT_REGISTRY:
            graph.add_node(agent_name, self._make_agent_node(agent_name))

        # 3. Entry point → intent detector
        graph.set_entry_point("detect_intent")

        # 4. Conditional edges from intent detector → agent nodes
        graph.add_conditional_edges(
            "detect_intent",
            self._route_to_agent,
            {name: name for name in AGENT_REGISTRY},
        )

        # 5. Each agent → END
        for agent_name in AGENT_REGISTRY:
            graph.add_edge(agent_name, END)

        return graph

    # ── Node functions ───────────────────────────────────────────

    def _detect_intent_node(self, state: HRGraphState) -> Dict:
        """Node: classify intent using the LLM-based detector."""
        user_input = state["user_input"]
        ctx = state["ctx"]

        # Skip detection if mid-slot-filling
        if ctx.active_agent:
            return {
                "intent": ctx.last_intent or "continuation",
                "confidence": 1.0,
                "target_agent": ctx.active_agent,
                "reasoning": f"Continuing slot-filling with {ctx.active_agent}",
            }

        result = self._intent_detector.detect(user_input, ctx.history)
        return {
            "intent": result["intent"],
            "confidence": result["confidence"],
            "target_agent": result["target_agent"],
            "reasoning": result.get("reasoning", ""),
        }

    def _route_to_agent(self, state: HRGraphState) -> str:
        """Conditional edge: return the target agent node name."""
        return state["target_agent"]

    def _make_agent_node(self, agent_name: str):
        """Factory: create a node function for a specific agent."""
        def agent_node(state: HRGraphState) -> Dict:
            agent = self._get_or_create_agent(agent_name)
            ctx = state["ctx"]

            # Log handoff
            previous = ctx.last_agent
            if previous and previous != agent_name:
                print(f"  [LangGraph] Agent handoff: {previous} → {agent_name}")

            ctx.last_intent = state["intent"]
            ctx.last_agent = agent_name

            reply = agent.handle(state["user_input"], ctx)
            return {"reply": reply}

        return agent_node

    def _get_or_create_agent(self, agent_name: str):
        if agent_name not in self._agent_cache:
            cls = AGENT_REGISTRY[agent_name]
            self._agent_cache[agent_name] = cls(self.llm, self.db)
        return self._agent_cache[agent_name]

    # ── Public API (overrides) ───────────────────────────────────

    def route_intent(self, user_input: str, ctx: SessionContext) -> Dict:
        """Not used directly — handled inside the graph."""
        pass

    def invoke_agent(self, agent_name: str, user_input: str, ctx: SessionContext) -> str:
        """Not used directly — handled inside the graph."""
        pass

    def handoff_context(self, from_agent: str, to_agent: str, ctx: SessionContext) -> None:
        """Handoff is logged inside the agent nodes."""
        print(f"  [LangGraph] 🔗 Graph edge traversal: {from_agent} → {to_agent}")

    def process(self, user_input: str, ctx: SessionContext) -> Dict:
        """Run the full graph pipeline."""
        initial_state: HRGraphState = {
            "user_input": user_input,
            "ctx": ctx,
            "intent": "",
            "confidence": 0.0,
            "target_agent": "",
            "reasoning": "",
            "reply": "",
        }

        result = self._compiled.invoke(initial_state)

        return {
            "reply":       result["reply"],
            "intent":      result["intent"],
            "confidence":  result["confidence"],
            "agent":       self._agent_display_name(result["target_agent"]),
            "agent_class": result["target_agent"],
            "reasoning":   result.get("reasoning", ""),
            "session_id":  ctx.session_id,
            "backend":     self.backend_name,
        }

    # ── Graph image export ───────────────────────────────────────

    def _save_graph_image(self) -> None:
        """Save the compiled graph as a PNG image."""
        try:
            # draw_mermaid_png uses an external service (mermaid.ink) by default,
            # which works without local graphviz installation.
            img_data = self._compiled.get_graph().draw_mermaid_png()
            print("ignoring graph image saved........")
        #     os.makedirs("data", exist_ok=True)
        #     image_path = os.path.join("data", "langgraph_flow.png")
        #     with open(image_path, "wb") as f:
        #         f.write(img_data)
        #     print(f"  [LangGraph] ✅ Graph image saved → {image_path}")
        except Exception as e:
            print(f"  [LangGraph] ⚠️  Could not save graph image: {e}")
            print("               Ensure you have internet access for mermaid renderer.")

    def save_graph_image(self, path: str = "data/langgraph_flow.png") -> str:
        """Public API: save graph image to a custom path. Returns the path."""
        try:
            img_data = self._compiled.get_graph().draw_mermaid_png()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(img_data)
            return path
        except Exception as e:
            return f"Error: {e}"
