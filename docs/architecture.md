# Multi-Agent Orchestration Architecture

A detailed breakdown of how the system operates under the hood: how the base
agents are defined, how the orchestrator abstraction works, and the precise user
flow for each of the four orchestration backends.

---

## 1. How Base Agents Are Defined

All "worker" agents inherit from a single `BaseAgent` class
([agents/base_agent.py](../agents/base_agent.py)).

### The `BaseAgent` Contract

- **Shared Resources:** It provides all sub-agents with shared access to the LLM
  (via `LLMWrapper`, [core/llm_wrapper.py](../core/llm_wrapper.py)) and the
  database (via `DatabaseManager`, [database/db.py](../database/db.py)).
- **Prompt Building:** It handles the repetitive task of injecting the last N
  turns of conversation history into the LLM prompt via `_build_messages()`.
- **The `handle()` Method:** Every agent *must* implement
  `handle(user_input, ctx)`. This is where the domain logic lives. The agent
  reads the user input, queries the database if needed, calls the LLM, and
  returns a plain string reply.

### State and Multi-Turn Dialogue

The most important part of the agent design is how multi-turn conversations are
handled — for example, `LeaveRequestAgent` asking for a start date, then an end
date, then a reason.

- If an agent needs to keep control of the conversation for the next turn, it
  sets `ctx.active_agent = self.__class__.__name__` and stores its working
  variables in `ctx.agent_state`.
- When the task is complete (or cancelled), the agent sets
  `ctx.active_agent = None`. This signals to the orchestrator that it can resume
  normal dynamic routing on the next message.

`SessionContext` ([core/session.py](../core/session.py)) is the single carrier of
this state. `SessionManager` persists `active_agent` and `agent_state` to the
`sessions` table between HTTP requests, so slot-filling survives a page reload.

---

## 2. Dynamic Routing (The Orchestrator Abstraction)

When a request hits the Flask `/api/chat` endpoint it does not talk to the agents
directly. It talks to an **orchestrator**, selected by `orchestrator_backend` in
`config.yaml` and constructed by [orchestration/factory.py](../orchestration/factory.py).

Every orchestrator implements the same `process(user_input, ctx)` pipeline
defined in [orchestration/base.py](../orchestration/base.py):

1. **Check for an active agent.** Is there an ongoing multi-turn conversation
   (is `ctx.active_agent` set)? If yes, bypass routing and send the message
   straight to that agent.
2. **Intent detection.** If no agent is active, use `IntentDetector`
   ([core/intent_detector.py](../core/intent_detector.py)) — a single
   JSON-mode LLM classification call — to determine the `target_agent`.
3. **Execution.** Hand the message to the target agent using the backend's
   own machinery.

The abstract base declares three hooks — `route_intent()`, `invoke_agent()`, and
`handoff_context()` — and `process()` composes them. A backend that cannot
express its flow as those three steps (LangGraph, ADK) overrides `process()`
directly, but still implements the three hooks so that the interface stays
honest and directly callable.

---

## 3. Backend Walkthroughs

### Native

A pure-Python dictionary mapping intent labels to agent classes. Fastest and
leanest, and the reference implementation for the other three.
See [orchestration/native.py](../orchestration/native.py).

### LangGraph

Compiles a directed `StateGraph`
([orchestration/langgraph_adapter.py](../orchestration/langgraph_adapter.py)):

- A `detect_intent` node is the entry point. It short-circuits to the active
  agent when mid-slot-fill, otherwise calls `IntentDetector`.
- One node per agent in `AGENT_REGISTRY`, each wrapping the *same* native agent
  instance — no logic is duplicated.
- A conditional edge maps the resolved `target_agent` onto the matching agent
  node, with an unknown name falling back to `GeneralAssistantAgent`.
- Every agent node terminates at `END`.

`SessionContext` is carried through the graph as an opaque value in the state
dict, so agents mutate the *same* live object the caller holds — which is what
makes slot-filling work identically to the native backend.

### CrewAI

Wraps the agents inside role-based CrewAI "personas"
([orchestration/crewai_adapter.py](../orchestration/crewai_adapter.py)). The
native agent runs first (it owns DB access and slot-filling state), then a
single-agent sequential `Crew` reviews and polishes the reply. CrewAI resolves
models through LiteLLM, so the model id is provider-prefixed.

### Google ADK

Uses **tool-based routing** rather than an explicit router
([orchestration/adk_adapter.py](../orchestration/adk_adapter.py)).

**Lazy init.** Google ADK has a large dependency chain, so the adapter defers
importing `google.adk` and constructing the `InMemoryRunner` until the first
request. This keeps Flask startup fast.

**The root agent.** A single Gemini-powered root agent is given five tools, each
a thin Python function wrapping one of the native agents.

**Routing.** Unlike LangGraph, ADK makes the routing decision itself: the Gemini
model reads the prompt, matches it against the tool docstrings, and invokes the
matching Python function natively. The native agent inside that tool does the
real work — for example `CompanyKnowledgeAgent` retrieving from
[data/company_policy.txt](../data/company_policy.txt) — and returns a string,
which the root agent is instructed to return verbatim.

**Slot-filling bypass.** If the user is mid leave request, the adapter bypasses
ADK entirely and calls the native `LeaveRequestAgent.handle()` to finish the
form, because ADK's own session store does not carry our `agent_state`.

**Fallback.** If ADK times out or hits a quota limit, the adapter catches the
error and falls back to native `IntentDetector` routing so the user always gets
an answer.

---

## 4. Summary of Framework Differences

The underlying `BaseAgent` logic (DB reads, system prompts, slot-filling) is
identical across all four. Only the wrapping differs:

| Backend | Routing decided by | Agent wrapped as |
|---|---|---|
| **Native** | `IntentDetector` | Plain Python class |
| **LangGraph** | `IntentDetector`, via a conditional graph edge | Graph node |
| **CrewAI** | `IntentDetector` | Role-based `Agent` in a sequential `Crew` |
| **Google ADK** | The Gemini root agent, natively | Tool function |
