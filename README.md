# 🏢 ACME HR – Multi-Agent System

A multi-agent HR assistant built around a **custom intent-based orchestrator**, with
the orchestration layer behind an interface so a different engine can be swapped in
by changing one line of config.

Five specialist agents share one `SessionContext`: leave requests (multi-turn
slot-filling with balance and overlap validation), leave balance, company-policy
Q&A grounded in a local document, generic HR tickets, and a general fallback.
State, conversation history, and submitted records live in SQLite.

**Four orchestration backends** implement the same `BaseOrchestrator` contract:

- **`native`** — the reference implementation. Plain-Python intent routing, no framework.
- **`langgraph`** — a real compiled `StateGraph` whose nodes call the same agents,
  reports the node path each turn took, and contains a failing agent instead of
  aborting. The most thoroughly covered backend.
- **`crewai`** — a real `Agent`/`Task`/`Crew` integration. Exercised manually, not
  by automated tests.
- **`adk`** — a real Google ADK root agent exposing the five agents as tools, with
  Gemini making the routing decision itself. **Not covered by automated tests.**

Swapping the backend changes the machinery, not the behaviour — there is a test
asserting that `native` and `langgraph` produce identical routing decisions for
the same input.

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/mallahim01/hr_management-multiagent_orchestration.git
cd hr_management-multiagent_orchestration

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

The default provider is **Groq** (OpenAI-compatible endpoint, generous free tier):

```bash
cp .env.example .env             # Windows: copy .env.example .env
# Edit .env and set: GROQ_API_KEY=gsk_...
```

To use OpenAI instead, set `OPENAI_API_KEY` in `.env` and `llm.provider: openai`
in `config.yaml`. See [Configuration](#configuration) for the full list.

### 3. Run

| Mode | Command |
|------|---------|
| **Web UI** (recommended) | `python main.py --web` |
| Interactive CLI | `python main.py` |
| Auto-demo (5 prompts) | `python main.py --demo` |
| CLI data preview | `python main.py --preview-cli` |
| HTML report | `python main.py --preview-html` |

Open **http://127.0.0.1:5000** for the web UI.

### 4. Run the tests

```bash
python verify.py             # imports, DB seeding, agent registry
python test_langgraph.py     # LangGraph orchestrator      – offline, no API key
python test_validation.py    # leave safeguards + DB rules – offline, no API key
python test_eval.py          # routing-eval scoring logic  – offline, no API key
python test_eval.py --live   # …plus a live judge probe against labelled data
python test_backends.py      # all 4 backends against the live LLM (costs tokens)
```

The three `test_*.py` suites above run against a stub LLM and a throwaway
database, so they are deterministic and need neither a key nor network access.
`test_backends.py` and `--live` consume tokens.

### 5. Evaluate the routing

```bash
python eval_routing.py --limit 20
```

Replays recent turns from the interaction log and asks the model, after the fact,
whether each one reached the right agent. See [Evaluation](#evaluation).

---

## Architecture

```
User Input
    │
    ▼
Orchestrator (selectable via config.yaml)
    │
    ├── IntentDetector (LLM)
    │     └── returns: intent + confidence + target_agent
    │
    ├── If mid-slot-fill → continue with active agent
    │
    └── Route to matched Agent
          ├── LeaveRequestAgent      (multi-turn slot-filling)
          ├── LeaveBalanceAgent      (DB lookup)
          ├── CompanyKnowledgeAgent  (policy-grounded)
          ├── HRRequestAgent         (DB insert + ticket)
          └── GeneralAssistantAgent  (fallback)
```

### Switching Orchestration Backend

Set the startup default in `config.yaml`:

```yaml
orchestrator_backend: native     # or: crewai | langgraph | adk
```

Or switch live from the web UI's sidebar (`POST /api/backend`). Session state
lives in SQLite rather than inside any backend, so a conversation survives the
swap — you can begin a leave request under LangGraph and finish it under Native.

### LangGraph specifics

The LangGraph backend reports the ordered nodes each turn traversed, both in the
chat UI under every reply and in the **Graph** tab alongside an ASCII rendering of
the compiled graph:

```
detect_intent → LeaveRequestAgent
```

An exception inside an agent node is contained rather than aborting the graph:
the user gets a plain apology, the session is released so a failing agent cannot
trap the conversation, and an `agent_node_failed` event is logged.

---

## Project Structure

```
.
├── agents/               # 5 specialised sub-agents
├── core/                 # LLM wrapper, intent detector, session, logger, routing judge
├── docs/                 # Architecture deep-dive
├── orchestration/        # 4 backends + abstract base + factory
├── database/             # SQLite schema + CRUD helpers
├── preview/              # CLI viewer + HTML report generator
├── frontend/             # Chat UI (HTML/CSS/JS)
├── data/                 # SQLite DB, company policy, logs, reports
├── config.yaml           # Backend, model, user, DB settings
├── .env                  # API key (not committed)
├── app.py                # Flask web server
└── main.py               # CLI entry point
```

See [docs/architecture.md](docs/architecture.md) for how agents, orchestrators,
and session state fit together.

---

## Demo Conversation Flow

| User says | Intent | Agent |
|-----------|--------|-------|
| "What is our WFH policy?" | `company_question` | CompanyKnowledgeAgent |
| "I'm sick, need leave" | `leave_request` | LeaveRequestAgent (slot-fill) |
| "How many leaves do I have?" | `leave_balance` | LeaveBalanceAgent |
| "I need help with reimbursement" | `hr_request` | HRRequestAgent |
| "Hello!" | `general` | GeneralAssistantAgent |

---

## Configuration

```yaml
# config.yaml
orchestrator_backend: native   # native | crewai | langgraph | adk
active_user_id: 3              # Auto-created on first run
llm:
  provider: groq               # groq | openai
  model: llama-3.3-70b-versatile
  max_retries: 3
  temperature: 0.7
database:
  path: data/hr_demo.db
conversation:
  history_size: 3              # Recent exchanges sent as context
```

`llm.provider` selects both the API endpoint and which `.env` keys are read
(see `PROVIDERS` in [core/llm_wrapper.py](core/llm_wrapper.py)). For Groq you may
set `GROQ_API_KEY_2` / `GROQ_API_KEY_3` as spares — the wrapper rotates to the
next key when one hits its free-tier rate limit.

---

## Database Tables

| Table | Purpose |
|-------|---------|
| `users` | Employee records |
| `leave_balance` | Annual leave quota |
| `leave_requests` | Submitted leave applications |
| `hr_requests` | Generic HR tickets |
| `conversation_history` | Full message log with intent/agent |
| `sessions` | Active slot-filling state per session |

Three dummy users are seeded automatically on first run.

### Leave-request safeguards

A leave request is refused, with a message naming the specific problem, when it:

- exceeds the user's remaining balance,
- overlaps a `Pending` or `Approved` request already on the calendar,
- carries dates that are unparseable or run backwards.

Validation runs twice — before the confirmation prompt and again on submission,
since the balance can move in between. Each rejection writes a structured line to
`data/interactions.log` (`"event": "leave_request_rejected"`) carrying the reason
code and the numbers, alongside the friendly reply the user sees. Storing the
request and deducting the balance share a single transaction, so a failure cannot
leave a request with no matching deduction. See `python test_validation.py`.

---

## API Endpoints (Flask)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Chat UI |
| POST | `/api/chat` | Send a message |
| GET | `/api/status` | System info |
| POST | `/api/backend` | Switch orchestration backend at runtime |
| GET | `/api/graph` | Compiled StateGraph + last node path (LangGraph only) |
| GET | `/api/eval` | LLM-as-judge scoring of recent routing |
| GET | `/api/history` | Conversation log |
| GET | `/api/preview/leave-balance` | Balance JSON |
| GET | `/api/preview/leave-requests` | Leave requests JSON |
| GET | `/api/preview/hr-requests` | HR requests JSON |
| GET | `/api/report` | Generate HTML report |

---

## Backend Status

| Backend | Implementation | Tested |
|---|---|---|
| `native` | Reference implementation — plain Python intent routing | Yes |
| `langgraph` | Real compiled `StateGraph`; nodes call the same agents as native, with node-level error containment and per-turn path reporting | Yes — `python test_langgraph.py` (10 offline checks, no API key needed) |
| `crewai` | Real `Agent`/`Task`/`Crew`; the native agent runs first, then a sequential Crew polishes the reply | Manually, against live Groq. No automated coverage |
| `adk` | Real `google.adk` root agent with the five agents exposed as tools; Gemini decides routing | Manually, against live Gemini. No automated coverage |

**`crewai` requires the `litellm` extra** (already in `requirements.txt`). Passing a
model *string* to a CrewAI agent makes it attach a `cache_breakpoint` field that
Groq rejects, so the adapter builds an explicit `crewai.LLM` object instead.

**`adk` requires `GOOGLE_API_KEY`** and a current Gemini model. Google retires model
ids on a rolling basis; the default is the `gemini-flash-latest` alias, overridable
with `ADK_MODEL`. A retired id returns 404 and sends every turn down the native
fallback path — which is now reported in the result's `reasoning` and logged as an
`adk_fallback` event, rather than passing silently for success.

All four share the same `BaseOrchestrator` contract and the same agents — only the
routing machinery differs. See [docs/architecture.md](docs/architecture.md).

---

## Evaluation

Routing is the one decision this system makes on its own, so it is the one thing
worth scoring. `python eval_routing.py` (or the **Evals** tab, or `GET /api/eval`)
replays recent turns out of the interaction log and asks the model, after the fact
and with the agent's actual reply in view, whether each turn reached the right
agent.

```
  [PASS] turn 1  (langgraph, conf 95%)
         user:     What is our work from home policy?
         routed:   CompanyKnowledgeAgent  (intent: company_question)
         judge:    The question is about a company policy.

  [FAIL] turn 2  (native, conf 90%)
         user:     How many annual leave days do I have left?
         routed:   CompanyKnowledgeAgent
         expected: LeaveBalanceAgent
         judge:    Asks for a personal balance, not a policy.

  routing accuracy: 87.5%
```

Three things make the number mean something:

- **The judge does not reuse the detector's prompt.** Grading the classifier with
  its own prompt would mostly measure it against itself, so the judge gets its own
  instructions, sees the reply as evidence, and judges after the fact.
- **Slot-fill continuations are excluded, not scored.** While an agent holds the
  session the orchestrator skips classification deliberately; there is no routing
  decision on those turns, and counting them would inflate accuracy.
- **Ambiguous and errored turns are excluded from the denominator** rather than
  counted as passes. `accuracy = correct / (correct + incorrect)`.

An evaluator that answers "correct" to everything would report 100% forever, so
`python test_eval.py --live` feeds the real judge eight synthetic turns with
hand-labelled good and bad routing and checks it separates them. Re-run it
whenever the judge prompt or the model changes.

`eval_routing.py` exits non-zero when it finds a misroute, so it can gate a
pipeline. Each run also writes a `routing_eval` summary back to the log.

---

## Design decisions & known limitations

### Why intent-based routing

A single large prompt with every capability described in it would have been less
code. Routing through a dedicated classifier was chosen because:

- **The decision is inspectable.** `IntentDetector` returns an intent, a
  confidence, and a one-sentence rationale, all of which are logged. When the
  assistant answers oddly you can see whether it was misrouted or the agent
  itself was wrong — with one big prompt those two failures look identical.
- **Each agent gets a small prompt.** `CompanyKnowledgeAgent` is grounded in the
  policy document and nothing else; `LeaveBalanceAgent` sees only the balance
  row. Smaller prompts mean fewer tokens and less room to hallucinate across
  domains.
- **Routing is cheap and swappable.** Classification is one short JSON-mode call.
  It could be replaced with keyword rules or a fine-tuned classifier without
  touching a single agent.

The cost is a second LLM round-trip per turn. That is mitigated by skipping
classification entirely while an agent holds the conversation for slot-filling.

### Why pluggable backends

The orchestration framework landscape moves quickly, and the interesting question
is usually "what does this framework actually buy me?" — which is hard to answer
if the business logic is welded to one of them. Here the agents, session handling,
database access, and prompts sit behind `BaseOrchestrator`, and each backend only
supplies routing machinery. That makes the comparison concrete: the LangGraph and
native backends are asserted by test to produce identical routing decisions, so
any difference between them is framework overhead, not behaviour.

### What is and isn't verified

| Area | Status |
|---|---|
| `native`, `langgraph` routing and state | Automated, offline, deterministic (`test_langgraph.py`) |
| Leave safeguards, DB validation, atomicity | Automated, offline (`test_validation.py`) |
| Routing quality | Scored by an LLM judge (`eval_routing.py`); the judge itself is checked against labelled data by `test_eval.py --live` |
| `crewai` backend | Run manually against live Groq; no automated coverage |
| `adk` backend | Run manually against live Gemini; no automated coverage |
| Agent answer quality | **Not evaluated.** The judge scores *which agent* handled a turn, not whether the answer was correct or well-grounded |

### Known limitations

- **Single hard-coded user.** `active_user_id` comes from `config.yaml`. There is
  no authentication, no authorisation, and no notion of "who is asking" beyond
  that integer — so anyone hitting the API acts as that employee.
- **Approval is cosmetic.** Requests are stored as `Pending` and nobody can
  approve them; there is no HR-side view. The balance is deducted at submission
  rather than on approval, which is the wrong policy for a real system but keeps
  the demo's numbers legible.
- **Policy retrieval is whole-document.** `CompanyKnowledgeAgent` loads
  `data/company_policy.txt` into the prompt rather than chunking and embedding it.
  Fine at ~6 KB; it will not scale to a real policy corpus.
- **No concurrency story.** SQLite in WAL mode tolerates concurrent reads, and the
  balance deduction is guarded against going negative, but there is no locking
  around the read-validate-write sequence across separate HTTP requests.
  The `adk` backend is additionally not concurrency-safe: ADK invokes its tool
  functions through its own runner, so the caller's `SessionContext` is parked in
  a single instance slot for the duration of a turn and simultaneous requests
  would read each other's.
- **Relative dates are resolved by the LLM.** "Next Tuesday" is turned into a date
  inside the extraction prompt. Malformed output is caught by validation, but a
  plausible-yet-wrong date is not.
- **Session state is unbounded.** `sessions` rows are never expired or cleaned up.
- **Git history carries two author identities**, one with a malformed email
  (`smallahimali.com`, missing the `@`). Left as-is rather than rewriting history.

### What I would harden next, in order

1. **Extend evaluation from routing to answers.** `eval_routing.py` scores which
   agent handled a turn; nothing scores whether the answer was right. The next
   step is a fixed prompt set with expected outputs, plus groundedness checks on
   the policy agent so a confident wrong answer fails the build.
2. **Automated coverage for `crewai` and `adk`**, using the same stub-LLM
   technique. Both are currently only verified by hand, which is how the ADK
   tool functions ran for a while against a hardcoded `user_id=1` — reporting
   one employee's leave balance to another — without anything catching it.
3. **Move approval into the model**: deduct on approval rather than submission,
   add a status transition path, and give HR a view.
4. **Real retrieval** for the policy document — chunk, embed, and retrieve, with
   citations back to the source section.
5. **Authentication**, so `user_id` comes from a session rather than a config file.
