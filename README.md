# 🏢 ACME HR – Multi-Agent System

A multi-agent HR assistant built around a **custom intent-based orchestrator**, with
the orchestration layer behind an interface so a different engine can be swapped in
by changing one line of config.

Five specialist agents share one `SessionContext`: leave requests (multi-turn
slot-filling with balance and overlap validation), leave balance, company-policy
Q&A answered by **hybrid RAG over Milvus** with cited sources, generic HR tickets,
and a general fallback. State, conversation history, and submitted records live
in SQLite; policy documents live in a Milvus collection that HR can upload to
from the web UI.

Groq runs the chat models, Google runs the embeddings.

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

Set `GOOGLE_API_KEY` too — it powers the embeddings for the knowledge base.

### 3. Start Milvus and load the policy document (optional)

The policy agent uses hybrid RAG over Milvus. Start a standalone instance:

```bash
# https://milvus.io/docs/install_standalone-docker.md
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.6.4 milvus run standalone

python ingest_knowledge.py          # loads data/company_policy.txt
```

**This step is optional.** Without Milvus the policy agent falls back to putting
`data/company_policy.txt` straight into the prompt, and says so in its reply. You
lose citations and uploaded documents, not the ability to run the project.

### 4. Run

| Mode | Command |
|------|---------|
| **Web UI** (recommended) | `python main.py --web` |
| Interactive CLI | `python main.py` |
| Auto-demo (5 prompts) | `python main.py --demo` |
| CLI data preview | `python main.py --preview-cli` |
| HTML report | `python main.py --preview-html` |

Open **http://127.0.0.1:5000** for the web UI.

### 5. Run the tests

```bash
python verify.py             # imports, DB seeding, agent registry
python test_langgraph.py     # LangGraph orchestrator      – offline, no API key
python test_validation.py    # leave safeguards + DB rules – offline, no API key
python test_eval.py          # routing-eval scoring logic  – offline, no API key
python test_rag.py           # chunking, grounding, citations – offline, no API key
python test_eval.py --live   # …plus a live judge probe against labelled data
python test_rag.py  --live   # …plus real Milvus + Gemini retrieval checks
python test_backends.py      # all 4 backends against the live LLM (costs tokens)
```

The four `test_*.py` suites run against stubs and a throwaway database, so they
are deterministic and need no key, no network and no Milvus. `test_backends.py`
and the `--live` passes consume tokens.

### 6. Evaluate the routing

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
├── knowledge/            # chunking, Gemini embeddings, Milvus hybrid store
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
| "What is our WFH policy?" | `company_question` | CompanyKnowledgeAgent (hybrid RAG + citations) |
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
| GET | `/api/knowledge/status` | Milvus/embedding availability, document counts |
| GET | `/api/knowledge/documents` | Documents in the knowledge base |
| POST | `/api/knowledge/ingest` | Upload a document (file or pasted text) |
| DELETE | `/api/knowledge/documents/<id>` | Remove a document |
| GET | `/api/knowledge/search` | Preview hybrid retrieval for a query |
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

## Knowledge base — hybrid RAG

`CompanyKnowledgeAgent` answers policy questions from a Milvus collection rather
than from a prompt containing the whole policy file. Each chunk is indexed twice
in the same collection:

| Arm | How | Good at |
|---|---|---|
| **Dense** | Gemini `gemini-embedding-001`, HNSW + COSINE | meaning — *"can I do my job from my house?"* finds the WFH clause despite sharing no words with it |
| **Sparse** | Milvus's built-in BM25 function over the same text | exact terms — *"LWP"*, a form number, a policy code, which dense vectors handle poorly |

Both arms retrieve `candidate_k` results and the two **rankings** are fused with
Reciprocal Rank Fusion. Fusing rankings rather than scores matters here: BM25 is
unbounded and cosine sits in [-1, 1], so a weighted sum of raw scores would need
per-corpus tuning to mean anything.

Retrieval quality is mostly decided by chunking, so headings are detected and
carried onto every chunk — both as metadata and prefixed onto the embedded text.
A chunk reading *"5 days for immediate family"* is far less findable than
`SECTION 1 – LEAVE POLICY › 1.5 Bereavement Leave: 5 days for immediate family`.

### Schema

`pk`, `text`, `sparse_vector` (BM25 output), `dense_vector`, plus the metadata
that makes an answer auditable: `doc_id`, `source`, `title`, `section`,
`chunk_index`, `total_chunks`, `uploaded_at`, `uploaded_by`. Every answer cites
the extracts it used, and every citation resolves back to a document and chunk.

### HR document upload

The **Knowledge** tab lets HR drop in a `.txt`, `.md`, `.csv`, `.json` or `.pdf`
file (or paste text), see what is stored, preview exactly what a query retrieves,
and remove a document. Re-uploading the same filename **replaces** the previous
copy rather than adding a second one — otherwise a corrected policy would sit in
the index next to the version it was meant to supersede, and the agent would
happily cite either.

From the CLI:

```bash
python ingest_knowledge.py                       # seed the bundled policy
python ingest_knowledge.py handbook.md           # add a document
python ingest_knowledge.py --search "LWP"        # preview hybrid retrieval
python ingest_knowledge.py --list                # what is stored
python ingest_knowledge.py --delete doc-abc123   # remove a document
```

### Grounding rules

- Retrieved extracts are numbered and the model is told to cite them inline.
- **If nothing is retrieved the agent does not call the LLM at all** — it returns
  the "no information" response and logs `knowledge_no_results`. Answering with an
  empty context is how a RAG system starts inventing policy.
- If Milvus or the embedding key is unavailable, the agent falls back to the
  bundled policy file and **says so in the reply**. The failure is logged as
  `knowledge_retrieval_failed`; it is never a silent downgrade.

All four backends resolve this agent through `AGENT_REGISTRY`, so native,
LangGraph, CrewAI and ADK all use RAG without knowing Milvus exists.

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
- **Each agent gets a small prompt.** `CompanyKnowledgeAgent` sees only the
  handful of policy extracts retrieved for the question; `LeaveBalanceAgent` sees
  only the balance row. Smaller prompts mean fewer tokens and less room to
  hallucinate across domains.
- **Routing is what decides whether RAG runs at all.** Retrieval only fires on
  `company_question`, so leave arithmetic and ticket creation never pay for an
  embedding call, and a policy answer is never grounded in a leave balance.
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
| RAG grounding and degradation | Automated, offline (`test_rag.py`) — citations, refusal on empty retrieval, fallback when Milvus is down |
| Hybrid retrieval against real Milvus | `test_rag.py --live` checks exact-term and paraphrase recall; **no labelled recall@k benchmark** |
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
- **Retrieval quality is unmeasured.** Hybrid search demonstrably finds the right
  clause on the queries tried, but there is no labelled retrieval set and so no
  recall@k or MRR figure. The routing evaluation does not cover it.
- **No reranker.** RRF fuses two rankings; a cross-encoder over the fused
  candidates would do better, at the cost of another model call per query.
- **Chunking is heading-driven and tuned to this document.** A policy file with
  no headings falls back to paragraph packing, which is workable but blunter.
- **Uploads are unauthenticated.** Anyone who can reach the app can add or delete
  policy documents that the assistant will then cite as authoritative. In a real
  deployment this endpoint needs to sit behind an HR role.
- **Scanned PDFs are not handled** — text extraction only, no OCR.
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

1. **Extend evaluation from routing to answers and retrieval.** `eval_routing.py`
   scores which agent handled a turn; nothing scores whether the answer was right
   or whether the right chunk was retrieved. The next step is a labelled
   question→chunk set for recall@k, plus a groundedness check that fails an answer
   making claims absent from its cited extracts.
2. **Put the upload endpoint behind an HR role.** Right now anyone who can reach
   the app can change what the assistant treats as company policy.
3. **Add a reranker** over the fused candidates.
2. **Automated coverage for `crewai` and `adk`**, using the same stub-LLM
   technique. Both are currently only verified by hand, which is how the ADK
   tool functions ran for a while against a hardcoded `user_id=1` — reporting
   one employee's leave balance to another — without anything catching it.
3. **Move approval into the model**: deduct on approval rather than submission,
   add a status transition path, and give HR a view.
4. **Real retrieval** for the policy document — chunk, embed, and retrieve, with
   citations back to the source section.
5. **Authentication**, so `user_id` comes from a session rather than a config file.
