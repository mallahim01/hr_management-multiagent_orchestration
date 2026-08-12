# Submission notes

A map of the system for someone reading the source. Written to be checkable:
every claim below names a file you can open.

---

## 1. Load-bearing file paths

### Orchestration and control flow

| Path | What it does |
|---|---|
| [`orchestration/base.py`](../orchestration/base.py) | The contract every backend implements — `route_intent`, `invoke_agent`, `handoff_context` — plus the `process()` pipeline that composes them. Also the slot-fill short-circuit: when `ctx.active_agent` is set, classification is skipped entirely. |
| [`orchestration/factory.py`](../orchestration/factory.py) | The single place backend selection happens. Unknown names fall back to native. |
| [`orchestration/native.py`](../orchestration/native.py) | Reference implementation. Plain-Python routing, no framework. |
| [`orchestration/langgraph_adapter.py`](../orchestration/langgraph_adapter.py) | The main backend. Compiles a `StateGraph` with a `detect_intent` entry node, one node per agent, and a conditional edge between them. Uses `stream()` rather than `invoke()` so each turn reports the nodes it actually visited. Contains agent exceptions at the node. |
| [`orchestration/crewai_adapter.py`](../orchestration/crewai_adapter.py) | Real CrewAI `Agent`/`Task`/`Crew`. Optional install. |
| [`orchestration/adk_adapter.py`](../orchestration/adk_adapter.py) | Real Google ADK root agent with the five agents exposed as tools; Gemini does the routing itself. Optional install. |
| [`core/intent_detector.py`](../core/intent_detector.py) | One JSON-mode LLM call returning intent, confidence, target agent and a rationale. Swallows its own errors and falls back to `general`. |
| [`agents/__init__.py`](../agents/__init__.py) | `AGENT_REGISTRY`. Every backend resolves agents through it, which is why swapping `CompanyKnowledgeAgent` for a RAG implementation reached all four with no change to any of them. |

### State handling

| Path | What it does |
|---|---|
| [`core/session.py`](../core/session.py) | `SessionContext` (the per-conversation state: `active_agent`, `agent_state`, history) and `SessionManager`, which persists it between HTTP requests. |
| [`database/schema.py`](../database/schema.py) | The `sessions` table where slot-filling state survives a reload, plus the rest of the schema. |
| [`database/db.py`](../database/db.py) | `save_session` / `load_session`. `load_session` tolerates a corrupt JSON blob by resetting the state rather than raising. |
| [`orchestration/langgraph_adapter.py`](../orchestration/langgraph_adapter.py) | `HRGraphState` carries `SessionContext` **by reference**, not serialised — agents mutate the same object the caller persists afterwards. Copying it would silently break multi-turn dialogue; there is a test for exactly that. |

### Tool and retrieval use

| Path | What it does |
|---|---|
| [`knowledge/store.py`](../knowledge/store.py) | The Milvus collection: schema, BM25 function, HNSW index, ingestion, hybrid search with RRF fusion, and per-document delete. |
| [`knowledge/embeddings.py`](../knowledge/embeddings.py) | Gemini embeddings with batching, key rotation, and re-normalisation of truncated vectors (Matryoshka output below 3072 dimensions is not unit-length, and cosine assumes it is). |
| [`knowledge/chunker.py`](../knowledge/chunker.py) | Heading-aware chunking. The heading trail is carried as metadata *and* prefixed onto the embedded text. |
| [`agents/company_knowledge_agent.py`](../agents/company_knowledge_agent.py) | The RAG agent: retrieve, ground, cite, and decline when nothing is retrieved. |
| [`agents/leave_request_agent.py`](../agents/leave_request_agent.py) | Multi-turn slot-filling with LLM slot extraction, then database reads for validation. |
| [`app.py`](../app.py) | The knowledge endpoints, including file upload and text extraction. |

### Guardrails and failure handling

| Path | What it does |
|---|---|
| [`agents/leave_request_agent.py`](../agents/leave_request_agent.py) | `_validate_request` / `_reject`. Refuses over-balance and overlapping requests, malformed dates, and missing balance rows. Runs twice — before the confirmation prompt and again at submission. Also the unconditional cancel path. |
| [`database/db.py`](../database/db.py) | `RecordValidationError`, the `PRAGMA foreign_keys=ON` that makes the declared FKs real, and `submit_leave_request`, which puts the insert and the balance deduction in one transaction with a guarded UPDATE. |
| [`agents/company_knowledge_agent.py`](../agents/company_knowledge_agent.py) | Declines rather than answering with an empty retrieval context; falls back to the bundled policy file when Milvus is down, and says so in the reply. |
| [`orchestration/langgraph_adapter.py`](../orchestration/langgraph_adapter.py) | Unknown-agent fallback on the conditional edge; node-level exception containment. |
| [`orchestration/adk_adapter.py`](../orchestration/adk_adapter.py) | The native fallback, made non-silent — it reports itself in `reasoning` and logs an `adk_fallback` event. |
| [`core/llm_wrapper.py`](../core/llm_wrapper.py) | Key rotation on rate limit before exponential backoff. |

### Tests and evals

| Path | What it does |
|---|---|
| [`test_langgraph.py`](../test_langgraph.py) | 10 checks: graph structure, contract conformance, routing, slot-fill continuation, unknown-agent fallback, detector failure, node containment, path reporting, native parity. |
| [`test_validation.py`](../test_validation.py) | 16 checks on the leave safeguards, DB validation, atomicity, and the cancel escape hatch. |
| [`test_rag.py`](../test_rag.py) | 9 offline checks on chunking, grounding, citations, refusal and degradation, plus a `--live` pass against real Milvus and Gemini. |
| [`test_api.py`](../test_api.py) | 11 checks on the Flask surface — the two routes holding mutable server state (user and backend switching), plus the error paths the UI depends on. |
| [`test_eval.py`](../test_eval.py) | 10 checks on the routing-eval scoring itself, plus a `--live` probe of the judge against labelled data. |
| [`eval_system.py`](../eval_system.py) + [`evals/golden_set.json`](../evals/golden_set.json) | Golden-set evaluation, 16 hand-labelled cases across four suites: routing and retrieval scored deterministically, groundedness and refusal by a narrow LLM judge. |
| [`eval_routing.py`](../eval_routing.py) + [`core/routing_judge.py`](../core/routing_judge.py) | LLM-as-judge scoring of routing on real logged traffic. |
| [`test_metrics.py`](../test_metrics.py) | 14 checks on cost accounting, the streaming sink and the groundedness judge, plus a `--live` probe that shows the judge four fabrications and checks it catches them. |
| [`eval_retrieval.py`](../eval_retrieval.py) + [`evals/retrieval_benchmark.json`](../evals/retrieval_benchmark.json) | 25 labelled question→section pairs over `evals/corpus/`; recall@1/3/5 and MRR for dense, sparse and three fusion configurations. |
| [`verify.py`](../verify.py) | Import and DB-seeding smoke check. |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Runs every suite on two Python versions, builds the Docker image and runs the suites inside it, and fails the build on a committed secret, a machine-specific path, or a tracked `.env`. No secrets configured — the offline suites need none. |

**93 offline checks** across seven suites, none needing an API key, network or
Milvus. Both LLM judges are themselves scored against hand-labelled data
(`test_eval.py --live`, `test_metrics.py --live`), so a judge that rubber-stamps
everything fails its own test.

### Cost, latency and streaming

| Path | What it does |
|---|---|
| [`core/metrics.py`](../core/metrics.py) | Per-turn token, cost and latency accounting, split by stage. Held in a `ContextVar` so no call signature changes; a missing rate costs zero rather than being guessed, and an estimated token count is flagged. |
| [`core/streaming.py`](../core/streaming.py) | The token sink, also in a `ContextVar`. `LLMWrapper` streams into it when one is active and still returns the finished string, so agents and orchestrators are untouched. Only free-text generation streams. |
| [`core/groundedness.py`](../core/groundedness.py) | Per-claim judge: breaks an answer apart and checks each claim against the extracts it cites. |

### Logging

| Path | What it does |
|---|---|
| [`core/logger.py`](../core/logger.py) | JSONL writer. `log()` for conversation turns — carrying the cost/latency block — and `log_event()` for structured domain events; both go to one file so a refusal and the turn that caused it sit adjacent. |
| [`logs/README.md`](../logs/README.md) | Record reference for all 11 event types. |
| [`logs/sample-session.log`](../logs/sample-session.log) | A real recorded session, committed so behaviour is inspectable without running anything. |
| [`read_log.py`](../read_log.py) | Renders the log as a table. |

### What does not exist

- **No distributed tracing.** No OpenTelemetry, no LangSmith, no span tree. Every
  turn does carry its own cost and per-stage latency (`core/metrics.py`), but
  there is nothing that correlates a request across processes.
- **No auth.** No login, no roles. `active_user_id` is an integer in
  `config.yaml`, and the document-upload endpoint is unauthenticated.
- **No deployment.** CI builds the image and runs the suites, but nothing is
  deployed anywhere; there is no staging environment and no live traffic.
- **No load or concurrency testing.** Latency is measured one request at a time on
  a laptop; there is no throughput figure, no saturation point, and no locking
  around the read-validate-write sequence leave submission depends on.
- **The retrieval benchmark is one corpus.** 25 labelled cases over 74 chunks is a
  real measurement, not a distribution, and the labels are mine.

---

## 2. What I built, and what I did not

**Not mine:**

- **Frameworks**, used as intended: LangGraph (`StateGraph`, `add_conditional_edges`,
  `stream`), CrewAI, Google ADK, Flask, Milvus and `pymilvus`, the OpenAI SDK,
  `google-genai`, `pypdf`.
- **BM25 is Milvus's**, not mine. `knowledge/store.py` declares a `Function` of
  type `BM25` and Milvus computes the sparse vectors server-side. I did not
  implement a scorer or a vocabulary.
- **RRF is a standard algorithm**; I use `pymilvus`'s `RRFRanker`.
- **The Milvus compose services** (etcd, MinIO, standalone wiring) follow Milvus's
  published standalone deployment.
- **Written with an AI coding assistant.** Design decisions, the failure modes
  worth guarding, what to test and what to measure were mine; a substantial share
  of the code and prose was drafted with assistance and then reviewed, corrected
  and verified by me. The git history shows the sequence.

**Mine:**

- The orchestration abstraction — that `BaseOrchestrator` contract, and the
  decision to keep agents, session handling and persistence outside it so
  backends are interchangeable. The native/LangGraph parity test exists to hold
  that claim honest.
- The slot-filling state model: `active_agent` + `agent_state` on a persisted
  `SessionContext`, and the rule that classification is skipped while an agent
  holds the session.
- All five agents' logic and prompts.
- The leave-request validation rules and the atomic submission path.
- Heading-aware chunking, and the decision to prefix the heading trail onto the
  embedded text.
- Both evaluation harnesses, the golden set, and the choice of what is scored
  deterministically versus by judge.
- The log record design.
- Deliberately *not* mine to claim: the routing accuracy numbers are measured on
  8 hand-written cases, and the retrieval numbers on 6. They are real, and they
  are small.

---

## 3. Data: ingestion, retrieval, and bad records

Two data paths.

### Policy documents → Milvus

Ingestion is [`knowledge/store.py::ingest_text`](../knowledge/store.py), reached
from the Knowledge tab, `POST /api/knowledge/ingest`, or `ingest_knowledge.py`.
Retrieval is `hybrid_search` in the same file.

**A bad record here** is a document that contradicts or supersedes one already
stored — HR uploads a corrected leave policy while the old one is still indexed.
Retrieval then returns both, and the agent cites whichever ranked higher. This is
worse than an error, because the answer looks authoritative and carries a real
citation.

**Handled:** re-ingesting the same `source` deletes the previous copy inside the
same operation (`replace_existing=True`, `delete_by_source`), and the API reports
how many chunks were replaced.

**Not handled:** the same policy uploaded under a *different* filename. Nothing
detects near-duplicate content, so `leave-policy.md` and `leave-policy-v2.md`
would both be retrievable and could be cited against each other. The system has
no notion of a document superseding another except by filename identity.

Also handled: unparseable or scanned PDFs, unsupported file types, oversized
uploads, and empty documents — all rejected with a message naming the reason
([`app.py::_read_upload`](../app.py)).

### Leave and HR records → SQLite

Writes go through [`database/db.py`](../database/db.py).

**A bad record here** is a leave request that conflicts with existing state: more
days than the employee has left, or dates overlapping leave already booked. Both
are rejected before anything is written, with a message naming the conflict and a
structured log line carrying the numbers. Malformed dates, reversed ranges, empty
reasons, invalid statuses and unknown `user_id`s are rejected at the database
boundary as `RecordValidationError`.

**A subtler one:** the request and the balance deduction used to be two separate
transactions, so a failure between them left a request with no matching deduction.
They now share one transaction and the UPDATE is guarded so the balance cannot go
negative.

**Not handled:** two simultaneous requests can both pass validation before either
deducts. The guarded UPDATE prevents a negative balance, so the second one fails
rather than corrupting anything — but it fails with a storage error rather than a
clean "someone just used those days" message. There is no row-level locking
across HTTP requests.

---

## 4. A failure mode designed against

**The one I designed against: answering a policy question with nothing retrieved.**

If Milvus returns no chunks — an empty collection, a failed embedding call, a
query matching nothing — the naive path is to call the LLM anyway with an empty
context block. The model will not say "I received no context". It will answer from
its own priors, in the assistant's voice, about *this company's* leave policy. That
is the worst failure this system could have: confident, fluent, specific, and
completely invented. And it is invisible, because the reply looks exactly like a
correct one.

**How I anticipated it:** it is the standard failure of retrieval-augmented
systems, and it gets worse the better the surrounding UX is — a citation block on
a fabricated answer makes it *more* credible, not less.

**How I designed around it,** in
[`agents/company_knowledge_agent.py`](../agents/company_knowledge_agent.py):

1. Empty retrieval returns a fixed refusal **without calling the LLM at all**.
   Not a prompt instruction — a code path the model cannot talk its way past.
2. Retrieval failure is separated from empty retrieval. A failure falls back to
   the bundled policy file and **says so in the reply**; it does not silently
   pretend to be grounded.
3. When retrieval succeeds, extracts are numbered and citations are appended from
   the retrieved metadata, not from anything the model wrote — so a citation
   cannot be hallucinated.
4. Both branches log a structured event.

**How I verified the safeguard:**

- `test_rag.py::test_empty_retrieval_refuses_to_answer` asserts the reply is the
  refusal, that the log line was written, **and that the LLM was never called** —
  a fake that counts invocations, so a regression reintroducing the ungrounded
  call fails the test even if the wording happens to look right.
- `test_rag.py::test_falls_back_when_milvus_is_down` asserts the degraded reply
  says it is degraded and does *not* carry a sources block.
- `eval_system.py`'s refusal suite runs two invented-premise questions ("pet
  iguana policy", "crypto-trading leave") through the live stack and has a judge
  confirm the reply declines rather than inventing one. Both pass.
- The refusal path is visible in
  [`logs/sample-session.log`](../logs/sample-session.log).

**The related failure I did *not* anticipate, and found by accident:** while an
agent holds the session for slot-filling, the orchestrator skips routing
entirely. After a rejected leave request the agent kept the session so the user
could retry dates — but `cancel` was only handled while awaiting confirmation.
So a user whose request was refused was **trapped**: every subsequent message, on
any subject, came back to the leave agent asking for a start date. I found this
while reading a captured log for the sample session, where an HR ticket request
was swallowed by the leave agent. Fixed with an unconditional escape check in
both multi-turn agents, covered by `test_user_can_always_leave_the_flow` and
`test_rejection_does_not_trap_the_user`, and both tests were confirmed to fail
against the old behaviour before the fix was kept.
