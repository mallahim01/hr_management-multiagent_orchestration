# Logs

One JSON object per line (JSONL), appended to `logs/interactions.log`. Written by
`core/logger.py`. Conversation turns and domain events share the file on purpose:
when a leave request is refused, the turn and the reason for the refusal sit next
to each other in timestamp order, which is what you want when reconstructing what
happened.

- **`interactions.log`** — the live log. Git-ignored; created on first run.
- **`sample-session.log`** — a real recorded session, committed so the system's
  behaviour is inspectable without running anything. It was captured from the
  Docker stack, not hand-written.

Read either with:

```bash
python read_log.py                            # the live log
python read_log.py logs/sample-session.log    # the committed sample
```

## Record types

Every record has `timestamp` and `event`.

| `event` | Written by | Carries |
|---|---|---|
| `interaction` | `main.py`, `app.py` | one conversation turn: `user_input`, `intent`, `confidence`, `target_agent`, `agent_response` (truncated), `backend`, `session_id` |
| `leave_request_rejected` | `agents/leave_request_agent.py` | `reason_code`, `detail`, the dates, and for balance failures `requested_days` / `remaining_leaves`; for clashes `conflicting_request_id` |
| `hr_request_rejected` | `agents/hr_request_agent.py` | `reason_code`, `detail` |
| `agent_node_failed` | `orchestration/langgraph_adapter.py` | `node`, the exception type and message — an agent that raised and was contained |
| `adk_fallback` | `orchestration/adk_adapter.py` | why ADK was unavailable and which native agent answered instead |
| `knowledge_document_ingested` | `app.py` | `doc_id`, `source`, `chunks`, `replaced_chunks`, `uploaded_by` |
| `knowledge_document_deleted` | `app.py` | `doc_id`, `chunks` |
| `knowledge_retrieval_failed` | `agents/company_knowledge_agent.py` | why retrieval failed before the answer fell back to the bundled policy file |
| `knowledge_no_results` | `agents/company_knowledge_agent.py` | the query that retrieved nothing, so the agent declined instead of answering ungrounded |
| `routing_eval` | `eval_routing.py`, `/api/eval` | `judged`, `correct`, `incorrect`, `accuracy`, and the misroutes |
| `system_eval` | `eval_system.py` | per-suite pass counts against the golden set |

`reason_code` values for rejections: `insufficient_balance`, `overlapping_request`,
`malformed_dates`, `no_balance_record`, `storage_rejected`.

## What the sample session shows

The committed sample is one continuous conversation covering the paths worth
inspecting:

1. a greeting routed to the general agent
2. a policy question answered from Milvus with cited sources
3. a question the policy does not cover — the agent declines rather than inventing one
4. a leave balance lookup against SQLite
5. a multi-turn leave request, filled and submitted
6. a second request overlapping the first — **rejected**, with the clash logged
7. a request beyond the remaining balance — **rejected**, with the numbers logged
8. `cancel` — releasing the agent so routing resumes
9. an HR ticket raised through a different agent, proving the release worked

Useful greps:

```bash
grep leave_request_rejected logs/sample-session.log     # every refusal, with reasons
grep '"backend": "langgraph"' logs/sample-session.log   # turns served by the graph
grep routing_eval logs/sample-session.log               # the self-evaluation result
```

## Known limitations

- **No rotation.** The file grows without bound. Fine for a demo; a real
  deployment wants `logging.handlers.RotatingFileHandler` or shipping to a
  collector.
- **No redaction.** Whatever an employee types is stored verbatim. Real HR
  traffic would contain personal and medical detail, and this would need a
  retention policy and PII handling before going anywhere near production.
- **Agent-level events use the default log path.** Agents are constructed by the
  orchestrators as `cls(llm, db)` with no handle on the app's logger, so they log
  through `InteractionLogger()`'s default — which is the same file `main.py` and
  `app.py` use. Passing a custom path to those entry points would split the two
  streams apart.
