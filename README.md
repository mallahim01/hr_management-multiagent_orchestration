# 🏢 ACME HR – Multi-Agent System

A lightweight multi-agent HR assistant that demonstrates **intent-based orchestration** with **pluggable backends**.

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
python verify.py           # imports, DB seeding, agent registry
python test_langgraph.py   # LangGraph orchestrator — offline, no API key needed
python test_backends.py    # all 4 backends against the live LLM (costs tokens)
```

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

Edit `config.yaml`:

```yaml
orchestrator_backend: native     # or: crewai | langgraph | adk
```

Restart the server. No other code changes needed.

---

## Project Structure

```
.
├── agents/               # 5 specialised sub-agents
├── core/                 # LLM wrapper, intent detector, session, logger
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

---

## API Endpoints (Flask)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Chat UI |
| POST | `/api/chat` | Send a message |
| GET | `/api/status` | System info |
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
| `langgraph` | Real compiled `StateGraph`; nodes call the same agents as native | Yes — `python test_langgraph.py` (8 offline checks, no API key needed) |
| `crewai` | Real `Agent`/`Task`/`Crew`; the native agent runs first, then a sequential Crew polishes the reply | Manually, via `test_backends.py` |
| `adk` | Real `google.adk` root agent with the five agents exposed as tools; Gemini decides routing | Not covered by automated tests |

All four share the same `BaseOrchestrator` contract and the same agents — only the
routing machinery differs. See [docs/architecture.md](docs/architecture.md).
