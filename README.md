# 🏢 ACME HR – Multi-Agent Demo System

A lightweight multi-agent HR assistant that demonstrates **intent-based orchestration** with **pluggable backends**. Designed for client demos.

---

## Quick Start

### 1. Install dependencies

```bash
cd E:\Projects\multiagent_
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
copy .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...
```

### 3. Run

| Mode | Command |
|------|---------|
| **Web UI** (recommended) | `python main.py --web` |
| Interactive CLI | `python main.py` |
| Auto-demo (5 prompts) | `python main.py --demo` |
| CLI data preview | `python main.py --preview-cli` |
| HTML report | `python main.py --preview-html` |

Open **http://127.0.0.1:5000** for the web UI.

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
multiagent_/
├── agents/               # 5 specialised sub-agents
├── core/                 # LLM wrapper, intent detector, session, logger
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
active_user_id: 1              # Auto-created on first run
llm:
  model: gpt-4o-mini
  max_retries: 3
  temperature: 0.7
database:
  path: data/hr_demo.db
conversation:
  history_size: 3              # Recent exchanges sent as context
```

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

> **Note:** CrewAI, LangGraph, and ADK adapters are implemented as stubs that display
> the integration pattern and delegate to native logic. Install `crewai` or `langgraph`
> and extend the respective adapter class to activate full framework integration.
