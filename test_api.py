"""
test_api.py – Tests for the Flask API surface.

Runs offline against Flask's test client with a fake LLM and a throwaway
database. No API key, no network, no Milvus.

Covers the routes that hold mutable server state — the backend switcher and the
user switcher — because those are the two places where a bug leaks one request's
context into the next. Also covers the error paths, since the UI relies on them
returning a usable message rather than a stack trace.

Usage: python test_api.py
"""

import json
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The knowledge base is not what these tests are about, and leaving it on makes
# every request wait for a Milvus connection that is not there.
os.environ["KNOWLEDGE__ENABLED"] = "false"

from core.logger import InteractionLogger
from core.session import SessionManager
from database.db import DatabaseManager
from database.schema import initialize_database
from orchestration.native import NativeOrchestrator

CONFIG = {
    "orchestrator_backend": "native",
    "active_user_id": 3,
    "llm": {"provider": "groq", "model": "fake-model", "max_retries": 1, "temperature": 0.0},
    "database": {"path": ":memory:"},
    "conversation": {"history_size": 3},
    "knowledge": {"enabled": False},
}


class FakeLLM:
    """Routes everything to the leave-balance agent and answers with a fixed line."""

    model = "fake-model"
    provider = "fake"

    def chat(self, messages, json_mode=False, temperature=None) -> str:
        if json_mode:
            return json.dumps({"start_date": None, "end_date": None, "reason": None})
        return "Canned reply."

    def chat_json(self, messages) -> dict:
        return {"intent": "leave_balance", "confidence": 0.9,
                "target_agent": "LeaveBalanceAgent", "reasoning": "fake"}


def make_client():
    """
    Build an isolated app. Returns (client, db, log_path).

    Each call gets its own directory: a test that inspected a shared or
    globbed log would pick up events from whichever test ran before it.
    """
    from app import create_app

    workdir = tempfile.mkdtemp(prefix="hr_api_test_")
    db_path = os.path.join(workdir, "t.db")
    log_path = os.path.join(workdir, "events.log")
    db = DatabaseManager(db_path)
    initialize_database(db, CONFIG["active_user_id"])
    llm = FakeLLM()
    config = dict(CONFIG, database={"path": db_path})
    app = create_app(
        config, db, llm,
        NativeOrchestrator(llm, db, 3),
        SessionManager(db, 3),
        InteractionLogger(log_path),
    )
    return app.test_client(), db, log_path


# ── Harness ──────────────────────────────────────────────────────────────────

_results: list = []


def check(label, fn):
    try:
        fn()
        print(f"  PASS  {label}")
        _results.append((label, None))
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        _results.append((label, e))


# ── Status ───────────────────────────────────────────────────────────────────

def test_status_reports_the_running_configuration() -> None:
    c, _, _ = make_client()
    s = c.get("/api/status").get_json()
    assert s["backend"] == "native", s
    assert s["provider"] == "groq", s
    assert s["active_user"]["name"] == "Carol Williams", s
    assert set(s["backends"]) == {"native", "langgraph", "crewai", "adk"}, s


# ── User switching ───────────────────────────────────────────────────────────

def test_lists_every_user_with_balance() -> None:
    c, _, _ = make_client()
    users = c.get("/api/users").get_json()
    assert len(users) == 3, users
    assert [u["name"] for u in users] == ["Alice Johnson", "Bob Martinez", "Carol Williams"], users
    assert sum(u["active"] for u in users) == 1, "exactly one user should be active"
    assert all(u["remaining_leaves"] is not None for u in users), users


def test_switching_user_changes_whose_data_is_served() -> None:
    """
    The important part: after a switch, every read *and* every write must be
    against the new employee. A stale user_id here would file one person's
    leave against another.
    """
    c, db, _ = make_client()
    assert c.get("/api/preview/leave-balance").get_json()["user_id"] == 3

    r = c.post("/api/user", json={"user_id": 1})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["active_user"]["name"] == "Alice Johnson", body
    assert body["reset_session"] is True, "client must be told to start a new conversation"

    assert c.get("/api/status").get_json()["active_user"]["id"] == 1
    assert c.get("/api/preview/leave-balance").get_json()["user_id"] == 1
    users = c.get("/api/users").get_json()
    assert [u["id"] for u in users if u["active"]] == [1], users

    # A turn taken after the switch must be recorded against Alice, not Carol.
    session_id = str(uuid.uuid4())
    result = c.post("/api/chat", json={"message": "how many days left?",
                                       "session_id": session_id}).get_json()
    assert result["user_id"] == 1, result
    rows = db.fetch_all(
        "SELECT DISTINCT user_id FROM conversation_history WHERE session_id = ?",
        (session_id,))
    assert [r["user_id"] for r in rows] == [1], rows


def test_switching_user_is_logged() -> None:
    c, _, log_path = make_client()
    c.post("/api/user", json={"user_id": 2})

    with open(log_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    switches = [r for r in records if r.get("event") == "active_user_changed"]
    assert len(switches) == 1, switches
    assert switches[0]["previous_user_id"] == 3, switches[0]
    assert switches[0]["user_id"] == 2, switches[0]
    assert switches[0]["name"] == "Bob Martinez", switches[0]

    # Re-selecting the same user is a no-op and must not log a second change.
    c.post("/api/user", json={"user_id": 2})
    with open(log_path, encoding="utf-8") as f:
        again = [json.loads(l) for l in f if l.strip() and
                 json.loads(l).get("event") == "active_user_changed"]
    assert len(again) == 1, f"a no-op switch logged an event: {again}"


def test_rejects_unknown_or_malformed_user() -> None:
    c, _, _ = make_client()
    assert c.post("/api/user", json={"user_id": 999}).status_code == 404
    assert c.post("/api/user", json={"user_id": "carol"}).status_code == 400
    assert c.post("/api/user", json={}).status_code == 400
    # A failed switch must not move the active user.
    assert c.get("/api/status").get_json()["active_user"]["id"] == 3


# ── Backend switching ────────────────────────────────────────────────────────

def test_switching_backend() -> None:
    c, _, _ = make_client()
    r = c.post("/api/backend", json={"backend": "langgraph"})
    assert r.status_code == 200 and r.get_json()["backend"] == "langgraph", r.get_json()
    assert c.get("/api/status").get_json()["backend"] == "langgraph"

    # Re-selecting the same backend is a no-op, not a rebuild.
    again = c.post("/api/backend", json={"backend": "langgraph"}).get_json()
    assert again["changed"] is False, again

    # A turn now reports the graph path, which only langgraph produces.
    result = c.post("/api/chat", json={"message": "how many days left?"}).get_json()
    assert result["backend"] == "langgraph", result
    assert result["graph_path"] == ["detect_intent", "LeaveBalanceAgent"], result


def test_rejects_unknown_backend_and_keeps_serving() -> None:
    c, _, _ = make_client()
    r = c.post("/api/backend", json={"backend": "nonsense"})
    assert r.status_code == 400, r.get_json()
    assert "nonsense" in r.get_json()["error"], r.get_json()
    assert c.get("/api/status").get_json()["backend"] == "native", "failed switch changed state"
    assert c.post("/api/chat", json={"message": "hello"}).status_code == 200


# ── Chat + graph + errors ────────────────────────────────────────────────────

def test_chat_rejects_empty_message() -> None:
    c, _, _ = make_client()
    assert c.post("/api/chat", json={"message": "   "}).status_code == 400
    assert c.post("/api/chat", json={}).status_code == 400


def test_graph_endpoint_is_backend_aware() -> None:
    c, _, _ = make_client()
    assert c.get("/api/graph").get_json()["available"] is False, "native has no graph"
    c.post("/api/backend", json={"backend": "langgraph"})
    g = c.get("/api/graph").get_json()
    assert g["available"] is True and "detect_intent" in g["ascii"], g


def test_knowledge_endpoints_degrade_without_milvus() -> None:
    """With the knowledge base off these must answer, not hang or 500."""
    c, _, _ = make_client()
    s = c.get("/api/knowledge/status")
    assert s.status_code == 200, s.status_code
    assert s.get_json()["enabled"] is False, s.get_json()
    assert c.get("/api/knowledge/documents").status_code == 200
    assert c.get("/api/knowledge/search").status_code == 400, "missing q must be a 400"


def test_serves_the_ui() -> None:
    c, _, _ = make_client()
    r = c.get("/")
    assert r.status_code == 200, r.status_code
    html = r.get_data(as_text=True)
    for marker in ("user-list", "panel-knowledge", "panel-graph", "panel-evals"):
        assert marker in html, f"UI is missing {marker}"


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== Flask API – Tests ===\n")
    check("status reports the configuration",   test_status_reports_the_running_configuration)
    check("lists every user with balance",      test_lists_every_user_with_balance)
    check("switching user changes reads+writes", test_switching_user_changes_whose_data_is_served)
    check("switching user is logged",           test_switching_user_is_logged)
    check("rejects unknown/malformed user",     test_rejects_unknown_or_malformed_user)
    check("switching backend",                  test_switching_backend)
    check("rejects unknown backend",            test_rejects_unknown_backend_and_keeps_serving)
    check("chat rejects an empty message",      test_chat_rejects_empty_message)
    check("graph endpoint is backend-aware",    test_graph_endpoint_is_backend_aware)
    check("knowledge endpoints degrade safely", test_knowledge_endpoints_degrade_without_milvus)
    check("serves the UI",                      test_serves_the_ui)

    failures = [label for label, err in _results if err is not None]
    print("\n" + "=" * 50)
    if failures:
        print(f"FAILED: {len(failures)}/{len(_results)} – {failures}")
        sys.exit(1)
    print(f"All {len(_results)} checks PASSED")


if __name__ == "__main__":
    main()
