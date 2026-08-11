"""
test_validation.py – Tests for the leave-request safeguards and DB validation.

Runs entirely offline against a fake LLM, a throwaway SQLite file, and a
throwaway log file, so it needs no API key and no network.

Covers:
  Safeguard (a) – a request exceeding the remaining balance is rejected,
                  nothing is written, and a structured log line is emitted
  Safeguard (b) – a request overlapping existing leave is rejected likewise
  Malformed input – unparseable or reversed dates, empty reason/text,
                    unknown status, unknown user_id
  Atomicity     – a failed balance deduction leaves no orphan request row
  Robustness    – a corrupt session state blob does not break the session
  Happy path    – a valid request still submits and deducts correctly

Usage: python test_validation.py
"""

import json
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.leave_request_agent import LeaveRequestAgent
from core.logger import InteractionLogger
from core.session import SessionContext
from database.db import DatabaseManager, RecordValidationError
from database.schema import initialize_database

TEST_USER_ID = 1          # seeded with total=20, used=5, remaining=15


# ── Test doubles ─────────────────────────────────────────────────────────────

class FakeLLM:
    """Extracts nothing, so the agent works purely from pre-seeded slot state."""

    def chat(self, messages, json_mode=False, temperature=None) -> str:
        if json_mode:
            return json.dumps({"start_date": None, "end_date": None, "reason": None})
        return "Canned assistant reply."

    def chat_json(self, messages) -> dict:
        return {"intent": "leave_request", "confidence": 0.95,
                "target_agent": "LeaveRequestAgent", "reasoning": "fake"}


class Fixture:
    """A seeded database, a throwaway log, and an agent wired to both."""

    def __init__(self) -> None:
        workdir = tempfile.mkdtemp(prefix="hr_val_test_")
        self.db = DatabaseManager(os.path.join(workdir, "test.db"))
        initialize_database(self.db, TEST_USER_ID)
        self.log_path = os.path.join(workdir, "events.log")
        self.logger = InteractionLogger(self.log_path)
        self.agent = LeaveRequestAgent(FakeLLM(), self.db, self.logger)

    def ctx(self, **state) -> SessionContext:
        return SessionContext(
            session_id=str(uuid.uuid4()),
            user_id=TEST_USER_ID,
            active_agent="LeaveRequestAgent",
            agent_state=dict(state),
        )

    def events(self) -> list:
        """Structured (non-interaction) records written so far."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return [r for r in records if r.get("event") != "interaction"]

    def balance(self) -> dict:
        return self.db.get_leave_balance(TEST_USER_ID)

    def requests(self) -> list:
        return self.db.get_leave_requests(TEST_USER_ID)


# ── Harness ──────────────────────────────────────────────────────────────────

_results: list = []


def check(label: str, fn) -> None:
    try:
        fn()
        print(f"  PASS  {label}")
        _results.append((label, None))
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        _results.append((label, e))


def assert_raises(exc_type, fn, *args, **kwargs) -> Exception:
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        return e
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


# ── Safeguard (a): balance ───────────────────────────────────────────────────

def test_rejects_request_exceeding_balance() -> None:
    """15 days remaining; ask for 31. Must be refused, logged, and not stored."""
    fx = Fixture()
    ctx = fx.ctx(start_date="2026-06-01", end_date="2026-07-01", reason="vacation")

    reply = fx.agent.handle("please submit it", ctx)

    # User-facing message names both numbers.
    assert "31 days" in reply, reply
    assert "15 days" in reply, reply
    assert "❌" in reply, reply

    # Nothing was written and the balance did not move.
    assert fx.requests() == [], fx.requests()
    assert fx.balance()["remaining_leaves"] == 15, fx.balance()

    # A structured line was logged.
    events = fx.events()
    assert len(events) == 1, f"expected 1 event, got {events}"
    event = events[0]
    assert event["event"] == "leave_request_rejected", event
    assert event["reason_code"] == "insufficient_balance", event
    assert event["requested_days"] == 31, event
    assert event["remaining_leaves"] == 15, event
    assert event["start_date"] == "2026-06-01", event
    assert event["timestamp"], event

    # The user can retry: dates cleared, agent still holding the conversation.
    assert ctx.active_agent == "LeaveRequestAgent", ctx.active_agent
    assert "start_date" not in ctx.agent_state, ctx.agent_state
    assert ctx.agent_state.get("reason") == "vacation", ctx.agent_state


def test_rejects_at_confirmation_if_balance_moved() -> None:
    """
    The balance can change between the summary turn and the yes. The second
    validation pass must catch it.
    """
    fx = Fixture()
    ctx = fx.ctx(start_date="2026-06-01", end_date="2026-06-10",
                 reason="vacation", awaiting_confirmation=True)

    # Someone else consumed the balance in the meantime.
    fx.db.execute(
        "UPDATE leave_balance SET remaining_leaves = 2 WHERE user_id = ?",
        (TEST_USER_ID,),
    )

    reply = fx.agent.handle("yes", ctx)

    assert "❌" in reply, reply
    assert fx.requests() == [], "a request was stored despite the balance check"
    assert fx.balance()["remaining_leaves"] == 2, fx.balance()
    assert fx.events()[0]["reason_code"] == "insufficient_balance", fx.events()


# ── Safeguard (b): overlap ───────────────────────────────────────────────────

def test_rejects_overlapping_request() -> None:
    """An approved booking on the calendar blocks an intersecting request."""
    fx = Fixture()
    existing = fx.db.insert_leave_request(
        TEST_USER_ID, "2026-06-08", "2026-06-12", "family wedding", status="Approved"
    )

    # 2026-06-10 → 06-11 sits inside the approved range.
    ctx = fx.ctx(start_date="2026-06-10", end_date="2026-06-11", reason="personal")
    reply = fx.agent.handle("please submit it", ctx)

    assert "❌" in reply, reply
    assert f"LR-{existing:04d}" in reply, reply
    assert "2026-06-08" in reply and "family wedding" in reply, reply

    # Only the pre-existing row is present; nothing new was stored.
    assert len(fx.requests()) == 1, fx.requests()
    assert fx.balance()["remaining_leaves"] == 15, fx.balance()

    events = fx.events()
    assert len(events) == 1, events
    assert events[0]["reason_code"] == "overlapping_request", events[0]
    assert events[0]["conflicting_request_id"] == existing, events[0]


def test_overlap_detection_covers_every_intersection_shape() -> None:
    """Identical, enclosing, enclosed, and both partial overlaps must all hit."""
    fx = Fixture()
    fx.db.insert_leave_request(
        TEST_USER_ID, "2026-06-10", "2026-06-20", "booked", status="Approved"
    )
    overlapping = [
        ("2026-06-10", "2026-06-20", "identical"),
        ("2026-06-01", "2026-06-30", "encloses"),
        ("2026-06-12", "2026-06-14", "enclosed by"),
        ("2026-06-05", "2026-06-11", "overlaps start"),
        ("2026-06-19", "2026-06-25", "overlaps end"),
        ("2026-06-20", "2026-06-20", "touches last day"),
    ]
    for start, end, shape in overlapping:
        clashes = fx.db.get_overlapping_leave_requests(TEST_USER_ID, start, end)
        assert clashes, f"{shape} ({start}→{end}) was not detected as an overlap"

    for start, end, shape in [("2026-06-01", "2026-06-09", "strictly before"),
                              ("2026-06-21", "2026-06-30", "strictly after")]:
        clashes = fx.db.get_overlapping_leave_requests(TEST_USER_ID, start, end)
        assert not clashes, f"{shape} ({start}→{end}) was wrongly flagged"


def test_rejected_statuses_do_not_block() -> None:
    """Leave that was rejected or cancelled no longer occupies the calendar."""
    fx = Fixture()
    for status in ("Rejected", "Cancelled"):
        fx.db.insert_leave_request(
            TEST_USER_ID, "2026-08-01", "2026-08-05", "old", status=status
        )
    clashes = fx.db.get_overlapping_leave_requests(
        TEST_USER_ID, "2026-08-02", "2026-08-03"
    )
    assert not clashes, f"a {clashes[0]['status']} request blocked a new one"


# ── Malformed records at the DB boundary ─────────────────────────────────────

def test_rejects_malformed_dates() -> None:
    db = Fixture().db
    for bad in ("not-a-date", "2026-13-01", "2026-02-30", "01/06/2026", "", None):
        assert_raises(RecordValidationError,
                      db.insert_leave_request, TEST_USER_ID, bad, "2026-06-02", "x")


def test_rejects_reversed_date_range() -> None:
    db = Fixture().db
    e = assert_raises(RecordValidationError, db.insert_leave_request,
                      TEST_USER_ID, "2026-06-10", "2026-06-01", "x")
    assert "before" in str(e), str(e)


def test_rejects_empty_reason_and_bad_status() -> None:
    db = Fixture().db
    assert_raises(RecordValidationError, db.insert_leave_request,
                  TEST_USER_ID, "2026-06-01", "2026-06-02", "   ")
    assert_raises(RecordValidationError, db.insert_leave_request,
                  TEST_USER_ID, "2026-06-01", "2026-06-02", "x", "Approvedd")


def test_rejects_unknown_user_id() -> None:
    """The users FK is now enforced, so an orphan request cannot be created."""
    db = Fixture().db
    assert_raises(RecordValidationError, db.insert_leave_request,
                  9999, "2026-06-01", "2026-06-02", "ghost")
    assert_raises(RecordValidationError, db.insert_hr_request, 9999, "ghost ticket")


def test_rejects_empty_hr_request() -> None:
    db = Fixture().db
    assert_raises(RecordValidationError, db.insert_hr_request, TEST_USER_ID, "  ")


# ── Atomicity ────────────────────────────────────────────────────────────────

def test_submit_is_atomic() -> None:
    """
    If the balance deduction cannot apply, the request row must be rolled back
    too — previously these were two independent transactions.
    """
    fx = Fixture()
    fx.db.execute(
        "UPDATE leave_balance SET remaining_leaves = 1 WHERE user_id = ?",
        (TEST_USER_ID,),
    )
    assert_raises(RecordValidationError, fx.db.submit_leave_request,
                  TEST_USER_ID, "2026-06-01", "2026-06-05", "too long")

    assert fx.requests() == [], "orphan request row survived a failed deduction"
    assert fx.balance()["remaining_leaves"] == 1, fx.balance()


def test_happy_path_submits_and_deducts() -> None:
    """The safeguards must not block a legitimate request."""
    fx = Fixture()
    ctx = fx.ctx(start_date="2026-06-01", end_date="2026-06-03",
                 reason="vacation", awaiting_confirmation=True)

    reply = fx.agent.handle("yes", ctx)

    assert "submitted successfully" in reply, reply
    assert "3 days deducted" in reply, reply

    rows = fx.requests()
    assert len(rows) == 1, rows
    assert rows[0]["start_date"] == "2026-06-01", rows[0]
    assert rows[0]["status"] == "Pending", rows[0]

    balance = fx.balance()
    assert balance["remaining_leaves"] == 12, balance   # 15 - 3
    assert balance["used_leaves"] == 8, balance         # 5 + 3

    assert fx.events() == [], f"a valid request logged a rejection: {fx.events()}"
    assert ctx.active_agent is None, "agent did not release the session"


# ── Escape hatch ─────────────────────────────────────────────────────────────

def test_user_can_always_leave_the_flow() -> None:
    """
    While this agent holds the session the orchestrator skips routing, so a
    user with no way out is stuck talking to it about nothing else. Cancelling
    must work at every stage, not only while a confirmation is pending.
    """
    stages = {
        "collecting slots":       {},
        "after partial slots":    {"start_date": "2026-06-01"},
        "awaiting confirmation":  {"start_date": "2026-06-01", "end_date": "2026-06-02",
                                   "reason": "x", "awaiting_confirmation": True},
    }
    for stage, state in stages.items():
        for word in ("cancel", "never mind", "stop", "forget it"):
            fx = Fixture()
            ctx = fx.ctx(**state)
            reply = fx.agent.handle(word, ctx)
            assert ctx.active_agent is None, f"{stage!r} + {word!r} left the agent holding the session"
            assert ctx.agent_state == {}, f"{stage!r} + {word!r} left stale state: {ctx.agent_state}"
            assert "cancel" in reply.lower(), reply


def test_rejection_does_not_trap_the_user() -> None:
    """
    A rejected request keeps the agent active so the user can retry dates —
    but they must still be able to walk away, which they could not before.
    """
    fx = Fixture()
    ctx = fx.ctx(start_date="2026-06-01", end_date="2026-07-01", reason="vacation")

    rejection = fx.agent.handle("submit it", ctx)
    assert "❌" in rejection, rejection
    assert ctx.active_agent == "LeaveRequestAgent", "agent should hold on for a retry"

    reply = fx.agent.handle("actually, cancel that", ctx)
    assert ctx.active_agent is None, "user could not escape after a rejection"
    assert fx.requests() == [], "nothing should have been stored"


def test_hr_agent_can_also_be_cancelled() -> None:
    from agents.hr_request_agent import HRRequestAgent

    fx = Fixture()
    agent = HRRequestAgent(FakeLLM(), fx.db, fx.logger)
    ctx = fx.ctx(request_description="a salary slip")
    ctx.active_agent = "HRRequestAgent"

    reply = agent.handle("never mind", ctx)
    assert ctx.active_agent is None, ctx.active_agent
    assert ctx.agent_state == {}, ctx.agent_state
    assert "cancel" in reply.lower(), reply
    assert fx.db.get_hr_requests(TEST_USER_ID) == [], "cancelled request was stored"


# ── Robustness ───────────────────────────────────────────────────────────────

def test_corrupt_session_state_is_survivable() -> None:
    """A truncated JSON blob must reset the session, not raise."""
    fx = Fixture()
    session_id = str(uuid.uuid4())
    fx.db.save_session(session_id, TEST_USER_ID, "LeaveRequestAgent", {"a": 1})
    fx.db.execute(
        "UPDATE sessions SET agent_state = ? WHERE session_id = ?",
        ('{"start_date": "2026-0', session_id),
    )

    row = fx.db.load_session(session_id)
    assert row is not None, "session vanished"
    assert row["agent_state"] == {}, row["agent_state"]
    assert row["active_agent"] is None, row["active_agent"]


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== Leave-Request Safeguards & DB Validation – Tests ===\n")

    check("(a) rejects request exceeding balance",   test_rejects_request_exceeding_balance)
    check("(a) re-checks balance at confirmation",   test_rejects_at_confirmation_if_balance_moved)
    check("(b) rejects overlapping request",         test_rejects_overlapping_request)
    check("(b) detects every overlap shape",         test_overlap_detection_covers_every_intersection_shape)
    check("(b) rejected/cancelled leave frees dates", test_rejected_statuses_do_not_block)
    check("rejects malformed dates",                 test_rejects_malformed_dates)
    check("rejects reversed date range",             test_rejects_reversed_date_range)
    check("rejects empty reason / bad status",       test_rejects_empty_reason_and_bad_status)
    check("rejects unknown user_id (FK enforced)",   test_rejects_unknown_user_id)
    check("rejects empty HR request text",           test_rejects_empty_hr_request)
    check("submit is atomic",                        test_submit_is_atomic)
    check("happy path submits and deducts",          test_happy_path_submits_and_deducts)
    check("user can always leave the flow",          test_user_can_always_leave_the_flow)
    check("rejection does not trap the user",        test_rejection_does_not_trap_the_user)
    check("HR agent can also be cancelled",          test_hr_agent_can_also_be_cancelled)
    check("corrupt session state is survivable",     test_corrupt_session_state_is_survivable)

    failures = [label for label, err in _results if err is not None]
    print("\n" + "=" * 50)
    if failures:
        print(f"FAILED: {len(failures)}/{len(_results)} – {failures}")
        sys.exit(1)
    print(f"All {len(_results)} checks PASSED")


if __name__ == "__main__":
    main()
