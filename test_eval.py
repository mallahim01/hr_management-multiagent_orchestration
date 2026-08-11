"""
test_eval.py – Tests for the LLM-as-judge routing evaluation.

Default mode is offline against a fake judge: it verifies the log parsing,
continuation handling, scoring arithmetic and failure behaviour, none of which
should depend on a live model.

    python test_eval.py           # offline, deterministic, no API key
    python test_eval.py --live    # also probe the real judge (costs tokens)

The --live probe is the one that matters for trusting the numbers: it feeds the
judge eight synthetic turns with known-good and known-bad routing and checks it
separates them. An evaluator that returns "correct" for everything would report
100% accuracy forever, so this is worth re-running whenever the judge prompt or
the model changes.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logger import InteractionLogger
from core.routing_judge import RoutingJudge

# (user message, agent it was routed to, verdict a competent judge should give)
GROUND_TRUTH = [
    ("How many annual leave days do I have left?", "CompanyKnowledgeAgent", "incorrect"),
    ("Hi there, how's your day going?",            "LeaveRequestAgent",     "incorrect"),
    ("What is the work from home policy?",         "LeaveBalanceAgent",     "incorrect"),
    ("I want to book leave for next Monday",       "GeneralAssistantAgent", "incorrect"),
    ("How many leave days do I have left?",        "LeaveBalanceAgent",     "correct"),
    ("What is our maternity leave policy?",        "CompanyKnowledgeAgent", "correct"),
    ("Good morning!",                              "GeneralAssistantAgent", "correct"),
    ("Please raise a ticket for my salary slip",   "HRRequestAgent",        "correct"),
]


# ── Test doubles ─────────────────────────────────────────────────────────────

class FakeJudgeLLM:
    """Returns a scripted verdict for every turn it is shown."""

    def __init__(self, verdict: str = "correct", fail: bool = False) -> None:
        self.verdict = verdict
        self.fail = fail
        self.calls = 0

    def chat_json(self, messages) -> dict:
        self.calls += 1
        if self.fail:
            raise RuntimeError("judge unavailable")
        # Count the turns in the prompt and answer one verdict each.
        payload = messages[-1]["content"]
        turn_numbers = [
            int(line.split("Turn ")[1].split(" ")[0])
            for line in payload.splitlines() if line.startswith("--- Turn ")
        ]
        return {"verdicts": [
            {"n": n, "verdict": self.verdict,
             "expected_agent": "LeaveBalanceAgent", "reason": "canned"}
            for n in turn_numbers
        ]}


def write_log(records) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="hr_eval_test_"), "log.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def turn(user_input, agent, intent="leave_balance", backend="native") -> dict:
    return {"timestamp": "2026-08-11T00:00:00Z", "event": "interaction",
            "session_id": "s", "user_id": 3, "user_input": user_input,
            "intent": intent, "confidence": 0.9, "target_agent": agent,
            "agent_response": "reply", "backend": backend}


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


# ── Log reading ──────────────────────────────────────────────────────────────

def test_reads_only_interaction_turns() -> None:
    """Domain events share the file and must not be scored as turns."""
    path = write_log([
        turn("a", "LeaveBalanceAgent"),
        {"timestamp": "…", "event": "leave_request_rejected", "reason_code": "x"},
        turn("b", "LeaveBalanceAgent"),
        {"timestamp": "…", "event": "routing_eval", "accuracy": 1.0},
    ])
    turns = RoutingJudge(FakeJudgeLLM(), log_path=path).load_turns(limit=10)
    assert [t["user_input"] for t in turns] == ["a", "b"], turns


def test_tolerates_malformed_and_missing_log() -> None:
    path = write_log([turn("a", "LeaveBalanceAgent")])
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"truncated": \n')          # a partially-written last line
    turns = RoutingJudge(FakeJudgeLLM(), log_path=path).load_turns(limit=10)
    assert len(turns) == 1, turns

    missing = RoutingJudge(FakeJudgeLLM(), log_path="does/not/exist.log")
    assert missing.load_turns(limit=5) == []
    assert missing.evaluate(limit=5)["judged"] == 0


def test_limit_takes_the_most_recent_oldest_first() -> None:
    path = write_log([turn(str(i), "LeaveBalanceAgent") for i in range(10)])
    turns = RoutingJudge(FakeJudgeLLM(), log_path=path).load_turns(limit=3)
    assert [t["user_input"] for t in turns] == ["7", "8", "9"], turns


# ── Continuation handling ────────────────────────────────────────────────────

def test_continuations_are_skipped_not_scored() -> None:
    """
    Mid slot-fill the orchestrator deliberately skips classification, so those
    turns carry no routing decision and must not dilute the score.
    """
    path = write_log([
        turn("start leave", "LeaveRequestAgent", intent="leave_request"),
        turn("yes", "LeaveRequestAgent", intent="continuation"),
        turn("same day", "LeaveRequestAgent", intent="continuation"),
    ])
    llm = FakeJudgeLLM("correct")
    report = RoutingJudge(llm, log_path=path).evaluate(limit=10)

    assert report["judged"] == 1, report
    assert report["skipped_continuations"] == 2, report
    assert report["accuracy"] == 1.0, report


def test_all_continuations_reports_nothing_to_grade() -> None:
    path = write_log([turn("yes", "LeaveRequestAgent", intent="continuation")])
    llm = FakeJudgeLLM()
    report = RoutingJudge(llm, log_path=path).evaluate(limit=10)
    assert report["judged"] == 0, report
    assert report["accuracy"] is None, report
    assert llm.calls == 0, "judge was called with nothing to grade"


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_accuracy_excludes_ambiguous() -> None:
    """Ambiguous turns must not be silently counted as passes."""
    path = write_log([turn(str(i), "LeaveBalanceAgent") for i in range(4)])
    report = RoutingJudge(FakeJudgeLLM("ambiguous"), log_path=path).evaluate(limit=4)
    assert report["ambiguous"] == 4, report
    assert report["accuracy"] is None, f"ambiguous turns inflated accuracy: {report}"

    report = RoutingJudge(FakeJudgeLLM("incorrect"), log_path=path).evaluate(limit=4)
    assert report["accuracy"] == 0.0, report
    assert report["incorrect"] == 4, report


def test_batching_covers_every_turn() -> None:
    path = write_log([turn(str(i), "LeaveBalanceAgent") for i in range(7)])
    llm = FakeJudgeLLM("correct")
    report = RoutingJudge(llm, log_path=path, batch_size=3).evaluate(limit=7)
    assert report["judged"] == 7, report
    assert llm.calls == 3, f"expected 3 batches for 7 turns, got {llm.calls}"
    assert [r["n"] for r in report["results"]] == list(range(1, 8)), report["results"]


def test_judge_failure_degrades_to_error_verdicts() -> None:
    """A flaky judge must not raise, and must not be scored as success."""
    path = write_log([turn(str(i), "LeaveBalanceAgent") for i in range(3)])
    report = RoutingJudge(FakeJudgeLLM(fail=True), log_path=path).evaluate(limit=3)
    assert report["errors"] == 3, report
    assert report["accuracy"] is None, report
    assert all(r["verdict"] == "error" for r in report["results"]), report["results"]


def test_unknown_expected_agent_is_normalised() -> None:
    """A hallucinated agent name must not leak into the report."""
    class HallucinatingLLM(FakeJudgeLLM):
        def chat_json(self, messages):
            return {"verdicts": [{"n": 1, "verdict": "incorrect",
                                  "expected_agent": "PayrollWizardAgent",
                                  "reason": "made up"}]}

    path = write_log([turn("a", "LeaveBalanceAgent")])
    report = RoutingJudge(HallucinatingLLM(), log_path=path).evaluate(limit=1)
    assert report["results"][0]["expected_agent"] == "LeaveBalanceAgent", report["results"]


# ── Persistence ──────────────────────────────────────────────────────────────

def test_report_is_logged_with_misroutes() -> None:
    path = write_log([turn("a", "LeaveBalanceAgent")])
    out = os.path.join(tempfile.mkdtemp(prefix="hr_eval_out_"), "events.log")
    judge = RoutingJudge(FakeJudgeLLM("incorrect"), log_path=path)
    report = judge.evaluate(limit=1)
    judge.log_report(report, InteractionLogger(out))

    with open(out, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert len(events) == 1, events
    assert events[0]["event"] == "routing_eval", events[0]
    assert events[0]["incorrect"] == 1, events[0]
    assert events[0]["misroutes"][0]["chosen"] == "LeaveBalanceAgent", events[0]


# ── Live probe (opt-in) ──────────────────────────────────────────────────────

def run_live_probe() -> int:
    """
    Score the real judge against hand-labelled routing decisions.

    Returns the number of disagreements with the ground truth.
    """
    from dotenv import load_dotenv
    import yaml
    load_dotenv()
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    from core.llm_wrapper import LLMWrapper, available_keys

    provider = config["llm"].get("provider", "groq")
    if not available_keys(provider):
        print(f"  SKIP  live probe – no API key for provider '{provider}'")
        return 0

    path = write_log([turn(msg, agent, intent="probe") for msg, agent, _ in GROUND_TRUTH])
    llm = LLMWrapper(model=config["llm"]["model"], provider=provider)
    report = RoutingJudge(llm, log_path=path, batch_size=4).evaluate(limit=len(GROUND_TRUTH))

    misses = 0
    for (msg, agent, want), r in zip(GROUND_TRUTH, report["results"]):
        agree = r["verdict"] == want
        misses += not agree
        print(f"  {'OK  ' if agree else 'MISS'}  want={want:9} got={r['verdict']:9}"
              f"  {msg[:40]:42} -> {agent}")
        if not agree:
            print(f"          judge said: {r['reason']}")
    print(f"\n  judge agreed with {len(GROUND_TRUTH) - misses}/{len(GROUND_TRUTH)} labels")
    return misses


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    live = "--live" in sys.argv

    print("\n=== Routing Evaluation – Tests (offline) ===\n")
    check("reads only interaction turns",         test_reads_only_interaction_turns)
    check("tolerates malformed / missing log",    test_tolerates_malformed_and_missing_log)
    check("limit takes most recent, oldest first", test_limit_takes_the_most_recent_oldest_first)
    check("continuations skipped, not scored",    test_continuations_are_skipped_not_scored)
    check("all-continuations grades nothing",     test_all_continuations_reports_nothing_to_grade)
    check("accuracy excludes ambiguous",          test_accuracy_excludes_ambiguous)
    check("batching covers every turn",           test_batching_covers_every_turn)
    check("judge failure degrades to errors",     test_judge_failure_degrades_to_error_verdicts)
    check("hallucinated agent normalised",        test_unknown_expected_agent_is_normalised)
    check("report logged with misroutes",         test_report_is_logged_with_misroutes)

    misses = 0
    if live:
        print("\n=== Live judge probe (ground truth) ===\n")
        misses = run_live_probe()
        # Allow one disagreement: these are judgement calls, not arithmetic.
        check("live judge separates good from bad routing",
              lambda: (_ for _ in ()).throw(AssertionError(
                  f"{misses} disagreements with ground truth")) if misses > 1 else None)
    else:
        print("\n  (run with --live to also probe the real judge against ground truth)")

    failures = [label for label, err in _results if err is not None]
    print("\n" + "=" * 50)
    if failures:
        print(f"FAILED: {len(failures)}/{len(_results)} – {failures}")
        sys.exit(1)
    print(f"All {len(_results)} checks PASSED")


if __name__ == "__main__":
    main()
