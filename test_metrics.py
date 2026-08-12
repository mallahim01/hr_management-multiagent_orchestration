"""
test_metrics.py – Tests for cost/latency accounting, streaming and groundedness.

Offline against fakes: no API key, no network, no Milvus.

    python test_metrics.py           # offline, deterministic
    python test_metrics.py --live    # also probe the real groundedness judge

The --live probe is the one that decides whether the groundedness numbers mean
anything. It shows the judge a faithful answer and four fabrications — an
inflated figure, two invented rules, an invented entitlement — and checks it
separates them. A judge that returns "grounded" for everything would report a
perfect score forever.
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import metrics
from core import streaming as streams
from core.groundedness import GroundednessJudge
from core.logger import InteractionLogger

# Extracts and answers used by both the offline shape checks and the live probe.
EXTRACTS = [
    {"section": "SECTION 2 – WFH › 2.2 WFH Entitlement", "source": "remote.txt",
     "text": "Employees may work from home up to 2 days per week with line manager approval."},
    {"section": "SECTION 2 – WFH › 2.1 Eligibility", "source": "remote.txt",
     "text": "Employees become eligible after completing the 6 month probation period."},
]
QUESTION = "How many days a week can I work from home?"
GROUND_TRUTH = [
    ("faithful", "You can work from home up to 2 days per week [1], once you have "
                 "completed your 6 month probation [2].", True),
    ("inflated number", "You can work from home up to 4 days per week [1], once you "
                        "have completed probation [2].", False),
    ("invented rule", "You can work from home up to 2 days per week [1]. You must also "
                      "give 48 hours notice and log it in the attendance portal [1].", False),
    ("invented entitlement", "You can work from home up to 2 days per week [1], and you "
                             "receive a 50,000 PKR home office allowance [2].", False),
]


class FakeJudge:
    """Returns a scripted claim list."""

    def __init__(self, claims=None, fail=False) -> None:
        self.claims = claims if claims is not None else []
        self.fail = fail

    def chat_json(self, messages) -> dict:
        if self.fail:
            raise RuntimeError("judge unavailable")
        self.last_prompt = messages[-1]["content"]
        return {"claims": self.claims}


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


# ── Metrics ──────────────────────────────────────────────────────────────────

def test_records_cost_and_splits_by_stage() -> None:
    pricing = {"m": {"input": 1.0, "output": 2.0}}    # $1/$2 per 1M tokens
    with metrics.turn(pricing=pricing) as m:
        with metrics.stage("classification"):
            metrics.record_call("m", 1_000_000, 0, 0.5)
        with metrics.stage("generation"):
            metrics.record_call("m", 0, 1_000_000, 1.5)

    s = m.summary()
    assert s["prompt_tokens"] == 1_000_000, s
    assert s["completion_tokens"] == 1_000_000, s
    assert s["llm_calls"] == 2, s
    assert abs(s["cost_usd"] - 3.0) < 1e-9, s["cost_usd"]
    assert abs(s["stages"]["classification"]["cost_usd"] - 1.0) < 1e-9, s["stages"]
    assert abs(s["stages"]["generation"]["cost_usd"] - 2.0) < 1e-9, s["stages"]
    assert s["stages"]["classification"]["seconds"] >= 0, s["stages"]


def test_unknown_model_costs_zero_not_a_guess() -> None:
    """An absent rate must not be invented — a wrong number is worse than none."""
    with metrics.turn(pricing={"known": {"input": 5.0, "output": 5.0}}) as m:
        metrics.record_call("never-heard-of-it", 1_000_000, 1_000_000, 0.1)
    assert m.summary()["cost_usd"] == 0.0, m.summary()


def test_nested_stage_wins() -> None:
    """Retrieval nested inside generation is attributed to retrieval."""
    with metrics.turn() as m:
        with metrics.stage("generation"):
            with metrics.stage("retrieval"):
                metrics.record_call("x", 10, 0, 0.01)
    stages = m.summary()["stages"]
    assert stages["retrieval"]["calls"] == 1, stages
    assert stages.get("generation", {}).get("calls", 0) == 0, stages


def test_estimated_tokens_are_flagged() -> None:
    with metrics.turn() as m:
        metrics.record_call("x", 100, 50, 0.1, estimated=True)
    assert m.summary()["estimated_tokens"] is True


def test_metrics_are_optional() -> None:
    """Recording outside a turn must be a no-op, so tests and the CLI stay clean."""
    metrics.record_call("x", 1, 1, 0.1)          # must not raise
    with metrics.stage("generation"):
        pass
    assert metrics.current() is None


def test_logger_carries_metrics() -> None:
    path = os.path.join(tempfile.mkdtemp(prefix="hr_metrics_"), "events.log")
    logger = InteractionLogger(path)
    logger.log("s", 1, "hi", "general", 0.9, "GeneralAssistantAgent", "hello",
               "native", metrics={"seconds": 1.2, "cost_usd": 0.0004})
    logger.log("s", 1, "hi", "general", 0.9, "GeneralAssistantAgent", "hello", "native")

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert rows[0]["metrics"]["cost_usd"] == 0.0004, rows[0]
    assert "metrics" not in rows[1], "absent metrics should not write an empty key"


# ── Streaming ────────────────────────────────────────────────────────────────

def test_sink_carries_tokens_and_events() -> None:
    sink = streams.TokenSink()
    sink.emit_event("stage", stage="generation")
    sink.emit_token("Hel")
    sink.emit_token("lo")
    sink.emit_event("result", reply="Hello")
    sink.close()

    events = list(sink.drain(timeout=2))
    assert [e["type"] for e in events] == ["stage", "token", "token", "result"], events
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello"


def test_streams_only_free_text_generation() -> None:
    """
    Classification is JSON mode — streaming it would emit fragments of a routing
    decision as if they were an answer.
    """
    assert streams.should_stream("generation", json_mode=False) is False, \
        "must not stream without a sink"
    with streams.sink(streams.TokenSink()):
        assert streams.should_stream("generation", json_mode=False) is True
        assert streams.should_stream("generation", json_mode=True) is False
        assert streams.should_stream("classification", json_mode=False) is False
        assert streams.should_stream(None, json_mode=False) is False


def test_stage_publishes_progress_to_an_active_stream() -> None:
    sink = streams.TokenSink()
    with streams.sink(sink):
        with metrics.stage("retrieval"):
            pass
    sink.close()
    events = list(sink.drain(timeout=2))
    assert any(e["type"] == "stage" and e["stage"] == "retrieval" for e in events), events


# ── Groundedness ─────────────────────────────────────────────────────────────

def test_unsupported_claim_fails_grounding() -> None:
    judge = GroundednessJudge(FakeJudge([
        {"claim": "2 days per week", "verdict": "supported", "evidence": "[1]"},
        {"claim": "50,000 PKR allowance", "verdict": "unsupported", "evidence": "absent"},
    ]))
    r = judge.evaluate(QUESTION, "answer", EXTRACTS)
    assert r["grounded"] is False, r
    assert r["unsupported"] == 1 and r["supported"] == 1, r
    assert r["support_rate"] == 0.5, r


def test_pleasantries_do_not_inflate_the_score() -> None:
    """not_a_claim entries are excluded from the denominator."""
    judge = GroundednessJudge(FakeJudge([
        {"claim": "Happy to help!", "verdict": "not_a_claim", "evidence": ""},
        {"claim": "Contact HR", "verdict": "not_a_claim", "evidence": ""},
        {"claim": "4 days per week", "verdict": "contradicted", "evidence": "[1] says 2"},
    ]))
    r = judge.evaluate(QUESTION, "answer", EXTRACTS)
    assert r["factual_claims"] == 1, r
    assert r["support_rate"] == 0.0, r
    assert r["grounded"] is False, r


def test_unreadable_verdict_is_not_a_pass() -> None:
    judge = GroundednessJudge(FakeJudge([
        {"claim": "something", "verdict": "probably fine?", "evidence": ""},
    ]))
    r = judge.evaluate(QUESTION, "answer", EXTRACTS)
    assert r["claims"][0]["verdict"] == "unsupported", r["claims"]
    assert r["grounded"] is False, r


def test_no_extracts_and_judge_failure_are_not_passes() -> None:
    judge = GroundednessJudge(FakeJudge([]))
    empty = judge.evaluate(QUESTION, "answer", [])
    assert empty["grounded"] is False and empty["error"], empty

    broken = GroundednessJudge(FakeJudge(fail=True)).evaluate(QUESTION, "a", EXTRACTS)
    assert broken["grounded"] is False and broken["error"], broken


def test_sources_block_is_not_audited_as_prose() -> None:
    """The citation list is metadata; auditing it would produce noise claims."""
    fake = FakeJudge([{"claim": "x", "verdict": "supported", "evidence": "[1]"}])
    judge = GroundednessJudge(fake)
    judge.evaluate(QUESTION,
                   "Up to 2 days.\n\n📚 **Sources**\n  [1] WFH — *remote.txt*",
                   EXTRACTS)
    assert "📚" not in fake.last_prompt, "sources block reached the judge"
    assert "Up to 2 days." in fake.last_prompt


# ── Live probe (opt-in) ──────────────────────────────────────────────────────

def run_live_probe() -> int:
    from dotenv import load_dotenv
    import yaml
    load_dotenv()
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    from core.llm_wrapper import LLMWrapper, available_keys

    provider = config["llm"].get("provider", "groq")
    if not available_keys(provider):
        print("  SKIP  live probe – no API key")
        return 0

    judge = GroundednessJudge(LLMWrapper(model=config["llm"]["model"],
                                         provider=provider, temperature=0.0))
    misses = 0
    for label, answer, expected in GROUND_TRUTH:
        r = judge.evaluate(QUESTION, answer, EXTRACTS)
        ok = r["grounded"] == expected
        misses += not ok
        print(f"  {'OK  ' if ok else 'MISS'}  {label:22} grounded={str(r['grounded']):5} "
              f"supported={r['supported']} unsupported={r['unsupported']} "
              f"contradicted={r['contradicted']}")
        for c in r["claims"]:
            if c["verdict"] in ("unsupported", "contradicted"):
                print(f"          caught → [{c['verdict']}] {c['claim'][:66]}")
    print(f"\n  judge matched {len(GROUND_TRUTH) - misses}/{len(GROUND_TRUTH)} expectations")
    return misses


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    live = "--live" in sys.argv
    print("\n=== Metrics, Streaming & Groundedness – Tests ===\n")

    check("records cost and splits by stage",     test_records_cost_and_splits_by_stage)
    check("unknown model costs zero, not a guess", test_unknown_model_costs_zero_not_a_guess)
    check("nested stage wins",                    test_nested_stage_wins)
    check("estimated tokens are flagged",         test_estimated_tokens_are_flagged)
    check("metrics are optional",                 test_metrics_are_optional)
    check("logger carries metrics",               test_logger_carries_metrics)
    check("sink carries tokens and events",       test_sink_carries_tokens_and_events)
    check("streams only free-text generation",    test_streams_only_free_text_generation)
    check("stage publishes progress",             test_stage_publishes_progress_to_an_active_stream)
    check("unsupported claim fails grounding",    test_unsupported_claim_fails_grounding)
    check("pleasantries do not inflate score",    test_pleasantries_do_not_inflate_the_score)
    check("unreadable verdict is not a pass",     test_unreadable_verdict_is_not_a_pass)
    check("no extracts / judge failure fail",     test_no_extracts_and_judge_failure_are_not_passes)
    check("sources block is not audited",         test_sources_block_is_not_audited_as_prose)

    if live:
        print("\n=== Live groundedness probe (ground truth) ===\n")
        misses = run_live_probe()
        check("live judge catches fabrications",
              lambda: (_ for _ in ()).throw(AssertionError(
                  f"{misses} disagreement(s) with ground truth")) if misses else None)
    else:
        print("\n  (run with --live to probe the real groundedness judge)")

    failures = [label for label, err in _results if err is not None]
    print("\n" + "=" * 50)
    if failures:
        print(f"FAILED: {len(failures)}/{len(_results)} – {failures}")
        sys.exit(1)
    print(f"All {len(_results)} checks PASSED")


if __name__ == "__main__":
    main()
