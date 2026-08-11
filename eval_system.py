"""
eval_system.py – End-to-end evaluation against a hand-labelled golden set.

Complements eval_routing.py. The two answer different questions:

  eval_routing.py  – "was the routing on real traffic correct?"   LLM as judge,
                     over whatever is in the log. Open-ended, no ground truth.
  eval_system.py   – "does the system still do the things we decided it should?"
                     Fixed cases from evals/golden_set.json with expected answers
                     written by hand, scored deterministically where possible.

Three suites:

  1. ROUTING    – deterministic. The expected agent is a string in the golden
                  set; the run either matches it or it does not. No judge.
  2. RETRIEVAL  – deterministic. Checks the expected policy section appears in
                  the retrieved chunks (recall@k) and that expected facts appear
                  in the answer. Also reports the dense-only and sparse-only
                  recall so the hybrid claim is measured, not asserted.
  3. REFUSAL    – LLM as judge, because "did it decline to invent a policy?"
                  cannot be a string match. The judge is asked one narrow
                  question with a fixed rubric, not "is this a good answer".

Usage:
    python eval_system.py                # everything
    python eval_system.py --suite rag    # routing | rag | refusal | all
    python eval_system.py --json         # machine-readable
"""

import argparse
import json
import os
import sys
import uuid

from dotenv import load_dotenv
import yaml

load_dotenv()

with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

from core.llm_wrapper import LLMWrapper, available_keys
from core.logger import InteractionLogger
from core.session import SessionContext
from database.db import DatabaseManager
from database.schema import initialize_database
from knowledge import build_store, knowledge_config

GOLDEN_PATH = os.path.join("evals", "golden_set.json")

REFUSAL_JUDGE_PROMPT = """You are checking one narrow thing about an HR assistant's reply.

The employee asked about something that is NOT in the company policy document.
A correct reply declines and points the employee to HR. An incorrect reply
invents a policy, states a number as if it were policy, or accepts the premise
of the question as true.

Answer ONLY with JSON:
{"refused": true|false, "reason": "<one short sentence>"}

"refused": true  – the reply declines, says it has no information, or redirects
                   to HR without asserting a policy
"refused": false – the reply states or implies a policy that was never provided
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_golden() -> dict:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def section_matches(hits, needles) -> bool:
    """True when any retrieved chunk's section contains all expected fragments."""
    if not needles:
        return True
    for hit in hits:
        haystack = f"{hit.get('section','')} {hit.get('title','')}".upper()
        if all(n.upper() in haystack for n in needles):
            return True
    return False


def rank_of_section(hits, needles):
    """1-based rank of the first matching chunk, or None."""
    if not needles:
        return None
    for hit in hits:
        haystack = f"{hit.get('section','')} {hit.get('title','')}".upper()
        if all(n.upper() in haystack for n in needles):
            return hit.get("rank")
    return None


class Suite:
    """Collects per-case results and prints a scored table."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.cases = []

    def add(self, case_id, passed, detail, skipped=False):
        self.cases.append({"id": case_id, "passed": bool(passed),
                           "skipped": skipped, "detail": detail})

    @property
    def scored(self):
        return [c for c in self.cases if not c["skipped"]]

    @property
    def passed(self):
        return sum(1 for c in self.scored if c["passed"])

    @property
    def accuracy(self):
        return round(self.passed / len(self.scored), 3) if self.scored else None

    def report(self) -> dict:
        return {"suite": self.name, "total": len(self.scored),
                "passed": self.passed, "accuracy": self.accuracy,
                "cases": self.cases}

    def print(self) -> None:
        print(f"\n{'─' * 78}\n  {self.name}\n{'─' * 78}")
        for c in self.cases:
            mark = "SKIP" if c["skipped"] else ("PASS" if c["passed"] else "FAIL")
            print(f"  [{mark}] {c['id']:<10} {c['detail']}")
        if self.scored:
            print(f"\n  {self.passed}/{len(self.scored)} passed "
                  f"({self.accuracy:.0%})")


# ── Suite 1: routing (deterministic) ─────────────────────────────────────────

def run_routing(golden, llm) -> Suite:
    from core.intent_detector import IntentDetector

    suite = Suite("ROUTING – expected agent per query (deterministic, no judge)")
    detector = IntentDetector(llm)

    for case in golden["routing"]:
        result = detector.detect(case["query"], [])
        actual = result["target_agent"]
        expected = case["expected_agent"]
        ok = actual == expected
        detail = (f"{case['query'][:52]:<54} -> {actual}"
                  + ("" if ok else f"  (expected {expected})"))
        suite.add(case["id"], ok, detail)
    return suite


# ── Suite 2: retrieval (deterministic) ───────────────────────────────────────

def run_retrieval(golden, llm, store, cfg) -> Suite:
    suite = Suite("RETRIEVAL – recall@k on expected sections (deterministic)")

    db = DatabaseManager(os.path.join("data", "eval_tmp.db"))
    initialize_database(db, CONFIG["active_user_id"])
    from agents.company_knowledge_agent import CompanyKnowledgeAgent
    agent = CompanyKnowledgeAgent(llm, db, InteractionLogger(), store=store,
                                  config=CONFIG)

    arm_stats = {"hybrid": 0, "dense": 0, "sparse": 0, "pinned": 0}

    for case in golden["retrieval"]:
        hits = store.hybrid_search(case["query"], top_k=cfg["top_k"],
                                   candidate_k=cfg["candidate_k"])
        needles = case["expect_section_contains"]
        section_ok = section_matches(hits, needles)
        rank = rank_of_section(hits, needles)

        # Answer check: does the grounded reply contain the expected facts?
        reply = agent.handle(case["query"],
                             SessionContext(session_id=str(uuid.uuid4()),
                                            user_id=CONFIG["active_user_id"]))
        answer_ok = all(f.lower() in reply.lower()
                        for f in case["expect_answer_contains"])

        ok = section_ok and answer_ok
        bits = []
        if needles:
            bits.append(f"section {'@' + str(rank) if rank else 'MISS'}")
        if case["expect_answer_contains"]:
            bits.append(f"facts {'ok' if answer_ok else 'MISS'}")
        detail = f"{case['query'][:52]:<54} {' | '.join(bits)}"
        suite.add(case["id"], ok, detail)

        # Per-arm recall@1, to show what fusion is actually buying.
        # Measured at k=1 on purpose: at k=5 over a small corpus both arms
        # trivially succeed, and the comparison says nothing.
        if needles:
            arm_stats["pinned"] += 1
            arm_stats["hybrid"] += int(rank == 1)
            for arm in ("dense", "sparse"):
                single = single_arm_search(store, case["query"], arm,
                                           1, cfg["candidate_k"])
                arm_stats[arm] += int(section_matches(single, needles))

    if arm_stats["pinned"]:
        n = arm_stats["pinned"]
        print(f"\n  recall@1 over {n} pinned case(s) — what fusion buys:"
              f"  dense-only {arm_stats['dense']}/{n}"
              f" | sparse-only {arm_stats['sparse']}/{n}"
              f" | hybrid {arm_stats['hybrid']}/{n}")
    suite.arm_stats = arm_stats

    for suffix in ("", "-wal", "-shm"):
        path = os.path.join("data", "eval_tmp.db") + suffix
        if os.path.exists(path):
            os.remove(path)
    return suite


def single_arm_search(store, query, arm, top_k, candidate_k):
    """Run one retrieval arm alone, to compare against the fused result."""
    client = store._connect()
    if arm == "dense":
        data, field, param = [store.embedder.embed_query(query)], "dense_vector", \
                             {"ef": max(64, candidate_k * 2)}
    else:
        data, field, param = [query], "sparse_vector", {"drop_ratio_search": 0.0}

    from knowledge.store import OUTPUT_FIELDS
    raw = client.search(store.collection, data=data, anns_field=field,
                        search_params=param, limit=top_k,
                        output_fields=OUTPUT_FIELDS)[0]
    return [{"rank": i + 1, **h.get("entity", {})} for i, h in enumerate(raw)]


# ── Suite 3: refusal (LLM as judge) ──────────────────────────────────────────

def run_refusal(golden, llm, store) -> Suite:
    suite = Suite("REFUSAL – declines questions the policy does not answer (LLM judge)")

    db = DatabaseManager(os.path.join("data", "eval_tmp2.db"))
    initialize_database(db, CONFIG["active_user_id"])
    from agents.company_knowledge_agent import CompanyKnowledgeAgent
    agent = CompanyKnowledgeAgent(llm, db, InteractionLogger(), store=store,
                                  config=CONFIG)

    for case in golden["refusal"]:
        reply = agent.handle(case["query"],
                             SessionContext(session_id=str(uuid.uuid4()),
                                            user_id=CONFIG["active_user_id"]))
        try:
            verdict = llm.chat_json([
                {"role": "system", "content": REFUSAL_JUDGE_PROMPT},
                {"role": "user", "content": f"Question: {case['query']}\n\nReply:\n{reply[:1200]}"},
            ])
            refused = bool(verdict.get("refused"))
            reason = str(verdict.get("reason", ""))[:70]
        except Exception as e:
            suite.add(case["id"], False, f"judge failed: {type(e).__name__}", skipped=True)
            continue

        ok = refused == case["expect_refusal"]
        suite.add(case["id"], ok,
                  f"{case['query'][:46]:<48} refused={refused}  {reason}")

    for suffix in ("", "-wal", "-shm"):
        path = os.path.join("data", "eval_tmp2.db") + suffix
        if os.path.exists(path):
            os.remove(path)
    return suite


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate against the golden set")
    parser.add_argument("--suite", default="all",
                        choices=["all", "routing", "rag", "refusal"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    provider = CONFIG["llm"].get("provider", "groq")
    if not available_keys(provider):
        print(f"❌ No API key for provider '{provider}' in .env")
        sys.exit(1)

    llm = LLMWrapper(model=CONFIG["llm"]["model"],
                     max_retries=CONFIG["llm"]["max_retries"],
                     temperature=CONFIG["llm"]["temperature"],
                     provider=provider)
    golden = load_golden()
    cfg = knowledge_config(CONFIG)

    suites = []
    if args.suite in ("all", "routing"):
        suites.append(run_routing(golden, llm))

    needs_store = args.suite in ("all", "rag", "refusal")
    store = None
    if needs_store:
        store = build_store(CONFIG)
        if not store.embedder.configured or not store.ensure_ready():
            print(f"\n⚠️  Knowledge base unavailable ({store.last_error or 'no GOOGLE_API_KEY'})."
                  f"\n   Start Milvus and run: python ingest_knowledge.py")
            if args.suite in ("rag", "refusal"):
                sys.exit(1)
            needs_store = False

    if needs_store and args.suite in ("all", "rag"):
        suites.append(run_retrieval(golden, llm, store, cfg))
    if needs_store and args.suite in ("all", "refusal"):
        suites.append(run_refusal(golden, llm, store))

    reports = [s.report() for s in suites]
    if args.json:
        print(json.dumps({"suites": reports}, indent=2))
    else:
        print("\n" + "=" * 78)
        print("  SYSTEM EVALUATION – evals/golden_set.json")
        print("=" * 78)
        for s in suites:
            s.print()

        total = sum(r["total"] for r in reports)
        passed = sum(r["passed"] for r in reports)
        print("\n" + "=" * 78)
        print(f"  OVERALL: {passed}/{total} "
              f"({passed / total:.0%})" if total else "  OVERALL: nothing scored")
        print("=" * 78 + "\n")

    InteractionLogger().log_event(
        "system_eval",
        suites={r["suite"].split(" –")[0]: {"passed": r["passed"], "total": r["total"]}
                for r in reports},
    )

    failed = sum(r["total"] - r["passed"] for r in reports)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
