"""
eval_routing.py – LLM-as-judge evaluation of the orchestrator's routing.

Replays the most recent turns from logs/interactions.log and asks the model,
after the fact, whether each one went to the right agent. Prints a scored table
and writes a `routing_eval` summary back to the log.

Usage:
    python eval_routing.py                 # judge the last 10 turns
    python eval_routing.py --limit 20      # judge the last 20
    python eval_routing.py --json          # machine-readable output
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
import yaml

load_dotenv()

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from core.llm_wrapper import LLMWrapper, available_keys
from core.logger import DEFAULT_LOG_PATH, InteractionLogger
from core.routing_judge import RoutingJudge

VERDICT_MARK = {
    "correct":   "PASS",
    "incorrect": "FAIL",
    "ambiguous": "----",
    "error":     "????",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge recent routing decisions")
    parser.add_argument("--limit", type=int, default=10, help="turns to judge (default 10)")
    parser.add_argument("--batch", type=int, default=5, help="turns per judge call (default 5)")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="log file to read")
    parser.add_argument("--json", action="store_true", help="print the raw report as JSON")
    args = parser.parse_args()

    provider = config["llm"].get("provider", "groq")
    if not available_keys(provider):
        print(f"❌ No API key for provider '{provider}' in .env")
        sys.exit(1)

    llm = LLMWrapper(
        model=config["llm"]["model"],
        max_retries=config["llm"]["max_retries"],
        temperature=config["llm"]["temperature"],
        provider=provider,
    )
    judge = RoutingJudge(llm, log_path=args.log, batch_size=args.batch)

    print(f"\nJudging the last {args.limit} turns from {args.log} …")
    report = judge.evaluate(limit=args.limit)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    if report["judged"]:
        judge.log_report(report, InteractionLogger(args.log))
        print("  (summary written back to the log as a 'routing_eval' event)")

    # Exit non-zero on any confirmed misroute so this can gate a pipeline.
    sys.exit(1 if report["incorrect"] else 0)


def print_report(report: dict) -> None:
    print("\n" + "=" * 78)
    print("  ROUTING EVALUATION – LLM as judge")
    print("=" * 78)

    if not report["results"]:
        print(f"\n  {report['note']}\n")
        return

    for r in report["results"]:
        mark = VERDICT_MARK.get(r["verdict"], "????")
        conf = r.get("confidence")
        conf_txt = f"{conf:.0%}" if isinstance(conf, (int, float)) else "  – "
        print(f"\n  [{mark}] turn {r['n']}  ({r.get('backend', '?')}, conf {conf_txt})")
        print(f"         user:     {r['user_input'][:70]}")
        print(f"         routed:   {r['chosen_agent']}  (intent: {r.get('intent')})")
        if r["verdict"] == "incorrect":
            print(f"         expected: {r['expected_agent']}")
        print(f"         judge:    {r['reason']}")

    print("\n" + "-" * 78)
    acc = report["accuracy"]
    acc_txt = f"{acc:.1%}" if acc is not None else "n/a (nothing gradeable)"
    print(f"  judged {report['judged']} turns  "
          f"({report['skipped_continuations']} slot-fill continuations skipped)")
    print(f"  correct {report['correct']} | incorrect {report['incorrect']} | "
          f"ambiguous {report['ambiguous']} | errors {report['errors']}")
    print(f"  routing accuracy: {acc_txt}")
    print(f"  {report['note']}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
