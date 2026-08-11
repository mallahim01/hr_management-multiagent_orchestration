"""
core/routing_judge.py
──────────────────────
LLM-as-judge evaluation of the orchestrator's routing decisions.

Reads recent turns back out of the interaction log and asks the model, after the
fact and with the agent's actual reply in view, whether each turn was sent to the
right agent. This is deliberately a *different* prompt from IntentDetector's — a
judge that reused the detector's prompt would mostly be measuring the detector
against itself.

Continuation turns are reported separately rather than scored: while an agent
holds the session for slot-filling the orchestrator skips classification on
purpose, so there is no routing decision to grade.
"""

import json
from typing import Any, Dict, List, Optional

from core.intent_detector import INTENT_TO_AGENT
from core.llm_wrapper import LLMWrapper
from core.logger import InteractionLogger

# Intents that mean "no routing decision was taken this turn".
CONTINUATION_INTENTS = {"continuation"}

AGENT_CATALOGUE = """\
- LeaveRequestAgent     (leave_request)    : applying for, requesting or submitting leave / time off / a sick day
- LeaveBalanceAgent     (leave_balance)    : how many leave days remain or have been used
- CompanyKnowledgeAgent (company_question) : company policies, WFH rules, benefits, reimbursement policy, code of conduct
- HRRequestAgent        (hr_request)       : raising an HR ticket — certificates, payroll, grievances, referrals, reimbursement claims
- GeneralAssistantAgent (general)          : greetings, chitchat, anything not covered above"""

JUDGE_SYSTEM_PROMPT = f"""You are an impartial evaluator auditing a corporate HR assistant.

The assistant routes each user message to exactly one specialist agent:

{AGENT_CATALOGUE}

You will be given numbered turns. For each, you see the user's message, the agent
the system chose, and the reply that agent produced.

Judge ONLY whether the chosen agent was the right one for that message. Do not
grade writing style, tone, or completeness of the reply — use the reply only as
evidence about whether that agent could actually serve the request.

Use these verdicts:
- "correct"   : the chosen agent is the best fit
- "incorrect" : a different agent was clearly the better fit
- "ambiguous" : the message genuinely fits more than one agent, or is too vague to place

Be strict about "incorrect" — reserve it for clear misroutes, not preferences.
Note the distinction between asking ABOUT a policy (CompanyKnowledgeAgent) and
asking the HR team TO DO something (HRRequestAgent).

Respond ONLY with a JSON object of this exact shape:
{{"verdicts": [{{"n": <turn number>, "verdict": "correct"|"incorrect"|"ambiguous",
                "expected_agent": "<agent class name, or the chosen one if correct>",
                "reason": "<one short sentence>"}}]}}

Return exactly one entry per turn given, with matching "n" values."""


class RoutingJudge:
    """Scores logged routing decisions using the LLM as an impartial judge."""

    def __init__(
        self,
        llm: LLMWrapper,
        log_path: str = "data/interactions.log",
        batch_size: int = 5,
    ) -> None:
        self.llm = llm
        self.log_path = log_path
        # Turns are judged in batches: one call per turn is more reliable but
        # burns tokens linearly, and batching keeps the whole audit to a
        # handful of calls.
        self.batch_size = batch_size

    # ── Log reading ──────────────────────────────────────────────────────────

    def load_turns(self, limit: int = 10) -> List[Dict]:
        """
        Return the most recent `limit` interaction turns, oldest first.

        Records predating the "event" field are all interactions, so a missing
        field is treated as one.
        """
        try:
            with open(self.log_path, encoding="utf-8") as f:
                records = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue        # tolerate a partially-written last line
        except FileNotFoundError:
            return []

        turns = [r for r in records if r.get("event", "interaction") == "interaction"]
        return turns[-limit:]

    # ── Judging ──────────────────────────────────────────────────────────────

    def evaluate(self, limit: int = 10) -> Dict[str, Any]:
        """
        Judge the most recent `limit` turns.

        Returns a report dict with per-turn results and an accuracy summary.
        Never raises: a failed batch is reported as an "error" verdict so a
        flaky judge call cannot take down whatever is displaying the report.
        """
        turns = self.load_turns(limit)
        if not turns:
            return self._empty_report("no interaction turns found in the log")

        scored, skipped = [], []
        for turn in turns:
            target = (turn.get("intent") or "").strip()
            if target in CONTINUATION_INTENTS:
                skipped.append(turn)
            else:
                scored.append(turn)

        if not scored:
            return self._empty_report(
                f"all {len(skipped)} recent turns were slot-fill continuations, "
                f"which take no routing decision"
            )

        results: List[Dict] = []
        for start in range(0, len(scored), self.batch_size):
            batch = scored[start:start + self.batch_size]
            results.extend(self._judge_batch(batch, offset=start))

        return self._summarise(results, skipped_count=len(skipped))

    def _judge_batch(self, batch: List[Dict], offset: int) -> List[Dict]:
        """Judge one batch of turns; degrade to 'error' verdicts on failure."""
        lines = []
        for i, turn in enumerate(batch):
            n = offset + i + 1
            lines.append(
                f"--- Turn {n} ---\n"
                f"User message: {turn.get('user_input', '')}\n"
                f"Chosen agent: {turn.get('target_agent', '?')}\n"
                f"Agent reply:  {(turn.get('agent_response') or '')[:300]}"
            )
        payload = "\n\n".join(lines)

        try:
            raw = self.llm.chat_json([
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Judge these {len(batch)} turns:\n\n{payload}"},
            ])
            by_n = {int(v["n"]): v for v in raw.get("verdicts", []) if "n" in v}
        except Exception as e:
            print(f"[RoutingJudge] Batch failed: {e}")
            by_n = {}

        out = []
        for i, turn in enumerate(batch):
            n = offset + i + 1
            verdict = by_n.get(n)
            out.append(self._normalise(n, turn, verdict))
        return out

    @staticmethod
    def _normalise(n: int, turn: Dict, verdict: Optional[Dict]) -> Dict:
        """Coerce one judge verdict into a stable shape, defaulting to 'error'."""
        chosen = turn.get("target_agent", "?")
        base = {
            "n": n,
            "user_input": turn.get("user_input", ""),
            "chosen_agent": chosen,
            "intent": turn.get("intent"),
            "confidence": turn.get("confidence"),
            "backend": turn.get("backend"),
            "timestamp": turn.get("timestamp"),
        }
        if not verdict:
            return {**base, "verdict": "error", "expected_agent": chosen,
                    "reason": "judge returned no verdict for this turn"}

        label = str(verdict.get("verdict", "")).lower().strip()
        if label not in ("correct", "incorrect", "ambiguous"):
            label = "error"
        expected = verdict.get("expected_agent") or chosen
        if expected not in INTENT_TO_AGENT.values():
            expected = chosen
        return {**base, "verdict": label, "expected_agent": expected,
                "reason": str(verdict.get("reason", ""))[:200]}

    # ── Reporting ────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_report(note: str) -> Dict[str, Any]:
        return {"judged": 0, "skipped_continuations": 0, "correct": 0,
                "incorrect": 0, "ambiguous": 0, "errors": 0,
                "accuracy": None, "note": note, "results": []}

    @staticmethod
    def _summarise(results: List[Dict], skipped_count: int) -> Dict[str, Any]:
        counts = {"correct": 0, "incorrect": 0, "ambiguous": 0, "error": 0}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

        # Accuracy is over turns that were actually gradeable: ambiguous turns
        # and failed judge calls are excluded rather than counted as passes.
        gradeable = counts["correct"] + counts["incorrect"]
        accuracy = round(counts["correct"] / gradeable, 3) if gradeable else None

        return {
            "judged": len(results),
            "skipped_continuations": skipped_count,
            "correct": counts["correct"],
            "incorrect": counts["incorrect"],
            "ambiguous": counts["ambiguous"],
            "errors": counts["error"],
            "accuracy": accuracy,
            "note": "accuracy = correct / (correct + incorrect); "
                    "ambiguous and errored turns are excluded",
            "results": results,
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def log_report(self, report: Dict[str, Any], logger: InteractionLogger) -> None:
        """Record the summary as a structured event so eval runs are auditable."""
        logger.log_event(
            "routing_eval",
            judged=report["judged"],
            correct=report["correct"],
            incorrect=report["incorrect"],
            ambiguous=report["ambiguous"],
            errors=report["errors"],
            accuracy=report["accuracy"],
            misroutes=[
                {"user_input": r["user_input"][:120],
                 "chosen": r["chosen_agent"],
                 "expected": r["expected_agent"]}
                for r in report["results"] if r["verdict"] == "incorrect"
            ],
        )
