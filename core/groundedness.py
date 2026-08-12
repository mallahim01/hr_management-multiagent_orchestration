"""
core/groundedness.py
─────────────────────
Does the answer only say things its cited extracts support?

This catches the failure the other evaluations cannot see. Routing can be right,
retrieval can return the correct chunks, citations can point at real documents —
and the answer can still assert a number, a deadline or an eligibility rule that
appears nowhere in the retrieved text. Everything upstream looks healthy, and
the reply is wrong in the most expensive way: specific, sourced and confident.

The check runs claim by claim rather than asking "is this grounded?" as one
question. A whole-answer verdict collapses to a vibe; splitting it forces the
judge to point at the sentence it objects to, which is both a stricter test and
a usable trace — you get the offending claim, not just a score.

Verdicts per claim:
  supported     – stated in the extracts, or a fair paraphrase of them
  unsupported   – plausible but absent; the failure this module exists to find
  contradicted  – the extracts say something different
  not_a_claim   – a greeting, a hedge, or a pointer to HR; carries no facts
"""

import json
from typing import Any, Dict, List, Optional

JUDGE_PROMPT = """You are auditing an HR assistant for factual grounding.

You are given policy EXTRACTS that were retrieved for a question, and the ANSWER
the assistant produced from them.

Break the answer into individual factual claims and judge each one ONLY against
the extracts. Company knowledge you may have from elsewhere is irrelevant and
must not be used to support a claim.

Verdicts:
- "supported"    : the extracts state this, or it is a fair paraphrase
- "unsupported"  : plausible, but not present in the extracts
- "contradicted" : the extracts say something materially different
- "not_a_claim"  : greeting, hedge, offer to help, or a pointer to contact HR

Be strict about numbers, durations, eligibility conditions and deadlines: a
figure that does not appear in the extracts is "unsupported" even if it sounds
reasonable. Do not treat a citation marker like [1] as evidence — check the text.

Respond ONLY with JSON:
{"claims": [{"claim": "<the claim, quoted or closely paraphrased>",
             "verdict": "supported"|"unsupported"|"contradicted"|"not_a_claim",
             "evidence": "<extract number, or why it is missing>"}]}
"""


class GroundednessJudge:
    """Scores an answer against the extracts it was supposed to be built from."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def evaluate(self, question: str, answer: str,
                 extracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Judge one answer.

        Returns a report with per-claim verdicts and a `grounded` flag that is
        False when anything is unsupported or contradicted. Never raises: a
        judge failure is reported as `error` so a flaky call cannot be mistaken
        for a clean pass.
        """
        if not extracts:
            return self._report([], error="no extracts were retrieved")

        numbered = "\n\n".join(
            f"[{i}] ({e.get('section') or e.get('title') or e.get('source', '?')})\n"
            f"{e.get('text', '')}"
            for i, e in enumerate(extracts, start=1)
        )
        # The sources block the agent appends is metadata, not a claim; leaving
        # it in makes the judge audit the citation list as if it were prose.
        answer_body = answer.split("📚 **Sources**")[0].strip()

        try:
            raw = self.llm.chat_json([
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content":
                    f"QUESTION:\n{question}\n\nEXTRACTS:\n{numbered}\n\nANSWER:\n{answer_body}"},
            ])
        except Exception as e:
            return self._report([], error=f"{type(e).__name__}: {e}")

        claims = []
        for item in raw.get("claims", []) or []:
            verdict = str(item.get("verdict", "")).lower().strip()
            if verdict not in ("supported", "unsupported", "contradicted", "not_a_claim"):
                verdict = "unsupported"      # an unreadable verdict is not a pass
            claims.append({
                "claim": str(item.get("claim", ""))[:300],
                "verdict": verdict,
                "evidence": str(item.get("evidence", ""))[:200],
            })
        return self._report(claims)

    @staticmethod
    def _report(claims: List[Dict], error: Optional[str] = None) -> Dict[str, Any]:
        counts = {"supported": 0, "unsupported": 0, "contradicted": 0, "not_a_claim": 0}
        for c in claims:
            counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1

        factual = counts["supported"] + counts["unsupported"] + counts["contradicted"]
        # Greetings and hedges are excluded from the denominator: padding an
        # answer with pleasantries should not raise its groundedness score.
        rate = round(counts["supported"] / factual, 3) if factual else None
        return {
            "grounded": error is None and factual > 0
                        and counts["unsupported"] == 0 and counts["contradicted"] == 0,
            "claims_total": len(claims),
            "factual_claims": factual,
            "supported": counts["supported"],
            "unsupported": counts["unsupported"],
            "contradicted": counts["contradicted"],
            "not_a_claim": counts["not_a_claim"],
            "support_rate": rate,
            "error": error,
            "claims": claims,
        }
