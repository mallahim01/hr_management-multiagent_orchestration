"""
core/metrics.py
────────────────
Per-turn cost and latency accounting.

A turn costs money and takes time in three distinguishable places — classifying
the intent, retrieving policy chunks, and generating the answer — and the useful
questions ("what does a turn cost?", "where does the latency actually go?",
"is skipping classification during slot-filling worth anything?") cannot be
answered without splitting them apart.

The collector is held in a ContextVar rather than threaded through every call
signature. `LLMWrapper` and the knowledge store look for an active turn and
record into it if one exists; if none does they behave exactly as before, which
is what keeps the tests and the CLI free of instrumentation plumbing.

    with metrics.turn() as m:
        with metrics.stage("classification"):
            ...                      # LLM calls here are attributed to it
        with metrics.stage("generation"):
            ...
    m.summary()

Costs are computed from a rate table that is a *snapshot*, not a live feed —
see PRICING below. Token counts come from the provider's usage field where it
reports one; embedding calls are estimated and flagged as such.
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import streaming as _streaming

# USD per 1,000,000 tokens, as published in August 2026. Rates change; these are
# a snapshot for order-of-magnitude reporting, not billing. Override in
# config.yaml under `pricing:` when they move, and treat an unknown model as
# free rather than guessing — a wrong number is worse than an absent one.
PRICING: Dict[str, Dict[str, float]] = {
    "llama-3.3-70b-versatile":  {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":     {"input": 0.05, "output": 0.08},
    "openai/gpt-oss-120b":      {"input": 0.15, "output": 0.75},
    "gpt-4o-mini":              {"input": 0.15, "output": 0.60},
    "gemini-embedding-001":     {"input": 0.15, "output": 0.00},
}

STAGES = ("classification", "retrieval", "generation", "other")


def price_for(model: str, pricing: Optional[Dict] = None) -> Dict[str, float]:
    """Rate for a model, or zeros when it is not in the table."""
    table = pricing or PRICING
    return table.get(model, {"input": 0.0, "output": 0.0})


@dataclass
class LLMCall:
    """One provider call, attributed to the stage that was active."""
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    cost_usd: float
    estimated: bool = False      # True when the provider reported no usage


@dataclass
class TurnMetrics:
    """Cost and latency for a single conversation turn."""

    calls: List[LLMCall] = field(default_factory=list)
    stage_seconds: Dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)
    ended: Optional[float] = None
    pricing: Optional[Dict] = None

    # ── Recording ────────────────────────────────────────────────────────────

    def record_call(self, model: str, prompt_tokens: int, completion_tokens: int,
                    seconds: float, stage: Optional[str] = None,
                    estimated: bool = False) -> None:
        rate = price_for(model, self.pricing)
        cost = (prompt_tokens / 1_000_000) * rate["input"] + \
               (completion_tokens / 1_000_000) * rate["output"]
        self.calls.append(LLMCall(
            stage=stage or current_stage() or "other",
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            seconds=round(seconds, 4),
            cost_usd=round(cost, 8),
            estimated=estimated,
        ))

    def add_stage_time(self, stage: str, seconds: float) -> None:
        self.stage_seconds[stage] = round(
            self.stage_seconds.get(stage, 0.0) + seconds, 4)

    def finish(self) -> None:
        if self.ended is None:
            self.ended = time.perf_counter()

    # ── Reporting ────────────────────────────────────────────────────────────

    @property
    def total_seconds(self) -> float:
        end = self.ended if self.ended is not None else time.perf_counter()
        return round(end - self.started, 4)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.calls), 8)

    def summary(self) -> Dict[str, Any]:
        """
        Flat dict suitable for the interaction log and the API.

        Per-stage cost and latency are reported side by side because they do not
        move together: classification is cheap but adds a full round trip, which
        is the entire reason the orchestrator skips it during slot-filling.
        """
        by_stage: Dict[str, Dict[str, Any]] = {}
        for call in self.calls:
            entry = by_stage.setdefault(call.stage, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "cost_usd": 0.0, "llm_seconds": 0.0,
            })
            entry["calls"] += 1
            entry["prompt_tokens"] += call.prompt_tokens
            entry["completion_tokens"] += call.completion_tokens
            entry["cost_usd"] = round(entry["cost_usd"] + call.cost_usd, 8)
            entry["llm_seconds"] = round(entry["llm_seconds"] + call.seconds, 4)

        # Wall-clock per stage covers work the provider is not doing — the Milvus
        # round trip, for instance — so it is tracked separately from llm_seconds.
        for stage, seconds in self.stage_seconds.items():
            by_stage.setdefault(stage, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "cost_usd": 0.0, "llm_seconds": 0.0,
            })["seconds"] = seconds

        total_prompt = sum(c.prompt_tokens for c in self.calls)
        total_completion = sum(c.completion_tokens for c in self.calls)
        return {
            "seconds": self.total_seconds,
            "cost_usd": self.total_cost_usd,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "llm_calls": len(self.calls),
            "estimated_tokens": any(c.estimated for c in self.calls),
            "stages": by_stage,
        }


# ── Ambient turn / stage ─────────────────────────────────────────────────────

_current_turn: ContextVar[Optional[TurnMetrics]] = ContextVar(
    "hr_current_turn", default=None)
_current_stage: ContextVar[Optional[str]] = ContextVar(
    "hr_current_stage", default=None)


def current() -> Optional[TurnMetrics]:
    """The turn being measured, or None when nothing is instrumented."""
    return _current_turn.get()


def current_stage() -> Optional[str]:
    return _current_stage.get()


@contextmanager
def turn(pricing: Optional[Dict] = None):
    """Measure one conversation turn."""
    m = TurnMetrics(pricing=pricing)
    token = _current_turn.set(m)
    try:
        yield m
    finally:
        m.finish()
        _current_turn.reset(token)


@contextmanager
def stage(name: str, progress: bool = True):
    """
    Attribute everything inside to `name`, and time it.

    Safe to use when no turn is active: it becomes a no-op, so agents and the
    store stay callable from tests and scripts without a metrics context.

    `progress=False` keeps the accounting but suppresses the user-facing
    progress event. The orchestrator opens "generation" around the whole agent
    call, and an agent that retrieves opens "retrieval" inside it — announcing
    both would tell the reader "writing", then "searching", then "writing"
    again. The outer wrapper stays silent; the stage that is really running
    speaks.
    """
    token = _current_stage.set(name)
    started = time.perf_counter()
    # Stage boundaries are the only progress signal a caller can get during the
    # ~70% of a warm turn that is not generation, so publish them to any active
    # stream. No-op when nothing is streaming.
    if progress:
        _streaming.emit_stage(name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        m = _current_turn.get()
        if m is not None:
            m.add_stage_time(name, elapsed)
        _current_stage.reset(token)


def record_call(model: str, prompt_tokens: int, completion_tokens: int,
                seconds: float, estimated: bool = False) -> None:
    """Record a provider call against the active turn, if there is one."""
    m = _current_turn.get()
    if m is not None:
        m.record_call(model, prompt_tokens, completion_tokens, seconds,
                      estimated=estimated)
