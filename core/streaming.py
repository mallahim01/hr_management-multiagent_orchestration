"""
core/streaming.py
──────────────────
Token streaming without rewriting the agents.

The obvious way to stream is to turn every agent's `handle()` into a generator
and thread that through all four orchestration backends. That is a large change
to the part of the system that is deliberately stable, for a presentation
concern.

Instead a sink is published in a ContextVar. When one is active, `LLMWrapper`
asks the provider for a streamed response, forwards each delta to the sink, and
still returns the complete string to its caller. Agents, orchestrators and the
session model are untouched — they never learn that streaming happened.

Only the generation stage streams. Classification is a JSON-mode call whose
partial output is meaningless to a reader, and streaming it would emit fragments
of a routing decision as if they were an answer.
"""

import queue
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

# Stages whose output is worth showing a token at a time.
STREAMABLE_STAGES = {"generation"}


class TokenSink:
    """
    Thread-safe conduit between the worker running a turn and the HTTP response.

    The turn executes on a worker thread; the Flask response generator drains
    this from the request thread.
    """

    _DONE = object()

    def __init__(self) -> None:
        self._q: "queue.Queue[Any]" = queue.Queue()

    # ── Producer side ────────────────────────────────────────────────────────

    def emit_token(self, text: str) -> None:
        if text:
            self._q.put({"type": "token", "text": text})

    def emit_event(self, event_type: str, **payload: Any) -> None:
        """Send a non-token event: a stage change, the final result, an error."""
        self._q.put({"type": event_type, **payload})

    def close(self) -> None:
        self._q.put(self._DONE)

    # ── Consumer side ────────────────────────────────────────────────────────

    def drain(self, timeout: float = 300.0) -> Iterator[Dict]:
        """Yield events until close() is called, or the timeout elapses."""
        while True:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                yield {"type": "error", "error": "timed out waiting for the model"}
                return
            if item is self._DONE:
                return
            yield item


_current_sink: ContextVar[Optional[TokenSink]] = ContextVar(
    "hr_token_sink", default=None)


def current_sink() -> Optional[TokenSink]:
    return _current_sink.get()


def should_stream(stage: Optional[str], json_mode: bool) -> bool:
    """Stream only a free-text generation with a sink attached."""
    if json_mode or _current_sink.get() is None:
        return False
    return stage in STREAMABLE_STAGES


@contextmanager
def sink(target: TokenSink):
    """Publish `target` so LLMWrapper streams into it for this context."""
    token = _current_sink.set(target)
    try:
        yield target
    finally:
        _current_sink.reset(token)


def emit(text: str) -> None:
    s = _current_sink.get()
    if s is not None:
        s.emit_token(text)


def emit_stage(name: str) -> None:
    """
    Announce that a stage started.

    Measurement showed generation is not the bottleneck on a warm turn —
    classification and retrieval together are roughly 70% of the latency — so
    telling the reader what is happening during that 70% is worth more than
    streaming the tail.
    """
    s = _current_sink.get()
    if s is not None:
        s.emit_event("stage", stage=name)
