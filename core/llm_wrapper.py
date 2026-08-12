"""
core/llm_wrapper.py
────────────────────
Thin, reusable chat-completions wrapper.
• Supports system + user message lists
• Exponential-backoff retry (configurable)
• Optional JSON-mode enforcement (response_format=json_object)
• Provider-agnostic: any OpenAI-compatible endpoint (OpenAI, Groq, …)

Provider selection lives in config.yaml (`llm.provider`). Groq is used by
default because it exposes an OpenAI-compatible endpoint, so the same client
library serves both without a second SDK dependency.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from core import metrics
from core import streaming as streams


# ── Provider registry ────────────────────────────────────────────────────────
# base_url=None means "use the SDK default" (i.e. api.openai.com).
# key_env is ordered: the first env var that is set becomes the active key, and
# the rest act as spares that the wrapper rotates to when a key is rate-limited.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": None,
        "key_env": ["OPENAI_API_KEY"],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"],
    },
}

DEFAULT_PROVIDER = "groq"


def available_keys(provider: str = DEFAULT_PROVIDER) -> List[str]:
    """
    Return every non-empty API key configured for `provider`, in priority order.

    Entry points use this to fail fast with a clear message instead of letting
    the SDK raise deep inside the first request.
    """
    spec = PROVIDERS.get(provider.lower().strip())
    if spec is None:
        raise ValueError(
            f"Unknown llm provider '{provider}'. Known: {', '.join(sorted(PROVIDERS))}"
        )
    return [key for key in (os.getenv(name) for name in spec["key_env"]) if key]


class LLMWrapper:
    """Wraps chat.completions with retry logic, key rotation and JSON-mode support."""

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        temperature: float = 0.7,
        provider: str = DEFAULT_PROVIDER,
    ) -> None:
        self.provider = provider.lower().strip()
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature

        spec = PROVIDERS.get(self.provider)
        if spec is None:
            raise ValueError(
                f"Unknown llm provider '{provider}'. Known: {', '.join(sorted(PROVIDERS))}"
            )
        self.base_url = spec["base_url"]

        self._keys = available_keys(self.provider)
        if not self._keys:
            raise RuntimeError(
                f"No API key found for provider '{self.provider}'. "
                f"Set one of: {', '.join(spec['key_env'])} in your .env file."
            )
        self._key_index = 0
        self.client = self._make_client()

    # ── Client / key rotation ────────────────────────────────────────────────

    def _make_client(self) -> OpenAI:
        """Build a client bound to the currently-selected API key."""
        kwargs: Dict[str, Any] = {"api_key": self._keys[self._key_index]}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _rotate_key(self) -> bool:
        """
        Switch to the next configured API key.

        Free-tier keys have per-key quotas, so exhausting one is a routine event
        rather than an error. Returns False when no spare key is left.
        """
        if self._key_index + 1 >= len(self._keys):
            return False
        self._key_index += 1
        self.client = self._make_client()
        print(
            f"[LLMWrapper] Rotating to spare API key "
            f"#{self._key_index + 1}/{len(self._keys)}"
        )
        return True

    def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a list of messages to the LLM and return the text reply.

        Args:
            messages:    List of {"role": "system"|"user"|"assistant", "content": "..."}
            json_mode:   If True, forces the model to return valid JSON.
            temperature: Override instance temperature for this call.

        Returns:
            The model's text response as a string.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        streaming = streams.should_stream(metrics.current_stage(), json_mode)

        last_error: Exception = RuntimeError("Unknown LLM error")
        attempt = 0
        while attempt < self.max_retries:
            started = time.perf_counter()
            try:
                if streaming:
                    return self._chat_streaming(kwargs, started)
                response = self.client.chat.completions.create(**kwargs)
                self._record_usage(response, time.perf_counter() - started,
                                   messages, response.choices[0].message.content)
                return response.choices[0].message.content or ""
            except RateLimitError as e:
                last_error = e
                # Prefer a spare key over waiting. Rotation does not burn a retry,
                # because a different key is a genuinely different request budget.
                if self._rotate_key():
                    continue
                attempt += 1
                wait = 2 ** attempt
                print(f"[LLMWrapper] Rate limit hit – retrying in {wait}s (attempt {attempt})")
                time.sleep(wait)
            except APIConnectionError as e:
                last_error = e
                attempt += 1
                print(f"[LLMWrapper] Connection error – retrying (attempt {attempt})")
                time.sleep(1)
            except APIError as e:
                # Non-retriable API error
                raise RuntimeError(f"{self.provider} API error: {e}") from e

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    # ── Streaming ────────────────────────────────────────────────────────────

    def _chat_streaming(self, kwargs: Dict[str, Any], started: float) -> str:
        """
        Stream deltas to the active sink while returning the complete string.

        The caller — an agent — receives exactly what it would have received
        from a non-streamed call, so nothing downstream needs to know.
        """
        chunks: List[str] = []
        usage = None
        stream = self.client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True})
        for event in stream:
            usage = getattr(event, "usage", None) or usage
            if not event.choices:
                continue                      # the final usage-only frame
            delta = event.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                chunks.append(piece)
                streams.emit(piece)

        reply = "".join(chunks)
        self._record_usage(usage, time.perf_counter() - started,
                           kwargs["messages"], reply, usage_is_object=False)
        return reply

    # ── Accounting ───────────────────────────────────────────────────────────

    def _record_usage(self, response: Any, seconds: float,
                      messages: List[Dict[str, str]], reply: Optional[str],
                      usage_is_object: bool = True) -> None:
        """
        Attribute this call's tokens and latency to the active turn.

        Falls back to a character-based estimate when the provider reports no
        usage, and flags it, so a number that was guessed is never presented as
        one that was measured.
        """
        usage = getattr(response, "usage", None) if usage_is_object else response
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        estimated = prompt_tokens is None or completion_tokens is None
        if estimated:
            prompt_chars = sum(len(m.get("content") or "") for m in messages)
            prompt_tokens = prompt_tokens if prompt_tokens is not None else prompt_chars // 4
            completion_tokens = completion_tokens if completion_tokens is not None \
                else len(reply or "") // 4

        metrics.record_call(self.model, prompt_tokens, completion_tokens,
                            seconds, estimated=estimated)

    def chat_json(self, messages: List[Dict[str, str]]) -> Dict:
        """
        Convenience method: sends messages in JSON mode and parses the result.

        Returns:
            Parsed dictionary from the model's JSON response.
        """
        raw = self.chat(messages, json_mode=True, temperature=0.0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned invalid JSON: {raw}") from e
