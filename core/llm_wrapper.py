"""
core/llm_wrapper.py
────────────────────
Thin, reusable OpenAI wrapper.
• Supports system + user message lists
• Exponential-backoff retry (configurable)
• Optional JSON-mode enforcement (response_format=json_object)
"""

import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, RateLimitError, APIConnectionError


class LLMWrapper:
    """Wraps openai.chat.completions with retry logic and JSON-mode support."""

    def __init__(self, model: str, max_retries: int = 3, temperature: float = 0.7) -> None:
        self.client = OpenAI()           # reads OPENAI_API_KEY from env
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature

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

        last_error: Exception = RuntimeError("Unknown LLM error")
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except RateLimitError as e:
                last_error = e
                wait = 2 ** attempt
                print(f"[LLMWrapper] Rate limit hit – retrying in {wait}s (attempt {attempt})")
                time.sleep(wait)
            except APIConnectionError as e:
                last_error = e
                print(f"[LLMWrapper] Connection error – retrying (attempt {attempt})")
                time.sleep(1)
            except APIError as e:
                # Non-retriable API error
                raise RuntimeError(f"OpenAI API error: {e}") from e

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

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
