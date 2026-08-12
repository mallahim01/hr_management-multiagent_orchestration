"""
knowledge/embeddings.py
────────────────────────
Gemini embeddings for the knowledge base.

Kept separate from core/llm_wrapper.py on purpose: chat generation runs on Groq
and embeddings run on Google, so a single wrapper would have to serve two
providers at once. The two share the same habits though — ordered key list with
rotation on rate limit, and a clear error when nothing is configured.

`gemini-embedding-001` returns 3072 dimensions by default and supports
Matryoshka truncation to smaller sizes. Truncated vectors come back
un-normalised, so this module re-normalises them: cosine similarity in Milvus
assumes unit length, and skipping this quietly degrades every search.
"""

import os
import time
from typing import List, Optional, Sequence

from core import metrics

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIMENSION = 768
KEY_ENV = ["GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3"]

# The API caps how many texts one request may carry.
BATCH_SIZE = 32


def available_keys() -> List[str]:
    """Every configured Google API key, in priority order."""
    return [k for k in (os.getenv(name) for name in KEY_ENV) if k]


class EmbeddingUnavailable(RuntimeError):
    """Raised when embeddings cannot be produced (no key, or the API refused)."""


class GeminiEmbedder:
    """Produces normalised dense vectors for documents and queries."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self.max_retries = max_retries
        self._keys = available_keys()
        self._key_index = 0
        self._client = None

    # ── Client / keys ────────────────────────────────────────────────────────

    @property
    def configured(self) -> bool:
        """True when at least one Google key is present."""
        return bool(self._keys)

    def _get_client(self):
        if self._client is None:
            if not self._keys:
                raise EmbeddingUnavailable(
                    "No Google API key found. Set GOOGLE_API_KEY in .env to enable "
                    "the knowledge base."
                )
            from google import genai
            self._client = genai.Client(api_key=self._keys[self._key_index])
        return self._client

    def _rotate_key(self) -> bool:
        """Move to the next spare key; False when none is left."""
        if self._key_index + 1 >= len(self._keys):
            return False
        self._key_index += 1
        self._client = None
        print(f"[GeminiEmbedder] Rotating to spare key "
              f"#{self._key_index + 1}/{len(self._keys)}")
        return True

    # ── Embedding ────────────────────────────────────────────────────────────

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed chunks for storage. Returns one vector per input, in order."""
        return self._embed(list(texts), task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a search query.

        The task type differs from documents deliberately — Gemini places
        queries and passages in compatible but distinct regions of the space,
        and using RETRIEVAL_DOCUMENT for both measurably weakens recall.
        """
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        if not texts:
            return []

        from google.genai import types

        vectors: List[List[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            vectors.extend(self._embed_batch(batch, task_type, types))
        return vectors

    def _embed_batch(self, batch: List[str], task_type: str, types) -> List[List[float]]:
        last_error: Optional[Exception] = None
        attempt = 0
        while attempt < self.max_retries:
            started = time.perf_counter()
            try:
                response = self._get_client().models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimension,
                        task_type=task_type,
                    ),
                )
                self._record(response, batch, time.perf_counter() - started)
                return [_normalise(list(e.values)) for e in response.embeddings]
            except EmbeddingUnavailable:
                raise
            except Exception as e:
                last_error = e
                message = str(e).lower()
                if ("429" in message or "quota" in message or "resource_exhausted" in message) \
                        and self._rotate_key():
                    continue          # a spare key is a fresh budget, not a retry
                attempt += 1
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"[GeminiEmbedder] Embed failed ({type(e).__name__}) – "
                          f"retrying in {wait}s")
                    time.sleep(wait)

        raise EmbeddingUnavailable(
            f"Embedding failed after {self.max_retries} attempts: {last_error}"
        )

    def _record(self, response, batch: List[str], seconds: float) -> None:
        """
        Attribute the embedding call to the active turn.

        The embeddings API reports token usage inconsistently across versions,
        so fall back to a character estimate and flag it rather than reporting
        a guess as a measurement. Embeddings have no output tokens.
        """
        usage = getattr(response, "usage_metadata", None)
        tokens = getattr(usage, "total_token_count", None) if usage else None
        estimated = tokens is None
        if estimated:
            tokens = sum(len(t) for t in batch) // 4
        metrics.record_call(self.model, int(tokens), 0, seconds, estimated=estimated)


def _normalise(vector: List[float]) -> List[float]:
    """Scale to unit length so COSINE distance behaves as expected."""
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm else vector
