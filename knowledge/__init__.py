"""
knowledge package
──────────────────
Hybrid retrieval over HR documents: chunking, Gemini embeddings, and a
Milvus collection combining dense vectors with BM25.

`build_store(config)` is the single construction point used by the agents,
the Flask app and the ingestion CLI, so they all share one configuration.
"""

from typing import Any, Dict, Optional

from knowledge.chunker import Chunk, chunk_document
from knowledge.embeddings import (EmbeddingUnavailable, GeminiEmbedder,
                                  available_keys as embedding_keys)
from knowledge.store import (KnowledgeStoreUnavailable, MilvusKnowledgeStore,
                             OUTPUT_FIELDS)

__all__ = [
    "Chunk", "chunk_document",
    "GeminiEmbedder", "EmbeddingUnavailable", "embedding_keys",
    "MilvusKnowledgeStore", "KnowledgeStoreUnavailable", "OUTPUT_FIELDS",
    "build_store", "knowledge_config",
]

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "milvus_uri": "http://localhost:19530",
    "milvus_token": "",
    "collection": "hr_knowledge_base",
    "embedding_model": "gemini-embedding-001",
    "dimension": 768,
    "chunk_max_chars": 1200,
    "chunk_overlap": 150,
    "top_k": 5,
    "candidate_k": 20,
    "rrf_k": 60,
}


def _load_config_file() -> Dict[str, Any]:
    """
    Read config.yaml directly.

    Agents are constructed by the orchestrators as cls(llm, db) and have no
    handle on the app config, so rather than threading it through all four
    backends the knowledge layer reads the file itself when nothing is passed.
    """
    import os
    import yaml

    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(os.getcwd(), "config.yaml"),
                 os.path.normpath(os.path.join(here, "..", "config.yaml"))):
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[knowledge] Could not read {path}: {e}")
    return {}


def knowledge_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge the `knowledge:` block of config.yaml over the defaults."""
    if not config:
        config = _load_config_file()
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in (config.get("knowledge") or {}).items()
                   if v is not None})
    return merged


def build_store(config: Optional[Dict[str, Any]] = None) -> MilvusKnowledgeStore:
    """Construct the store from config.yaml. Does not connect — that is lazy."""
    cfg = knowledge_config(config)
    embedder = GeminiEmbedder(model=cfg["embedding_model"], dimension=cfg["dimension"])
    return MilvusKnowledgeStore(
        uri=cfg["milvus_uri"],
        collection=cfg["collection"],
        embedder=embedder,
        dimension=cfg["dimension"],
        token=cfg["milvus_token"],
        rrf_k=cfg["rrf_k"],
    )
