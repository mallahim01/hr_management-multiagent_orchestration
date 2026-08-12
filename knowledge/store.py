"""
knowledge/store.py
───────────────────
Milvus-backed hybrid store for HR policy documents.

Each chunk is indexed twice in one collection:

  • dense_vector  – Gemini embedding, HNSW + COSINE, for meaning
                    ("can I work remotely?" → the WFH clause)
  • sparse_vector – Milvus's built-in BM25 function over the same text,
                    for exact terms ("LWP", "form 16", a policy number)

A query runs both and the results are fused with Reciprocal Rank Fusion, which
combines rankings rather than scores — the two are on entirely different scales
(BM25 is unbounded, cosine is [-1,1]), so a weighted sum of raw scores would be
meaningless without per-corpus tuning.

Every chunk carries provenance (doc_id, source, title, section, chunk position,
uploader, timestamp) so answers can cite where they came from and a document can
be replaced or removed as a unit.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from knowledge.chunker import chunk_document
from knowledge.embeddings import EmbeddingUnavailable, GeminiEmbedder

# Fields returned by a search — everything an answer needs to cite a source.
OUTPUT_FIELDS = [
    "text", "doc_id", "source", "title", "section",
    "chunk_index", "total_chunks", "uploaded_at", "uploaded_by",
]

MAX_TEXT_CHARS = 8000        # must stay under the VARCHAR max_length below


class KnowledgeStoreUnavailable(RuntimeError):
    """Raised when Milvus cannot be reached or the collection cannot be prepared."""


class MilvusKnowledgeStore:
    """Hybrid (dense + BM25) document store over a single Milvus collection."""

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection: str = "hr_knowledge_base",
        embedder: Optional[GeminiEmbedder] = None,
        dimension: int = 768,
        token: str = "",
        rrf_k: int = 60,
        connect_timeout: float = 5.0,
        fusion: str = "rrf",
        dense_weight: float = 0.85,
    ) -> None:
        self.uri = uri
        self.collection = collection
        self.dimension = dimension
        self.token = token
        self.rrf_k = rrf_k
        # How the two arms are combined: "rrf" (rank fusion, both arms equal),
        # "weighted" (dense_weight vs the remainder), or "dense" (skip BM25).
        # eval_retrieval.py measures all three — on the benchmark corpus dense
        # alone currently scores highest, so this is deliberately a setting
        # rather than a hardcoded assumption.
        self.fusion = fusion
        self.dense_weight = dense_weight
        # Without an explicit timeout the client blocks for a long time when
        # Milvus is down, which turns a degraded knowledge base into a hung UI.
        # Failing fast lets the agent fall back and the Knowledge tab say why.
        self.connect_timeout = connect_timeout
        self.embedder = embedder or GeminiEmbedder(dimension=dimension)
        self._client = None
        self._ready = False
        self._last_error: Optional[str] = None

    # ── Connection / schema ──────────────────────────────────────────────────

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _connect(self):
        if self._client is None:
            from pymilvus import MilvusClient
            kwargs: Dict[str, Any] = {"uri": self.uri, "timeout": self.connect_timeout}
            if self.token:
                kwargs["token"] = self.token
            self._client = MilvusClient(**kwargs)
        return self._client

    def ensure_ready(self) -> bool:
        """
        Connect and create the collection if needed. Returns False (rather than
        raising) so callers can degrade gracefully when Milvus is not running.
        """
        if self._ready:
            return True
        try:
            client = self._connect()
            if not client.has_collection(self.collection):
                self._create_collection(client)
                print(f"  [KnowledgeStore] Created collection '{self.collection}'")
            client.load_collection(self.collection)
            self._ready = True
            self._last_error = None
            return True
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            self._client = None
            return False

    def _create_collection(self, client) -> None:
        from pymilvus import DataType, Function, FunctionType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        # enable_analyzer is what lets the BM25 function tokenise this field.
        schema.add_field("text", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("title", DataType.VARCHAR, max_length=256)
        schema.add_field("section", DataType.VARCHAR, max_length=256)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("total_chunks", DataType.INT64)
        schema.add_field("uploaded_at", DataType.VARCHAR, max_length=32)
        schema.add_field("uploaded_by", DataType.VARCHAR, max_length=128)

        # Milvus computes the sparse vector itself from `text` on insert, so
        # nothing here has to ship a BM25 implementation or a vocabulary.
        schema.add_function(Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        ))

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="sparse_vector",
                               index_type="SPARSE_INVERTED_INDEX",
                               metric_type="BM25")
        index_params.add_index(field_name="dense_vector",
                               index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 200})

        client.create_collection(self.collection, schema=schema,
                                 index_params=index_params)

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_text(
        self,
        text: str,
        source: str,
        title: str = "",
        uploaded_by: str = "hr",
        doc_id: Optional[str] = None,
        replace_existing: bool = True,
        max_chars: int = 1200,
        overlap: int = 150,
    ) -> Dict[str, Any]:
        """
        Chunk, embed and store a document.

        Re-ingesting the same `source` replaces the previous copy by default, so
        uploading a corrected policy does not leave the old text retrievable
        alongside the new one — the classic way a RAG system starts citing
        superseded rules.

        Raises:
            KnowledgeStoreUnavailable: Milvus unreachable.
            EmbeddingUnavailable:      embeddings could not be produced.
            ValueError:                the document is empty.
        """
        if not text or not text.strip():
            raise ValueError("Document is empty")
        if not self.ensure_ready():
            raise KnowledgeStoreUnavailable(
                f"Milvus unavailable at {self.uri} ({self._last_error})"
            )

        chunks = chunk_document(text, max_chars=max_chars, overlap=overlap)
        if not chunks:
            raise ValueError("Document produced no chunks")

        vectors = self.embedder.embed_documents([c.embedding_text() for c in chunks])

        replaced = 0
        if replace_existing:
            replaced = self.delete_by_source(source)

        doc_id = doc_id or f"doc-{uuid.uuid4().hex[:12]}"
        uploaded_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        rows = [{
            "text":         chunk.embedding_text()[:MAX_TEXT_CHARS],
            "dense_vector": vector,
            "doc_id":       doc_id,
            "source":       source[:512],
            "title":        (title or source)[:256],
            "section":      chunk.section[:256],
            "chunk_index":  i,
            "total_chunks": len(chunks),
            "uploaded_at":  uploaded_at,
            "uploaded_by":  uploaded_by[:128],
        } for i, (chunk, vector) in enumerate(zip(chunks, vectors))]

        client = self._connect()
        client.insert(self.collection, rows)
        client.load_collection(self.collection)

        return {
            "doc_id": doc_id, "source": source, "title": title or source,
            "chunks": len(rows), "replaced_chunks": replaced,
            "uploaded_at": uploaded_at, "uploaded_by": uploaded_by,
        }

    # ── Retrieval ────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        filter_expr: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Run dense and BM25 retrieval and fuse the rankings with RRF.

        Each arm retrieves `candidate_k` before fusion: fusing two top-5 lists
        throws away the cases hybrid search exists for, where a chunk ranks
        mid-table on both arms but top overall.
        """
        if not query or not query.strip():
            return []
        if not self.ensure_ready():
            raise KnowledgeStoreUnavailable(
                f"Milvus unavailable at {self.uri} ({self._last_error})"
            )

        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        query_vector = self.embedder.embed_query(query)
        client = self._connect()

        if self.fusion == "dense":
            # Skip BM25 entirely — one arm, one query.
            raw = client.search(
                self.collection, data=[query_vector], anns_field="dense_vector",
                search_params={"ef": max(64, candidate_k * 2)}, limit=top_k,
                filter=filter_expr or "", output_fields=OUTPUT_FIELDS,
            )[0]
            hits = raw
        else:
            requests = [
                AnnSearchRequest(
                    data=[query_vector], anns_field="dense_vector",
                    param={"ef": max(64, candidate_k * 2)},
                    limit=candidate_k, expr=filter_expr or None,
                ),
                AnnSearchRequest(
                    data=[query], anns_field="sparse_vector",
                    param={"drop_ratio_search": 0.0},
                    limit=candidate_k, expr=filter_expr or None,
                ),
            ]
            ranker = (WeightedRanker(self.dense_weight,
                                     round(1.0 - self.dense_weight, 3))
                      if self.fusion == "weighted" else RRFRanker(self.rrf_k))
            hits = client.hybrid_search(
                self.collection, reqs=requests, ranker=ranker,
                limit=top_k, output_fields=OUTPUT_FIELDS,
            )[0]

        results = []
        for rank, hit in enumerate(hits, start=1):
            entity = hit.get("entity", {})
            results.append({
                "rank": rank,
                "score": float(hit.get("distance", 0.0)),
                "text": entity.get("text", ""),
                "doc_id": entity.get("doc_id", ""),
                "source": entity.get("source", ""),
                "title": entity.get("title", ""),
                "section": entity.get("section", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "total_chunks": entity.get("total_chunks", 0),
                "uploaded_at": entity.get("uploaded_at", ""),
                "uploaded_by": entity.get("uploaded_by", ""),
            })
        return results

    # ── Document management ──────────────────────────────────────────────────

    def list_documents(self) -> List[Dict[str, Any]]:
        """One row per stored document, aggregated from its chunks."""
        if not self.ensure_ready():
            return []
        client = self._connect()
        rows = client.query(
            self.collection, filter="chunk_index >= 0",
            output_fields=["doc_id", "source", "title", "uploaded_at",
                           "uploaded_by", "total_chunks"],
            limit=16384, consistency_level="Strong",
        )
        documents: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            doc_id = row.get("doc_id", "")
            entry = documents.setdefault(doc_id, {
                "doc_id": doc_id,
                "source": row.get("source", ""),
                "title": row.get("title", ""),
                "uploaded_at": row.get("uploaded_at", ""),
                "uploaded_by": row.get("uploaded_by", ""),
                "chunks": 0,
            })
            entry["chunks"] += 1
        return sorted(documents.values(), key=lambda d: d["uploaded_at"], reverse=True)

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of one document. Returns the number deleted."""
        if not doc_id or not self.ensure_ready():
            return 0
        return self._delete(f'doc_id == "{_escape(doc_id)}"')

    def delete_by_source(self, source: str) -> int:
        """Remove every chunk that came from `source` (used when replacing it)."""
        if not source or not self.ensure_ready():
            return 0
        return self._delete(f'source == "{_escape(source)}"')

    def _delete(self, expr: str) -> int:
        client = self._connect()
        # Count first: Milvus deletes are applied asynchronously, so counting
        # afterwards can report rows that are already gone.
        existing = client.query(self.collection, filter=expr, output_fields=["pk"],
                                limit=16384, consistency_level="Strong")
        if not existing:
            return 0
        client.delete(self.collection, filter=expr)
        return len(existing)

    def stats(self) -> Dict[str, Any]:
        """Availability and size, safe to call when Milvus is down."""
        if not self.ensure_ready():
            return {"available": False, "uri": self.uri,
                    "collection": self.collection, "error": self._last_error,
                    "documents": 0, "chunks": 0,
                    "embedding_configured": self.embedder.configured}
        try:
            client = self._connect()
            rows = client.query(self.collection, filter="chunk_index >= 0",
                                output_fields=["doc_id"], limit=16384,
                                consistency_level="Strong")
            return {"available": True, "uri": self.uri,
                    "collection": self.collection, "error": None,
                    "documents": len({r.get("doc_id") for r in rows}),
                    "chunks": len(rows),
                    "embedding_model": self.embedder.model,
                    "dimension": self.dimension,
                    "embedding_configured": self.embedder.configured}
        except Exception as e:
            return {"available": False, "uri": self.uri,
                    "collection": self.collection,
                    "error": f"{type(e).__name__}: {e}",
                    "documents": 0, "chunks": 0,
                    "embedding_configured": self.embedder.configured}


def _escape(value: str) -> str:
    """Escape a value for use inside a double-quoted Milvus filter literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')
