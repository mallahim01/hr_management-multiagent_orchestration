"""
ingest_knowledge.py – Load documents into the Milvus knowledge base.

Seeds data/company_policy.txt on first run, and accepts any additional
.txt/.md files. Re-ingesting the same source replaces the previous copy rather
than duplicating it, so a corrected policy does not sit alongside the old one.

Usage:
    python ingest_knowledge.py                          # seed the bundled policy
    python ingest_knowledge.py path/to/handbook.md      # add a document
    python ingest_knowledge.py --list                   # list stored documents
    python ingest_knowledge.py --delete doc-abc123      # remove one document
    python ingest_knowledge.py --search "wfh policy"    # preview retrieval
    python ingest_knowledge.py --reset                  # drop the collection
"""

import argparse
import os
import sys

from dotenv import load_dotenv
import yaml

load_dotenv()

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from knowledge import build_store, knowledge_config

DEFAULT_DOC = os.path.join("data", "company_policy.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the HR knowledge base")
    parser.add_argument("paths", nargs="*", help="document(s) to ingest")
    parser.add_argument("--title", default="", help="title for the ingested document")
    parser.add_argument("--by", default="cli", help="who is uploading (metadata)")
    parser.add_argument("--list", action="store_true", help="list stored documents")
    parser.add_argument("--delete", metavar="DOC_ID", help="delete one document")
    parser.add_argument("--search", metavar="QUERY", help="preview hybrid retrieval")
    parser.add_argument("--reset", action="store_true", help="drop the whole collection")
    args = parser.parse_args()

    cfg = knowledge_config(config)
    store = build_store(config)
    print(f"\nMilvus:     {cfg['milvus_uri']}")
    print(f"Collection: {cfg['collection']}")

    if not store.embedder.configured:
        print("\n❌ No GOOGLE_API_KEY in .env — embeddings are required for ingestion.")
        sys.exit(1)
    if not store.ensure_ready():
        print(f"\n❌ Milvus unavailable: {store.last_error}")
        print("   Start it with your Docker Desktop 'milvus-standalone' container.")
        sys.exit(1)

    if args.reset:
        confirm = input(f"Drop collection '{cfg['collection']}'? [y/N] ").strip().lower()
        if confirm == "y":
            store._connect().drop_collection(cfg["collection"])
            print("Collection dropped.")
        else:
            print("Cancelled.")
        return

    if args.delete:
        removed = store.delete_document(args.delete)
        print(f"\nDeleted {removed} chunk(s) for {args.delete}.")
        return

    if args.search:
        results = store.hybrid_search(args.search, top_k=cfg["top_k"],
                                      candidate_k=cfg["candidate_k"])
        print(f"\nHybrid results for {args.search!r}:\n")
        for r in results:
            print(f"  [{r['rank']}] rrf={r['score']:.4f}  {r['section'] or r['title']}")
            print(f"      source: {r['source']} (chunk {r['chunk_index'] + 1}/{r['total_chunks']})")
            print(f"      {r['text'][:150].replace(chr(10), ' ')}…\n")
        if not results:
            print("  (nothing retrieved — is anything ingested?)")
        return

    if args.list:
        print_documents(store)
        return

    paths = args.paths or [DEFAULT_DOC]
    for path in paths:
        if not os.path.isfile(path):
            print(f"\n❌ Not a file: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        print(f"\nIngesting {path} ({len(text)} chars)…")
        result = store.ingest_text(
            text=text, source=source,
            title=args.title or source, uploaded_by=args.by,
            max_chars=cfg["chunk_max_chars"], overlap=cfg["chunk_overlap"],
        )
        print(f"  ✅ {result['chunks']} chunks stored as {result['doc_id']}")
        if result["replaced_chunks"]:
            print(f"     (replaced {result['replaced_chunks']} chunk(s) from a previous copy)")

    print_documents(store)


def print_documents(store) -> None:
    documents = store.list_documents()
    print(f"\nStored documents ({len(documents)}):")
    if not documents:
        print("  (none)")
        return
    print(f"  {'DOC ID':<20} {'CHUNKS':>7}  {'UPLOADED':<21} {'SOURCE'}")
    for d in documents:
        print(f"  {d['doc_id']:<20} {d['chunks']:>7}  {d['uploaded_at']:<21} {d['source']}")
    stats = store.stats()
    print(f"\n  total chunks: {stats['chunks']}  |  "
          f"model: {stats.get('embedding_model')}  |  dim: {stats.get('dimension')}")


if __name__ == "__main__":
    main()
