"""
eval_retrieval.py – Retrieval benchmark: recall@k for dense, sparse and hybrid.

Answers the question the earlier evaluation could not: does fusing dense and
BM25 rankings actually beat either arm alone? On the single bundled policy
(29 chunks) it did not — both arms already ranked the right chunk first, so
there was nothing for fusion to add. That result is only meaningful on a corpus
large and confusable enough to make retrieval hard, which is what
evals/corpus/ and the 25 labelled cases in evals/retrieval_benchmark.json exist
to provide.

The benchmark ingests into its own Milvus collection, so the app's knowledge
base is never touched.

Usage:
    python eval_retrieval.py                # ingest if needed, then score
    python eval_retrieval.py --reingest     # rebuild the benchmark collection
    python eval_retrieval.py --json         # machine-readable
    python eval_retrieval.py --drop         # remove the benchmark collection
"""

import argparse
import json
import os
import sys
from typing import Dict, List

from dotenv import load_dotenv
import yaml

load_dotenv()

with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

from knowledge import build_store, knowledge_config
from knowledge.store import OUTPUT_FIELDS

BENCHMARK_PATH = os.path.join("evals", "retrieval_benchmark.json")
K_VALUES = (1, 3, 5)


def load_benchmark() -> Dict:
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return json.load(f)


def matches(hits: List[Dict], needles: List[str]) -> bool:
    for hit in hits:
        haystack = f"{hit.get('section','')} {hit.get('title','')}".upper()
        if all(n.upper() in haystack for n in needles):
            return True
    return False


def rank_of(hits: List[Dict], needles: List[str]):
    for i, hit in enumerate(hits, start=1):
        haystack = f"{hit.get('section','')} {hit.get('title','')}".upper()
        if all(n.upper() in haystack for n in needles):
            return i
    return None


# ── Single-arm search ────────────────────────────────────────────────────────

def search_arm(store, query: str, arm: str, limit: int, candidate_k: int) -> List[Dict]:
    """Run one retrieval arm on its own, so it can be compared against fusion."""
    if arm == "hybrid_rrf":
        return store.hybrid_search(query, top_k=limit, candidate_k=candidate_k)
    if arm.startswith("hybrid_w"):
        weight = float(arm.split("hybrid_w")[1]) / 100.0
        return weighted_search(store, query, limit, candidate_k, weight)

    client = store._connect()
    if arm == "dense":
        data, field, params = [store.embedder.embed_query(query)], "dense_vector", \
                              {"ef": max(64, candidate_k * 2)}
    else:
        data, field, params = [query], "sparse_vector", {"drop_ratio_search": 0.0}

    raw = client.search(store.collection, data=data, anns_field=field,
                        search_params=params, limit=limit,
                        output_fields=OUTPUT_FIELDS)[0]
    return [{"rank": i + 1, **h.get("entity", {})} for i, h in enumerate(raw)]


def weighted_search(store, query: str, limit: int, candidate_k: int,
                    dense_weight: float) -> List[Dict]:
    """
    Fuse with WeightedRanker instead of RRF.

    RRF treats both arms as equally trustworthy. When one is materially stronger
    that is the wrong prior — the weak arm gets an equal vote and pulls correct
    top-1 results down. Weighting is the direct test of that explanation.
    """
    from pymilvus import AnnSearchRequest, WeightedRanker

    client = store._connect()
    reqs = [
        AnnSearchRequest(data=[store.embedder.embed_query(query)],
                         anns_field="dense_vector",
                         param={"ef": max(64, candidate_k * 2)}, limit=candidate_k),
        AnnSearchRequest(data=[query], anns_field="sparse_vector",
                         param={"drop_ratio_search": 0.0}, limit=candidate_k),
    ]
    hits = client.hybrid_search(
        store.collection, reqs=reqs,
        ranker=WeightedRanker(dense_weight, round(1.0 - dense_weight, 3)),
        limit=limit, output_fields=OUTPUT_FIELDS)[0]
    return [{"rank": i + 1, **h.get("entity", {})} for i, h in enumerate(hits)]


# ── Ingestion ────────────────────────────────────────────────────────────────

def ingest_corpus(store, benchmark: Dict, cfg: Dict) -> int:
    corpus_dir = benchmark["corpus"]
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith((".txt", ".md")))
    total = 0
    for name in files:
        with open(os.path.join(corpus_dir, name), encoding="utf-8") as f:
            text = f.read()
        result = store.ingest_text(
            text=text, source=name, title=name, uploaded_by="benchmark",
            max_chars=cfg["chunk_max_chars"], overlap=cfg["chunk_overlap"])
        total += result["chunks"]
        print(f"  {name:32} {result['chunks']:>3} chunks")
    return total


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval benchmark")
    parser.add_argument("--reingest", action="store_true", help="rebuild the collection")
    parser.add_argument("--drop", action="store_true", help="drop the collection and exit")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    benchmark = load_benchmark()
    cfg = knowledge_config(CONFIG)

    # A separate collection so the benchmark never disturbs the app's data.
    store = build_store({**CONFIG, "knowledge": {
        **(CONFIG.get("knowledge") or {}), "collection": benchmark["collection"]}})

    if not store.embedder.configured:
        print("❌ No GOOGLE_API_KEY — embeddings are required.")
        sys.exit(1)
    if not store.ensure_ready():
        print(f"❌ Milvus unavailable: {store.last_error}")
        sys.exit(1)

    if args.drop:
        store._connect().drop_collection(benchmark["collection"])
        print(f"Dropped {benchmark['collection']}.")
        return

    existing = store.list_documents()
    if args.reingest and existing:
        store._connect().drop_collection(benchmark["collection"])
        store._ready = False
        store.ensure_ready()
        existing = []
    if not existing:
        print(f"\nIngesting {benchmark['corpus']} into {benchmark['collection']}…")
        total = ingest_corpus(store, benchmark, cfg)
        print(f"  → {total} chunks\n")

    stats = store.stats()
    corpus_size = stats["chunks"]

    arms = ("dense", "sparse", "hybrid_rrf",
            "hybrid_w50", "hybrid_w70", "hybrid_w85")
    max_k = max(K_VALUES)
    results = {arm: [] for arm in arms}

    for case in benchmark["cases"]:
        for arm in arms:
            hits = search_arm(store, case["query"], arm, max_k, cfg["candidate_k"])
            results[arm].append({
                "id": case["id"], "probe": case["probe"],
                "rank": rank_of(hits, case["expect_section_contains"]),
            })

    report = {
        "corpus_documents": stats["documents"],
        "corpus_chunks": corpus_size,
        "cases": len(benchmark["cases"]),
        "arms": {},
        "by_probe": {},
    }
    for arm in arms:
        rows = results[arm]
        arm_report = {}
        for k in K_VALUES:
            hits = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k)
            arm_report[f"recall@{k}"] = round(hits / len(rows), 3)
        found = [r["rank"] for r in rows if r["rank"] is not None]
        # MRR over all cases; a miss contributes zero rather than being dropped.
        arm_report["mrr"] = round(
            sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows), 3)
        arm_report["misses"] = [r["id"] for r in rows if r["rank"] is None]
        report["arms"][arm] = arm_report

    probes = sorted({c["probe"] for c in benchmark["cases"]})
    for probe in probes:
        report["by_probe"][probe] = {}
        for arm in arms:
            rows = [r for r in results[arm] if r["probe"] == probe]
            hits = sum(1 for r in rows if r["rank"] == 1)
            report["by_probe"][probe][arm] = f"{hits}/{len(rows)}"

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, probes)

    from core.logger import InteractionLogger
    InteractionLogger().log_event(
        "retrieval_benchmark",
        corpus_chunks=corpus_size, cases=report["cases"],
        arms={a: v["recall@1"] for a, v in report["arms"].items()})


def print_report(report: Dict, probes: List[str]) -> None:
    print("=" * 74)
    print("  RETRIEVAL BENCHMARK")
    print("=" * 74)
    print(f"  corpus: {report['corpus_documents']} documents, "
          f"{report['corpus_chunks']} chunks   |   {report['cases']} labelled cases\n")

    arms = list(report["arms"])
    print(f"  {'arm':<12} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9} {'MRR':>7}   misses")
    print("  " + "-" * 74)
    for arm, v in report["arms"].items():
        misses = ",".join(v["misses"]) or "–"
        print(f"  {arm:<12} {v['recall@1']:>9.3f} {v['recall@3']:>9.3f} "
              f"{v['recall@5']:>9.3f} {v['mrr']:>7.3f}   {misses}")

    print(f"\n  recall@1 by what the case stresses:")
    header = "  " + f"{'probe':<12}" + "".join(f"{a:>13}" for a in arms)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for probe in probes:
        row = report["by_probe"][probe]
        print("  " + f"{probe:<12}" + "".join(f"{row[a]:>13}" for a in arms))

    single = max(report["arms"]["dense"]["recall@1"],
                 report["arms"]["sparse"]["recall@1"])
    best_single = "dense" if report["arms"]["dense"]["recall@1"] >= \
        report["arms"]["sparse"]["recall@1"] else "sparse"
    fusions = {a: v["recall@1"] for a, v in report["arms"].items()
               if a.startswith("hybrid")}
    best_fusion = max(fusions, key=fusions.get)

    print()
    print(f"  best single arm : {best_single} @ {single:.3f}")
    print(f"  best fusion     : {best_fusion} @ {fusions[best_fusion]:.3f}")
    if fusions[best_fusion] > single:
        print(f"  → Fusion wins. Weighting matters: RRF scores "
              f"{report['arms']['hybrid_rrf']['recall@1']:.3f}.")
    elif fusions[best_fusion] == single:
        print(f"  → Fusion ties {best_single}. It costs a second query for no "
              f"measured gain at k=1.")
    else:
        print(f"  → Fusion does NOT beat {best_single} on this corpus. Equal-weight "
              f"RRF is the worst fusion here, which is what you would expect when "
              f"one arm is much stronger: the weak arm gets an equal vote.")
    print("=" * 74)


if __name__ == "__main__":
    main()
