"""
test_rag.py – Tests for the hybrid RAG knowledge base.

Offline mode uses a fake store and a fake LLM, so chunking, grounding,
citation and degradation behaviour are all deterministic and need no Milvus,
no Google key and no network.

    python test_rag.py           # offline, deterministic
    python test_rag.py --live    # also exercise the real Milvus + Gemini stack

The --live pass is what proves hybrid retrieval is actually hybrid: it checks
that an exact-term query ("LWP") finds the right clause, which dense vectors
alone are poor at, and that a paraphrased query finds a clause sharing none of
its words, which BM25 alone cannot do.
"""

import json
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# knowledge_config() lets KNOWLEDGE__* environment variables override whatever a
# caller passes, which is what makes the container able to point at a different
# Milvus. That also means a stray KNOWLEDGE__ENABLED in the environment would
# silently decide the outcome of these tests, so the offline suite clears them.
_CLEARED_ENV = {k: os.environ.pop(k) for k in list(os.environ)
                if k.startswith("KNOWLEDGE__")}
if _CLEARED_ENV:
    print(f"  (ignoring ambient {', '.join(sorted(_CLEARED_ENV))} for these tests)")

from agents.company_knowledge_agent import NO_RESULTS_REPLY, CompanyKnowledgeAgent
from core.logger import InteractionLogger
from core.session import SessionContext
from database.db import DatabaseManager
from database.schema import initialize_database
from knowledge.chunker import chunk_document
from knowledge.store import KnowledgeStoreUnavailable

TEST_USER_ID = 1

POLICY = """ACME Corporation – Employee HR Policy Document
================================================

SECTION 1 – LEAVE POLICY
────────────────────────

1.1 Annual Leave
   • All full-time employees are entitled to 20 days of paid annual leave per year.
   • Leave accrues at 1.67 days per month.

1.7 Leave Without Pay (LWP)
   • Employees may apply for LWP after exhausting entitled leave.
   • LWP requires HR manager approval.

---

SECTION 2 – WORK FROM HOME (WFH) POLICY
───────────────────────────────────────

2.2 WFH Entitlement
   • You can work from home up to 2 days per week with line manager approval.
"""


# ── Test doubles ─────────────────────────────────────────────────────────────

class FakeLLM:
    """Echoes back that it was called, and records the prompt it received."""

    def __init__(self) -> None:
        self.last_system_prompt = None
        self.calls = 0

    def chat(self, messages, json_mode=False, temperature=None) -> str:
        self.calls += 1
        self.last_system_prompt = messages[0]["content"]
        return "Canned grounded answer [1]."

    def chat_json(self, messages) -> dict:
        return {}


class FakeStore:
    """Stands in for MilvusKnowledgeStore without touching Milvus."""

    def __init__(self, results=None, raises=None) -> None:
        self._results = results if results is not None else []
        self._raises = raises
        self.queries = []

    def hybrid_search(self, query, top_k=5, candidate_k=20, filter_expr=""):
        self.queries.append(query)
        if self._raises:
            raise self._raises
        return self._results[:top_k]


def hit(rank, text, section, source="company_policy.txt", idx=0, total=3):
    return {"rank": rank, "score": 0.03, "text": text, "doc_id": "doc-1",
            "source": source, "title": "ACME Policy", "section": section,
            "chunk_index": idx, "total_chunks": total,
            "uploaded_at": "2026-08-11T00:00:00Z", "uploaded_by": "hr"}


def make_agent(store, config=None):
    db_path = os.path.join(tempfile.mkdtemp(prefix="hr_rag_"), "t.db")
    db = DatabaseManager(db_path)
    initialize_database(db, TEST_USER_ID)
    log_path = os.path.join(tempfile.mkdtemp(prefix="hr_rag_log_"), "events.log")
    logger = InteractionLogger(log_path)
    llm = FakeLLM()
    agent = CompanyKnowledgeAgent(llm, db, logger, store=store,
                                  config=config or {"knowledge": {"enabled": True}})
    return agent, llm, log_path


def ctx() -> SessionContext:
    return SessionContext(session_id=str(uuid.uuid4()), user_id=TEST_USER_ID)


def events(log_path):
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Harness ──────────────────────────────────────────────────────────────────

_results: list = []


def check(label, fn):
    try:
        fn()
        print(f"  PASS  {label}")
        _results.append((label, None))
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        _results.append((label, e))


# ── Chunking ─────────────────────────────────────────────────────────────────

def test_chunks_keep_their_heading() -> None:
    """A chunk that has lost its heading is much harder to retrieve correctly."""
    chunks = chunk_document(POLICY, max_chars=1200, overlap=100)
    assert chunks, "no chunks produced"

    lwp = [c for c in chunks if "LWP" in c.text or "LWP" in c.section]
    assert lwp, f"LWP clause not found in {[c.section for c in chunks]}"
    assert "SECTION 1" in lwp[0].section, lwp[0].section
    assert "1.7" in lwp[0].section, lwp[0].section
    # The heading is prefixed onto what gets embedded and BM25-indexed.
    assert "Leave Without Pay" in lwp[0].embedding_text()

    wfh = [c for c in chunks if "2 days per week" in c.text]
    assert wfh, "WFH clause missing"
    assert "WORK FROM HOME" in wfh[0].section, wfh[0].section


def test_chunker_handles_unstructured_and_empty() -> None:
    plain = "Just one paragraph of text with no headings at all."
    chunks = chunk_document(plain)
    assert len(chunks) == 1, chunks
    assert chunks[0].text == plain, chunks[0].text

    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []


def test_long_block_is_split_with_overlap() -> None:
    body = "\n\n".join(f"Paragraph {i} " + "x" * 200 for i in range(12))
    chunks = chunk_document(body, max_chars=500, overlap=80)
    assert len(chunks) > 1, "oversized block was not split"
    assert all(len(c.text) <= 700 for c in chunks), [len(c.text) for c in chunks]
    # Nothing is lost: every paragraph marker survives somewhere.
    joined = " ".join(c.text for c in chunks)
    for i in range(12):
        assert f"Paragraph {i} " in joined, f"lost paragraph {i}"


# ── Grounding and citation ───────────────────────────────────────────────────

def test_answer_is_grounded_and_cites_sources() -> None:
    store = FakeStore([
        hit(1, "You can work from home up to 2 days per week.",
            "SECTION 2 – WFH › 2.2 WFH Entitlement"),
        hit(2, "Eligibility requires completing probation.",
            "SECTION 2 – WFH › 2.1 Eligibility", idx=1),
    ])
    agent, llm, _ = make_agent(store)

    reply = agent.handle("can I work remotely?", ctx())

    assert store.queries == ["can I work remotely?"], store.queries
    # Retrieved text reached the prompt, numbered for citation.
    assert "[1]" in llm.last_system_prompt
    assert "2 days per week" in llm.last_system_prompt
    assert "RETRIEVED POLICY EXTRACTS" in llm.last_system_prompt
    # The whole policy file is NOT in the prompt — that is the point of RAG.
    assert "SECTION 1 – LEAVE POLICY" not in llm.last_system_prompt

    assert "📚 **Sources**" in reply, reply
    assert "WFH Entitlement" in reply, reply
    assert "company_policy.txt" in reply, reply
    assert len(agent.last_sources) == 2, agent.last_sources
    assert agent.last_mode == "rag", agent.last_mode


def test_empty_retrieval_refuses_to_answer() -> None:
    """With nothing retrieved, answering at all would mean answering ungrounded."""
    store = FakeStore([])
    agent, llm, log_path = make_agent(store)

    reply = agent.handle("what is the pet bereavement policy?", ctx())

    assert reply == NO_RESULTS_REPLY, reply
    assert llm.calls == 0, "LLM was called with no context to ground the answer"
    logged = events(log_path)
    assert len(logged) == 1 and logged[0]["event"] == "knowledge_no_results", logged


def test_agent_releases_the_session() -> None:
    agent, _, _ = make_agent(FakeStore([hit(1, "text", "S")]))
    c = ctx()
    c.active_agent = "CompanyKnowledgeAgent"
    c.agent_state = {"stale": True}
    agent.handle("anything", c)
    assert c.active_agent is None and c.agent_state == {}, (c.active_agent, c.agent_state)


# ── Degradation ──────────────────────────────────────────────────────────────

def test_falls_back_when_milvus_is_down() -> None:
    """
    A reviewer without Docker must still get a working system — but the answer
    has to say it is degraded rather than pretend retrieval happened.
    """
    store = FakeStore(raises=KnowledgeStoreUnavailable("connection refused"))
    agent, llm, log_path = make_agent(store)

    reply = agent.handle("what is the WFH policy?", ctx())

    assert llm.calls == 1, "fallback did not answer"
    assert "knowledge base is unavailable" in reply, reply
    assert "📚 **Sources**" not in reply, "cited sources it never retrieved"
    assert agent.last_mode == "fallback", agent.last_mode

    logged = events(log_path)
    assert any(e["event"] == "knowledge_retrieval_failed" for e in logged), logged
    assert "connection refused" in logged[0]["detail"], logged[0]


def test_rag_can_be_disabled_by_config() -> None:
    store = FakeStore([hit(1, "should not be used", "S")])
    agent, llm, _ = make_agent(store, config={"knowledge": {"enabled": False}})

    reply = agent.handle("what is the WFH policy?", ctx())

    assert store.queries == [], "searched despite being disabled"
    assert llm.calls == 1, "no answer produced"
    assert "knowledge base is unavailable" not in reply, "disabled is not degraded"
    assert agent.last_mode == "fallback", agent.last_mode


# ── Registry wiring ──────────────────────────────────────────────────────────

def test_every_backend_resolves_the_rag_agent() -> None:
    """
    The four orchestrators all construct agents through AGENT_REGISTRY, so the
    RAG path reaches every backend without any of them knowing about Milvus.
    """
    from agents import AGENT_REGISTRY
    assert AGENT_REGISTRY["CompanyKnowledgeAgent"] is CompanyKnowledgeAgent

    from orchestration.langgraph_adapter import LangGraphOrchestrator
    from orchestration.native import NativeOrchestrator

    db_path = os.path.join(tempfile.mkdtemp(prefix="hr_rag_reg_"), "t.db")
    db = DatabaseManager(db_path)
    initialize_database(db, TEST_USER_ID)

    for orchestrator in (NativeOrchestrator(FakeLLM(), db),
                         LangGraphOrchestrator(FakeLLM(), db)):
        agent = orchestrator._get_agent("CompanyKnowledgeAgent")
        assert isinstance(agent, CompanyKnowledgeAgent), type(agent)
        assert hasattr(agent, "store"), "agent has no knowledge store"


# ── Live probe (opt-in) ──────────────────────────────────────────────────────

def run_live_probe() -> int:
    """Exercise the real Milvus + Gemini stack. Returns the failure count."""
    from dotenv import load_dotenv
    import yaml
    load_dotenv()
    os.environ.update(_CLEARED_ENV)      # the live probe should honour them
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))

    from knowledge import build_store
    store = build_store(config)

    if not store.embedder.configured:
        print("  SKIP  live probe – no GOOGLE_API_KEY")
        return 0
    if not store.ensure_ready():
        print(f"  SKIP  live probe – Milvus unavailable ({store.last_error})")
        return 0

    failures = 0
    source = f"__probe_{uuid.uuid4().hex[:8]}.txt"
    try:
        result = store.ingest_text(POLICY, source=source, title="Probe Policy",
                                   uploaded_by="test")
        print(f"  ingested {result['chunks']} chunks as {result['doc_id']}")

        # Exact-term recall: BM25's job. Dense vectors handle acronyms poorly.
        hits = store.hybrid_search("LWP", top_k=3)
        top = hits[0] if hits else {}
        ok = "LWP" in (top.get("section", "") + top.get("text", ""))
        failures += not ok
        print(f"  {'OK  ' if ok else 'MISS'}  exact term 'LWP' -> {top.get('section', '(nothing)')}")

        # Semantic recall: dense's job. Shares no content words with the clause.
        hits = store.hybrid_search("am I allowed to do my job from my house?", top_k=3)
        top = hits[0] if hits else {}
        ok = "WORK FROM HOME" in top.get("section", "").upper()
        failures += not ok
        print(f"  {'OK  ' if ok else 'MISS'}  paraphrase -> {top.get('section', '(nothing)')}")

        # Provenance is present on every hit.
        ok = all(h.get("source") and h.get("doc_id") for h in hits)
        failures += not ok
        print(f"  {'OK  ' if ok else 'MISS'}  every hit carries source metadata")

        # Re-ingesting the same source replaces rather than duplicates.
        again = store.ingest_text(POLICY, source=source, title="Probe Policy",
                                  uploaded_by="test")
        ok = again["replaced_chunks"] == result["chunks"]
        failures += not ok
        print(f"  {'OK  ' if ok else 'MISS'}  re-ingest replaced "
              f"{again['replaced_chunks']}/{result['chunks']} chunks")
    finally:
        removed = store.delete_by_source(source)
        print(f"  cleaned up {removed} probe chunk(s)")

    return failures


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    live = "--live" in sys.argv

    print("\n=== Hybrid RAG – Tests (offline) ===\n")
    check("chunks keep their heading",            test_chunks_keep_their_heading)
    check("chunker handles unstructured/empty",   test_chunker_handles_unstructured_and_empty)
    check("long block split with overlap",        test_long_block_is_split_with_overlap)
    check("answer is grounded and cites sources", test_answer_is_grounded_and_cites_sources)
    check("empty retrieval refuses to answer",    test_empty_retrieval_refuses_to_answer)
    check("agent releases the session",           test_agent_releases_the_session)
    check("falls back when Milvus is down",       test_falls_back_when_milvus_is_down)
    check("RAG can be disabled by config",        test_rag_can_be_disabled_by_config)
    check("every backend resolves the RAG agent", test_every_backend_resolves_the_rag_agent)

    if live:
        print("\n=== Live probe (Milvus + Gemini) ===\n")
        misses = run_live_probe()
        check("live hybrid retrieval works",
              lambda: (_ for _ in ()).throw(AssertionError(
                  f"{misses} live check(s) failed")) if misses else None)
    else:
        print("\n  (run with --live to exercise the real Milvus + Gemini stack)")

    failures = [label for label, err in _results if err is not None]
    print("\n" + "=" * 50)
    if failures:
        print(f"FAILED: {len(failures)}/{len(_results)} – {failures}")
        sys.exit(1)
    print(f"All {len(_results)} checks PASSED")


if __name__ == "__main__":
    main()
