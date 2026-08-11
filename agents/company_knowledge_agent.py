"""
agents/company_knowledge_agent.py
─────────────────────────────────
Answers company policy questions using hybrid RAG over Milvus.

Retrieval path: the question is embedded with Gemini and searched against the
knowledge collection twice — dense vectors for meaning, BM25 for exact terms —
with the rankings fused. The top chunks become the grounding context, and the
answer cites the sections it used.

If Milvus or the embedding key is unavailable the agent falls back to the older
behaviour of putting data/company_policy.txt straight into the prompt. That is
worse (no citations, no uploaded documents, whole corpus every call) but it
keeps the project runnable for anyone who clones it without Docker, and the
degraded mode is announced rather than silent.

Because every backend resolves this agent through AGENT_REGISTRY, native,
LangGraph, CrewAI and ADK all pick up the RAG path with no change of their own.
"""

import os
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from core.session import SessionContext

RAG_SYSTEM_PROMPT = """You are the ACME Corporation HR Knowledge Assistant.

Answer the employee's question using ONLY the numbered policy extracts below.
These were retrieved from the company's policy library for this question.

Rules:
- Base every statement on the extracts. Do not add policies from general knowledge.
- Cite the extract you used inline as [1], [2], … immediately after the claim.
- If the extracts do not answer the question, say exactly: "I don't have specific
  information about that in our current policy documents. Please contact
  hr@acmecorp.com for clarification." Do not guess.
- If extracts disagree, say so and cite both.
- Be friendly, clear and concise. Use bullet points where they help.

━━━━━━━━━━━━ RETRIEVED POLICY EXTRACTS ━━━━━━━━━━━━

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

FALLBACK_SYSTEM_PROMPT = """You are the ACME Corporation HR Knowledge Assistant.
Your role is to answer employee questions about company policies accurately and helpfully.

You MUST base your answers ONLY on the company policy document provided below.
If the answer is not in the document, say: "I don't have specific information about that in our
current policy documents. Please contact hr@acmecorp.com for clarification."

Keep answers friendly, clear, and concise. Use bullet points where appropriate.
Do NOT make up policies or details not found in the document.

━━━━━━━━━━━━ COMPANY POLICY DOCUMENT ━━━━━━━━━━━━

{policy_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

NO_RESULTS_REPLY = (
    "I don't have specific information about that in our current policy documents. "
    "Please contact hr@acmecorp.com for clarification."
)


class CompanyKnowledgeAgent(BaseAgent):
    """Answers HR policy questions grounded in retrieved policy extracts."""

    display_name = "Company Knowledge Agent"
    colour = "purple"

    def __init__(self, llm, db, logger=None, store=None, config=None) -> None:
        super().__init__(llm, db, logger)
        self.config = config or {}
        self._store = store
        self._policy_text: Optional[str] = None      # loaded only if needed
        # Sources backing the most recent answer, for the API and the UI.
        self.last_sources: List[Dict[str, Any]] = []
        self.last_mode: str = "unknown"

    # ── Store access ─────────────────────────────────────────────────────────

    @property
    def store(self):
        """The knowledge store, built on first use so startup stays fast."""
        if self._store is None:
            from knowledge import build_store
            self._store = build_store(self.config)
        return self._store

    def _rag_enabled(self) -> bool:
        from knowledge import knowledge_config
        return bool(knowledge_config(self.config)["enabled"])

    # ── Main entry point ─────────────────────────────────────────────────────

    def handle(self, user_input: str, ctx: SessionContext) -> str:
        ctx.active_agent = None
        ctx.agent_state = {}
        self.last_sources = []

        results: List[Dict] = []
        if self._rag_enabled():
            try:
                cfg = self._retrieval_config()
                results = self.store.hybrid_search(
                    user_input, top_k=cfg["top_k"], candidate_k=cfg["candidate_k"]
                )
                self.last_mode = "rag"
            except Exception as e:
                # Retrieval is the whole point of this agent, so a failure is
                # worth recording even though the user still gets an answer.
                print(f"  [CompanyKnowledgeAgent] ⚠️  Retrieval failed: {e}")
                self.logger.log_event(
                    "knowledge_retrieval_failed",
                    session_id=ctx.session_id,
                    user_id=ctx.user_id,
                    reason_code="retrieval_error",
                    detail=f"{type(e).__name__}: {e}",
                )
                return self._answer_from_full_document(user_input, ctx, degraded=True)
        else:
            return self._answer_from_full_document(user_input, ctx, degraded=False)

        if not results:
            # Nothing retrieved: answering anyway would mean answering ungrounded.
            self.logger.log_event(
                "knowledge_no_results",
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                reason_code="empty_retrieval",
                detail=f"query: {user_input[:160]}",
            )
            return NO_RESULTS_REPLY

        self.last_sources = results
        print(f"  [CompanyKnowledgeAgent] Retrieved {len(results)} chunks "
              f"from {len({r['source'] for r in results})} document(s)")

        messages = self._build_messages(
            RAG_SYSTEM_PROMPT.format(context=self._format_context(results)),
            user_input, ctx,
        )
        reply = self.llm.chat(messages)
        return f"{reply}\n\n{self._format_sources(results)}"

    # ── Context / citation formatting ────────────────────────────────────────

    @staticmethod
    def _format_context(results: List[Dict]) -> str:
        """Number the extracts so the model has something concrete to cite."""
        blocks = []
        for r in results:
            heading = r.get("section") or r.get("title") or r.get("source")
            blocks.append(f"[{r['rank']}] ({heading})\n{r['text']}")
        return "\n\n".join(blocks)

    @staticmethod
    def _format_sources(results: List[Dict]) -> str:
        """A compact source list appended under every grounded answer."""
        lines = ["📚 **Sources**"]
        for r in results:
            label = r.get("section") or r.get("title") or "—"
            lines.append(
                f"  [{r['rank']}] {label} — *{r.get('source', 'unknown')}* "
                f"(chunk {r.get('chunk_index', 0) + 1}/{r.get('total_chunks', 1)})"
            )
        return "\n".join(lines)

    def _retrieval_config(self) -> Dict[str, Any]:
        from knowledge import knowledge_config
        return knowledge_config(self.config)

    # ── Degraded path ────────────────────────────────────────────────────────

    def _answer_from_full_document(
        self, user_input: str, ctx: SessionContext, degraded: bool
    ) -> str:
        """
        Answer from the whole policy file, as the agent did before Milvus.

        Kept so the project runs without Docker. It cannot cite sections and
        cannot see uploaded documents, so the reply says so when it is standing
        in for a failed retrieval.
        """
        self.last_mode = "fallback"
        policy = self._load_policy()
        messages = self._build_messages(
            FALLBACK_SYSTEM_PROMPT.format(policy_text=policy), user_input, ctx
        )
        reply = self.llm.chat(messages)
        if degraded:
            reply += ("\n\n_⚠️ Answered from the bundled policy file — the knowledge "
                      "base is unavailable, so uploaded documents were not searched._")
        return reply

    def _load_policy(self) -> str:
        if self._policy_text is not None:
            return self._policy_text
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(os.getcwd(), "data", "company_policy.txt"),
            os.path.normpath(os.path.join(here, "..", "data", "company_policy.txt")),
        ]
        for path in candidates:
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    self._policy_text = f.read()
                return self._policy_text
        print("  [CompanyKnowledgeAgent] ⚠️  Policy file not found in any candidate path!")
        self._policy_text = (
            "(Company policy document not found. Please add data/company_policy.txt)"
        )
        return self._policy_text
