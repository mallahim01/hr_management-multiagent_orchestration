"""
agents/company_knowledge_agent.py
─────────────────────────────────
Answers company policy questions grounded in data/company_policy.txt.
Uses a long system prompt embedding the full policy document so the LLM
responds only based on documented company rules.
"""

import os

from agents.base_agent import BaseAgent
from core.session import SessionContext

SYSTEM_PROMPT_TEMPLATE = """You are the ACME Corporation HR Knowledge Assistant.
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


class CompanyKnowledgeAgent(BaseAgent):
    """Answers HR policy questions grounded in the company policy document."""

    display_name = "Company Knowledge Agent"
    colour = "purple"

    def __init__(self, llm, db):
        super().__init__(llm, db)
        self._policy_text = self._load_policy()
        print(f"  [CompanyKnowledgeAgent] Policy loaded: {len(self._policy_text)} chars")
        self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            policy_text=self._policy_text
        )

    def handle(self, user_input: str, ctx: SessionContext) -> str:
        messages = self._build_messages(self._system_prompt, user_input, ctx)
        reply = self.llm.chat(messages)
        ctx.active_agent = None
        ctx.agent_state = {}
        return reply

    @staticmethod
    def _load_policy() -> str:
        # Try multiple paths to find the policy file reliably
        candidates = [
            os.path.join(os.getcwd(), "data", "company_policy.txt"),
            os.path.join(os.path.dirname(__file__), "..", "data", "company_policy.txt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "company_policy.txt"),
        ]
        for path in candidates:
            norm = os.path.normpath(path)
            if os.path.isfile(norm):
                with open(norm, encoding="utf-8") as f:
                    return f.read()
        print("  [CompanyKnowledgeAgent] ⚠️  Policy file not found in any candidate path!")
        return "(Company policy document not found. Please add data/company_policy.txt)"
