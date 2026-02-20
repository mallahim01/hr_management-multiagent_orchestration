"""
core/intent_detector.py
────────────────────────
Uses the LLM to classify user intent and determine target agent routing.
Returns a structured dict with intent, confidence, and target_agent.
"""

from typing import Dict, List
from core.llm_wrapper import LLMWrapper

# Maps intent labels to the agent class name that will handle them
INTENT_TO_AGENT = {
    "leave_request":    "LeaveRequestAgent",
    "leave_balance":    "LeaveBalanceAgent",
    "company_question": "CompanyKnowledgeAgent",
    "hr_request":       "HRRequestAgent",
    "general":          "GeneralAssistantAgent",
}

SYSTEM_PROMPT = """You are an intent classification engine for a corporate HR assistant.

Classify the user message into exactly one of these intents:
- leave_request    : User wants to apply for, request, or submit leave / time off / sick day
- leave_balance    : User wants to know how many leaves they have remaining or used
- company_question : User is asking about company policies, WFH, reimbursements, benefits, rules
- hr_request       : User wants help with HR matters like reimbursements, payroll, grievances, referrals
- general          : Chitchat, greetings, or anything not fitting the above

Respond ONLY with a valid JSON object in this exact format:
{
  "intent": "<one of the five labels above>",
  "confidence": <float between 0.0 and 1.0>,
  "target_agent": "<agent class name>",
  "reasoning": "<one sentence explanation>"
}

Agent class names to use:
- leave_request    → LeaveRequestAgent
- leave_balance    → LeaveBalanceAgent
- company_question → CompanyKnowledgeAgent
- hr_request       → HRRequestAgent
- general          → GeneralAssistantAgent
"""


class IntentDetector:
    """Classifies user messages into HR intents using the LLM."""

    def __init__(self, llm: LLMWrapper) -> None:
        self.llm = llm

    def detect(
        self,
        user_message: str,
        recent_history: List[Dict[str, str]] | None = None,
    ) -> Dict:
        """
        Detect intent from the user message plus optional recent context.

        Args:
            user_message:   The latest user input string.
            recent_history: Optional list of {"role", "content"} dicts for context.

        Returns:
            Dict with keys: intent, confidence, target_agent, reasoning
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include recent history as context (summarised, not full messages)
        if recent_history:
            context_lines = []
            for msg in recent_history[-4:]:  # last 4 turns max for detection
                context_lines.append(f"{msg['role'].upper()}: {msg['content']}")
            context_block = "\n".join(context_lines)
            messages.append({
                "role": "user",
                "content": f"Recent conversation:\n{context_block}\n\nNew message to classify: {user_message}",
            })
        else:
            messages.append({"role": "user", "content": f"Message to classify: {user_message}"})

        try:
            result = self.llm.chat_json(messages)
            # Validate and normalise
            intent = result.get("intent", "general")
            if intent not in INTENT_TO_AGENT:
                intent = "general"
            return {
                "intent": intent,
                "confidence": float(result.get("confidence", 0.8)),
                "target_agent": INTENT_TO_AGENT[intent],
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            # Never crash the demo – fall back to general
            print(f"[IntentDetector] Error: {e} – defaulting to general")
            return {
                "intent": "general",
                "confidence": 0.5,
                "target_agent": "GeneralAssistantAgent",
                "reasoning": "Fallback due to detection error",
            }
