"""
test_backends.py - Runs a focused demo through each orchestration backend.

Usage: python test_backends.py
"""

import os
import sys
import uuid

from dotenv import load_dotenv
import yaml

load_dotenv()

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from database.db import DatabaseManager
from database.schema import initialize_database
from core.llm_wrapper import LLMWrapper, available_keys
from core.session import SessionManager
from core.logger import InteractionLogger
from orchestration.factory import get_orchestrator

provider = config["llm"].get("provider", "groq")
if not available_keys(provider):
    print(f"❌ No API key for provider '{provider}' in .env")
    sys.exit(1)

# Init shared resources
db_path = config["database"]["path"]
os.makedirs(os.path.dirname(db_path), exist_ok=True)
db = DatabaseManager(db_path)
initialize_database(db, config["active_user_id"])

llm = LLMWrapper(
    model=config["llm"]["model"],
    max_retries=config["llm"]["max_retries"],
    temperature=config["llm"]["temperature"],
    provider=provider,
)
logger = InteractionLogger()
user_id = config["active_user_id"]
history_size = config["conversation"]["history_size"]

# Test prompts
TEST_PROMPTS = [
    ("Company Policy",   "What is our work from home policy?"),
    ("Leave Balance",    "How many leave days do I have remaining?"),
    ("Leave Request",    "I want to take sick leave tomorrow"),
    ("HR Request",       "I need help with my travel reimbursement"),
    ("General",          "Hello, how are you?"),
]

BACKENDS = ["native", "crewai", "langgraph", "adk"]

# ── Run tests ────────────────────────────────────────────────────

def test_backend(backend_name: str) -> bool:
    """
    Run all test prompts through a single backend.

    Returns False when the backend could not start — crewai and adk are optional
    installs (requirements-backends.txt), so their absence is reported and
    skipped rather than failing the run.
    """
    print(f"\n{'='*70}")
    print(f"  TESTING BACKEND: {backend_name.upper()}")
    print(f"{'='*70}\n")

    try:
        orchestrator = get_orchestrator(backend_name, llm, db, history_size)
    except ImportError as e:
        print(f"  ⊘  Skipped – {backend_name} is not installed ({e}).")
        print(f"     Install with: pip install -r requirements-backends.txt\n")
        return False
    session_mgr = SessionManager(db, history_size)
    session_id = str(uuid.uuid4())

    for label, prompt in TEST_PROMPTS:
        print(f"\n{'─'*50}")
        print(f"  📝 Test: {label}")
        print(f"  User: \"{prompt}\"")
        print(f"{'─'*50}")

        ctx = session_mgr.get_or_create(session_id, user_id)
        result = orchestrator.process(prompt, ctx)

        # Save messages
        db.save_message(session_id, user_id, "user", prompt)
        db.save_message(
            session_id, user_id, "assistant",
            result["reply"],
            agent=result["agent_class"],
            intent=result["intent"],
        )
        session_mgr.save(ctx)

        print(f"\n  Intent:     {result['intent']}")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Agent:      {result['agent_class']}")
        print(f"  Backend:    {result['backend']}")
        print(f"  Response:\n")

        # Print response indented
        for line in result["reply"].split("\n"):
            print(f"    {line}")

        print()

    print(f"\n  ✅  Backend '{backend_name}' test complete.\n")
    return True


def main():
    print("\n" + "█"*70)
    print("  Multi-Agent HR Demo – Backend Test Suite")
    print("  Testing each installed orchestration backend sequentially")
    print("█"*70)

    tested = [b for b in BACKENDS if test_backend(b)]
    skipped = [b for b in BACKENDS if b not in tested]

    print("\n" + "█"*70)
    print(f"  TESTED: {', '.join(tested) or 'none'}")
    if skipped:
        print(f"  SKIPPED (not installed): {', '.join(skipped)}")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
