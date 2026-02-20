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

if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not set in .env")
    sys.exit(1)

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from database.db import DatabaseManager
from database.schema import initialize_database
from core.llm_wrapper import LLMWrapper
from core.session import SessionManager
from core.logger import InteractionLogger
from orchestration.factory import get_orchestrator

# Init shared resources
db_path = config["database"]["path"]
os.makedirs(os.path.dirname(db_path), exist_ok=True)
db = DatabaseManager(db_path)
initialize_database(db, config["active_user_id"])

llm = LLMWrapper(
    model=config["llm"]["model"],
    max_retries=config["llm"]["max_retries"],
    temperature=config["llm"]["temperature"],
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

def test_backend(backend_name: str) -> None:
    """Run all test prompts through a single backend."""
    print(f"\n{'='*70}")
    print(f"  TESTING BACKEND: {backend_name.upper()}")
    print(f"{'='*70}\n")

    orchestrator = get_orchestrator(backend_name, llm, db, history_size)
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


def main():
    print("\n" + "█"*70)
    print("  Multi-Agent HR Demo – Backend Test Suite")
    print("  Testing all 4 orchestration backends sequentially")
    print("█"*70)

    for backend in BACKENDS:
        test_backend(backend)

    print("\n" + "█"*70)
    print("  ALL 4 BACKENDS TESTED SUCCESSFULLY")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
