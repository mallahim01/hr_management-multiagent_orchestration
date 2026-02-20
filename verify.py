"""verify.py – Quick sanity check for imports and DB seeding. Run with venv python."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

errors = []

def check(label, fn):
    try:
        fn()
        print(f"  OK  {label}")
    except Exception as e:
        print(f"  ERR {label}: {e}")
        errors.append(label)

print("\n=== Multi-Agent HR Demo – Verification ===\n")

# Imports
check("database.db",           lambda: __import__("database.db"))
check("database.schema",       lambda: __import__("database.schema"))
check("core.llm_wrapper",      lambda: __import__("core.llm_wrapper"))
check("core.intent_detector",  lambda: __import__("core.intent_detector"))
check("core.session",          lambda: __import__("core.session"))
check("core.logger",           lambda: __import__("core.logger"))
check("agents registry",       lambda: __import__("agents"))
check("orchestration.base",    lambda: __import__("orchestration.base"))
check("orchestration.native",  lambda: __import__("orchestration.native"))
check("orchestration.crewai",  lambda: __import__("orchestration.crewai_adapter"))
check("orchestration.langgraph",lambda: __import__("orchestration.langgraph_adapter"))
check("orchestration.adk",     lambda: __import__("orchestration.adk_adapter"))
check("orchestration.factory", lambda: __import__("orchestration.factory"))
check("preview.cli_viewer",    lambda: __import__("preview.cli_viewer"))
check("preview.html_report",   lambda: __import__("preview.html_report"))
check("app",                   lambda: __import__("app"))

# DB operations
def db_test():
    from database.db import DatabaseManager
    from database.schema import initialize_database
    os.makedirs("data", exist_ok=True)
    db = DatabaseManager("data/verify_test.db")
    initialize_database(db, 1)
    user = db.get_user(1)
    assert user and user["name"] == "Alice Johnson", f"Got: {user}"
    bal = db.get_leave_balance(1)
    assert bal and bal["remaining_leaves"] == 15, f"Got: {bal}"
    db.insert_leave_request(1, "2025-03-10", "2025-03-12", "Medical")
    rows = db.get_leave_requests(1)
    assert len(rows) >= 1
    os.remove("data/verify_test.db")

check("DB seed + CRUD",  db_test)

# Agent registry
def agent_test():
    from agents import AGENT_REGISTRY
    expected = {"LeaveRequestAgent","LeaveBalanceAgent","CompanyKnowledgeAgent","HRRequestAgent","GeneralAssistantAgent"}
    assert set(AGENT_REGISTRY.keys()) == expected, f"Got: {AGENT_REGISTRY.keys()}"

check("Agent registry",  agent_test)

print("\n" + "="*43)
if errors:
    print(f"FAILED: {len(errors)} checks: {errors}")
    sys.exit(1)
else:
    print("All checks PASSED ✓")
