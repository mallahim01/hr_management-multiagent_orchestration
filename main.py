"""
main.py
────────
CLI entry point for the multi-agent HR demo system.

Usage:
    python main.py                  # interactive REPL
    python main.py --demo           # auto-run 5 demo prompts
    python main.py --preview-cli    # print data tables to terminal
    python main.py --preview-html   # generate HTML report and open browser
    python main.py --web            # start Flask web server (requires app.py)
"""

import argparse
import os
import sys

from dotenv import load_dotenv
import yaml

# ── Bootstrap ────────────────────────────────────────────────────────────────

load_dotenv()

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from database.db import DatabaseManager
from database.schema import initialize_database
from core import metrics
from core.llm_wrapper import LLMWrapper, available_keys

_provider = config["llm"].get("provider", "groq")
if not available_keys(_provider):
    print(f"❌ No API key for provider '{_provider}'. "
          f"Copy .env.example to .env and add your key.")
    sys.exit(1)

from core.session import SessionManager
from core.logger import InteractionLogger
from orchestration.factory import get_orchestrator
from preview.cli_viewer import CLIViewer
from preview.html_report import HTMLReportGenerator

db_path = config["database"]["path"]
os.makedirs(os.path.dirname(db_path), exist_ok=True)

db = DatabaseManager(db_path)
initialize_database(db, config["active_user_id"])

llm = LLMWrapper(
    model=config["llm"]["model"],
    max_retries=config["llm"]["max_retries"],
    temperature=config["llm"]["temperature"],
    provider=_provider,
)

orchestrator = get_orchestrator(
    backend=config["orchestrator_backend"],
    llm=llm,
    db=db,
    history_size=config["conversation"]["history_size"],
)

session_manager = SessionManager(db, history_size=config["conversation"]["history_size"])
logger = InteractionLogger()
user_id = config["active_user_id"]

# ── Demo prompts ──────────────────────────────────────────────────────────────

DEMO_PROMPTS = [
    "Hi! What can you help me with?",
    "What is our company's work from home policy?",
    "I need to take sick leave tomorrow and the day after",
    "How many leave days do I have left?",
    "I need help with my travel reimbursement from last week",
]


def run_turn(session_id: str, user_input: str) -> dict:
    """Process one conversation turn and return the result dict."""
    ctx = session_manager.get_or_create(session_id, user_id)

    with metrics.turn(pricing=config.get("pricing")) as turn_metrics:
        result = orchestrator.process(user_input, ctx)
    result["metrics"] = turn_metrics.summary()

    # Persist messages and session
    db.save_message(session_id, user_id, "user", user_input)
    db.save_message(
        session_id, user_id, "assistant",
        result["reply"],
        agent=result["agent_class"],
        intent=result["intent"],
    )
    ctx.history = db.get_recent_messages(session_id, limit=config["conversation"]["history_size"] * 2)
    session_manager.save(ctx)

    logger.log(
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        intent=result["intent"],
        confidence=result["confidence"],
        target_agent=result["agent_class"],
        agent_response=result["reply"],
        backend=result["backend"],
        metrics=result["metrics"],
    )

    return result


def interactive_repl(session_id: str) -> None:
    """Run an interactive command-line chat loop."""
    user = db.get_user(user_id)
    print(f"\n{'━'*60}")
    print(f"  🤖 ACME HR Multi-Agent Demo")
    print(f"  Backend: {config['orchestrator_backend'].upper()}")
    print(f"  Active User: {user['name'] if user else user_id}")
    print(f"  Session: {session_id[:8]}...")
    print(f"{'━'*60}")
    print("  Type 'exit' to quit, 'preview' for data tables.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break
        if user_input.lower() == "preview":
            CLIViewer(db, user_id).show_all()
            continue

        result = run_turn(session_id, user_input)
        print(f"\n[{result['agent']}] (intent: {result['intent']}, "
              f"confidence: {result['confidence']:.0%}, backend: {result['backend']})")
        print(f"Assistant: {result['reply']}\n")


def run_demo(session_id: str) -> None:
    """Auto-run the 5 demo prompts sequentially."""
    print("\n" + "━"*60)
    print("  🎬 Running Demo (5 prompts)")
    print("━"*60)
    for prompt in DEMO_PROMPTS:
        print(f"\nYou: {prompt}")
        result = run_turn(session_id, prompt)
        print(f"[{result['agent']}] intent={result['intent']} confidence={result['confidence']:.0%}")
        print(f"Assistant: {result['reply']}\n")
        print("─"*60)

    print("\n✅ Demo complete. Run --preview-cli or --preview-html to see stored data.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ACME HR Multi-Agent Demo")
    parser.add_argument("--demo",         action="store_true", help="Run 5 auto demo prompts")
    parser.add_argument("--preview-cli",  action="store_true", help="Show data tables in terminal")
    parser.add_argument("--preview-html", action="store_true", help="Generate HTML report")
    parser.add_argument("--web",          action="store_true", help="Start Flask web server")
    parser.add_argument("--session",      default=None,        help="Resume session by ID")
    args = parser.parse_args()

    import uuid
    session_id = args.session or str(uuid.uuid4())

    if args.web:
        os.environ["FLASK_CONFIG_LOADED"] = "1"
        from app import create_app
        app = create_app(config, db, llm, orchestrator, session_manager, logger)
        server_cfg = config.get("server", {})
        # SERVER_HOST/PORT override config.yaml so a container can bind 0.0.0.0
        # without the local default becoming 0.0.0.0 too — binding every
        # interface should be an explicit choice, not the default for a laptop.
        host = os.getenv("SERVER_HOST", server_cfg.get("host", "127.0.0.1"))
        port = int(os.getenv("SERVER_PORT", server_cfg.get("port", 5000)))
        debug = os.getenv("SERVER_DEBUG", str(server_cfg.get("debug", False))).lower() \
            in ("1", "true", "yes", "on")
        print(f"[main] Serving on http://{host}:{port} (debug={debug})")
        app.run(host=host, port=port, debug=debug)
    elif args.demo:
        run_demo(session_id)
    elif args.preview_cli:
        CLIViewer(db, user_id).show_all()
    elif args.preview_html:
        HTMLReportGenerator(db, user_id).generate(open_browser=True)
    else:
        interactive_repl(session_id)


if __name__ == "__main__":
    main()
