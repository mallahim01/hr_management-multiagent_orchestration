"""
app.py
───────
Flask web server for the HR multi-agent demo.
Serves the frontend chat UI and exposes REST API endpoints.

Endpoints:
  GET  /                          – serve frontend/index.html
  POST /api/chat                  – process a user message
  GET  /api/status                – system status (user, backend, session)
  POST /api/backend               – switch orchestration backend at runtime
  GET  /api/graph                 – LangGraph structure (langgraph backend only)
  GET  /api/eval                  – LLM-as-judge scoring of recent routing
  GET  /api/history               – conversation history for active user
  GET  /api/preview/leave-balance – leave balance
  GET  /api/preview/leave-requests– leave requests
  GET  /api/preview/hr-requests   – HR requests
  GET  /api/report                – generate and return HTML report path
"""

import os
import uuid

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv
import yaml

load_dotenv()


def create_app(config, db, llm, orchestrator, session_manager, logger):
    """Application factory – accepts pre-initialised shared objects."""

    app = Flask(__name__, static_folder="frontend")
    user_id = config["active_user_id"]

    # The active orchestrator is held in a one-slot dict rather than the closure
    # variable so /api/backend can swap it without rebuilding the app.
    active = {
        "backend": config["orchestrator_backend"],
        "orchestrator": orchestrator,
    }

    def _run_turn(session_id: str, user_input: str) -> dict:
        """Shared pipeline: detect intent, invoke agent, persist, log."""
        ctx = session_manager.get_or_create(session_id, user_id)
        result = active["orchestrator"].process(user_input, ctx)

        db.save_message(session_id, user_id, "user", user_input)
        db.save_message(
            session_id, user_id, "assistant",
            result["reply"],
            agent=result["agent_class"],
            intent=result["intent"],
        )
        history_limit = config["conversation"]["history_size"] * 2
        ctx.history = db.get_recent_messages(session_id, limit=history_limit)
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
        )
        return result

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return send_from_directory("frontend", "index.html")

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True)
        user_input = (data.get("message") or "").strip()
        session_id = data.get("session_id") or str(uuid.uuid4())

        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        result = _run_turn(session_id, user_input)
        return jsonify(result)

    @app.route("/api/status")
    def status():
        user = db.get_user(user_id) or {"name": "Unknown", "department": "–", "email": "–"}
        return jsonify({
            "backend":     active["backend"],
            "model":       config["llm"]["model"],
            "provider":    config["llm"].get("provider", "groq"),
            "active_user": user,
            "db_path":     config["database"]["path"],
            "backends":    ["native", "langgraph", "crewai", "adk"],
        })

    @app.route("/api/backend", methods=["POST"])
    def switch_backend():
        """
        Swap the orchestration backend for subsequent turns.

        Session state lives in SQLite rather than in any backend, so switching
        mid-conversation is safe — the next turn picks up the same slot-filling
        state through a different engine. config.yaml is left untouched, so a
        restart returns to the configured default.
        """
        data = request.get_json(silent=True) or {}
        backend = (data.get("backend") or "").lower().strip()
        if backend not in ("native", "langgraph", "crewai", "adk"):
            return jsonify({"error": f"Unknown backend '{backend}'"}), 400

        if backend == active["backend"]:
            return jsonify({"backend": backend, "changed": False})

        try:
            from orchestration.factory import get_orchestrator
            active["orchestrator"] = get_orchestrator(
                backend, llm, db, config["conversation"]["history_size"]
            )
            active["backend"] = backend
            return jsonify({"backend": backend, "changed": True})
        except Exception as e:
            # crewai/adk pull in heavy optional dependencies and their own
            # credentials; a failure here must not take the running app down.
            return jsonify({
                "error": f"Could not start '{backend}': {type(e).__name__}: {e}",
                "backend": active["backend"],
            }), 502

    @app.route("/api/graph")
    def graph():
        """ASCII rendering of the compiled StateGraph, when langgraph is active."""
        orch = active["orchestrator"]
        if not hasattr(orch, "graph_ascii"):
            return jsonify({
                "available": False,
                "backend": active["backend"],
                "message": "Graph view is specific to the langgraph backend.",
            })
        return jsonify({
            "available": True,
            "backend": active["backend"],
            "ascii": orch.graph_ascii(),
            "last_path": getattr(orch, "last_path", []),
        })

    @app.route("/api/eval")
    def evaluate_routing():
        """Score recent routing decisions with the LLM acting as judge."""
        from core.routing_judge import RoutingJudge

        try:
            limit = max(1, min(int(request.args.get("limit", 10)), 50))
        except (TypeError, ValueError):
            limit = 10

        judge = RoutingJudge(llm, batch_size=5)
        report = judge.evaluate(limit=limit)
        if report["judged"]:
            judge.log_report(report, logger)
        return jsonify(report)

    @app.route("/api/history")
    def history():
        rows = db.get_all_messages(user_id)
        return jsonify(rows)

    @app.route("/api/preview/leave-balance")
    def leave_balance():
        data = db.get_leave_balance(user_id)
        return jsonify(data or {})

    @app.route("/api/preview/leave-requests")
    def leave_requests():
        return jsonify(db.get_leave_requests(user_id))

    @app.route("/api/preview/hr-requests")
    def hr_requests():
        return jsonify(db.get_hr_requests(user_id))

    @app.route("/api/report")
    def generate_report():
        from preview.html_report import HTMLReportGenerator
        gen = HTMLReportGenerator(db, user_id)
        path = gen.generate(open_browser=False)
        return jsonify({"path": path, "url": f"/report-file"})

    @app.route("/report-file")
    def report_file():
        return send_from_directory("data", "report.html")

    return app


# ── Standalone launch (without main.py) ──────────────────────────────────────

if __name__ == "__main__":
    import sys

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
        print(f"❌ No API key for provider '{provider}'.")
        sys.exit(1)

    db_path = config["database"]["path"]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    db = DatabaseManager(db_path)
    initialize_database(db, config["active_user_id"])

    llm = LLMWrapper(
        **{k: config["llm"][k] for k in ("model", "max_retries", "temperature")},
        provider=provider,
    )
    orchestrator = get_orchestrator(
        config["orchestrator_backend"], llm, db,
        config["conversation"]["history_size"],
    )
    session_manager = SessionManager(db, config["conversation"]["history_size"])

    app = create_app(config, db, llm, orchestrator, session_manager, InteractionLogger())
    srv = config.get("server", {})
    app.run(host=srv.get("host", "127.0.0.1"), port=srv.get("port", 5000), debug=srv.get("debug", False))
