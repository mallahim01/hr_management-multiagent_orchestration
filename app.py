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
  GET  /api/knowledge/status      – Milvus / embedding availability + counts
  GET  /api/knowledge/documents   – documents in the knowledge base
  POST /api/knowledge/ingest      – upload a document (file or pasted text)
  DEL  /api/knowledge/documents/<id> – remove a document
  GET  /api/knowledge/search      – preview hybrid retrieval for a query
  GET  /api/history               – conversation history for active user
  GET  /api/preview/leave-balance – leave balance
  GET  /api/preview/leave-requests– leave requests
  GET  /api/preview/hr-requests   – HR requests
  GET  /api/report                – generate and return HTML report path
"""

import json
import os
import uuid

from flask import Flask, Response, jsonify, request, send_from_directory
from dotenv import load_dotenv
import yaml

from core import metrics
from core import streaming as streams

load_dotenv()

# Formats accepted for knowledge-base uploads. PDFs need pypdf, which is an
# optional install — the status endpoint advertises whether it is present.
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".rst"}
MAX_UPLOAD_CHARS = 2_000_000


def _pdf_supported() -> bool:
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return False


def _read_upload(upload) -> tuple:
    """
    Extract text from an uploaded file.

    Returns (text, source_name). Raises ValueError with a message meant for the
    user on anything it cannot read.
    """
    name = os.path.basename(upload.filename or "").strip()
    if not name:
        raise ValueError("The upload has no filename")
    extension = os.path.splitext(name)[1].lower()

    if extension == ".pdf":
        if not _pdf_supported():
            raise ValueError("PDF support needs `pip install pypdf`")
        from pypdf import PdfReader
        try:
            reader = PdfReader(upload.stream)
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            raise ValueError(f"Could not read that PDF: {e}") from e
        if not text.strip():
            raise ValueError(
                "No text found in that PDF — scanned images need OCR, which "
                "this project does not do"
            )
    elif extension in TEXT_EXTENSIONS:
        raw = upload.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
    else:
        accepted = ", ".join(sorted(TEXT_EXTENSIONS | ({".pdf"} if _pdf_supported() else set())))
        raise ValueError(f"Unsupported file type '{extension}'. Accepted: {accepted}")

    if len(text) > MAX_UPLOAD_CHARS:
        raise ValueError(f"Document is too large ({len(text)} chars, "
                         f"limit {MAX_UPLOAD_CHARS})")
    if not text.strip():
        raise ValueError("That file is empty")
    return text, name


def create_app(config, db, llm, orchestrator, session_manager, logger):
    """Application factory – accepts pre-initialised shared objects."""

    app = Flask(__name__, static_folder="frontend")

    # Backend and active user are held in a one-slot dict rather than closure
    # variables so /api/backend and /api/user can swap them without rebuilding
    # the app. There is no authentication here — see the README; this is a demo
    # persona switch, not a login.
    active = {
        "backend": config["orchestrator_backend"],
        "orchestrator": orchestrator,
        "user_id": config["active_user_id"],
    }

    pricing = config.get("pricing") or None

    def _run_turn(session_id: str, user_input: str) -> dict:
        """Shared pipeline: detect intent, invoke agent, persist, log."""
        user_id = active["user_id"]
        ctx = session_manager.get_or_create(session_id, user_id)

        with metrics.turn(pricing=pricing) as turn_metrics:
            result = active["orchestrator"].process(user_input, ctx)
        result["metrics"] = turn_metrics.summary()

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
            metrics=result["metrics"],
        )
        result["user_id"] = user_id
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

    @app.route("/api/chat/stream", methods=["POST"])
    def chat_stream():
        """
        Server-sent events for one turn: stage updates, then tokens, then the
        same result dict `/api/chat` returns.

        The turn runs on a worker thread while this generator drains the sink
        from the request thread. `copy_context()` carries the ContextVars across
        — without it the worker would not see the sink, and streaming would
        silently fall back to a blocking call that still worked, which is a
        worse failure than an error.
        """
        import threading
        from contextvars import copy_context

        data = request.get_json(force=True, silent=True) or {}
        user_input = (data.get("message") or "").strip()
        session_id = data.get("session_id") or str(uuid.uuid4())
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        token_sink = streams.TokenSink()

        def worker():
            try:
                with streams.sink(token_sink):
                    token_sink.emit_event("stage", stage="routing")
                    result = _run_turn(session_id, user_input)
                    token_sink.emit_event("result", **result)
            except Exception as e:
                print(f"  [stream] turn failed: {type(e).__name__}: {e}")
                token_sink.emit_event(
                    "error", error=f"{type(e).__name__}: {e}")
            finally:
                token_sink.close()

        threading.Thread(target=copy_context().run, args=(worker,),
                         daemon=True).start()

        def events():
            for item in token_sink.drain():
                yield f"data: {json.dumps(item)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"

        return Response(events(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # stop nginx buffering the stream
            "Connection": "keep-alive",
        })

    @app.route("/api/status")
    def status():
        user = db.get_user(active["user_id"]) or \
            {"name": "Unknown", "department": "–", "email": "–"}
        return jsonify({
            "backend":     active["backend"],
            "model":       config["llm"]["model"],
            "provider":    config["llm"].get("provider", "groq"),
            "active_user": user,
            "db_path":     config["database"]["path"],
            "backends":    ["native", "langgraph", "crewai", "adk"],
        })

    @app.route("/api/users")
    def list_users():
        """Every seeded employee, with their leave balance, for the switcher."""
        users = db.fetch_all("SELECT * FROM users ORDER BY id")
        for user in users:
            balance = db.get_leave_balance(user["id"]) or {}
            user["remaining_leaves"] = balance.get("remaining_leaves")
            user["total_leaves"] = balance.get("total_leaves")
            user["active"] = user["id"] == active["user_id"]
        return jsonify(users)

    @app.route("/api/user", methods=["POST"])
    def switch_user():
        """
        Change which employee the assistant is acting for.

        This is a demo persona switch, not authentication — there is no login
        and no authorisation anywhere in this project.

        The caller is told to start a new conversation, because slot-filling
        state is keyed by session rather than by user: continuing a half-filled
        leave request after switching would file it against the new employee.
        """
        data = request.get_json(silent=True) or {}
        try:
            new_id = int(data.get("user_id"))
        except (TypeError, ValueError):
            return jsonify({"error": "user_id must be an integer"}), 400

        user = db.get_user(new_id)
        if not user:
            return jsonify({"error": f"No user with id {new_id}"}), 404

        previous = active["user_id"]
        active["user_id"] = new_id
        if previous != new_id:
            logger.log_event("active_user_changed",
                             previous_user_id=previous, user_id=new_id,
                             name=user.get("name"))
        return jsonify({"active_user": user, "changed": previous != new_id,
                        "reset_session": True})

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

    # ── Knowledge base ────────────────────────────────────────────────────────

    def _store():
        """Shared knowledge store, built once per app."""
        if "store" not in active:
            from knowledge import build_store
            active["store"] = build_store(config)
        return active["store"]

    @app.route("/api/metrics/summary")
    def metrics_summary():
        """
        Aggregate cost and latency over recent turns, read back from the log.

        Percentiles rather than a mean: LLM latency has a long tail, and a mean
        that a cold start or one retry can move is not a number worth quoting.
        """
        try:
            limit = max(1, min(int(request.args.get("limit", 50)), 500))
        except (TypeError, ValueError):
            limit = 50

        try:
            with open(logger.log_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError):
            records = []

        turns = [r for r in records
                 if r.get("event", "interaction") == "interaction" and r.get("metrics")]
        turns = turns[-limit:]
        if not turns:
            return jsonify({"turns": 0, "note": "no measured turns in the log yet"})

        def pct(values, p):
            if not values:
                return None
            ordered = sorted(values)
            idx = min(int(round((p / 100) * (len(ordered) - 1))), len(ordered) - 1)
            return round(ordered[idx], 3)

        seconds = [t["metrics"]["seconds"] for t in turns]
        costs = [t["metrics"]["cost_usd"] for t in turns]
        tokens = [t["metrics"]["total_tokens"] for t in turns]

        stage_totals: dict = {}
        for t in turns:
            for name, v in (t["metrics"].get("stages") or {}).items():
                entry = stage_totals.setdefault(
                    name, {"cost_usd": 0.0, "seconds": 0.0, "calls": 0})
                entry["cost_usd"] += v.get("cost_usd", 0.0)
                entry["seconds"] += v.get("seconds", v.get("llm_seconds", 0.0))
                entry["calls"] += v.get("calls", 0)

        total_cost = sum(costs) or 1e-12
        total_stage_seconds = sum(s["seconds"] for s in stage_totals.values()) or 1e-12
        for name, v in stage_totals.items():
            v["cost_usd"] = round(v["cost_usd"], 8)
            v["seconds"] = round(v["seconds"], 3)
            v["cost_share"] = round(v["cost_usd"] / total_cost, 3)
            v["latency_share"] = round(v["seconds"] / total_stage_seconds, 3)

        by_backend: dict = {}
        for t in turns:
            b = by_backend.setdefault(t.get("backend", "?"), [])
            b.append(t["metrics"]["seconds"])

        return jsonify({
            "turns": len(turns),
            "latency_p50": pct(seconds, 50),
            "latency_p95": pct(seconds, 95),
            "cost_median": pct(costs, 50),
            "cost_total": round(sum(costs), 6),
            "tokens_median": pct(tokens, 50),
            "estimated_any": any(t["metrics"].get("estimated_tokens") for t in turns),
            "stages": stage_totals,
            "by_backend": {k: {"turns": len(v), "p50": pct(v, 50)}
                           for k, v in by_backend.items()},
        })

    @app.route("/api/knowledge/status")
    def knowledge_status():
        from knowledge import knowledge_config
        cfg = knowledge_config(config)
        stats = _store().stats()
        stats["enabled"] = bool(cfg["enabled"])
        stats["accepted_formats"] = sorted(TEXT_EXTENSIONS | ({".pdf"} if _pdf_supported() else set()))
        return jsonify(stats)

    @app.route("/api/knowledge/documents")
    def knowledge_documents():
        return jsonify(_store().list_documents())

    @app.route("/api/knowledge/documents/<doc_id>", methods=["DELETE"])
    def knowledge_delete(doc_id):
        removed = _store().delete_document(doc_id)
        if not removed:
            return jsonify({"error": f"No document '{doc_id}'", "deleted": 0}), 404
        logger.log_event("knowledge_document_deleted", doc_id=doc_id,
                         chunks=removed, deleted_by="hr")
        return jsonify({"doc_id": doc_id, "deleted": removed})

    @app.route("/api/knowledge/ingest", methods=["POST"])
    def knowledge_ingest():
        """
        Add a document. Accepts either a multipart file upload or JSON
        {text, source, title} for pasted content.
        """
        from knowledge import knowledge_config
        from knowledge.embeddings import EmbeddingUnavailable
        from knowledge.store import KnowledgeStoreUnavailable

        cfg = knowledge_config(config)
        uploaded_by = (request.form.get("uploaded_by")
                       or (request.get_json(silent=True) or {}).get("uploaded_by")
                       or "hr")

        try:
            if "file" in request.files:
                upload = request.files["file"]
                text, source = _read_upload(upload)
                title = request.form.get("title") or source
            else:
                data = request.get_json(silent=True) or {}
                text = (data.get("text") or "").strip()
                source = (data.get("source") or "").strip()
                title = (data.get("title") or source).strip()
                if not text:
                    return jsonify({"error": "No file uploaded and no text provided"}), 400
                if not source:
                    return jsonify({"error": "A source name is required"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        try:
            result = _store().ingest_text(
                text=text, source=source, title=title, uploaded_by=uploaded_by,
                max_chars=cfg["chunk_max_chars"], overlap=cfg["chunk_overlap"],
            )
        except KnowledgeStoreUnavailable as e:
            return jsonify({"error": f"Knowledge base unavailable: {e}"}), 503
        except EmbeddingUnavailable as e:
            return jsonify({"error": f"Embeddings unavailable: {e}"}), 503
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        logger.log_event("knowledge_document_ingested", **result)
        return jsonify(result), 201

    @app.route("/api/knowledge/search")
    def knowledge_search():
        """Preview what retrieval returns for a query, without asking the LLM."""
        from knowledge import knowledge_config
        from knowledge.store import KnowledgeStoreUnavailable

        query = (request.args.get("q") or "").strip()
        if not query:
            return jsonify({"error": "Missing query parameter 'q'"}), 400
        cfg = knowledge_config(config)
        try:
            top_k = max(1, min(int(request.args.get("k", cfg["top_k"])), 20))
        except (TypeError, ValueError):
            top_k = cfg["top_k"]
        try:
            results = _store().hybrid_search(query, top_k=top_k,
                                             candidate_k=cfg["candidate_k"])
        except KnowledgeStoreUnavailable as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 502
        return jsonify({"query": query, "results": results})

    @app.route("/api/history")
    def history():
        rows = db.get_all_messages(active["user_id"])
        return jsonify(rows)

    @app.route("/api/preview/leave-balance")
    def leave_balance():
        data = db.get_leave_balance(active["user_id"])
        return jsonify(data or {})

    @app.route("/api/preview/leave-requests")
    def leave_requests():
        return jsonify(db.get_leave_requests(active["user_id"]))

    @app.route("/api/preview/hr-requests")
    def hr_requests():
        return jsonify(db.get_hr_requests(active["user_id"]))

    @app.route("/api/report")
    def generate_report():
        from preview.html_report import HTMLReportGenerator
        gen = HTMLReportGenerator(db, active["user_id"])
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
