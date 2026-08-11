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

import os
import uuid

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv
import yaml

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

    # ── Knowledge base ────────────────────────────────────────────────────────

    def _store():
        """Shared knowledge store, built once per app."""
        if "store" not in active:
            from knowledge import build_store
            active["store"] = build_store(config)
        return active["store"]

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
