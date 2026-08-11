#!/usr/bin/env bash
# Container entrypoint.
#
#   web     (default) wait for Milvus, seed the knowledge base, serve the app
#   ingest  ingest documents and exit
#   test    run the offline suites and exit
#   eval    run the golden-set evaluation and exit
#   shell   drop into bash
#
# Seeding is idempotent: the policy document is only ingested when the
# collection has no documents, so restarting the stack does not re-embed it.

set -euo pipefail

MILVUS_URI="${KNOWLEDGE__MILVUS_URI:-http://milvus:19530}"
MILVUS_HEALTH="${MILVUS_HEALTH_URL:-http://milvus:9091/healthz}"
WAIT_SECONDS="${MILVUS_WAIT_SECONDS:-90}"

log() { echo "[entrypoint] $*"; }

wait_for_milvus() {
  log "waiting for Milvus at ${MILVUS_HEALTH} (up to ${WAIT_SECONDS}s)…"
  local waited=0
  until curl -fsS "${MILVUS_HEALTH}" >/dev/null 2>&1; do
    if [ "${waited}" -ge "${WAIT_SECONDS}" ]; then
      log "Milvus did not become healthy in ${WAIT_SECONDS}s."
      log "The app will still start — the policy agent falls back to the"
      log "bundled policy file and says so in its replies."
      return 1
    fi
    sleep 3
    waited=$((waited + 3))
  done
  log "Milvus is healthy after ${waited}s."
  return 0
}

seed_knowledge_base() {
  if [ -z "${GOOGLE_API_KEY:-}" ]; then
    log "GOOGLE_API_KEY not set — skipping ingestion (embeddings need it)."
    return 0
  fi
  log "checking whether the knowledge base already has documents…"
  if python - <<'PY'
import sys, yaml
from knowledge import build_store
config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
store = build_store(config)
sys.exit(0 if store.ensure_ready() and store.list_documents() else 1)
PY
  then
    log "knowledge base already seeded — leaving it alone."
  else
    log "seeding data/company_policy.txt…"
    python ingest_knowledge.py --by docker-seed || \
      log "ingestion failed; the agent will fall back to the bundled policy file."
  fi
}

case "${1:-web}" in
  web)
    wait_for_milvus || true
    seed_knowledge_base || true
    log "starting Flask on 0.0.0.0:5000"
    exec python main.py --web
    ;;
  ingest)
    wait_for_milvus || true
    shift
    exec python ingest_knowledge.py "$@"
    ;;
  test)
    log "running the offline suites (no API key or Milvus required)…"
    python verify.py
    python test_langgraph.py
    python test_validation.py
    python test_api.py
    python test_eval.py
    exec python test_rag.py
    ;;
  eval)
    wait_for_milvus || true
    shift
    exec python eval_system.py "$@"
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    exec "$@"
    ;;
esac
