# syntax=docker/dockerfile:1
#
# Application image: Flask API + chat UI + the agents.
# Milvus runs as its own service — see docker-compose.yml.
#
# The crewai and adk backends are excluded by default because they pull in
# roughly 2 GB of transitive dependencies. The native and langgraph backends
# (langgraph being the one this project leads with) need nothing extra. Build
# with --build-arg INSTALL_ALL_BACKENDS=true to include them.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by the container healthcheck and the entrypoint's readiness wait.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, so application edits do not invalidate the install layer.
COPY requirements.txt requirements-backends.txt ./

ARG INSTALL_ALL_BACKENDS=false
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && if [ "$INSTALL_ALL_BACKENDS" = "true" ]; then \
        echo "Installing optional crewai + adk backends…" ; \
        pip install -r requirements-backends.txt ; \
    else \
        echo "Skipping crewai + adk (build with --build-arg INSTALL_ALL_BACKENDS=true)" ; \
    fi

COPY . .

# The entrypoint's executable bit does not survive a checkout on Windows, so set
# it here rather than relying on the repository's file mode.
RUN chmod +x /app/docker/entrypoint.sh

# Run as a non-root user; data/ and logs/ are bind-mounted at runtime so they
# must be writable by it.
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/data /app/logs \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:5000/api/status || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["web"]
