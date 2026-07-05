#!/bin/sh
# =============================================================
# start.sh — AI Medical Diagnosis Assistant container entrypoint
# =============================================================
# Starts both services inside the single container:
#   1. FastAPI  on port 8000  (background)
#   2. Streamlit on port 8501 (foreground — keeps container alive)
#
# The FastAPI process is started first and we wait for it to
# become healthy before starting Streamlit, so that Streamlit
# can immediately reach the API on startup.
#
# Usage (Docker automatically calls this as ENTRYPOINT):
#   /start.sh
# =============================================================

set -e

# ─── Ensure venv binaries are on PATH ────────────────────────
# Explicitly export /opt/venv/bin so that python, uvicorn, and
# streamlit are found even when the script is invoked in a
# restricted environment (e.g. `docker exec`, supervisord, etc.)
# where the Dockerfile ENV PATH may not be inherited.
export PATH="/opt/venv/bin:$PATH"
export PYTHONPATH="/app:${PYTHONPATH:-}"

echo "======================================================="
echo "  AI Medical Diagnosis Assistant — Container Starting"
echo "======================================================="
echo ""
echo "[ENV]   Python  : $(python --version 2>&1)"
echo "[ENV]   PATH    : $PATH"
echo ""

# ─── Helper: wait for a port to be open ─────────────────────
wait_for_port() {
    local host="$1"
    local port="$2"
    local service="$3"
    local max_attempts=30
    local attempt=1

    echo "[WAIT] Waiting for $service to be ready on $host:$port ..."
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "[TIMEOUT] $service did not start within ${max_attempts}s. Continuing anyway."
            return 1
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo "[OK] $service is up on $host:$port"
    return 0
}

# ─── 1. Start FastAPI (uvicorn) in background ────────────────
echo "[START] Launching FastAPI (uvicorn) on port 8000 ..."
python -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --no-access-log &

FASTAPI_PID=$!
echo "[INFO]  FastAPI PID: $FASTAPI_PID"

# ─── 2. Wait for FastAPI to be ready ─────────────────────────
wait_for_port localhost 8000 "FastAPI"

# ─── 3. Start Streamlit in foreground ────────────────────────
echo "[START] Launching Streamlit on port 8501 ..."
exec python -m streamlit run src/frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none
