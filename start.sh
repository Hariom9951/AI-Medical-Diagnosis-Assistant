#!/bin/sh
# =============================================================
# start.sh — AI Medical Diagnosis Assistant container entrypoint
# =============================================================
# Starts both services inside the single container:
#   1. Streamlit on port 8501 (background)
#   2. FastAPI (Uvicorn) on port 8000 (foreground - PID 1)
#
# Usage (Docker automatically calls this as ENTRYPOINT):
#   /start.sh
# =============================================================

set -e

# ─── Ensure venv binaries are on PATH ────────────────────────
export PATH="/opt/venv/bin:$PATH"
export PYTHONPATH="/app:${PYTHONPATH:-}"

echo "======================================================="
echo "  AI Medical Diagnosis Assistant — Container Starting"
echo "======================================================="
echo ""
echo "[ENV]   Python  : $(python --version 2>&1)"
echo "[ENV]   PATH    : $PATH"
echo ""

# ─── 1. Start Streamlit in background ────────────────────────
echo "[START] Launching Streamlit on port 8501 (background) ..."
python -m streamlit run src/frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none &

# Brief pause to allow Streamlit worker to initiate
sleep 2

# ─── 2. Start FastAPI (uvicorn) in foreground as PID 1 ──────
echo "[START] Launching FastAPI (uvicorn) on port ${PORT:-8000} (PID 1) ..."
exec python -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --log-level info \
    --no-access-log

