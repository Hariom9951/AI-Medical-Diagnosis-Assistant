#!/bin/sh
# =============================================================
# start.sh — AI Medical Diagnosis Assistant container entrypoint
# =============================================================
# Starts Streamlit for Hugging Face Spaces.
#
# Usage (Docker automatically calls this as ENTRYPOINT):
#   /start.sh
# =============================================================

set -e

# ─── Ensure venv binaries are on PATH ────────────────────────
export PATH=/opt/venv/bin:$PATH
export PYTHONPATH=/app

echo "======================================================="
echo "  AI Medical Diagnosis Assistant — Container Starting"
echo "======================================================="
echo ""
echo "[ENV]   Python  : $(python --version 2>&1)"
echo "[ENV]   PATH    : $PATH"
echo ""

# ─── Start Streamlit only ────────────────────────
echo "[START] Launching Streamlit on port ${PORT:-7860} ..."
exec streamlit run src/frontend/app.py \
    --server.port ${PORT:-7860} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false


