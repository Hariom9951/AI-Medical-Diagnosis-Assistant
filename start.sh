#!/bin/sh
# =============================================================
# start.sh — AI Medical Diagnosis Assistant container entrypoint
# =============================================================
# Starts Streamlit for Hugging Face Spaces / Docker.
#
# At startup this script:
#   1. Validates environment
#   2. Pre-warms model checkpoints via ModelDownloader
#      (downloads from HF_MODEL_REPO_ID if not cached)
#   3. Launches Streamlit on port 7860
#
# Required environment variables:
#   HF_TOKEN         — Hugging Face read token (for private repos)
#   HF_MODEL_REPO_ID — (optional) override model repo ID
#                      defaults to Hariom9951/AI-Medical-Diagnosis-Models
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
echo "[ENV]   Repo    : ${HF_MODEL_REPO_ID:-Hariom9951/AI-Medical-Diagnosis-Models}"
echo ""

# ─── Pre-warm model checkpoints ──────────────────────────────
# Downloads all model artifacts from the Hugging Face model
# repository on first startup. Cached files are verified via
# SHA256 and skipped on subsequent starts.
# Set CI_SKIP_MODEL_DOWNLOAD=true to skip this step (e.g., in CI).
if [ "${CI_SKIP_MODEL_DOWNLOAD:-false}" = "true" ]; then
  echo "[MODELS] CI_SKIP_MODEL_DOWNLOAD=true — skipping model pre-warm."
else
  echo "[MODELS] Pre-warming model checkpoints from Hugging Face..."
  python - <<'PYEOF'
import sys
import os

# Allow the downloader to reach the HF model repo even though
# TRANSFORMERS_OFFLINE=1 is set (that flag only affects the
# transformers library, not our custom downloader).
os.environ.pop("HF_DATASETS_OFFLINE", None)

from src.utils.downloader import ModelDownloader, EXPECTED_SHA256
from pathlib import Path

project_root = Path("/app")

MODEL_FILES = [
    # (repo_path, local_relative_path)
    ("image/best_model.pth",              "artifacts/checkpoints/best_model.pth"),
    ("image/checkpoint_epoch_050.pth",    "artifacts/checkpoints/checkpoint_epoch_050.pth"),
    ("nlp/best_model.pt",                 "artifacts/checkpoints_nlp/best_model.pt"),
    ("nlp/tokenizer.json",                "artifacts/checkpoints_nlp/tokenizer.json"),
    ("nlp/tokenizer_config.json",         "artifacts/checkpoints_nlp/tokenizer_config.json"),
    ("nlp/label_encoder.pkl",             "artifacts/checkpoints_nlp/label_encoder.pkl"),
    ("nlp/model_metadata.json",           "artifacts/checkpoints_nlp/model_metadata.json"),
    ("nlp/temperature_scaler.json",       "artifacts/checkpoints_nlp/temperature_scaler.json"),
    ("nlp/clinical_explanations.json",    "artifacts/checkpoints_nlp/clinical_explanations.json"),
]

downloader = ModelDownloader()
errors = []

for repo_path, local_rel in MODEL_FILES:
    local_abs = project_root / local_rel
    try:
        downloader.download_file(repo_path, local_abs)
        print(f"[MODELS] OK: {repo_path}")
    except Exception as e:
        print(f"[MODELS] WARNING: Could not download {repo_path}: {e}", file=sys.stderr)
        errors.append(repo_path)

if errors:
    print(f"[MODELS] {len(errors)} file(s) could not be downloaded. "
          f"The app may fail if required models are missing.", file=sys.stderr)
else:
    print("[MODELS] All model checkpoints ready.")
PYEOF
fi

echo ""

# ─── Run startup diagnostics ─────────────────────────────────
echo "[DIAG] Running startup diagnostics..."
python scripts/diagnostics.py || true
echo ""

# ─── Start Streamlit ─────────────────────────────────────────
echo "[START] Launching Streamlit on port ${PORT:-7860} ..."
exec streamlit run src/frontend/app.py \
    --server.port ${PORT:-7860} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
