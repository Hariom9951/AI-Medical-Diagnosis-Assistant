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
#                      defaults to Hariom51/AI-Medical-Diagnosis-Models
#
# Usage (Docker automatically calls this as ENTRYPOINT):
#   /start.sh
# =============================================================

set -e

# ─── Ensure venv binaries are on PATH ────────────────────────
# Put venv binaries on PATH
export PATH=/opt/venv/bin:$PATH
export PYTHONPATH=/app

echo "======================================================="
echo "  AI Medical Diagnosis Assistant — Container Starting"
echo "======================================================="
echo ""
echo "[ENV]   Python  : $(python --version 2>&1)"
echo "[ENV]   PATH    : $PATH"
echo "[ENV]   Repo    : ${HF_MODEL_REPO_ID:-Hariom51/AI-Medical-Diagnosis-Models}"
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
from pathlib import Path

# Allow the downloader to reach the HF model repo even though
# TRANSFORMERS_OFFLINE=1 is set (that flag only affects the
# transformers library, not our custom downloader).
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

from src.utils.downloader import ModelDownloader

project_root = Path("/app")

# Flat files stored at the root of the Hugging Face repository
# Hariom51/AI-Medical-Diagnosis-Models
MODEL_FILES = [
    # (repo_path, local_relative_path)
    ("checkpoint_epoch_050.pth",          "artifacts/checkpoints/checkpoint_epoch_050.pth"),
    ("best_model.pt",                     "artifacts/checkpoints_nlp/best_model.pt"),
    ("tokenizer.json",                    "artifacts/checkpoints_nlp/tokenizer.json"),
    ("tokenizer_config.json",             "artifacts/checkpoints_nlp/tokenizer_config.json"),
    ("label_encoder.pkl",                 "artifacts/checkpoints_nlp/label_encoder.pkl"),
    ("model_metadata.json",               "artifacts/checkpoints_nlp/model_metadata.json"),
    ("temperature_scaler.json",           "artifacts/checkpoints_nlp/temperature_scaler.json"),
    ("clinical_explanations.json",        "artifacts/checkpoints_nlp/clinical_explanations.json"),
]

downloader = ModelDownloader()
errors = []

for repo_path, local_rel in MODEL_FILES:
    local_abs = project_root / local_rel
    try:
        print(f"[MODELS] Checking: {repo_path} -> {local_rel}")
        res_path = downloader.download_file(repo_path, local_abs)
        size_bytes = res_path.stat().st_size
        print(f"[MODELS] VERIFIED: {repo_path} ({size_bytes} bytes)")
    except Exception as e:
        print(f"[MODELS] ERROR: Failed to download/verify {repo_path}: {e}", file=sys.stderr)
        errors.append((repo_path, str(e)))

# Create best_model.pth alias for compatibility if epoch model downloaded successfully
epoch_model_path = project_root / "artifacts/checkpoints/checkpoint_epoch_050.pth"
best_model_path = project_root / "artifacts/checkpoints/best_model.pth"
if epoch_model_path.exists():
    try:
        import shutil
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(epoch_model_path), str(best_model_path))
        print(f"[MODELS] Created compatibility alias: best_model.pth -> checkpoint_epoch_050.pth")
    except Exception as e:
        print(f"[MODELS] WARNING: Could not create best_model.pth alias: {e}", file=sys.stderr)

if errors:
    print(f"[MODELS] FATAL: {len(errors)} required file(s) failed verification/download:", file=sys.stderr)
    for r_path, err_msg in errors:
        print(f"  - {r_path}: {err_msg}", file=sys.stderr)
    sys.exit(1)
else:
    print("[MODELS] All model checkpoints successfully verified and ready.")
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
