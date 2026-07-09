# ============================================================
# AI Medical Diagnosis Assistant — Production Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder)  — install Python dependencies and
#                        pre-download BioBERT base weights into
#                        a clean virtual-env to keep the final
#                        image lean and cache-friendly.
#   Stage 2 (runtime)  — copy only the venv + app code.
#                        Model checkpoints are NOT baked in;
#                        they are fetched from the Hugging Face
#                        model repository at container start via
#                        the ModelDownloader service.
#
# Ports:
#   7860  Streamlit
#
# Required Environment Variables at Runtime:
#   HF_TOKEN           — Hugging Face API token (write access not
#                        required; read-only token for model repo)
#   HF_MODEL_REPO_ID   — (optional) override model repo, defaults
#                        to "Hariom9951/AI-Medical-Diagnosis-Models"
# ============================================================

# ─── Stage 1: dependency builder ────────────────────────────
FROM python:3.11-slim AS builder

# System build deps (needed by some pip wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ── Copy only what pip needs first (layer cache optimisation) ─
COPY requirements.txt .

# Create an isolated venv inside /opt/venv and install CPU-only PyTorch + requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip wheel setuptools --timeout 120 --retries 5 && \
    /opt/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --timeout 120 --retries 5 && \
    /opt/venv/bin/pip install -r requirements.txt --timeout 120 --retries 5

# ── Pre-download BioBERT base weights into the builder image ──
# Internet is available during build; we cache the model so the
# runtime container can load BioBERT architecture offline.
# NOTE: This does NOT download the fine-tuned checkpoint weights —
#       those are fetched at runtime from the HF model repository.
ENV HF_HOME=/build/hf_cache
RUN /opt/venv/bin/python - <<'EOF'
import sys
from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer

model_name = "dmis-lab/biobert-base-cased-v1.1"
print(f"[BUILD] Downloading {model_name} config + weights ...")

try:
    # 1. Download tokenizer with fallback to use_fast=False
    try:
        print("[BUILD] Attempting fast tokenizer download...")
        AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"[BUILD] Fast tokenizer failed: {e}. Retrying with use_fast=False...")
        try:
            AutoTokenizer.from_pretrained(model_name, use_fast=False)
            print("[BUILD] Slow tokenizer cached successfully.")
        except Exception as e2:
            print(f"[BUILD] Warning: Tokenizer download failed: {e2}")

    # 2. Download config
    try:
        BertConfig.from_pretrained(model_name)
        print("[BUILD] Config cached successfully.")
    except Exception as e:
        print(f"[BUILD] Warning: Config download failed: {e}")

    # 3. Pre-cache model weights
    try:
        BertForSequenceClassification.from_pretrained(model_name, num_labels=41)
        print("[BUILD] BioBERT weights cached successfully.")
    except Exception as e:
        print(f"[BUILD] Warning: model weights cache got warning: {e}")

except Exception as e:
    print(f"[BUILD] Warning: BioBERT caching step failed but continuing: {e}")

print("[BUILD] BioBERT caching check complete.")
EOF


# ─── Stage 2: lean runtime image ────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Hariom Sharma <hariom9951@github.com>" \
      version="3.0.0" \
      description="AI Medical Diagnosis Assistant — Streamlit (runtime model download)"

# Runtime system libraries (no build tools)
# libgl1 is required by opencv-python-headless at runtime (links against libGL.so.1)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        curl \
        netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# ── Copy pre-built venv from builder ────────────────────────
COPY --from=builder /opt/venv /opt/venv

# ── Copy pre-downloaded HuggingFace model cache from builder ─
# Allows the runtime container to load dmis-lab/biobert-base-cased-v1.1
# architecture weights fully offline (TRANSFORMERS_OFFLINE=1 is set below).
COPY --from=builder /build/hf_cache /opt/hf_cache

# Put venv binaries first on PATH
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    # Streamlit env tweaks for non-interactive / headless mode
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    # Torch — use CPU inside container (no GPU drivers expected)
    CUDA_VISIBLE_DEVICES="" \
    # Suppress tokenizers parallelism warning
    TOKENIZERS_PARALLELISM=false \
    # HuggingFace cache — points at the BioBERT base weights baked into the image
    HF_HOME=/opt/hf_cache \
    # Transformers offline mode — use local BioBERT base weights, no HuggingFace downloads
    # (fine-tuned checkpoint weights are fetched from HF_MODEL_REPO_ID separately)
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    # Model repository for fine-tuned checkpoints (can be overridden via docker run -e)
    HF_MODEL_REPO_ID="Hariom51/AI-Medical-Diagnosis-Models"

WORKDIR /app

# ── Copy project source code ─────────────────────────────────
COPY src/           ./src/
COPY configs/       ./configs/
COPY scripts/       ./scripts/
COPY setup.py       .
COPY pyproject.toml .
COPY requirements.txt .

# ── Copy disease mapping used by NLP pipeline ─────────────────
COPY data/processed/disease_mapping_41.json ./data/processed/disease_mapping_41.json
COPY data/processed/disease_mapping.json    ./data/processed/disease_mapping.json

# ── Install the project package (editable-style, no deps) ────
RUN pip install --no-cache-dir --no-deps -e .

# ── Create writable runtime directories ──────────────────────
# Model checkpoints are downloaded at startup into these directories.
RUN mkdir -p \
        artifacts/checkpoints \
        artifacts/checkpoints_nlp \
        artifacts/gradcam \
        artifacts/evaluation \
        logs && \
    chmod -R 777 artifacts logs

# ── Copy startup script ───────────────────────────────────────
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ── Expose ports ──────────────────────────────────────────────
EXPOSE 7860

# ── Default entrypoint ────────────────────────────────────────
ENTRYPOINT ["/start.sh"]
