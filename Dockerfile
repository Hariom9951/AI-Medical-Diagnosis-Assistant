# ============================================================
# AI Medical Diagnosis Assistant — Production Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder)  — install Python dependencies into a
#                        clean virtual-env to keep the final
#                        image lean and cache-friendly.
#   Stage 2 (runtime)  — copy only the venv + app code.
#
# Ports:
#   8000  FastAPI  (uvicorn)
#   8501  Streamlit
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

# Create an isolated venv inside /build/venv
RUN python -m venv /build/venv && \
    /build/venv/bin/pip install --upgrade pip wheel && \
    /build/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Pre-download DistilBERT base weights into the builder image ──
# Internet is available during build; we cache the model so the
# runtime container can run fully offline (TRANSFORMERS_OFFLINE=1).
ENV HF_HOME=/build/hf_cache
RUN /build/venv/bin/python - <<'EOF'
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer
print("[BUILD] Downloading distilbert-base-uncased config + weights ...")
DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
DistilBertConfig.from_pretrained("distilbert-base-uncased")
DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
print("[BUILD] distilbert-base-uncased cached successfully.")
EOF

# ─── Stage 2: lean runtime image ────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Hariom Sharma <hariom9951@github.com>" \
      version="1.0.0" \
      description="AI Medical Diagnosis Assistant — FastAPI + Streamlit"

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
COPY --from=builder /build/venv /opt/venv

# ── Copy pre-downloaded HuggingFace model cache from builder ─
# Allows the runtime container to load distilbert-base-uncased
# fully offline (TRANSFORMERS_OFFLINE=1 is set below).
COPY --from=builder /build/hf_cache /opt/hf_cache

# Put venv binaries first on PATH
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    # Streamlit env tweaks for non-interactive / headless mode
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    # Torch — use CPU inside container (no GPU drivers expected)
    CUDA_VISIBLE_DEVICES="" \
    # Suppress tokenizers parallelism warning
    TOKENIZERS_PARALLELISM=false \
    # HuggingFace cache — points at the model weights baked into the image
    HF_HOME=/opt/hf_cache \
    # Transformers offline mode — use local files, no HuggingFace downloads
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

WORKDIR /app

# ── Copy project source code ─────────────────────────────────
COPY src/           ./src/
COPY configs/       ./configs/
COPY setup.py       .
COPY pyproject.toml .
COPY requirements.txt .

# ── Copy trained model artefacts ─────────────────────────────
# EfficientNet image checkpoint
COPY artifacts/checkpoints/   ./artifacts/checkpoints/

# DistilBERT NLP checkpoint + tokenizer
COPY artifacts/checkpoints_nlp/ ./artifacts/checkpoints_nlp/

# ── Copy disease mapping used by NLP pipeline ─────────────────
COPY data/processed/disease_mapping_41.json ./data/processed/disease_mapping_41.json
COPY data/processed/disease_mapping.json    ./data/processed/disease_mapping.json

# ── Install the project package (editable-style, no deps) ────
RUN pip install --no-cache-dir --no-deps -e .

# ── Create writable runtime directories ──────────────────────
RUN mkdir -p artifacts/gradcam artifacts/evaluation logs && \
    chmod -R 777 artifacts logs

# ── Copy startup script ───────────────────────────────────────
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ── Expose ports ──────────────────────────────────────────────
EXPOSE 8000 8501

# ── Docker health-check (polls FastAPI /health endpoint) ──────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default entrypoint ────────────────────────────────────────
ENTRYPOINT ["/start.sh"]
