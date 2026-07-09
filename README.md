---
title: AI Medical Diagnosis Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AI Medical Diagnosis Assistant

[![CI/CD Pipeline](https://github.com/Hariom9951/AI-Medical-Diagnosis-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Hariom9951/AI-Medical-Diagnosis-Assistant/actions/workflows/ci.yml)

An enterprise-grade, clean-architecture clinical decision support system that predicts diseases using patient symptoms and medical images. Featuring visual explanation overlay via Grad-CAM and automated clinical PDF report generation.

## 🚀 Key Features

- **Multimodal Prediction**: Merges vision data (chest X-rays) and symptom text for comprehensive diagnosis.
- **Explainability**: Focus visualizations using Grad-CAM heatmaps.
- **Automation**: Instantly compiles clinical PDF reports.
- **Lineage Tracking**: Powered by DVC and MLflow.
- **Production-Grade Deployment**: Automatic model checkpoint management via Hugging Face Hub.

---

## 🗂️ Model Repository Architecture

Model checkpoints are stored **outside this repository** in a dedicated Hugging Face Model Repository:

> **[Hariom9951/AI-Medical-Diagnosis-Models](https://huggingface.co/Hariom9951/AI-Medical-Diagnosis-Models)**

### Stored Artifacts

| Path in HF Repo | Description |
|---|---|
| `image/best_model.pth` | EfficientNet-B0 best image classifier checkpoint |
| `image/checkpoint_epoch_050.pth` | EfficientNet-B0 epoch 50 checkpoint |
| `nlp/best_model.pt` | Fine-tuned BioBERT symptom classifier |
| `nlp/tokenizer.json` | Saved tokenizer vocabulary |
| `nlp/tokenizer_config.json` | Tokenizer configuration |
| `nlp/label_encoder.pkl` | Scikit-learn LabelEncoder with 41 disease classes |
| `nlp/model_metadata.json` | Model training metadata |
| `nlp/temperature_scaler.json` | Confidence calibration temperature |
| `nlp/clinical_explanations.json` | Clinical explanation registry per disease |

### How It Works

The `ModelDownloader` service ([`src/utils/downloader.py`](src/utils/downloader.py)) handles:

1. **First startup** — Downloads missing files from the HF model repo with SHA256 verification
2. **Subsequent starts** — Verifies local cache; skips re-download if checksums match
3. **Retry logic** — Automatic retry with exponential backoff on network failures
4. **Cross-platform** — Works on Windows (local dev), Docker, Render, and Hugging Face Spaces

---

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.11
- Git
- Docker & Docker Compose (for containerised deployment)
- [Hugging Face account](https://huggingface.co) with a read-access API token

### 1. Clone the Repository

```bash
git clone https://github.com/Hariom9951/AI-Medical-Diagnosis-Assistant.git
cd AI-Medical-Diagnosis-Assistant
```

### 2. Initialize Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt -r requirements-dev.txt
```

### 3. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Hugging Face API token (read access is sufficient)
HF_TOKEN=hf_your_token_here

# (Optional) Override the model repository if needed
HF_MODEL_REPO_ID=Hariom9951/AI-Medical-Diagnosis-Models
```

### 4. Download Model Checkpoints (First Time)

The `ModelDownloader` service fetches all model artifacts automatically on first startup.
You can also trigger it manually:

```bash
python - <<'EOF'
from src.utils.downloader import ModelDownloader
from pathlib import Path

dl = ModelDownloader()
files = [
    ("image/best_model.pth",           "artifacts/checkpoints/best_model.pth"),
    ("image/checkpoint_epoch_050.pth", "artifacts/checkpoints/checkpoint_epoch_050.pth"),
    ("nlp/best_model.pt",              "artifacts/checkpoints_nlp/best_model.pt"),
    ("nlp/tokenizer.json",             "artifacts/checkpoints_nlp/tokenizer.json"),
    ("nlp/tokenizer_config.json",      "artifacts/checkpoints_nlp/tokenizer_config.json"),
    ("nlp/label_encoder.pkl",          "artifacts/checkpoints_nlp/label_encoder.pkl"),
    ("nlp/model_metadata.json",        "artifacts/checkpoints_nlp/model_metadata.json"),
    ("nlp/temperature_scaler.json",    "artifacts/checkpoints_nlp/temperature_scaler.json"),
    ("nlp/clinical_explanations.json", "artifacts/checkpoints_nlp/clinical_explanations.json"),
]
for repo_path, local_path in files:
    dl.download_file(repo_path, local_path)
    print(f"✅ {repo_path}")
EOF
```

### 5. Run the Application

```bash
streamlit run src/frontend/app.py
```

---

## 🧪 Testing and Code Quality

### Code Formatting Checks

```bash
black --check src tests
isort --check src tests

# Auto-format
black src tests
isort src tests
```

### Static Analysis & Linting

```bash
flake8 src tests
mypy src
```

### Unit Tests

```bash
python -m pytest tests/unit -v
```

### Integration Tests

```bash
# Run with mocked downloads (no network required)
CI_MOCK_DOWNLOADER=true python -m pytest tests/integration -v

# Run with real model downloads (requires HF_TOKEN and network access)
python -m pytest tests/integration -v -m "not slow"
```

---

## 🐳 Docker Deployment

The application uses a multi-stage Docker build. **Model checkpoints are not baked into the image** — they are downloaded at container startup via the `ModelDownloader` service.

### Required Secrets / Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes (private repo) | Hugging Face read token |
| `HF_MODEL_REPO_ID` | No | Override model repo (default: `Hariom9951/AI-Medical-Diagnosis-Models`) |

### 1. Build the Docker Image

```bash
docker build -t ai-medical-diagnosis-assistant:latest .
```

### 2. Run Locally

```bash
docker run -d \
  --name medical-assistant-app \
  -p 7860:7860 \
  -e HF_TOKEN=hf_your_token_here \
  -e HF_MODEL_REPO_ID=Hariom9951/AI-Medical-Diagnosis-Models \
  ai-medical-diagnosis-assistant:latest
```

On **first** startup, models are automatically downloaded (may take a few minutes for large files).
On **subsequent** starts, cached and verified files are reused immediately.

### 3. Health Check

```bash
curl -f http://localhost:7860/_stcore/health
```

---

## ⚙️ GitHub Actions CI/CD Pipeline

### CI Workflow (`ci.yml`)

Triggers on all pushes and pull requests to `main`:

1. **Environment Setup** — Python 3.11, dependency caching
2. **Format Validation** — `black` and `isort` dry-runs
3. **Lint Verification** — `flake8` and `mypy`
4. **Unit Testing** — `pytest tests/unit`
5. **Integration Testing** — Mocked downloader (`CI_MOCK_DOWNLOADER=true`)
6. **Docker Build & Health Check** — Image built without checkpoint files; Streamlit started with `CI_SKIP_MODEL_DOWNLOAD=true`

### HuggingFace Deploy Workflow (`deploy-huggingface.yml`)

Triggers on push to `main`. Syncs code to HF Spaces (no model binaries are uploaded):

1. Clones the HF Space repo
2. Rsyncs source code **excluding** all `.pth`, `.pt`, `.bin`, `.pkl` files
3. Creates empty `artifacts/checkpoints/` and `artifacts/checkpoints_nlp/` directories
4. Writes `HF_MODEL_REPO_ID` to Space configuration
5. Commits and pushes to the Space

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `HF_TOKEN` | Hugging Face API token |
| `HF_USERNAME` | Hugging Face username |

---

## 📁 System Architecture

```
AI-Medical-Diagnosis-Assistant/          ← Application repository (no model weights)
├── src/
│   ├── inference/
│   │   ├── predict.py                   ← Image inference pipeline
│   │   └── nlp_predict.py              ← NLP inference pipeline
│   └── utils/
│       ├── downloader.py               ← ModelDownloader service
│       └── common.py                   ← download_if_needed() bridge
├── tests/
│   ├── unit/                           ← Unit tests
│   └── integration/                    ← Integration tests (mocked + real)
├── .github/workflows/
│   ├── ci.yml                          ← CI pipeline
│   └── deploy-huggingface.yml          ← HF Spaces deployment
└── Dockerfile                          ← Multi-stage build (no model weights baked in)

Hariom9951/AI-Medical-Diagnosis-Models  ← Separate HF Model Repository
├── image/
│   ├── best_model.pth
│   └── checkpoint_epoch_050.pth
└── nlp/
    ├── best_model.pt
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── label_encoder.pkl
    ├── model_metadata.json
    ├── temperature_scaler.json
    └── clinical_explanations.json
```

---

## 🔐 Uploading Models to Hugging Face

To upload your trained checkpoints to the model repository:

```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()

repo_id = "Hariom9951/AI-Medical-Diagnosis-Models"

# Create repository if it doesn't exist
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Upload image checkpoints
api.upload_file(path_or_fileobj="artifacts/checkpoints/best_model.pth",
                path_in_repo="image/best_model.pth", repo_id=repo_id)
api.upload_file(path_or_fileobj="artifacts/checkpoints/checkpoint_epoch_050.pth",
                path_in_repo="image/checkpoint_epoch_050.pth", repo_id=repo_id)

# Upload NLP checkpoints and tokenizer
for fname in ["best_model.pt", "tokenizer.json", "tokenizer_config.json",
              "label_encoder.pkl", "model_metadata.json",
              "temperature_scaler.json", "clinical_explanations.json"]:
    api.upload_file(path_or_fileobj=f"artifacts/checkpoints_nlp/{fname}",
                    path_in_repo=f"nlp/{fname}", repo_id=repo_id)
    print(f"✅ Uploaded nlp/{fname}")
print("All models uploaded successfully!")
EOF
```

---

## 📄 License

This project is licensed under the MIT License.
