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
- **Multimodal Prediction**: Merges vision data and symptoms.
- **Explainability**: Focus visualizations using Grad-CAM heatmaps.
- **Automation**: Instantly compiles PDFs for clinicians.
- **Lineage Tracking**: Powered by DVC and MLflow.

## 📁 System Architecture
Refer to our software architecture guidelines in the project documentation.
- `src/domain/`: Core business models.
- `src/infrastructure/`: Concrete database and ML service providers.

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.11
- Git & DVC
- Docker & Docker Compose

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Hariom9951/AI-Medical-Diagnosis-Assistant.git
   cd AI-Medical-Diagnosis-Assistant
   ```
2. Initialize virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install --upgrade pip wheel setuptools
   pip install -r requirements.txt -r requirements-dev.txt
   ```
3. Pull data version references:
   ```bash
   dvc pull
   ```

## 🧪 Testing and Code Quality

You can validate code formatting, linting rules, and execute tests locally using the following commands:

### 1. Code Formatting Checks
Run Black and Isort to check if code meets formatting guidelines:
```bash
# Check formatting
black --check src tests
isort --check src tests

# Auto-format files
black src tests
isort src tests
```

### 2. Static Analysis & Linting
Run Flake8 and Mypy to verify code syntax, style conformity, and static typing correctness:
```bash
# Check code style and PEP8 violations
flake8 src tests

# Verify static typing
mypy src tests
```

### 3. Unit Tests
Run the Pytest suite to execute all unit tests:
```bash
python -m pytest tests/unit
```

## 🐳 Docker Deployment & Validation

The application contains a multi-stage production Dockerfile configured for offline inference mode. 

### 1. Build the Docker Image
To test building the container locally, execute:
```bash
docker build -t ai-medical-diagnosis-assistant:latest .
```

### 2. Run the Container Local Validation
Run the application container locally to verify successful startup and health check status:
```bash
# Run container in detached mode exposing default ports
docker run -d --name medical-assistant-app -p 8000:8000 -p 8501:8501 ai-medical-diagnosis-assistant:latest

# Wait 5 seconds and poll the health check endpoint
curl -f http://localhost:8000/health

# Clean up container
docker stop medical-assistant-app
docker rm medical-assistant-app
```

## ⚙️ GitHub Actions CI/CD Pipeline

The project includes an automated GitHub Actions CI workflow named `ci.yml` that triggers on:
- All pushes to the `main` branch
- All pull requests targeting the `main` branch

### Workflow Stages:
1. **Checkout & Environment Setup**: Checks out the repository and sets up Python 3.11 with automatic dependency caching (`actions/cache`).
2. **Dependency Installation**: Upgrades packaging utilities and installs dependencies (caching them to accelerate future runs).
3. **Format Validation**: Runs `black` and `isort` dry-runs to enforce visual style rules.
4. **Lint Verification**: Executes `flake8` and `mypy` checks to capture syntax bugs or type mismatches.
5. **Unit Testing**: Runs the unit tests suite via `pytest`.
6. **Container Verification**: Automatically generates mocks for weights/checkpoints and verifies that `docker build` builds and boots the runtime container successfully.
