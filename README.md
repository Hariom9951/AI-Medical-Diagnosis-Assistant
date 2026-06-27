# AI Medical Diagnosis Assistant

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
   git clone https://github.com/your-org/ai-medical-diagnosis-assistant.git
   cd ai-medical-diagnosis-assistant
   ```
2. Initialize virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Pull data version references:
   ```bash
   dvc pull
   ```

## 🧪 Testing
```bash
pytest tests/
```
