"""Unit Tests for Phase 15 — NLP Symptom Classification Pipeline.

Tests cover pipeline execution, splitting, dataloader construction,
and saving of artifacts.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from src.pipeline.nlp_training_pipeline import NLPTrainingPipeline


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary folder directory."""
    return tmp_path


@pytest.fixture
def test_pipeline_config(temp_dir: Path) -> Path:
    """Creates a minimal config for pipeline testing."""
    csv_path = temp_dir / "symptoms_cleaned.csv"
    mapping_path = temp_dir / "disease_mapping.json"
    checkpoint_dir = temp_dir / "checkpoints_nlp"

    # Create dummy symptom CSV
    data = {
        "Disease": ["COVID", "Allergy", "Normal", "Viral Pneumonia"] * 5,
        "Symptom_1": ["cough", "sneezing", "none", "fever"] * 5,
        "Symptom_2": ["fever", "shivering", "none", "cough"] * 5,
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    cfg_content = f"""
model_name: "distilbert-base-uncased"
tokenizer_name: "distilbert-base-uncased"
dropout: 0.2
max_length: 8

validated_symptoms_csv: "{str(csv_path).replace('\\', '/')}"
disease_mapping_file: "{str(mapping_path).replace('\\', '/')}"

train_split: 0.50
val_split: 0.25
test_split: 0.25
random_state: 42

optimizer: "adamw"
learning_rate: 0.00002
weight_decay: 0.01

scheduler: "cosine"
step_size: 2
gamma: 0.1
t_max: 2

epochs: 2
batch_size: 2
early_stopping_patience: 2
early_stopping_min_delta: 0.001
max_grad_norm: 1.0

checkpoint_dir: "{str(checkpoint_dir).replace('\\', '/')}"
best_model_path: "{str(checkpoint_dir / 'best_model.pth').replace('\\', '/')}"

mlflow_tracking_uri: "sqlite:///{str(temp_dir / 'mlflow.db').replace('\\', '/')}"
mlflow_experiment_name: "test-pipeline-experiment"
"""
    cfg_file = temp_dir / "nlp_training_config.yaml"
    cfg_file.write_text(cfg_content)
    return cfg_file


@patch("src.pipeline.nlp_training_pipeline.NLPClassifierTrainer")
@patch("src.pipeline.nlp_training_pipeline.DistilBertTokenizer.from_pretrained")
@patch("src.components.nlp_model_trainer.mlflow")
def test_nlp_training_pipeline_run(
    mock_mlflow: MagicMock,
    mock_tokenizer_class: MagicMock,
    mock_trainer_class: MagicMock,
    test_pipeline_config: Path
) -> None:
    """Verifies that NLPTrainingPipeline executes steps, creates loaders, and saves artifacts."""
    # 1. Setup mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 30522
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros((1, 8), dtype=torch.long),
        "attention_mask": torch.ones((1, 8), dtype=torch.long)
    }
    mock_tokenizer.save_pretrained = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    mock_tokenizer_class.return_value = mock_tokenizer

    # 2. Setup mock trainer
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = [
        {"train_loss": 0.5, "train_acc": 0.6, "val_loss": 0.4, "val_acc": 0.7, "learning_rate": 2e-5, "epoch_time_s": 1.2},
        {"train_loss": 0.3, "train_acc": 0.8, "val_loss": 0.2, "val_acc": 0.9, "learning_rate": 2e-5, "epoch_time_s": 1.1}
    ]
    mock_trainer_class.return_value = mock_trainer

    # Initialize and execute pipeline
    pipeline = NLPTrainingPipeline(config_path=test_pipeline_config)
    results = pipeline.run(max_epochs=2)

    # 3. Asserts on outputs
    assert results["num_classes"] == 4
    assert results["vocab_size"] == 30522
    assert len(results["training_history"]) == 2
    assert results["training_history"][-1]["val_acc"] == 0.9

    # Check files created in checkpoints
    chk_dir = Path(results["tokenizer_path"]).parent
    assert chk_dir.exists()
    assert (chk_dir / "training_curves.png").exists()
    assert (chk_dir / "nlp_training_config_snapshot.yaml").exists()
    assert (chk_dir / "tokenizer").exists()

    # Validate calls
    mock_tokenizer_class.assert_called_once_with("distilbert-base-uncased")
    mock_trainer.train.assert_called_once()
