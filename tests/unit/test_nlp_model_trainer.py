"""Unit Tests for Phase 15 — NLP Symptom Classifier.

Tests cover config parsing, data preprocessing, custom text dataset,
model architecture forward pass, early stopping, trainer optimizer/scheduler
factory methods, and checkpoint save/load.
"""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from transformers import DistilBertTokenizer

from src.components.nlp_model_trainer import (
    EarlyStopping,
    NLPClassifierConfig,
    NLPClassifierTrainer,
    NLPTextDataset,
    SymptomClassifier,
    SymptomDataPreprocessor,
)
from src.utils.exceptions import AppValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary folder directory."""
    return tmp_path


@pytest.fixture
def nlp_config_yaml(temp_dir: Path) -> Path:
    """Creates a minimal valid nlp_training_config.yaml for testing."""
    cfg_content = """
model_name: "distilbert-base-uncased"
tokenizer_name: "distilbert-base-uncased"
dropout: 0.2
max_length: 16

validated_symptoms_csv: "{csv_path}"
disease_mapping_file: "{mapping_path}"

train_split: 0.70
val_split: 0.15
test_split: 0.15
random_state: 42

optimizer: "adamw"
learning_rate: 0.00002
weight_decay: 0.01

scheduler: "cosine"
step_size: 3
gamma: 0.1
t_max: 5

epochs: 2
batch_size: 2
early_stopping_patience: 3
early_stopping_min_delta: 0.001
max_grad_norm: 1.0

checkpoint_dir: "{checkpoint_dir}"
best_model_path: "{best_model_path}"

mlflow_tracking_uri: "sqlite:///{mlruns_dir}/mlflow.db"
mlflow_experiment_name: "test-nlp-experiment"
""".format(
        csv_path=str(temp_dir / "symptoms.csv").replace("\\", "/"),
        mapping_path=str(temp_dir / "mapping.json").replace("\\", "/"),
        checkpoint_dir=str(temp_dir / "checkpoints").replace("\\", "/"),
        best_model_path=str(temp_dir / "checkpoints" / "best_model.pth").replace("\\", "/"),
        mlruns_dir=str(temp_dir).replace("\\", "/"),
    )
    cfg_file = temp_dir / "nlp_training_config.yaml"
    cfg_file.write_text(cfg_content)
    return cfg_file


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_symptom_data_preprocessor(temp_dir: Path) -> None:
    """Verifies preprocessor cleans strings, handles NaNs, and sets up mapping."""
    data = {
        "Disease": ["Fungal infection", "Allergy", "Allergy"],
        "Symptom_1": ["itching", " continuous_sneezing", "shivering"],
        "Symptom_2": [" skin_rash", "shivering", None],
        "Symptom_3": [" nodal_skin_eruptions ", None, None]
    }
    df = pd.DataFrame(data)
    mapping_file = temp_dir / "disease_mapping.json"

    symptom_strings, labels, disease_mapping = SymptomDataPreprocessor.preprocess_df(
        df=df,
        disease_mapping_file=mapping_file
    )

    assert len(symptom_strings) == 3
    assert symptom_strings[0] == "itching, skin rash, nodal skin eruptions"
    assert symptom_strings[1] == "continuous sneezing, shivering"
    assert symptom_strings[2] == "shivering"
    assert labels == [1, 0, 0]  # sorted unique: Allergy=0, Fungal infection=1
    assert disease_mapping == {"Allergy": 0, "Fungal infection": 1}
    assert mapping_file.exists()


def test_symptom_data_preprocessor_invalid_input(temp_dir: Path) -> None:
    """Confirms preprocessor raises error on empty input or missing columns."""
    df_empty = pd.DataFrame()
    mapping_file = temp_dir / "disease_mapping.json"

    with pytest.raises(AppValidationError):
        SymptomDataPreprocessor.preprocess_df(df_empty, mapping_file)

    df_no_disease = pd.DataFrame({"Symptom_1": ["cough"]})
    with pytest.raises(AppValidationError):
        SymptomDataPreprocessor.preprocess_df(df_no_disease, mapping_file)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_nlp_dataset_tokenization() -> None:
    """Asserts NLPTextDataset returns token ids, attention masks, and integer label."""
    symptom_strings = ["cough, fever"]
    labels = [2]

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102, 103, 104]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]])
    }

    dataset = NLPTextDataset(
        symptom_strings=symptom_strings,
        labels=labels,
        tokenizer=mock_tokenizer,
        max_length=4
    )

    assert len(dataset) == 1
    input_ids, attention_mask, label = dataset[0]
    assert label == 2
    assert torch.equal(input_ids, torch.tensor([101, 102, 103, 104]))
    assert torch.equal(attention_mask, torch.tensor([1, 1, 1, 0]))


# ─────────────────────────────────────────────────────────────────────────────
# Model Tests
# ─────────────────────────────────────────────────────────────────────────────

@patch("src.components.nlp_model_trainer.DistilBertForSequenceClassification.from_pretrained")
@patch("src.components.nlp_model_trainer.DistilBertConfig.from_pretrained")
def test_symptom_classifier_init_and_forward(
    mock_config_from_pretrained: MagicMock,
    mock_model_from_pretrained: MagicMock
) -> None:
    """Tests model instantiation, forward mock output, and summary values."""
    mock_config = MagicMock()
    mock_config_from_pretrained.return_value = mock_config

    mock_inner_model = MagicMock()
    # Mock forward return value to act like sequence classification output
    mock_logits = MagicMock()
    mock_logits.logits = torch.zeros(2, 3)
    mock_inner_model.return_value = mock_logits

    # Set parameters mock for model summary
    mock_param = nn.Parameter(torch.zeros(5))
    mock_inner_model.parameters.return_value = [mock_param]
    mock_model_from_pretrained.return_value = mock_inner_model

    classifier = SymptomClassifier(model_name="distilbert-base-uncased", num_classes=3, dropout=0.2)
    assert classifier.num_classes == 3

    # Forward
    dummy_input = torch.zeros(2, 10, dtype=torch.long)
    dummy_mask = torch.zeros(2, 10, dtype=torch.long)
    logits = classifier(dummy_input, dummy_mask)

    assert logits.shape == (2, 3)
    summary = classifier.model_summary()
    assert summary["num_classes"] == 3
    assert summary["total_parameters"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_early_stopping_nlp() -> None:
    """Asserts early stopping stops after patience epochs without improvement."""
    es = EarlyStopping(patience=2, min_delta=0.001)
    es.step(1.0)  # best
    es.step(1.0)  # counter = 1
    es.step(1.0)  # counter = 2, stop = True
    assert es.stop is True


# ─────────────────────────────────────────────────────────────────────────────
# Trainer Factory & Checkpoints Tests
# ─────────────────────────────────────────────────────────────────────────────

@patch("src.components.nlp_model_trainer.SymptomClassifier")
def test_trainer_build_utilities(
    mock_classifier_class: MagicMock,
    nlp_config_yaml: Path
) -> None:
    """Verifies optimizer and scheduler instantiation in trainer."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    # Set parameters mock for trainer initialization
    mock_param = nn.Parameter(torch.zeros(5))
    mock_model.parameters.return_value = [mock_param]
    mock_classifier_class.return_value = mock_model

    trainer = NLPClassifierTrainer(nlp_config_yaml, num_classes=5)

    assert isinstance(trainer.optimizer, AdamW)
    assert isinstance(trainer.scheduler, CosineAnnealingLR)


@patch("src.components.nlp_model_trainer.SymptomClassifier")
def test_trainer_checkpoint_roundtrip(
    mock_classifier_class: MagicMock,
    nlp_config_yaml: Path,
    temp_dir: Path
) -> None:
    """Validates save and load round-trip of checkpoint files."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.num_classes = 5
    mock_param = nn.Parameter(torch.zeros(5))
    mock_model.parameters.return_value = [mock_param]
    state_dict = {"weights": torch.tensor([1.0, 2.0])}
    mock_model.state_dict.return_value = state_dict
    mock_classifier_class.return_value = mock_model

    trainer = NLPClassifierTrainer(nlp_config_yaml, num_classes=5)

    ckpt_path = temp_dir / "nlp_checkpoint.pth"
    metrics = {"val_acc": 0.95, "val_loss": 0.12}

    # Save
    trainer.save_checkpoint(epoch=2, metrics=metrics, path=ckpt_path)
    assert ckpt_path.exists()

    # Load
    loaded = trainer.load_checkpoint(ckpt_path)
    assert loaded["epoch"] == 2
    assert loaded["metrics"]["val_acc"] == pytest.approx(0.95)
    assert loaded["config"]["num_classes"] == 5
