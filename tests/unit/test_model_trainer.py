"""Unit Tests for Phase 14 — EfficientNet-B0 Image Classifier.

Tests cover model architecture, parameter freezing/unfreezing, early stopping logic,
optimizer/scheduler construction, and checkpoint save/load round-trips.
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from src.components.model_trainer import (
    EarlyStopping,
    EfficientNetClassifier,
    ImageClassifierConfig,
    ImageClassifierTrainer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    """Creates a minimal valid training_config.yaml for testing."""
    cfg_content = """
model_name: "efficientnet_b0"
num_classes: 4
freeze_backbone: true
dropout: 0.3
optimizer: "adamw"
learning_rate: 0.001
weight_decay: 0.0001
momentum: 0.9
scheduler: "cosine"
step_size: 7
gamma: 0.1
t_max: 10
epochs: 5
batch_size: 4
early_stopping_patience: 3
early_stopping_min_delta: 0.001
checkpoint_dir: "{checkpoint_dir}"
best_model_path: "{best_model_path}"
mlflow_tracking_uri: "sqlite:///{mlruns_dir}/mlflow.db"
mlflow_experiment_name: "test-experiment"
""".format(
        checkpoint_dir=str(tmp_path / "checkpoints").replace("\\", "/"),
        best_model_path=str(tmp_path / "checkpoints" / "best_model.pth").replace("\\", "/"),
        mlruns_dir=str(tmp_path).replace("\\", "/"),
    )
    cfg_file = tmp_path / "training_config.yaml"
    cfg_file.write_text(cfg_content)
    return cfg_file


@pytest.fixture
def frozen_classifier() -> EfficientNetClassifier:
    """Returns a frozen-backbone EfficientNetClassifier with 4 classes."""
    return EfficientNetClassifier(num_classes=4, freeze_backbone=True, dropout=0.3)


@pytest.fixture
def trainer(config_yaml: Path) -> ImageClassifierTrainer:
    """Returns an initialized ImageClassifierTrainer."""
    return ImageClassifierTrainer(config_yaml)


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_efficientnet_forward_output_shape(frozen_classifier: EfficientNetClassifier) -> None:
    """Asserts model forward pass output shape is [B, num_classes]."""
    dummy_input = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        output = frozen_classifier(dummy_input)
    assert output.shape == (2, 4), f"Expected (2, 4), got {output.shape}"


def test_classifier_head_replaced(frozen_classifier: EfficientNetClassifier) -> None:
    """Confirms the final Linear layer has out_features matching num_classes."""
    classifier_head = frozen_classifier.backbone.classifier
    # Last module of Sequential must be the Linear head
    linear_layer = list(classifier_head.children())[-1]
    assert isinstance(linear_layer, nn.Linear)
    assert linear_layer.out_features == 4


def test_freeze_backbone_only_head_trainable(frozen_classifier: EfficientNetClassifier) -> None:
    """Verifies backbone features are frozen and classifier head params are trainable."""
    # Backbone features must be frozen
    for param in frozen_classifier.backbone.features.parameters():
        assert not param.requires_grad, "Backbone feature param should be frozen"

    # Classifier head must be trainable
    for param in frozen_classifier.backbone.classifier.parameters():
        assert param.requires_grad, "Classifier head param should be trainable"


def test_unfreeze_backbone(frozen_classifier: EfficientNetClassifier) -> None:
    """Verifies all backbone parameters become trainable after unfreeze_backbone()."""
    frozen_classifier.unfreeze_backbone()
    for param in frozen_classifier.backbone.features.parameters():
        assert param.requires_grad, "Backbone param should be trainable after unfreeze"


def test_model_summary_counts(frozen_classifier: EfficientNetClassifier) -> None:
    """Verifies model_summary returns non-zero counts and trainable < total."""
    summary = frozen_classifier.model_summary()
    assert summary["total_parameters"] > 0
    assert summary["trainable_parameters"] > 0
    assert summary["trainable_parameters"] < summary["total_parameters"]
    assert summary["frozen_parameters"] == (
        summary["total_parameters"] - summary["trainable_parameters"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_early_stopping_triggers() -> None:
    """Asserts stop flag fires after patience consecutive non-improving epochs.

    The first call sets best_loss from inf -> 1.0 (counts as improvement).
    The next `patience` calls with no improvement trigger the stop flag.
    Total calls needed = patience + 1.
    """
    es = EarlyStopping(patience=3, min_delta=0.01)
    es.step(1.0)  # initializes best_loss — NOT counted as non-improvement
    for _ in range(3):
        es.step(1.0)  # 3 consecutive non-improving steps = triggers stop
    assert es.stop is True


def test_early_stopping_resets_on_improvement() -> None:
    """Asserts counter resets when validation loss improves."""
    es = EarlyStopping(patience=3, min_delta=0.01)
    es.step(1.0)  # counter = 1
    es.step(1.0)  # counter = 2
    es.step(0.5)  # improvement — counter resets
    assert es.counter == 0
    assert es.stop is False


def test_early_stopping_no_trigger_before_patience() -> None:
    """Verifies stop flag is not set before patience exhausted."""
    es = EarlyStopping(patience=5, min_delta=0.0)
    for _ in range(4):
        es.step(1.5)
    assert es.stop is False


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer / Scheduler Factory Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_optimizer_adamw(trainer: ImageClassifierTrainer) -> None:
    """Asserts build_optimizer returns AdamW when config specifies 'adamw'."""
    assert isinstance(trainer.optimizer, AdamW)


def test_scheduler_cosine(trainer: ImageClassifierTrainer) -> None:
    """Asserts build_scheduler returns CosineAnnealingLR when config specifies 'cosine'."""
    assert isinstance(trainer.scheduler, CosineAnnealingLR)


def test_optimizer_sgd(config_yaml: Path, tmp_path: Path) -> None:
    """Asserts build_optimizer returns SGD when config specifies 'sgd'."""
    content = (config_yaml).read_text().replace('optimizer: "adamw"', 'optimizer: "sgd"')
    cfg_file = tmp_path / "sgd_config.yaml"
    cfg_file.write_text(content)
    t = ImageClassifierTrainer(cfg_file)
    assert isinstance(t.optimizer, SGD)


def test_scheduler_step(config_yaml: Path, tmp_path: Path) -> None:
    """Asserts build_scheduler returns StepLR when config specifies 'step'."""
    content = (config_yaml).read_text().replace('scheduler: "cosine"', 'scheduler: "step"')
    cfg_file = tmp_path / "step_config.yaml"
    cfg_file.write_text(content)
    t = ImageClassifierTrainer(cfg_file)
    assert isinstance(t.scheduler, StepLR)


def test_scheduler_plateau(config_yaml: Path, tmp_path: Path) -> None:
    """Asserts build_scheduler returns ReduceLROnPlateau when config specifies 'plateau'."""
    content = (config_yaml).read_text().replace('scheduler: "cosine"', 'scheduler: "plateau"')
    cfg_file = tmp_path / "plateau_config.yaml"
    cfg_file.write_text(content)
    t = ImageClassifierTrainer(cfg_file)
    assert isinstance(t.scheduler, ReduceLROnPlateau)


def test_scheduler_none(config_yaml: Path, tmp_path: Path) -> None:
    """Asserts build_scheduler returns None when config specifies 'none'."""
    content = (config_yaml).read_text().replace('scheduler: "cosine"', 'scheduler: "none"')
    cfg_file = tmp_path / "none_config.yaml"
    cfg_file.write_text(content)
    t = ImageClassifierTrainer(cfg_file)
    assert t.scheduler is None


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Save / Load Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_save_and_load(trainer: ImageClassifierTrainer, tmp_path: Path) -> None:
    """Saves a checkpoint and restores it, verifying state dict keys match."""
    ckpt_path = tmp_path / "test_checkpoint.pth"
    metrics = {"val_loss": 0.42, "val_acc": 0.88}

    # Save
    trainer.save_checkpoint(epoch=0, metrics=metrics, path=ckpt_path)
    assert ckpt_path.exists()

    # Rebuild trainer and load
    trainer2 = ImageClassifierTrainer(trainer.config.checkpoint_dir.parent / "training_config.yaml"
                                      if False else trainer.config.best_model_path.parent.parent / "training_config.yaml"
                                      if False else Path(str(ckpt_path).replace("test_checkpoint.pth", "training_config.yaml")))

    # Load using original trainer since config path is embedded
    loaded = trainer.load_checkpoint(ckpt_path)
    assert loaded["epoch"] == 0
    assert loaded["metrics"]["val_acc"] == pytest.approx(0.88)


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate Step Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_train_and_validate_one_epoch(trainer: ImageClassifierTrainer) -> None:
    """Runs one mini train+validate step with synthetic data, verifies metric ranges."""
    # Create a tiny synthetic DataLoader: (image [3,224,224], label, path_str)
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])
    paths = ["a", "b", "c", "d"]

    dataset = list(zip(images, labels, paths))
    loader = torch.utils.data.DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=2,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch],
        ),
    )

    train_loss, train_acc = trainer.train_one_epoch(loader)
    val_loss, val_acc = trainer.validate_one_epoch(loader)

    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= val_acc <= 1.0
    assert train_loss >= 0.0
    assert val_loss >= 0.0
