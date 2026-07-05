"""Unit Tests for Phase 15 — Model Evaluation.

Tests cover model loading, metric calculation, reports writing, and MLflow logging.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.components.model_evaluation import ImageClassifierEvaluator
from src.utils.exceptions import AppStorageError


# ─────────────────────────────────────────────────────────────────────────────
# Toy Dataset for Testing
# ─────────────────────────────────────────────────────────────────────────────

class DummyEvalDataset(Dataset):
    """Simple toy dataset that yields deterministic tensors for evaluation testing."""

    def __init__(self, size: int = 10) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        # Return 3-channel image, label (0 to 3), and dummy file path string
        dummy_img = torch.ones((3, 224, 224), dtype=torch.float32) * (idx / self.size)
        label = idx % 4
        path = f"dummy/path/{idx}.png"
        return dummy_img, label, path


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    """Creates a minimal valid training_config.yaml for testing."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_content = f"""
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
checkpoint_dir: "{checkpoint_dir.as_posix()}"
best_model_path: "{(checkpoint_dir / 'best_model.pth').as_posix()}"
mlflow_tracking_uri: "sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
mlflow_experiment_name: "test-evaluation-experiment"
"""
    cfg_file = tmp_path / "training_config.yaml"
    cfg_file.write_text(cfg_content)
    return cfg_file


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluator_init(config_yaml: Path, tmp_path: Path) -> None:
    """Verifies that ImageClassifierEvaluator initializes correctly."""
    reports_dir = tmp_path / "reports"
    evaluator = ImageClassifierEvaluator(
        training_config_path=config_yaml,
        reports_dir=reports_dir,
        class_names=["ClassA", "ClassB", "ClassC", "ClassD"]
    )
    assert evaluator.config.num_classes == 4
    assert evaluator.reports_dir == reports_dir
    assert evaluator.class_names == ["ClassA", "ClassB", "ClassC", "ClassD"]


def test_evaluator_missing_checkpoint_raises_error(config_yaml: Path, tmp_path: Path) -> None:
    """Asserts that AppStorageError is raised if the best model checkpoint file does not exist."""
    reports_dir = tmp_path / "reports"
    evaluator = ImageClassifierEvaluator(training_config_path=config_yaml, reports_dir=reports_dir)
    
    # Checkpoint doesn't exist, so loading should raise AppStorageError
    with pytest.raises(AppStorageError) as exc_info:
        evaluator.load_best_model()
    assert "Best model checkpoint not found" in str(exc_info.value)


@patch("src.components.model_evaluation.EfficientNetClassifier")
@patch("src.components.model_evaluation.torch.load")
def test_evaluator_load_best_model_success(
    mock_torch_load: MagicMock,
    mock_classifier: MagicMock,
    config_yaml: Path,
    tmp_path: Path
) -> None:
    """Verifies evaluator loads model weights successfully from checkpoint file."""
    # Write a dummy checkpoint file so the check path.exists() succeeds
    best_path = Path(tmp_path / "checkpoints" / "best_model.pth")
    best_path.touch()

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_classifier.return_value = mock_model
    mock_torch_load.return_value = {"model_state_dict": {"dummy": 1}}

    reports_dir = tmp_path / "reports"
    evaluator = ImageClassifierEvaluator(training_config_path=config_yaml, reports_dir=reports_dir)
    model = evaluator.load_best_model()
    
    # Assert
    assert model == mock_model
    mock_torch_load.assert_called_once_with(best_path, map_location=evaluator.device, weights_only=False)
    mock_model.load_state_dict.assert_called_once_with({"dummy": 1})
    mock_model.eval.assert_called_once()


def test_calculate_metrics() -> None:
    """Tests metric calculation function using deterministic numpy arrays."""
    # Setup dummy arrays: 8 samples, 4 classes
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 0])  # last prediction is wrong
    
    # Probabilities: 8 samples, 4 classes
    y_prob = np.zeros((8, 4))
    for i, val in enumerate(y_pred):
        y_prob[i, val] = 0.9
        for other in range(4):
            if other != val:
                y_prob[i, other] = 0.033

    evaluator = MagicMock(spec=ImageClassifierEvaluator)
    evaluator.class_names = ["A", "B", "C", "D"]
    
    metrics = ImageClassifierEvaluator._calculate_metrics(evaluator, y_true, y_pred, y_prob)

    assert metrics["test_accuracy"] == 7 / 8
    assert "macro_roc_auc" in metrics
    assert "weighted_roc_auc" in metrics
    assert "class_metrics" in metrics
    assert metrics["class_metrics"]["A"]["support"] == 2
    assert metrics["class_metrics"]["D"]["support"] == 2


@patch("src.components.model_evaluation.ImageClassifierEvaluator.load_best_model")
@patch("src.components.model_evaluation.mlflow")
def test_evaluate_and_save_artifacts(
    mock_mlflow: MagicMock,
    mock_load_model: MagicMock,
    config_yaml: Path,
    tmp_path: Path
) -> None:
    """Verifies complete evaluation loop: reports and plots are written, and MLflow is called."""
    # 1. Setup mock model that predicts labels deterministically
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    # Model forward pass outputs logits
    # We will pass 8 inputs, returning logits that argmax to 0, 1, 2, 3
    dummy_logits = torch.zeros((4, 4))
    for i in range(4):
        dummy_logits[i, i] = 10.0 # High value for class i
    
    # Dataloader with batch size 4, size 8
    dataset = DummyEvalDataset(size=8)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Make the mock model return appropriate logits for each batch
    mock_model.side_effect = [dummy_logits, dummy_logits]

    reports_dir = tmp_path / "reports"
    evaluator = ImageClassifierEvaluator(training_config_path=config_yaml, reports_dir=reports_dir)
    
    # 2. Run evaluation
    metrics = evaluator.evaluate(test_loader)
    
    # 3. Assertions on calculated metrics
    assert metrics["test_accuracy"] == 1.0  # Perfect predictions in our mock setup
    
    # 4. Verify output files exist
    assert (reports_dir / "Metrics.json").exists()
    assert (reports_dir / "Classification_Report.csv").exists()
    assert (reports_dir / "Evaluation_Report.md").exists()
    assert (reports_dir / "confusion_matrix.png").exists()
    assert (reports_dir / "confusion_matrix_normalized.png").exists()
    assert (reports_dir / "roc_curves.png").exists()
    assert (reports_dir / "precision_recall_curves.png").exists()

    # 5. Check markdown content
    md_content = (reports_dir / "Evaluation_Report.md").read_text()
    assert "Model Evaluation Report" in md_content
    assert "Total Test Accuracy" in md_content
    assert "COVID" in md_content
    assert "Lung_Opacity" in md_content

    # 6. Verify MLflow logging is called
    mock_mlflow.log_metrics.assert_called()
    mock_mlflow.log_artifact.assert_called()
