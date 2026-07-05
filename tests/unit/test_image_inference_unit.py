"""Unit Tests for the Image Inference Pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.predict import ImageInferencePipeline
from src.utils.exceptions import AppValidationError


@pytest.fixture
def test_config(tmp_path: Path) -> Path:
    """Fixture providing a mock training configuration YAML file."""
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
image_size: 224
imagenet_mean: [0.485, 0.456, 0.406]
imagenet_std: [0.229, 0.224, 0.225]
checkpoint_dir: "artifacts/checkpoints"
best_model_path: "artifacts/checkpoints/best_model.pth"
mlflow_tracking_uri: "sqlite:///mlflow.db"
mlflow_experiment_name: "test-experiment"
"""
    cfg_file = tmp_path / "training_config.yaml"
    cfg_file.write_text(cfg_content)
    return cfg_file


@patch("src.inference.predict.load_best_checkpoint")
def test_pipeline_initialization(
    mock_load_ckpt: MagicMock, test_config: Path, tmp_path: Path
) -> None:
    """Verifies that the inference pipeline initializes config, model, and loads checkpoint."""
    mock_load_ckpt.return_value = {"epoch": 1, "metrics": {"val_loss": 0.2}}

    pipeline = ImageInferencePipeline(
        config_path=test_config, checkpoint_dir=tmp_path
    )

    assert pipeline.checkpoint_dir == tmp_path
    assert pipeline.config.num_classes == 4
    mock_load_ckpt.assert_called_once()


@patch("src.inference.predict.load_best_checkpoint")
def test_preprocess_inputs(
    mock_load_ckpt: MagicMock, test_config: Path, tmp_path: Path
) -> None:
    """Verifies preprocessing works for NumPy array, PIL image, and grayscale inputs."""
    mock_load_ckpt.return_value = {"epoch": 1, "metrics": {"val_loss": 0.2}}
    pipeline = ImageInferencePipeline(
        config_path=test_config, checkpoint_dir=tmp_path
    )

    # 1. Test NumPy image input (H, W, C)
    np_img = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor = pipeline.preprocess(np_img)
    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(tensor, torch.Tensor)

    # 2. Test PIL Image input
    pil_img = Image.new("RGB", (100, 100))
    tensor2 = pipeline.preprocess(pil_img)
    assert tensor2.shape == (1, 3, 224, 224)

    # 3. Test grayscale NumPy image input (H, W)
    gray_img = np.zeros((100, 100), dtype=np.uint8)
    tensor3 = pipeline.preprocess(gray_img)
    assert tensor3.shape == (1, 3, 224, 224)


@patch("src.inference.predict.load_best_checkpoint")
def test_predict_flow(
    mock_load_ckpt: MagicMock, test_config: Path, tmp_path: Path
) -> None:
    """Verifies predict returns predicted disease, confidence, and class probabilities."""
    mock_load_ckpt.return_value = {"epoch": 1, "metrics": {"val_loss": 0.2}}
    pipeline = ImageInferencePipeline(
        config_path=test_config, checkpoint_dir=tmp_path
    )

    # Mock the forward pass of model to return constant logits where COVID wins
    pipeline.model = MagicMock()
    pipeline.model.return_value = torch.tensor([[10.0, 1.0, 1.0, 1.0]])

    np_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = pipeline.predict(np_img)

    assert res["predicted_disease"] == "COVID"
    assert res["confidence"] > 0.99
    assert "COVID" in res["class_probabilities"]
    assert len(res["class_probabilities"]) == 4


@patch("src.inference.predict.load_best_checkpoint")
def test_predict_invalid_image_path(
    mock_load_ckpt: MagicMock, test_config: Path, tmp_path: Path
) -> None:
    """Verifies validation error is raised if input image file path does not exist."""
    mock_load_ckpt.return_value = {"epoch": 1, "metrics": {"val_loss": 0.2}}
    pipeline = ImageInferencePipeline(
        config_path=test_config, checkpoint_dir=tmp_path
    )

    with pytest.raises(AppValidationError) as exc:
        pipeline.predict(tmp_path / "nonexistent.png")
    assert "does not exist" in str(exc.value)


@patch("src.inference.predict.load_best_checkpoint")
def test_predict_unsupported_type(
    mock_load_ckpt: MagicMock, test_config: Path, tmp_path: Path
) -> None:
    """Verifies validation error is raised when input image type is not supported."""
    mock_load_ckpt.return_value = {"epoch": 1, "metrics": {"val_loss": 0.2}}
    pipeline = ImageInferencePipeline(
        config_path=test_config, checkpoint_dir=tmp_path
    )

    with pytest.raises(AppValidationError) as exc:
        pipeline.predict(12345)  # type: ignore[arg-type]
    assert "Unsupported image input type" in str(exc.value)
