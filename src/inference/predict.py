"""Production-ready inference pipeline for the Image Model."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.components.model_trainer import EfficientNetClassifier, ImageClassifierConfig
from src.utils.common import load_best_checkpoint
from src.utils.exceptions import AppInferenceError, AppValidationError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class ImageInferencePipeline:
    """Production-grade inference pipeline for EfficientNet-B0 Medical Image classification."""

    CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

    # Path to the single best trained checkpoint (manually placed after training).
    BEST_CHECKPOINT = Path("artifacts/checkpoints/best_model.pth")

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/training_config.yaml",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes the inference pipeline, constructs the model architecture, and loads
        the best trained checkpoint.

        Args:
            config_path: Path to the training configuration YAML.
            checkpoint_path: Direct path to a specific .pth checkpoint file.
                             Defaults to ``artifacts/checkpoints/best_model.pth``.
        """
        logger.info("Initializing Image Inference Pipeline.")

        # ── Resolve project-relative paths and perform auto-downloads ───────
        project_root = Path(__file__).resolve().parent.parent.parent

        # Helper to convert to absolute path using project_root
        def to_absolute(p: Union[str, Path]) -> Path:
            p_path = Path(p)
            if not p_path.is_absolute():
                return project_root / p_path
            return p_path

        resolved_config_path = to_absolute(config_path)
        try:
            self.config = ImageClassifierConfig.from_yaml(resolved_config_path)
        except Exception as e:
            raise AppValidationError(
                message=f"Failed to load configuration in inference pipeline: {e}",
                details={"config_path": str(resolved_config_path)},
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Build Model Architecture
        try:
            logger.info("Building image model architecture...")
            self.model = EfficientNetClassifier(
                num_classes=self.config.num_classes,
                freeze_backbone=self.config.freeze_backbone,
                dropout=self.config.dropout,
            )
        except Exception as e:
            raise AppInferenceError(message=f"Failed to build image model architecture: {e}")

        # ── Resolve the checkpoint file path ────────────────────────────────
        from src.utils.common import download_if_needed

        best_ckpt_default = to_absolute(self.BEST_CHECKPOINT)
        if checkpoint_path is not None:
            resolved_ckpt = to_absolute(checkpoint_path)
            # Trigger dynamic auto-download if running in HF Space
            checkpoint_rel_name = "artifacts/checkpoints/" + resolved_ckpt.name
            resolved_ckpt = download_if_needed(resolved_ckpt, checkpoint_rel_name)
        else:
            resolved_ckpt = best_ckpt_default
            # Trigger dynamic auto-download if running in HF Space
            checkpoint_rel_name = "artifacts/checkpoints/" + resolved_ckpt.name
            resolved_ckpt = download_if_needed(resolved_ckpt, checkpoint_rel_name)

            if not resolved_ckpt.exists() or resolved_ckpt.stat().st_size == 0:
                ckpt_dir = resolved_ckpt.parent
                candidates = list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.pt"))
                # Filter out any files that are 0-byte placeholders
                candidates = [c for c in candidates if c.stat().st_size > 0]
                if len(candidates) == 1:
                    resolved_ckpt = candidates[0]
                    logger.info(
                        "best_model.pth not found or empty; using only available checkpoint: %s",
                        resolved_ckpt.name,
                    )
                elif len(candidates) > 1:
                    # Pick the one with the lowest val_loss among available checkpoints
                    best = None
                    best_loss = float("inf")
                    for c in candidates:
                        try:
                            data = torch.load(c, map_location="cpu", weights_only=False)
                            loss = data.get("metrics", {}).get("val_loss", float("inf"))
                            if loss < best_loss:
                                best_loss = loss
                                best = c
                        except Exception:
                            continue
                    resolved_ckpt = best if best else candidates[-1]
                    logger.info(
                        "Multiple checkpoints found; selected best by val_loss: %s",
                        resolved_ckpt.name,
                    )
                else:
                    raise AppInferenceError(
                        message=(
                            f"No valid checkpoint file found in {ckpt_dir}.\n"
                            "Place the trained checkpoint at "
                            "artifacts/checkpoints/best_model.pth and re-run inference."
                        ),
                        details={"checkpoint_dir": str(ckpt_dir)},
                    )

        logger.info("Loading checkpoint: %s", resolved_ckpt)

        # ── Load checkpoint directly ─────────────────────────────────────────
        try:
            logger.info("Loading model weights from checkpoint: %s", resolved_ckpt)
            raw = torch.load(resolved_ckpt, map_location=self.device, weights_only=False)

            saved_state = raw["model_state_dict"]
            model_keys = set(self.model.state_dict().keys())

            # Detect key prefix mismatch and remap automatically.
            # Training saved weights directly from the backbone (keys: "features.*",
            # "classifier.*"), but our EfficientNetClassifier wraps the backbone
            # under self.backbone (keys: "backbone.features.*", "backbone.classifier.*").
            # We remap transparently so neither the checkpoint nor the training
            # code ever needs to change.
            sample_ckpt_key = next(iter(saved_state))
            sample_model_key = next(iter(model_keys))
            ckpt_prefix = sample_ckpt_key.split(".")[0]
            model_prefix = sample_model_key.split(".")[0]

            if ckpt_prefix != model_prefix:
                logger.info(
                    "Key prefix mismatch detected — checkpoint prefix: '%s', "
                    "model prefix: '%s'. Remapping keys automatically.",
                    ckpt_prefix,
                    model_prefix,
                )
                saved_state = {f"{model_prefix}.{k}": v for k, v in saved_state.items()}

            self.model.load_state_dict(saved_state)

            # Extract and print metadata
            epoch = raw.get("epoch", "N/A")
            metrics = raw.get("metrics", {})
            val_loss = metrics.get("val_loss", "N/A")
            val_acc = metrics.get("val_acc", "N/A")

            print(f"checkpoint filename: {resolved_ckpt.name}")
            print(f"epoch number:        {epoch}")
            print(f"validation loss:     {val_loss}")
            print(f"validation accuracy: {val_acc}")

            # Keep metadata available for callers
            self.checkpoint_info = {
                "checkpoint_path": resolved_ckpt,
                "epoch": epoch,
                "metrics": metrics,
            }
            logger.info(
                "Checkpoint loaded — epoch %s | val_loss %s | val_acc %s",
                epoch,
                val_loss,
                val_acc,
            )

        except AppInferenceError:
            raise
        except Exception as e:
            raise AppInferenceError(
                message=f"Failed to load checkpoint {resolved_ckpt}: {e}",
                details={"checkpoint_path": str(resolved_ckpt)},
            )

        self.model.eval()

        # Load preprocessing parameters from transformation config or use defaults
        try:
            trans_config_path = to_absolute("configs/transformation_config.yaml")
            if trans_config_path.exists():
                import yaml

                with open(trans_config_path, "r", encoding="utf-8") as f:
                    cfg_trans = yaml.safe_load(f) or {}
                self.image_size = int(cfg_trans.get("image_size", 224))
                self.imagenet_mean = list(cfg_trans.get("imagenet_mean", [0.485, 0.456, 0.406]))
                self.imagenet_std = list(cfg_trans.get("imagenet_std", [0.229, 0.224, 0.225]))
            else:
                self.image_size = 224
                self.imagenet_mean = [0.485, 0.456, 0.406]
                self.imagenet_std = [0.229, 0.224, 0.225]
        except Exception as e:
            logger.warning("Failed to load transformation config, falling back to defaults: %s", e)
            self.image_size = 224
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]

        # Build the exact preprocessing pipeline used in validation/testing
        self.transform = A.Compose(
            [
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(
                    mean=tuple(self.imagenet_mean),
                    std=tuple(self.imagenet_std),
                ),
                ToTensorV2(),
            ]
        )
        logger.info(
            "Image preprocessing transforms initialized successfully (size=%d).", self.image_size
        )

    def preprocess(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocesses the input image according to validation requirements.

        Args:
            image_input: Path to the image, a PIL Image, or a NumPy array.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape [1, 3, H, W] mapped to target device.
        """
        try:
            # 1. Load image if path is passed
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.exists():
                    raise AppValidationError(message=f"Input image file does not exist: {img_path}")
                with Image.open(img_path) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    image_np = np.array(rgb_img)
            elif isinstance(image_input, Image.Image):
                rgb_img = image_input.convert("RGB")
                image_np = np.array(rgb_img)
            elif isinstance(image_input, np.ndarray):
                # Ensure it has 3 channels
                if len(image_input.shape) == 2:
                    # Grayscale to RGB
                    image_np = np.stack([image_input] * 3, axis=-1)
                elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
                    # RGBA to RGB
                    image_np = image_input[:, :, :3]
                else:
                    image_np = image_input
            else:
                raise AppValidationError(
                    message=f"Unsupported image input type: {type(image_input)}"
                )

            # 2. Apply validation transforms
            augmented = self.transform(image=image_np)
            image_tensor: torch.Tensor = augmented["image"]

            # Add batch dimension and move to device
            return image_tensor.unsqueeze(0).to(self.device)

        except Exception as e:
            if isinstance(e, AppValidationError):
                raise e
            raise AppValidationError(message=f"Failed to preprocess image input: {e}")

    def predict(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Performs disease prediction on a single medical image scan.

        Args:
            image_input: Path to the image, PIL Image, or NumPy array.

        Returns:
            Dict[str, Any]: Prediction results containing:
                - "predicted_disease": The class label string.
                - "confidence": Float score (0.0 to 1.0).
                - "class_probabilities": Dictionary mapping disease name to probability.
        """
        logger.info("Executing image model inference.")
        try:
            # 1. Preprocess input
            image_tensor = self.preprocess(image_input)

            # 2. Model inference
            with torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze(0)

            # 3. Format result
            pred_idx = int(torch.argmax(probabilities).item())
            predicted_disease = self.CLASSES[pred_idx]
            confidence = probabilities[pred_idx].item()

            class_probabilities = {
                self.CLASSES[i]: probabilities[i].item() for i in range(len(self.CLASSES))
            }

            logger.info(
                "Inference complete. Predicted: %s (Confidence: %.4f)",
                predicted_disease,
                confidence,
            )

            return {
                "predicted_disease": predicted_disease,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
            }

        except Exception as e:
            if isinstance(e, AppValidationError):
                raise e
            raise AppInferenceError(message=f"Image inference execution failed: {e}")
