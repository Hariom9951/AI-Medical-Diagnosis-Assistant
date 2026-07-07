"""Grad-CAM Explanation Generation Module.

Generates visual diagnostic attention maps for image classifier backbones.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class GradCAMExplainer:
    """Handles visual attention heatmap computation using Grad-CAM on PyTorch models."""

    def __init__(
        self, model: torch.nn.Module, target_layers: Optional[List[torch.nn.Module]] = None
    ) -> None:
        """Initializes the Grad-CAM explainer.

        Args:
            model: PyTorch classification model.
            target_layers: Optional explicit list of model target layers.
                           If None, auto-selects the last convolution block of EfficientNet.
        """
        self.model = model
        self.device = next(model.parameters()).device

        # Auto-resolve target layers for EfficientNet-B0 if not explicitly provided
        if target_layers is None:
            try:
                if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
                    # EfficientNetClassifier wrapper
                    target_layers = [model.backbone.features[-1]]
                elif hasattr(model, "features"):
                    # Standard torchvision EfficientNet
                    target_layers = [model.features[-1]]
                else:
                    logger.warning("Unsupported model architecture structure for auto-GradCAM.")
            except Exception as e:
                logger.error(f"Error resolving target layers automatically: {e}")

        self.target_layers = target_layers
        self.cam: Optional[GradCAM] = None

        if self.target_layers:
            try:
                self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
                logger.info("Grad-CAM explainer successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize GradCAM: {e}")
        else:
            logger.warning(
                "Grad-CAM explainer initialized without target layers (unsupported model)."
            )

    def generate_heatmap(
        self, input_tensor: torch.Tensor, target_class_idx: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Generates raw grayscale Grad-CAM heatmap array.

        Args:
            input_tensor: Preprocessed image tensor of shape [1, 3, H, W] on matching device.
            target_class_idx: Index of target class to visualize. If None, uses top predicted class.

        Returns:
            2D numpy array of shape (H, W) normalized to [0, 1], or None if generation failed.
        """
        if self.cam is None:
            logger.warning(
                "Grad-CAM generation skipped (explainer not fully initialized/supported)."
            )
            return None

        # Verify input shape
        if len(input_tensor.shape) != 4 or input_tensor.shape[0] != 1:
            logger.error(f"Input tensor must have shape [1, 3, H, W], got {input_tensor.shape}")
            return None

        try:
            # Force requires_grad = True on the input tensor.
            # This is critical because backbone parameters are frozen during training (requires_grad = False),
            # so intermediate activations will not track gradients unless the input tensor explicitly does.
            input_tensor = input_tensor.clone().detach().to(self.device)
            input_tensor.requires_grad = True

            # If target class index is not specified, run a forward pass to determine top class
            if target_class_idx is None:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    target_class_idx = int(torch.argmax(logits, dim=1).item())

            targets = [ClassifierOutputTarget(target_class_idx)]

            # Generate grayscale activation map
            # input_tensor must match model device and have requires_grad = True
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

            # Extract the 2D map for the batch element
            heatmap = grayscale_cam[0, :]
            return heatmap

        except Exception as e:
            logger.error(f"Error generating Grad-CAM heatmap: {e}")
            return None
