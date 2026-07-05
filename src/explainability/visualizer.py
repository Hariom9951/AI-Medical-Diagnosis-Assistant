"""Grad-CAM Heatmap Visualization Module.

Provides utilities to post-process, overlay, save, and visualize
Grad-CAM attention heatmaps on top of medical images.
"""

import uuid
from pathlib import Path
from typing import Any, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class GradCAMVisualizer:
    """Handles image processing, color mapping, and saving of Grad-CAM visualizations."""

    def __init__(self, output_dir: Union[str, Path] = "artifacts/gradcam") -> None:
        """Initializes the visualizer and ensures output directory exists.

        Args:
            output_dir: Target directory where visualization images will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_visualization(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[Image.Image, Image.Image, Path]:
        """Applies colormap, overlays it on the original image, and saves results.

        Args:
            original_image: Original RGB PIL image.
            heatmap: Grayscale attention heatmap from Grad-CAM (shape HxW, values 0.0-1.0).
            alpha: Transparency weight of the original image (0.0 to 1.0).

        Returns:
            Tuple containing:
              - PIL Image of the JET colormap heatmap
              - PIL Image of the overlay image
              - Path to the saved overlay file
        """
        # Ensure original image is RGB
        original_image = original_image.convert("RGB")
        original_np = np.array(original_image)
        h, w, _ = original_np.shape

        # 1. Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # 2. Convert grayscale heatmap to BGR JET colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # 3. Perform overlay math using BGR representations
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(original_bgr, alpha, heatmap_bgr, 1.0 - alpha, 0)

        # 4. Convert back to RGB for PIL representation
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        heatmap_pil = Image.fromarray(heatmap_rgb)
        overlay_pil = Image.fromarray(overlay_rgb)

        # 5. Save the overlay image under artifacts/gradcam/
        unique_id = uuid.uuid4().hex[:8]
        filename = f"gradcam_overlay_{unique_id}.png"
        save_path = self.output_dir / filename
        overlay_pil.save(save_path)

        logger.info(f"Saved Grad-CAM overlay image to: {save_path}")
        return heatmap_pil, overlay_pil, save_path
