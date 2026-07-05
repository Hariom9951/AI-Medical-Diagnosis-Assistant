"""API utilities — model loading, Grad-CAM helpers, and singleton management.

This module owns the two global inference pipeline singletons so that models
are loaded once at startup and reused across every request.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton references
# ---------------------------------------------------------------------------

_image_pipeline: Optional["ImageInferencePipeline"] = None  # type: ignore[name-defined]
_nlp_pipeline: Optional["NLPInferencePipeline"] = None  # type: ignore[name-defined]


def get_image_pipeline():
    """Returns the singleton ImageInferencePipeline, initialising it on first call."""
    global _image_pipeline
    if _image_pipeline is None:
        from src.inference.predict import ImageInferencePipeline

        logger.info("Loading ImageInferencePipeline for the first time …")
        _image_pipeline = ImageInferencePipeline()
        logger.info("ImageInferencePipeline ready.")
    return _image_pipeline


def get_nlp_pipeline():
    """Returns the singleton NLPInferencePipeline, initialising it on first call."""
    global _nlp_pipeline
    if _nlp_pipeline is None:
        from src.inference.nlp_predict import NLPInferencePipeline

        logger.info("Loading NLPInferencePipeline for the first time …")
        _nlp_pipeline = NLPInferencePipeline()
        logger.info("NLPInferencePipeline ready.")
    return _nlp_pipeline


def is_image_pipeline_loaded() -> bool:
    return _image_pipeline is not None


def is_nlp_pipeline_loaded() -> bool:
    return _nlp_pipeline is not None


# ---------------------------------------------------------------------------
# Grad-CAM helper
# ---------------------------------------------------------------------------


def run_gradcam(
    image_pipeline,
    image_np: np.ndarray,
    original_pil: Image.Image,
    pred_class_idx: int,
) -> Optional[str]:
    """Generates a Grad-CAM overlay image and returns its file path as a string.

    Args:
        image_pipeline: Loaded ImageInferencePipeline instance.
        image_np: RGB numpy array of the original image.
        original_pil: PIL Image of the original image (used for overlay sizing).
        pred_class_idx: Predicted class index to highlight.

    Returns:
        Absolute path string of the saved overlay PNG, or None on failure.
    """
    try:
        from src.explainability.gradcam import GradCAMExplainer
        from src.explainability.visualizer import GradCAMVisualizer

        explainer = GradCAMExplainer(model=image_pipeline.model)
        input_tensor = image_pipeline.preprocess(image_np)  # [1, 3, H, W]

        heatmap = explainer.generate_heatmap(
            input_tensor=input_tensor,
            target_class_idx=pred_class_idx,
        )
        if heatmap is None:
            logger.warning("Grad-CAM heatmap generation returned None.")
            return None

        visualizer = GradCAMVisualizer(output_dir="artifacts/gradcam")
        _, _, save_path = visualizer.create_visualization(
            original_image=original_pil,
            heatmap=heatmap,
        )
        return str(save_path.resolve())

    except Exception as exc:
        logger.error("Grad-CAM generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Image reading helper
# ---------------------------------------------------------------------------


def read_upload_as_pil(file_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
    """Converts raw file bytes into a PIL Image and an RGB numpy array.

    Args:
        file_bytes: Raw bytes from the uploaded file.

    Returns:
        Tuple of (PIL Image in RGB mode, numpy RGB uint8 array).

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        return pil_img, img_np
    except Exception as exc:
        raise ValueError(f"Cannot decode uploaded file as an image: {exc}") from exc


# ---------------------------------------------------------------------------
# Checkpoint metadata helper
# ---------------------------------------------------------------------------


def extract_checkpoint_info(pipeline, pipeline_type: str = "image") -> dict:
    """Extracts serialisable checkpoint metadata from a loaded pipeline.

    Args:
        pipeline: ImageInferencePipeline or NLPInferencePipeline instance.
        pipeline_type: 'image' or 'nlp' — used to select the right attr names.

    Returns:
        Dict with keys: checkpoint_file, epoch, metrics.
    """
    try:
        info = pipeline.checkpoint_info
        raw_path = info.get("checkpoint_path", "unknown")
        # Path objects are not JSON-serialisable
        ckpt_file = Path(raw_path).name if raw_path != "unknown" else "unknown"
        epoch = info.get("epoch", "N/A")
        metrics = info.get("metrics", {})
        # Ensure all metric values are JSON-safe (convert tensors / ndarrays)
        safe_metrics: dict = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                safe_metrics[k] = v.item()
            else:
                safe_metrics[k] = v
        return {"checkpoint_file": ckpt_file, "epoch": epoch, "metrics": safe_metrics}
    except Exception as exc:
        logger.warning("Could not extract checkpoint info: %s", exc)
        return {"checkpoint_file": "unknown", "epoch": "N/A", "metrics": {}}
