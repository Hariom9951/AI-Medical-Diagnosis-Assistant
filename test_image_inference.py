"""Test script for the Image Model inference pipeline.

Loads the best trained checkpoint from artifacts/checkpoints/,
runs inference on a sample image, and prints the required output.

──────────────────────────────────────────────────────────────────────────
HOW TO EXECUTE
──────────────────────────────────────────────────────────────────────────
  Windows (virtual environment):
      .\\venv\\Scripts\\python test_image_inference.py

  Google Colab:
      !python test_image_inference.py

  Checkpoint resolution order (automatic, no configuration needed):
      1.  artifacts/checkpoints/best_model.pth       (canonical name)
      2.  Any single .pth / .pt file in that folder  (e.g. checkpoint_epoch_050.pth)
      3.  Best by val_loss when multiple files exist
──────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

from src.inference.predict import ImageInferencePipeline
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("artifacts/checkpoints")

SAMPLE_IMAGE = Path(
    "data/raw/covid19-radiography-database"
    "/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png"
)


def _check_checkpoints() -> None:
    """Exit early with a clear message if no checkpoint files are found at all."""
    if not CHECKPOINT_DIR.exists():
        logger.error(
            "Checkpoint directory not found: %s\n"
            "Create the directory and place a trained .pth file inside it.",
            CHECKPOINT_DIR,
        )
        sys.exit(1)

    candidates = list(CHECKPOINT_DIR.glob("*.pth")) + list(CHECKPOINT_DIR.glob("*.pt"))
    if not candidates:
        logger.error(
            "No .pth / .pt checkpoint files found in: %s\n"
            "Place the trained checkpoint (e.g. best_model.pth or "
            "checkpoint_epoch_050.pth) inside that directory and re-run.",
            CHECKPOINT_DIR,
        )
        sys.exit(1)


def main() -> None:
    logger.info("=== Image Inference Test ===")

    # 1. Pre-flight: at least one checkpoint must exist
    _check_checkpoints()

    # 2. Confirm sample image exists
    if not SAMPLE_IMAGE.exists():
        logger.error("Sample image not found: %s", SAMPLE_IMAGE)
        sys.exit(1)

    # 3. Initialize inference pipeline
    #    The pipeline auto-resolves the best checkpoint inside artifacts/checkpoints/
    try:
        pipeline = ImageInferencePipeline(
            config_path=Path("configs/training_config.yaml"),
        )
    except Exception as e:
        logger.exception("Pipeline initialization failed: %s", e)
        sys.exit(1)

    # 4. Run inference on the sample image
    try:
        results = pipeline.predict(SAMPLE_IMAGE)
    except Exception as e:
        logger.exception("Inference execution failed: %s", e)
        sys.exit(1)

    # 5. Print results
    print(f"predicted disease:  {results['predicted_disease']}")
    print(f"confidence score:   {results['confidence']:.6f}")


if __name__ == "__main__":
    main()
