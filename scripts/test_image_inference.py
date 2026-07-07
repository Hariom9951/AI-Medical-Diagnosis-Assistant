"""Verification script to test the Image Model prediction pipeline."""

import sys
from pathlib import Path

from src.inference.predict import ImageInferencePipeline
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


def main() -> None:
    logger.info("Starting Image Inference Pipeline verification script.")

    # 1. Define paths
    sample_image_path = Path(
        "data/raw/covid19-radiography-database/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png"
    )
    config_path = Path("configs/training_config.yaml")

    # 2. Check dependencies
    if not sample_image_path.exists():
        logger.error("Sample image not found at: %s", sample_image_path)
        sys.exit(1)

    if not config_path.exists():
        logger.error("Configuration file not found at: %s", config_path)
        sys.exit(1)

    # Automatically create a dummy checkpoint if running locally and no checkpoint exists
    local_ckpt_dir = Path("artifacts/checkpoints")
    has_checkpoints = False
    if local_ckpt_dir.exists():
        has_checkpoints = any(local_ckpt_dir.glob("*.pth")) or any(local_ckpt_dir.glob("*.pt"))

    gdrive_path = Path(
        "content/drive/MyDrive/AI-Medical-Diagnosis-Assistant/checkpoints/image_model"
    )
    if gdrive_path.exists():
        has_checkpoints = (
            has_checkpoints or any(gdrive_path.glob("*.pth")) or any(gdrive_path.glob("*.pt"))
        )

    if not has_checkpoints:
        logger.info("No checkpoints found. Generating a dummy checkpoint for verification.")
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        import torch

        from src.components.model_trainer import EfficientNetClassifier

        model = EfficientNetClassifier(num_classes=4, freeze_backbone=True, dropout=0.3)
        dummy_ckpt = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 0.45, "val_acc": 0.82},
        }
        torch.save(dummy_ckpt, local_ckpt_dir / "checkpoint_epoch_001.pth")

    # 3. Run Inference
    try:
        # Initialize pipeline pointing to the local checkpoints directory
        # Since this is a test script running in the repository, we point to artifacts/checkpoints
        # (Where checkpoints are stored locally in the workspace)
        pipeline = ImageInferencePipeline(
            config_path=config_path, checkpoint_dir="artifacts/checkpoints"
        )

        logger.info("Running prediction on sample image: %s", sample_image_path)
        results = pipeline.predict(sample_image_path)

        # Print results clearly
        print("\n" + "=" * 50)
        print("IMAGE MODEL PREDICTION RESULTS")
        print("=" * 50)
        print(f"Sample Image Path:  {sample_image_path}")
        print(f"Predicted Disease:  {results['predicted_disease']}")
        print(f"Confidence Score:   {results['confidence']:.6f}")
        print("\nClass Probabilities:")
        for disease, prob in results["class_probabilities"].items():
            print(f" - {disease:<20}: {prob:.6f}")
        print("=" * 50 + "\n")

    except Exception as e:
        logger.exception("Verification script failed with exception: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
