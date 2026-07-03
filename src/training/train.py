"""Training Execution Entrypoint — Phase 14.

Allows invoking the complete training pipeline from CLI.
Usage:
    python -m src.training.train --epochs 1
"""

import argparse
import sys
from pathlib import Path

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


def main() -> None:
    """Main execution function for the training entrypoint."""
    parser = argparse.ArgumentParser(description="AI Medical Diagnosis Assistant Model Trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train (overrides training config epochs if set)",
    )
    args = parser.parse_args()

    try:
        pipeline = TrainingPipeline(
            ingestion_config_path=Path("configs/ingestion_config.yaml"),
            validation_config_path=Path("configs/validation_config.yaml"),
            transformation_config_path=Path("configs/transformation_config.yaml"),
            training_config_path=Path("configs/training_config.yaml"),
        )
        
        # Execute the pipeline
        pipeline.run(max_epochs=args.epochs)
        logger.info("Training pipeline execution finished successfully.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Pipeline execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
