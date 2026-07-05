"""Evaluation Execution Entrypoint — Phase 15.

Allows invoking the evaluation pipeline from CLI.
Usage:
    python -m src.evaluation.evaluate
"""

import sys
from pathlib import Path

from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


def main() -> None:
    """Main execution function for the evaluation entrypoint."""
    try:
        pipeline = EvaluationPipeline(
            transformation_config_path=Path("configs/transformation_config.yaml"),
            training_config_path=Path("configs/training_config.yaml"),
        )
        pipeline.run()
        logger.info("Evaluation pipeline execution finished successfully.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Evaluation pipeline execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
