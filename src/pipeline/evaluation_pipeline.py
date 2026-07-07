"""Evaluation Pipeline — Phase 15.

Orchestrates the evaluation flow:
1. Loads transformation and training configurations.
2. Creates data splits and initializes test DataLoaders.
3. Loads the trained EfficientNet-B0 model checkpoint.
4. Calculates metrics on the test dataset and exports plots/reports.
"""

from pathlib import Path
from typing import Any, Dict

from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ImageClassifierEvaluator
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class EvaluationPipeline:
    """Orchestrates test data transformation and model evaluation."""

    def __init__(
        self,
        transformation_config_path: Path = Path("configs/transformation_config.yaml"),
        training_config_path: Path = Path("configs/training_config.yaml"),
    ) -> None:
        """Initializes configuration paths for evaluation stages."""
        self.transformation_config_path = transformation_config_path
        self.training_config_path = training_config_path

    def run(self) -> Dict[str, Any]:
        """Runs the complete model evaluation pipeline.

        Returns:
            Dict[str, Any]: Calculated metrics.
        """
        logger.info("======================================================================")
        logger.info("Starting Phase 15 Evaluation Pipeline")
        logger.info("======================================================================")

        # 1. Transform/Split data to get test loaders
        logger.info("STAGE 1/2: Preparing Datasets & Dataloaders...")
        transformation = DataTransformation(self.transformation_config_path)
        loaders_dict = transformation.create_dataloaders()
        test_img_loader = loaders_dict.get("test_img_loader")

        if test_img_loader is None:
            logger.error("No test_img_loader was returned by DataTransformation.")
            return {}

        logger.info("Loaded test dataset with %d samples.", len(test_img_loader.dataset))

        # 2. Evaluate model
        logger.info("STAGE 2/2: Evaluating Best Checkpoint...")
        evaluator = ImageClassifierEvaluator(
            training_config_path=self.training_config_path,
            reports_dir=transformation.config.reports_dir,
        )

        metrics = evaluator.evaluate(test_img_loader)

        logger.info("======================================================================")
        logger.info("Phase 15 Evaluation Pipeline Finished Successfully")
        logger.info("======================================================================")

        return metrics
