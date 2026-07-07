"""Training Pipeline — Phase 14.

Orchestrates the entire training flow:
1. Ingests raw data from Kaggle (or verifies existing download)
2. Validates datasets and creates report
3. Transforms images and symptoms text to PyTorch DataLoaders
4. Trains the image classifier model, logging progress to MLflow
"""

from pathlib import Path
from typing import Any, Dict

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ImageClassifierTrainer
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class TrainingPipeline:
    """Orchestrates ingestion, validation, transformation, and model training."""

    def __init__(
        self,
        ingestion_config_path: Path = Path("configs/ingestion_config.yaml"),
        validation_config_path: Path = Path("configs/validation_config.yaml"),
        transformation_config_path: Path = Path("configs/transformation_config.yaml"),
        training_config_path: Path = Path("configs/training_config.yaml"),
    ) -> None:
        """Initializes configuration paths for all components in the pipeline."""
        self.ingestion_config_path = ingestion_config_path
        self.validation_config_path = validation_config_path
        self.transformation_config_path = transformation_config_path
        self.training_config_path = training_config_path

    def run(self, max_epochs: int | None = None) -> Dict[str, Any]:
        """Runs the complete training pipeline.

        Args:
            max_epochs (Optional[int]): Override epoch limit for quick testing.

        Returns:
            Dict[str, Any]: Combined results of pipeline execution.
        """
        logger.info("======================================================================")
        logger.info("Starting Phase 14 Training Pipeline")
        logger.info("======================================================================")

        # 1. Ingest data
        logger.info("STAGE 1/4: Ingesting Data...")
        ingestion = DataIngestion(self.ingestion_config_path)
        ingestion_results = ingestion.initiate_data_ingestion()
        logger.info("Ingestion completed: %s", ingestion_results)

        # 2. Validate data
        logger.info("STAGE 2/4: Validating Data...")
        validation = DataValidation(self.validation_config_path)
        validation.run_validation_pipeline()
        logger.info("Validation completed. Reports generated.")

        # 3. Transform data
        logger.info("STAGE 3/4: Transforming Data...")
        transformation = DataTransformation(self.transformation_config_path)
        loaders_dict = transformation.create_dataloaders()
        transformation.generate_transformation_report(loaders_dict)
        logger.info("Transformation completed and dataloaders ready.")

        # 4. Train model
        logger.info("STAGE 4/4: Model Training...")
        trainer = ImageClassifierTrainer(self.training_config_path)

        # We only train using the image loader (as text models are out-of-scope for Phase 14 model trainer)
        history = trainer.train(
            train_loader=loaders_dict["train_img_loader"],
            val_loader=loaders_dict["val_img_loader"],
            max_epochs=max_epochs,
        )
        logger.info("======================================================================")
        logger.info("Phase 14 Training Pipeline Finished Successfully")
        logger.info("======================================================================")

        return {
            "ingestion_results": ingestion_results,
            "loaders": loaders_dict,
            "training_history": history,
        }
