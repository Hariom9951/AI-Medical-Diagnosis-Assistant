"""NLP Training Pipeline — Phase 15.

Orchestrates the entire training workflow for the DistilBERT symptom classifier:
1. Loads YAML configurations.
2. Ingests and preprocesses the validated symptom narratives.
3. Performs stratified data splits.
4. Tokenizes narratives and creates PyTorch DataLoaders.
5. Performs DistilBERT model training with MLflow tracking.
6. Saves training curves, model checkpoints, tokenizer parameters, and configuration snapshots.
"""

import shutil
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

from src.components.nlp_model_trainer import (
    NLPClassifierConfig,
    NLPClassifierTrainer,
    NLPTextDataset,
    SymptomDataPreprocessor,
)
from src.components.pytorch_dataset import create_pytorch_dataloader
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class NLPTrainingPipeline:
    """Orchestrates the DistilBERT text classification pipeline."""

    def __init__(self, config_path: Path = Path("configs/nlp_training_config.yaml")) -> None:
        """Initializes configuration path.

        Args:
            config_path (Path): Path to nlp_training_config.yaml.
        """
        self.config_path = config_path
        self.config = NLPClassifierConfig.from_yaml(config_path)

    def plot_training_curves(self, history: list[dict[str, float]], save_path: Path) -> None:
        """Plots training and validation metrics curves and saves to disk.

        Args:
            history (list[dict[str, float]]): List of metrics per epoch.
            save_path (Path): Target file path for the plot.
        """
        epochs = list(range(1, len(history) + 1))
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        train_acc = [h["train_acc"] for h in history]
        val_acc = [h["val_acc"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss Curve
        ax1.plot(epochs, train_loss, "b-o", label="Train Loss")
        ax1.plot(epochs, val_loss, "r-s", label="Val Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy Curve
        ax2.plot(epochs, train_acc, "b-o", label="Train Acc")
        ax2.plot(epochs, val_acc, "r-s", label="Val Acc")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info("Saved training curves plot to %s", save_path)

    def run(self, max_epochs: int | None = None) -> Dict[str, Any]:
        """Runs the complete training pipeline.

        Args:
            max_epochs (Optional[int]): Override epoch count (useful for quick tests).

        Returns:
            Dict[str, Any]: Run statistics and metrics dict.
        """
        logger.info("======================================================================")
        logger.info("Starting Phase 15 DistilBERT NLP Training Pipeline")
        logger.info("======================================================================")

        # 1. Load Data
        logger.info("Loading validated symptoms dataset: %s", self.config.validated_symptoms_csv)
        if not self.config.validated_symptoms_csv.exists():
            raise FileNotFoundError(
                f"Symptom dataset not found at: {self.config.validated_symptoms_csv}"
            )

        df = pd.read_csv(self.config.validated_symptoms_csv)

        # 2. Preprocess Data
        logger.info("Cleaning symptoms and loading label mapping...")
        symptom_texts, labels, disease_mapping = SymptomDataPreprocessor.preprocess_df(
            df=df, disease_mapping_file=self.config.disease_mapping_file
        )
        num_classes = len(disease_mapping)
        logger.info(
            "Preprocessed %d records across %d diagnostic classes.", len(symptom_texts), num_classes
        )

        # 3. Stratified Data Split
        logger.info(
            "Splitting dataset (Train: %.2f | Val: %.2f | Test: %.2f)...",
            self.config.train_split,
            self.config.val_split,
            self.config.test_split,
        )

        temp_ratio = self.config.val_split + self.config.test_split

        try:
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                symptom_texts,
                labels,
                test_size=temp_ratio,
                stratify=labels,
                random_state=self.config.random_state,
            )
        except ValueError:
            logger.warning("Stratified split failed. Falling back to non-stratified split.")
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                symptom_texts, labels, test_size=temp_ratio, random_state=self.config.random_state
            )

        val_test_ratio = self.config.val_split / temp_ratio
        try:
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts,
                temp_labels,
                test_size=1.0 - val_test_ratio,
                stratify=temp_labels,
                random_state=self.config.random_state,
            )
        except ValueError:
            logger.warning(
                "Stratified val/test split failed. Falling back to non-stratified split."
            )
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts,
                temp_labels,
                test_size=1.0 - val_test_ratio,
                random_state=self.config.random_state,
            )

        logger.info(
            "Split allocation counts - Train: %d, Val: %d, Test: %d",
            len(train_texts),
            len(val_texts),
            len(test_texts),
        )

        # 4. Tokenizer & Dataset Creation
        logger.info("Loading DistilBERT tokenizer: %s", self.config.tokenizer_name)
        tokenizer = DistilBertTokenizer.from_pretrained(self.config.tokenizer_name)

        train_dataset = NLPTextDataset(train_texts, train_labels, tokenizer, self.config.max_length)
        val_dataset = NLPTextDataset(val_texts, val_labels, tokenizer, self.config.max_length)
        test_dataset = NLPTextDataset(test_texts, test_labels, tokenizer, self.config.max_length)

        # 5. Dataloaders
        train_loader = create_pytorch_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Safe on Windows
            pin_memory=True,
            persistent_workers=False,
        )
        val_loader = create_pytorch_dataloader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        test_loader = create_pytorch_dataloader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

        # 6. Model Training
        logger.info("Initializing NLP Classifier Trainer...")
        trainer = NLPClassifierTrainer(self.config_path, num_classes=num_classes)

        logger.info("Starting model training loop...")
        history = trainer.train(
            train_loader=train_loader, val_loader=val_loader, max_epochs=max_epochs
        )

        # 7. Post-Training Artifacts & Metrics Verification
        logger.info("Saving training curves plot...")
        curves_path = self.config.checkpoint_dir / "training_curves.png"
        self.plot_training_curves(history, curves_path)

        logger.info("Saving tokenizer to checkpoint directory...")
        tokenizer_save_dir = self.config.checkpoint_dir / "tokenizer"
        tokenizer.save_pretrained(tokenizer_save_dir)

        logger.info("Saving configuration snapshot...")
        config_snapshot_path = self.config.checkpoint_dir / "nlp_training_config_snapshot.yaml"
        shutil.copy(self.config_path, config_snapshot_path)

        logger.info("======================================================================")
        logger.info("Phase 15 NLP Training Pipeline Finished Successfully")
        logger.info("======================================================================")

        return {
            "num_classes": num_classes,
            "vocab_size": tokenizer.vocab_size,
            "training_history": history,
            "best_model_path": str(self.config.best_model_path),
            "tokenizer_path": str(tokenizer_save_dir),
            "curves_path": str(curves_path),
            "config_snapshot_path": str(config_snapshot_path),
        }
