"""NLP Symptom Classification Model Trainer — Phase 15.

Implements DistilBERT symptom sequence classifier with text preprocessing,
tokenization validation, early stopping, LR scheduling, checkpoint management,
and MLflow experiment tracking.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)

from src.utils.exceptions import (
    AppConfigurationError,
    AppStorageError,
    AppValidationError,
)
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NLPClassifierConfig:
    """Typed, immutable configuration for the DistilBERT symptom classifier."""

    # Model & Tokenizer
    model_name: str
    tokenizer_name: str
    dropout: float
    max_length: int

    # Dataset Paths
    validated_symptoms_csv: Path
    disease_mapping_file: Path

    # Splits
    train_split: float
    val_split: float
    test_split: float
    random_state: int

    # Optimizer
    optimizer: str
    learning_rate: float
    weight_decay: float

    # Tuning Search Space
    tune_learning_rates: List[float]
    tune_batch_sizes: List[int]
    tuning_epochs: int

    # Scheduler
    scheduler: str
    warmup_ratio: float
    step_size: int
    gamma: float
    t_max: int

    # Training
    epochs: int
    batch_size: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    max_grad_norm: float
    use_class_weights: bool

    # Paths
    checkpoint_dir: Path
    best_model_path: Path

    # MLflow
    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    @classmethod
    def from_yaml(cls, config_path: Path) -> "NLPClassifierConfig":
        """Loads and validates training configuration from a YAML file.

        Args:
            config_path (Path): Path to the YAML configuration file.

        Returns:
            NLPClassifierConfig: Validated, immutable configuration instance.

        Raises:
            AppConfigurationError: If the YAML is malformed or values are invalid.
        """
        if not config_path.exists():
            raise AppConfigurationError(
                message=f"Training config file not found: {config_path}",
                details={"path": str(config_path)},
            )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg: Dict[str, Any] = yaml.safe_load(f)

            return cls(
                model_name=str(cfg["model_name"]),
                tokenizer_name=str(cfg["tokenizer_name"]),
                dropout=float(cfg["dropout"]),
                max_length=int(cfg["max_length"]),
                validated_symptoms_csv=Path(cfg["validated_symptoms_csv"]),
                disease_mapping_file=Path(cfg["disease_mapping_file"]),
                train_split=float(cfg["train_split"]),
                val_split=float(cfg["val_split"]),
                test_split=float(cfg["test_split"]),
                random_state=int(cfg["random_state"]),
                optimizer=str(cfg["optimizer"]),
                learning_rate=float(cfg["learning_rate"]),
                weight_decay=float(cfg["weight_decay"]),
                tune_learning_rates=list(cfg.get("tune_learning_rates", [2e-5])),
                tune_batch_sizes=list(cfg.get("tune_batch_sizes", [16])),
                tuning_epochs=int(cfg.get("tuning_epochs", 2)),
                scheduler=str(cfg["scheduler"]),
                warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
                step_size=int(cfg["step_size"]),
                gamma=float(cfg["gamma"]),
                t_max=int(cfg["t_max"]),
                epochs=int(cfg["epochs"]),
                batch_size=int(cfg["batch_size"]),
                early_stopping_patience=int(cfg["early_stopping_patience"]),
                early_stopping_min_delta=float(cfg["early_stopping_min_delta"]),
                max_grad_norm=float(cfg["max_grad_norm"]),
                use_class_weights=bool(cfg.get("use_class_weights", False)),
                checkpoint_dir=Path(cfg["checkpoint_dir"]),
                best_model_path=Path(cfg["best_model_path"]),
                mlflow_tracking_uri=str(cfg["mlflow_tracking_uri"]),
                mlflow_experiment_name=str(cfg["mlflow_experiment_name"]),
            )
        except yaml.YAMLError as e:
            raise AppConfigurationError(
                message=f"Failed to parse training config YAML: {e}",
                details={"error": str(e)},
            )
        except (KeyError, TypeError, ValueError) as e:
            raise AppConfigurationError(
                message=f"Invalid value in training config: {e}",
                details={"error": str(e)},
            )


# ─────────────────────────────────────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class SymptomDataPreprocessor:
    """Production-grade symptom text dataset preprocessor and validator."""

    @staticmethod
    def preprocess_df(
        df: pd.DataFrame,
        disease_mapping_file: Path,
        disease_col: str = "Disease"
    ) -> Tuple[List[str], List[int], Dict[str, int]]:
        """Cleans and validates the disease symptom dataframe.

        Args:
            df (pd.DataFrame): Raw dataframe loaded from CSV.
            disease_mapping_file (Path): Path to disease mapping JSON.
            disease_col (str): Column name for the disease label.

        Returns:
            Tuple[List[str], List[int], Dict[str, int]]:
                - Cleaned narrative symptom strings list.
                - Encoded disease labels list.
                - Label mapping dictionary.

        Raises:
            AppValidationError: If validation fails.
        """
        # 1. Input Validation
        if df.empty:
            raise AppValidationError("Input symptom dataframe is empty.")

        if disease_col not in df.columns:
            raise AppValidationError(
                f"Missing required label column '{disease_col}' in dataframe.",
                details={"columns": list(df.columns)}
            )

        symptom_cols = [c for c in df.columns if c != disease_col]
        if not symptom_cols:
            raise AppValidationError("No symptom columns found in dataframe.")

        # 2. Missing Value Handling
        df_clean = df.copy()
        df_clean[symptom_cols] = df_clean[symptom_cols].fillna("")

        # 3. Disease mapping validation and label encoding
        unique_diseases = sorted(df_clean[disease_col].dropna().unique())

        if disease_mapping_file.exists():
            try:
                with open(disease_mapping_file, "r", encoding="utf-8") as f:
                    disease_to_idx = json.load(f)

                if not isinstance(disease_to_idx, dict):
                    raise AppValidationError(
                        f"Disease mapping file is invalid. Expected dict, got {type(disease_to_idx)}",
                        details={"path": str(disease_mapping_file)}
                    )

                missing_classes = [d for d in unique_diseases if d not in disease_to_idx]
                if missing_classes:
                    raise AppValidationError(
                        f"Symptom dataset contains diseases missing from mapping: {missing_classes}",
                        details={"missing": missing_classes, "mapping_keys": list(disease_to_idx.keys())}
                    )
            except Exception as e:
                if isinstance(e, AppValidationError):
                    raise e
                raise AppValidationError(f"Failed to load disease mapping file: {e}")
        else:
            disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
            try:
                disease_mapping_file.parent.mkdir(parents=True, exist_ok=True)
                with open(disease_mapping_file, "w", encoding="utf-8") as f:
                    json.dump(disease_to_idx, f, indent=4)
                logger.info("Created and saved new disease mapping to: %s", disease_mapping_file)
            except Exception as e:
                raise AppStorageError(f"Failed to save disease mapping file: {e}")

        # 4. Text cleaning loop
        symptom_strings: List[str] = []
        labels: List[int] = []

        for _, row in df_clean.iterrows():
            disease = row[disease_col]
            if pd.isna(disease):
                continue

            labels.append(disease_to_idx[disease])

            patient_symptoms: List[str] = []
            for col in symptom_cols:
                val = row[col]
                cleaned = str(val).strip()
                if not cleaned:
                    continue

                # Clean: lowercase conversion, normalize whitespace, remove punctuation
                cleaned = cleaned.lower()
                cleaned = re.sub(r"[^\w\s_]", "", cleaned)  # remove formatting punctuation except spaces/underscores
                cleaned = cleaned.replace("_", " ")
                cleaned = re.sub(r"\s+", " ", cleaned).strip()

                # Remove duplicates
                if cleaned and cleaned not in ["none", "nan", ""] and cleaned not in patient_symptoms:
                    patient_symptoms.append(cleaned)

            if not patient_symptoms:
                symptom_text = "no symptoms reported"
            else:
                symptom_text = ", ".join(patient_symptoms)

            symptom_strings.append(symptom_text)

        return symptom_strings, labels, disease_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NLPTextDataset(Dataset):
    """Custom PyTorch Dataset for clinical symptoms tokenization."""

    def __init__(
        self,
        symptom_strings: List[str],
        labels: List[int],
        tokenizer: DistilBertTokenizer,
        max_length: int = 64
    ) -> None:
        """Initializes the symptoms text dataset.

        Args:
            symptom_strings (List[str]): Preprocessed narrative texts.
            labels (List[int]): Integer class labels.
            tokenizer (DistilBertTokenizer): HuggingFace tokenizer instance.
            max_length (int): Max padding/truncation length boundaries.
        """
        if len(symptom_strings) != len(labels):
            raise AppValidationError(
                f"Symptom strings length ({len(symptom_strings)}) does not match labels length ({len(labels)})."
            )
        self.symptom_strings = symptom_strings
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.symptom_strings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        text = self.symptom_strings[idx]
        label = self.labels[idx]

        try:
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return (
                encoded["input_ids"].squeeze(0),
                encoded["attention_mask"].squeeze(0),
                label
            )
        except Exception as e:
            logger.error("Tokenization error for text '%s': %s", text, e)
            return (
                torch.zeros((self.max_length,), dtype=torch.long),
                torch.zeros((self.max_length,), dtype=torch.long),
                label
            )


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class SymptomClassifier(nn.Module):
    """DistilBERT text classifier for symptoms.

    Loads pretrained sequence classification weights, configures classes and dropout.
    """

    def __init__(self, model_name: str, num_classes: int, dropout: float) -> None:
        """Initializes the DistilBERT sequence classifier.

        Args:
            model_name (str): Hugging Face model repository ID.
            num_classes (int): Count of target disease classes.
            dropout (float): Dropout probability for sequence classification head.
        """
        super().__init__()
        self.num_classes = num_classes

        try:
            config = DistilBertConfig.from_pretrained(
                model_name,
                num_labels=num_classes,
                seq_classif_dropout=dropout
            )
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                config=config
            )
        except Exception as e:
            logger.error("Failed to load DistilBert model: %s", e)
            raise AppConfigurationError(f"Hugging Face model load failed: {e}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through DistilBERT transformer.

        Args:
            input_ids (torch.Tensor): [B, max_length] input token indices.
            attention_mask (torch.Tensor): [B, max_length] mask tensor.

        Returns:
            torch.Tensor: Logits tensor [B, num_classes].
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def model_summary(self) -> Dict[str, Any]:
        """Calculates parameters counts.

        Returns:
            Dict[str, Any]: Total, trainable, and frozen parameters.
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "model_name": "DistilBertForSequenceClassification",
            "num_classes": self.num_classes,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Monitors validation loss and signals training to stop if no improvement."""

    def __init__(self, patience: int, min_delta: float) -> None:
        """Initializes early stopping parameters."""
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.stop: bool = False

    def step(self, val_loss: float) -> None:
        """Increments patience counter or resets based on loss threshold validation."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info("EarlyStopping: no improvement for %d/%d epochs.", self.counter, self.patience)
            if self.counter >= self.patience:
                self.stop = True
                logger.warning("EarlyStopping limit reached. Triggering halt.")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class NLPClassifierTrainer:
    """Orchestrates DistilBERT training with early stopping, gradient clipping, and MLflow."""

    def __init__(
        self,
        config_path: Path,
        num_classes: int,
        class_weights: Optional[List[float]] = None,
        num_training_steps: Optional[int] = None
    ) -> None:
        """Initializes the NLP trainer.

        Args:
            config_path (Path): Path to nlp_training_config.yaml.
            num_classes (int): Number of diagnostic classes.
            class_weights (Optional[List[float]]): Class weights for imbalance mitigation.
            num_training_steps (Optional[int]): Total training steps for warmup schedules.
        """
        self.config: Final[NLPClassifierConfig] = NLPClassifierConfig.from_yaml(config_path)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("NLP Trainer initialized. Selected Device: %s", self.device)

        self.model = SymptomClassifier(
            model_name=self.config.model_name,
            num_classes=num_classes,
            dropout=self.config.dropout
        ).to(self.device)

        if class_weights is not None and self.config.use_class_weights:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            logger.info("Using weighted CrossEntropyLoss for NLP class imbalance.")
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer: Optimizer = self.build_optimizer()
        self.scheduler: Optional[LRScheduler | ReduceLROnPlateau] = self.build_scheduler(num_training_steps)
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta
        )

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def build_optimizer(self) -> Optimizer:
        """Constructs the optimizer from config parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt_name = self.config.optimizer.lower()

        if opt_name == "adamw":
            return AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        if opt_name == "adam":
            return Adam(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        if opt_name == "sgd":
            return SGD(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        raise AppConfigurationError(
            message=f"Unsupported optimizer: '{self.config.optimizer}'. Use adamw | adam | sgd.",
            details={"optimizer": self.config.optimizer}
        )

    def build_scheduler(self, num_training_steps: Optional[int] = None) -> Optional[LRScheduler | ReduceLROnPlateau]:
        """Constructs the learning rate scheduler from config."""
        sched_name = self.config.scheduler.lower()
        if sched_name == "linear_warmup":
            if num_training_steps is None:
                num_training_steps = 100
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)
            logger.info("Building linear warmup schedule: %d warmup steps, %d total steps.", warmup_steps, num_training_steps)
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        if sched_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.t_max)
        if sched_name == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        if sched_name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=3,
                factor=self.config.gamma
            )
        if sched_name == "none":
            return None
        raise AppConfigurationError(
            message=f"Unsupported scheduler: '{self.config.scheduler}'. Use linear_warmup | cosine | step | plateau | none.",
            details={"scheduler": self.config.scheduler}
        )

    def train_one_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Runs one training epoch, executing backpropagation and gradient clipping."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def validate_one_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Runs one validation epoch under eval mode (no grad)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += input_ids.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], path: Path) -> None:
        """Saves current trainer weights and optimization states."""
        try:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": {
                    "num_classes": self.model.num_classes,
                    "model_name": self.config.model_name,
                },
            }
            torch.save(checkpoint, path)
            logger.info("Saved NLP Checkpoint: %s", path)
        except Exception as e:
            raise AppStorageError(
                message=f"Failed to save NLP checkpoint: {e}",
                details={"path": str(path)}
            )

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Restores model and optimizer states from a checkpoint file."""
        if not path.exists():
            raise AppStorageError(
                message=f"NLP checkpoint not found: {path}",
                details={"path": str(path)}
            )
        try:
            checkpoint: Dict[str, Any] = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded NLP checkpoint: %s (epoch %d)", path, checkpoint["epoch"])
            return checkpoint
        except Exception as e:
            raise AppStorageError(
                message=f"Failed to load NLP checkpoint: {e}",
                details={"path": str(path)}
            )

    def _step_scheduler(self, val_loss: float) -> None:
        """Steps scheduler based on scheduler type signature checks."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: Optional[int] = None,
        resume_from_checkpoint: Optional[Path] = None
    ) -> List[Dict[str, float]]:
        """Full training loop execution logging to MLflow."""
        epochs = max_epochs if max_epochs is not None else self.config.epochs
        best_val_acc: float = 0.0
        start_epoch: int = 0
        history: List[Dict[str, float]] = []

        # Resume logic
        if resume_from_checkpoint is not None:
            loaded_ckpt = self.load_checkpoint(resume_from_checkpoint)
            start_epoch = loaded_ckpt["epoch"] + 1
            if "metrics" in loaded_ckpt:
                best_val_acc = loaded_ckpt["metrics"].get("val_acc", 0.0)
            logger.info("Resuming training from epoch %d. Best validation accuracy: %.4f", start_epoch, best_val_acc)

        # ── MLflow Tracking Setup ─────────────────────────────────────
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        total_training_start = time.time()

        with mlflow.start_run() as run:
            logger.info("NLP MLflow run started. ID: %s", run.info.run_id)

            # Log Hyperparameters
            mlflow.log_params({
                "model_name": self.config.model_name,
                "dropout": self.config.dropout,
                "max_length": self.config.max_length,
                "optimizer": self.config.optimizer,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "scheduler": self.config.scheduler,
                "epochs": epochs,
                "early_stopping_patience": self.config.early_stopping_patience,
                "max_grad_norm": self.config.max_grad_norm
            })

            summary = self.model.model_summary()
            mlflow.log_params({
                "total_parameters": summary["total_parameters"],
                "trainable_parameters": summary["trainable_parameters"],
                "num_classes": summary["num_classes"]
            })

            best_epoch_index = start_epoch

            for epoch in range(start_epoch, epochs):
                epoch_start = time.time()

                train_loss, train_acc = self.train_one_epoch(train_loader)
                val_loss, val_acc = self.validate_one_epoch(val_loader)

                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]

                metrics: Dict[str, float] = {
                    "train_loss": round(train_loss, 6),
                    "train_acc": round(train_acc, 6),
                    "val_loss": round(val_loss, 6),
                    "val_acc": round(val_acc, 6),
                    "learning_rate": round(current_lr, 8),
                    "epoch_time_s": round(epoch_time, 2)
                }
                history.append(metrics)

                # MLflow metrics log
                mlflow.log_metrics(metrics, step=epoch)

                logger.info(
                    "Epoch [%02d/%02d] | Train Loss: %.4f | Train Acc: %.4f | "
                    "Val Loss: %.4f | Val Acc: %.4f | LR: %.6f | Time: %.1fs",
                    epoch + 1, epochs,
                    train_loss, train_acc,
                    val_loss, val_acc,
                    current_lr, epoch_time
                )

                # Save best checkpoint
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch_index = epoch + 1
                    self.save_checkpoint(epoch, metrics, self.config.best_model_path)
                    logger.info("New best NLP model saved (val_acc=%.4f): %s", val_acc, self.config.best_model_path)

                # Period check checkpoint saves
                epoch_ckpt = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"
                self.save_checkpoint(epoch, metrics, epoch_ckpt)

                # Step scheduler
                self._step_scheduler(val_loss)

                # Early stopping check
                self.early_stopping.step(val_loss)
                if self.early_stopping.stop:
                    logger.warning("Early stopping activated at epoch %d.", epoch + 1)
                    break

            total_training_time = time.time() - total_training_start
            mlflow.log_metric("best_val_acc", best_val_acc)
            mlflow.log_metric("best_epoch", float(best_epoch_index))
            mlflow.log_metric("total_training_time_s", round(total_training_time, 2))

            # Log checkpoints as run artifacts
            if self.config.best_model_path.exists():
                mlflow.log_artifact(str(self.config.best_model_path), artifact_path="model_checkpoints")

            logger.info("NLP Training Completed. Best Val Acc: %.4f | Run ID: %s", best_val_acc, run.info.run_id)

        return history
