"""Image Classification Model Trainer — Phase 14.

Implements EfficientNet-B0 classifier with configurable frozen/unfrozen backbone,
full training loop with early stopping, LR scheduling, checkpoint management,
and MLflow experiment tracking.

Text/symptom models are out of scope for this phase.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import yaml
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from src.utils.exceptions import AppConfigurationError, AppStorageError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ImageClassifierConfig:
    """Typed, immutable configuration for the EfficientNet-B0 classifier."""

    # Model
    model_name: str
    num_classes: int
    freeze_backbone: bool
    dropout: float

    # Optimizer
    optimizer: str
    learning_rate: float
    weight_decay: float
    momentum: float

    # Scheduler
    scheduler: str
    step_size: int
    gamma: float
    t_max: int

    # Training
    epochs: int
    batch_size: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    max_grad_norm: float
    use_amp: bool

    # Paths
    checkpoint_dir: Path
    best_model_path: Path

    # MLflow
    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ImageClassifierConfig":
        """Loads and validates training configuration from a YAML file.

        Args:
            config_path (Path): Absolute path to the YAML configuration file.

        Returns:
            ImageClassifierConfig: Validated, immutable configuration instance.

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
                num_classes=int(cfg["num_classes"]),
                freeze_backbone=bool(cfg["freeze_backbone"]),
                dropout=float(cfg["dropout"]),
                optimizer=str(cfg["optimizer"]),
                learning_rate=float(cfg["learning_rate"]),
                weight_decay=float(cfg["weight_decay"]),
                momentum=float(cfg["momentum"]),
                scheduler=str(cfg["scheduler"]),
                step_size=int(cfg["step_size"]),
                gamma=float(cfg["gamma"]),
                t_max=int(cfg["t_max"]),
                epochs=int(cfg["epochs"]),
                batch_size=int(cfg["batch_size"]),
                early_stopping_patience=int(cfg["early_stopping_patience"]),
                early_stopping_min_delta=float(cfg["early_stopping_min_delta"]),
                max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
                use_amp=bool(cfg.get("use_amp", False)),
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
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 image classifier with a replaceable classification head.

    The backbone is loaded with ImageNet pretrained weights. The final classifier
    block is replaced with Dropout + Linear(1280 → num_classes). Backbone layers
    can be frozen or unfrozen independently of the classification head.
    """

    BACKBONE_OUT_FEATURES: Final[int] = 1280  # EfficientNet-B0 penultimate features

    def __init__(self, num_classes: int, freeze_backbone: bool, dropout: float) -> None:
        """Initializes the EfficientNet-B0 classifier.

        Args:
            num_classes (int): Number of output disease categories.
            freeze_backbone (bool): If True, all backbone layers are frozen.
            dropout (float): Dropout probability before the linear classification head.
        """
        super().__init__()
        self.num_classes = num_classes

        # 1. Load backbone with pretrained ImageNet weights
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # 2. Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(self.BACKBONE_OUT_FEATURES, num_classes),
        )

        # 3. Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone_layers()

        logger.info(
            "EfficientNetClassifier initialized. Classes: %d | Backbone frozen: %s",
            num_classes,
            freeze_backbone,
        )

    def _freeze_backbone_layers(self) -> None:
        """Freezes all EfficientNet backbone layers (features block only)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes all EfficientNet backbone layers for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        logger.info("EfficientNet backbone unfrozen. All parameters are now trainable.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classifier head.

        Args:
            x (torch.Tensor): Input image batch of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Logits tensor of shape [B, num_classes].
        """
        return self.backbone(x)

    def model_summary(self) -> Dict[str, Any]:
        """Returns model parameter statistics.

        Returns:
            Dict[str, Any]: Total, trainable, and frozen parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "model_name": "EfficientNet-B0",
            "num_classes": self.num_classes,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────


class EarlyStopping:
    """Monitors validation loss and signals training to stop if no improvement.

    Sets the `stop` flag to True after `patience` epochs without sufficient
    improvement (defined by `min_delta`).
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        """Initializes the early stopping monitor.

        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.stop: bool = False

    def step(self, val_loss: float) -> None:
        """Evaluates the current validation loss and updates the stop flag.

        Args:
            val_loss (float): Validation loss from the current epoch.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d epochs.",
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                self.stop = True
                logger.warning("EarlyStopping triggered. Halting training.")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class ImageClassifierTrainer:
    """Orchestrates EfficientNet-B0 training with checkpointing and MLflow logging.

    Manages the full training lifecycle:
    - Optimizer and LR scheduler construction from config
    - Per-epoch train/validation loops with loss and accuracy tracking
    - Best model preservation and periodic checkpointing
    - Early stopping monitoring
    - MLflow parameter and metric logging
    """

    def __init__(self, config_path: Path, class_weights: Optional[List[float]] = None) -> None:
        """Initializes the trainer.

        Args:
            config_path (Path): Path to training_config.yaml.
            class_weights (Optional[List[float]]): Optional class weights to handle imbalance.
        """
        self.config: Final[ImageClassifierConfig] = ImageClassifierConfig.from_yaml(config_path)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Trainer initialized. Device: %s", self.device)

        # Build model and move to device
        self.model = EfficientNetClassifier(
            num_classes=self.config.num_classes,
            freeze_backbone=self.config.freeze_backbone,
            dropout=self.config.dropout,
        ).to(self.device)

        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            logger.info("Using weighted CrossEntropyLoss for class imbalance.")
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer: Optimizer = self.build_optimizer()
        self.scheduler: Optional[LRScheduler | ReduceLROnPlateau] = self.build_scheduler()
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.config.use_amp and self.device.type == "cuda"
        )

        # Ensure checkpoint directory exists
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def build_optimizer(self) -> Optimizer:
        """Constructs the optimizer from config.

        Returns:
            Optimizer: Configured PyTorch optimizer.

        Raises:
            AppConfigurationError: If the optimizer name is unsupported.
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt_name = self.config.optimizer.lower()

        if opt_name == "adamw":
            return AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if opt_name == "adam":
            return Adam(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if opt_name == "sgd":
            return SGD(
                trainable_params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        raise AppConfigurationError(
            message=f"Unsupported optimizer: '{self.config.optimizer}'. Use adamw | adam | sgd.",
            details={"optimizer": self.config.optimizer},
        )

    def build_scheduler(self) -> Optional[LRScheduler | ReduceLROnPlateau]:
        """Constructs the LR scheduler from config.

        Returns:
            Optional scheduler instance, or None if scheduler = 'none'.
        """
        sched_name = self.config.scheduler.lower()
        if sched_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.t_max)
        if sched_name == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma,
            )
        if sched_name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=3,
                factor=self.config.gamma,
            )
        if sched_name == "none":
            return None
        raise AppConfigurationError(
            message=f"Unsupported scheduler: '{self.config.scheduler}'. Use cosine | step | plateau | none.",
            details={"scheduler": self.config.scheduler},
        )

    def train_one_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Runs one training epoch.

        Args:
            train_loader (DataLoader): Training DataLoader returning (image, label, path).

        Returns:
            Tuple[float, float]: Average training loss and accuracy for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_images, batch_labels, _ in train_loader:
            images = batch_images.to(self.device)
            labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision autocast
            with torch.cuda.amp.autocast(
                enabled=self.config.use_amp and self.device.type == "cuda"
            ):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # Mixed Precision scaling and backward pass
            self.scaler.scale(loss).backward()

            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.max_grad_norm
            )

            # Step and update scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def validate_one_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Runs one validation epoch in eval mode (no gradient computation).

        Args:
            val_loader (DataLoader): Validation DataLoader returning (image, label, path).

        Returns:
            Tuple[float, float]: Average validation loss and accuracy.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_images, batch_labels, _ in val_loader:
                images = batch_images.to(self.device)
                labels = batch_labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], path: Path) -> None:
        """Saves full training state to a checkpoint file.

        Args:
            epoch (int): Current epoch index (0-based).
            metrics (Dict[str, float]): Metrics dict to embed in checkpoint.
            path (Path): Destination file path for the checkpoint.

        Raises:
            AppStorageError: If saving fails.
        """
        try:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": {
                    "num_classes": self.config.num_classes,
                    "model_name": self.config.model_name,
                },
            }
            torch.save(checkpoint, path)
            logger.info("Checkpoint saved: %s", path)
        except Exception as e:
            raise AppStorageError(
                message=f"Failed to save checkpoint: {e}",
                details={"path": str(path)},
            )

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Restores model and optimizer state from a checkpoint file.

        Args:
            path (Path): Path to the checkpoint file.

        Returns:
            Dict[str, Any]: Full checkpoint dict including metrics and epoch.

        Raises:
            AppStorageError: If the checkpoint file is missing or corrupt.
        """
        if not path.exists():
            raise AppStorageError(
                message=f"Checkpoint not found: {path}",
                details={"path": str(path)},
            )
        try:
            checkpoint: Dict[str, Any] = torch.load(
                path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Checkpoint loaded from: %s (epoch %d)", path, checkpoint["epoch"])
            return checkpoint
        except Exception as e:
            raise AppStorageError(
                message=f"Failed to load checkpoint: {e}",
                details={"path": str(path)},
            )

    def _step_scheduler(self, val_loss: float) -> None:
        """Steps the LR scheduler appropriately based on its type."""
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
        resume_from_checkpoint: Optional[Path] = None,
    ) -> List[Dict[str, float]]:
        """Full training loop with early stopping, best model saving, and MLflow logging.

        Args:
            train_loader (DataLoader): Training DataLoader.
            val_loader (DataLoader): Validation DataLoader.
            max_epochs (Optional[int]): Override epoch count (useful for quick tests).
            resume_from_checkpoint (Optional[Path]): Checkpoint path to resume from.

        Returns:
            List[Dict[str, float]]: Per-epoch metric history.
        """
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
            logger.info(
                "Resuming image training from epoch %d. Best validation accuracy: %.4f",
                start_epoch,
                best_val_acc,
            )

        # ── MLflow setup ──────────────────────────────────────────────────
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        with mlflow.start_run() as run:
            logger.info("MLflow run started. Run ID: %s", run.info.run_id)

            # Log all hyperparameters once
            mlflow.log_params(
                {
                    "model_name": self.config.model_name,
                    "num_classes": self.config.num_classes,
                    "freeze_backbone": self.config.freeze_backbone,
                    "optimizer": self.config.optimizer,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "scheduler": self.config.scheduler,
                    "dropout": self.config.dropout,
                    "epochs": epochs,
                    "early_stopping_patience": self.config.early_stopping_patience,
                    "max_grad_norm": self.config.max_grad_norm,
                    "use_amp": self.config.use_amp,
                }
            )

            summary = self.model.model_summary()
            mlflow.log_params(
                {
                    "total_parameters": summary["total_parameters"],
                    "trainable_parameters": summary["trainable_parameters"],
                }
            )

            # ── Training Loop ─────────────────────────────────────────────
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
                    "epoch_time_s": round(epoch_time, 2),
                }
                history.append(metrics)

                # MLflow metric logging
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items()},
                    step=epoch,
                )

                logger.info(
                    "Epoch [%02d/%02d] | Train Loss: %.4f | Train Acc: %.4f | "
                    "Val Loss: %.4f | Val Acc: %.4f | LR: %.6f | Time: %.1fs",
                    epoch + 1,
                    epochs,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    current_lr,
                    epoch_time,
                )

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(epoch, metrics, self.config.best_model_path)
                    logger.info(
                        "New best model saved (val_acc=%.4f): %s",
                        best_val_acc,
                        self.config.best_model_path,
                    )

                # Save periodic epoch checkpoint
                epoch_ckpt = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"
                self.save_checkpoint(epoch, metrics, epoch_ckpt)

                # Step scheduler
                self._step_scheduler(val_loss)

                # Early stopping
                self.early_stopping.step(val_loss)
                if self.early_stopping.stop:
                    logger.warning("Early stopping activated at epoch %d.", epoch + 1)
                    break

            mlflow.log_metric("best_val_acc", best_val_acc)
            logger.info(
                "Training complete. Best val_acc: %.4f | MLflow run: %s",
                best_val_acc,
                run.info.run_id,
            )
            return history
