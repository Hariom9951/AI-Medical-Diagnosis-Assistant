"""Model Evaluation Component — Phase 15.

Provides performance metrics calculation, reporting, plotting, and MLflow logging
for the medical image classification models.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.components.model_trainer import EfficientNetClassifier, ImageClassifierConfig
from src.utils.exceptions import AppStorageError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class ImageClassifierEvaluator:
    """Evaluates a trained image classification model on a test dataset."""

    def __init__(
        self,
        training_config_path: Path,
        reports_dir: Path = Path("docs/reports"),
        class_names: List[str] | None = None,
    ) -> None:
        """Initializes the evaluator with training config and export settings.

        Args:
            training_config_path (Path): Path to training_config.yaml.
            reports_dir (Path): Output directory for evaluation reports and plots.
            class_names (Optional[List[str]]): List of class names for reporting.
        """
        self.config = ImageClassifierConfig.from_yaml(training_config_path)
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names or ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ImageClassifierEvaluator initialized. Device: %s", self.device)

    def load_best_model(self) -> nn.Module:
        """Loads the model and applies the best saved checkpoint weights.

        Returns:
            nn.Module: Loaded classifier model in eval mode.

        Raises:
            AppStorageError: If checkpoint loading fails or file is missing.
        """
        model = EfficientNetClassifier(
            num_classes=self.config.num_classes,
            freeze_backbone=self.config.freeze_backbone,
            dropout=self.config.dropout,
        ).to(self.device)

        checkpoint_path = self.config.best_model_path
        if not checkpoint_path.exists():
            raise AppStorageError(
                message=f"Best model checkpoint not found at: {checkpoint_path}",
                details={"best_model_path": str(checkpoint_path)},
            )

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Successfully loaded best model checkpoint from %s", checkpoint_path)
        except Exception as e:
            raise AppStorageError(
                message=f"Failed to load best model checkpoint: {e}",
                details={"best_model_path": str(checkpoint_path)},
            )

        model.eval()
        return model

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Runs predictions, computes evaluation metrics, saves reports and plots, and logs to MLflow.

        Args:
            test_loader (DataLoader): Test dataset DataLoader returning (images, labels, paths).

        Returns:
            Dict[str, Any]: Computed metrics dictionary.
        """
        model = self.load_best_model()

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[np.ndarray] = []

        logger.info("Running evaluation inference on test dataset...")
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(self.device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        if len(y_true) == 0:
            logger.warning("Empty test loader provided. Skipping metric calculation.")
            return {}

        metrics = self._calculate_metrics(y_true, y_pred, y_prob)

        # Save local reports and plots
        self._generate_plots(y_true, y_prob)
        self._write_reports(metrics, y_true, y_pred)

        # MLflow logging
        self._log_to_mlflow(metrics)

        return metrics

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates macro, weighted, and overall accuracy/F1/ROC-AUC metrics.

        Args:
            y_true (np.ndarray): True target class label indices.
            y_pred (np.ndarray): Predicted class label indices.
            y_prob (np.ndarray): Predicted class soft probabilities.

        Returns:
            Dict[str, Any]: Nested dictionary of computed metrics.
        """
        accuracy = float(np.mean(y_true == y_pred))

        # One-vs-Rest ROC-AUC calculation
        try:
            macro_roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            weighted_roc_auc = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
        except Exception as e:
            logger.warning("Failed to calculate ROC-AUC score: %s", e)
            macro_roc_auc = 0.0
            weighted_roc_auc = 0.0

        # Extract per-class precision, recall, f1
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        metrics = {
            "test_accuracy": accuracy,
            "macro_roc_auc": macro_roc_auc,
            "weighted_roc_auc": weighted_roc_auc,
            "macro_f1": report_dict["macro avg"]["f1-score"],
            "macro_precision": report_dict["macro avg"]["precision"],
            "macro_recall": report_dict["macro avg"]["recall"],
            "weighted_f1": report_dict["weighted avg"]["f1-score"],
            "weighted_precision": report_dict["weighted avg"]["precision"],
            "weighted_recall": report_dict["weighted avg"]["recall"],
            "class_metrics": {},
        }

        for idx, name in enumerate(self.class_names):
            str_idx = str(idx)
            if str_idx in report_dict:
                metrics["class_metrics"][name] = {
                    "precision": report_dict[str_idx]["precision"],
                    "recall": report_dict[str_idx]["recall"],
                    "f1-score": report_dict[str_idx]["f1-score"],
                    "support": int(report_dict[str_idx]["support"]),
                }
            else:
                metrics["class_metrics"][name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1-score": 0.0,
                    "support": 0,
                }

        return metrics

    def _generate_plots(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Generates confusion matrix and ROC/PR curve plots.

        Args:
            y_true (np.ndarray): True target class label indices.
            y_prob (np.ndarray): Predicted class soft probabilities.
        """
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_prob.argmax(axis=1))
        cm_norm = confusion_matrix(y_true, y_prob.argmax(axis=1), normalize="true")

        self._plot_confusion_matrix(cm, "confusion_matrix.png", title="Confusion Matrix")
        self._plot_confusion_matrix(
            cm_norm,
            "confusion_matrix_normalized.png",
            fmt=".2f",
            title="Normalized Confusion Matrix",
        )

        # 2. ROC Curves (OVR)
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-Rest ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.reports_dir / "roc_curves.png", dpi=150)
        plt.close()

        # 3. Precision-Recall Curves
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true == i, y_prob[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"{class_name} (AUC = {pr_auc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.reports_dir / "precision_recall_curves.png", dpi=150)
        plt.close()

        logger.info("Saved evaluation plots to: %s", self.reports_dir)

    def _plot_confusion_matrix(
        self, cm: np.ndarray, filename: str, fmt: str = "d", title: str = ""
    ) -> None:
        """Helper to plot confusion matrix using pure matplotlib.

        Args:
            cm (np.ndarray): Squared confusion matrix data.
            filename (str): Name of the output image file.
            fmt (str): String format specification for cell annotations.
            title (str): Header title for the plot.
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        plt.savefig(self.reports_dir / filename, dpi=150)
        plt.close()

    def _write_reports(
        self, metrics: Dict[str, Any], y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """Exports metrics.json, classification_report.csv, and model_evaluation_report.md.

        Args:
            metrics (Dict[str, Any]): Dictionary of calculated metrics.
            y_true (np.ndarray): True target indices.
            y_pred (np.ndarray): Predicted indices.
        """
        # 1. Export Metrics JSON
        json_path = self.reports_dir / "Metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        # 2. Export Classification Report CSV
        csv_path = self.reports_dir / "Classification_Report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
            for name, m in metrics["class_metrics"].items():
                writer.writerow(
                    [
                        name,
                        f"{m['precision']:.4f}",
                        f"{m['recall']:.4f}",
                        f"{m['f1-score']:.4f}",
                        m["support"],
                    ]
                )

        # 3. Export Evaluation Report Markdown
        md_path = self.reports_dir / "Evaluation_Report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Model Evaluation Report — Phase 15\n\n")
            f.write(
                "Presents evaluation metrics, per-class performance matrix tables, and support distributions.\n\n"
            )
            f.write("---\n\n")
            f.write("## 1. Overall Performance Summary\n")
            f.write(f"*   **Total Test Accuracy:** `{metrics['test_accuracy']:.6f}`\n")
            f.write(f"*   **Top-1 Accuracy:** `{metrics['test_accuracy']:.6f}`\n")
            f.write(f"*   **Macro ROC-AUC:** `{metrics['macro_roc_auc']:.6f}`\n")
            f.write(f"*   **Macro F1-Score:** `{metrics['macro_f1']:.6f}`\n")
            f.write(f"*   **Weighted F1-Score:** `{metrics['weighted_f1']:.6f}`\n\n")
            f.write("---\n\n")
            f.write("## 2. Per-class Support Table\n\n")
            f.write(
                "| Diagnostic Disease Class | Precision | Recall | F1-Score | Support Count |\n"
            )
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for name, m in metrics["class_metrics"].items():
                f.write(
                    f"| **{name}** | `{m['precision']:.4f}` | `{m['recall']:.4f}` | `{m['f1-score']:.4f}` | `{m['support']}` |\n"
                )
            f.write("\n---\n\n")
            f.write("## 3. Visual Performance Artifacts\n")
            f.write("*   **Confusion Matrix:** Refer to `confusion_matrix.png`\n")
            f.write("*   **Normalized Matrix:** Refer to `confusion_matrix_normalized.png`\n")
            f.write("*   **ROC Curve (OVR):** Refer to `roc_curves.png`\n")
            f.write("*   **Precision-Recall Curve:** Refer to `precision_recall_curves.png`\n")

        logger.info("Saved local evaluation reports to: %s", self.reports_dir)

    def _log_to_mlflow(self, metrics: Dict[str, Any]) -> None:
        """Logs metrics and report files to the active MLflow run or a new run if none is active.

        Args:
            metrics (Dict[str, Any]): Evaluated metrics dict.
        """
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        active_run = mlflow.active_run()
        if active_run:
            self._mlflow_log_all(metrics)
        else:
            with mlflow.start_run(run_name="model_evaluation") as run:
                logger.info("Started new MLflow run for evaluation: %s", run.info.run_id)
                self._mlflow_log_all(metrics)

    def _mlflow_log_all(self, metrics: Dict[str, Any]) -> None:
        """Helper to record metrics and plots inside active run context.

        Args:
            metrics (Dict[str, Any]): Evaluated metrics dict.
        """
        mlflow.log_metrics(
            {
                "test_accuracy": metrics["test_accuracy"],
                "test_macro_roc_auc": metrics["macro_roc_auc"],
                "test_weighted_roc_auc": metrics["weighted_roc_auc"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
            }
        )

        for name, m in metrics["class_metrics"].items():
            prefix = f"test_class_{name.lower().replace(' ', '_')}"
            mlflow.log_metrics(
                {
                    f"{prefix}_precision": m["precision"],
                    f"{prefix}_recall": m["recall"],
                    f"{prefix}_f1": m["f1-score"],
                }
            )

        # Log plots and markdown artifacts
        mlflow.log_artifact(str(self.reports_dir / "confusion_matrix.png"), "evaluation_plots")
        mlflow.log_artifact(
            str(self.reports_dir / "confusion_matrix_normalized.png"), "evaluation_plots"
        )
        mlflow.log_artifact(str(self.reports_dir / "roc_curves.png"), "evaluation_plots")
        mlflow.log_artifact(
            str(self.reports_dir / "precision_recall_curves.png"), "evaluation_plots"
        )
        mlflow.log_artifact(str(self.reports_dir / "Evaluation_Report.md"), "evaluation_reports")
        mlflow.log_artifact(str(self.reports_dir / "Metrics.json"), "evaluation_reports")
        mlflow.log_artifact(
            str(self.reports_dir / "Classification_Report.csv"), "evaluation_reports"
        )
        logger.info("Successfully logged all evaluation metrics and artifacts to MLflow.")


class NLPClassifierEvaluator:
    """Evaluates a trained DistilBERT symptom classifier on a test dataset."""

    def __init__(
        self,
        nlp_config_path: Path,
        reports_dir: Path = Path("docs/reports"),
        class_names: List[str] | None = None,
    ) -> None:
        """Initializes the NLP evaluator.

        Args:
            nlp_config_path (Path): Path to nlp_training_config.yaml.
            reports_dir (Path): Output directory for evaluation reports and plots.
            class_names (Optional[List[str]]): List of diagnostic category names.
        """
        from src.components.nlp_model_trainer import NLPClassifierConfig

        self.config = NLPClassifierConfig.from_yaml(nlp_config_path)
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load class names from mapping JSON
        if class_names is not None:
            self.class_names = class_names
        elif self.config.disease_mapping_file.exists():
            try:
                with open(self.config.disease_mapping_file, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                self.class_names = sorted(list(mapping.keys()), key=lambda x: mapping[x])
            except Exception:
                self.class_names = [f"Class_{i}" for i in range(38)]
        else:
            self.class_names = [f"Class_{i}" for i in range(38)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("NLPClassifierEvaluator initialized. Device: %s", self.device)

    def load_best_model(self) -> nn.Module:
        """Loads the model and applies the best saved checkpoint weights."""
        from src.components.nlp_model_trainer import SymptomClassifier

        model = SymptomClassifier(
            model_name=self.config.model_name,
            num_classes=len(self.class_names),
            dropout=self.config.dropout,
        ).to(self.device)

        checkpoint_path = self.config.best_model_path
        if not checkpoint_path.exists():
            raise AppStorageError(
                message=f"Best NLP model checkpoint not found at: {checkpoint_path}",
                details={"best_model_path": str(checkpoint_path)},
            )

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Successfully loaded best NLP model checkpoint from %s", checkpoint_path)
        except Exception as e:
            raise AppStorageError(
                message=f"Failed to load best NLP model checkpoint: {e}",
                details={"best_model_path": str(checkpoint_path)},
            )

        model.eval()
        return model

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Runs predictions, computes evaluation metrics, saves reports and plots, and logs to MLflow."""
        model = self.load_best_model()

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[np.ndarray] = []

        logger.info("Running evaluation inference on NLP test dataset...")
        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        if len(y_true) == 0:
            logger.warning("Empty test loader provided. Skipping metric calculation.")
            return {}

        metrics = self._calculate_metrics(y_true, y_pred, y_prob)

        # Save local reports and plots
        self._generate_plots(y_true, y_prob)
        self._write_reports(metrics, y_true, y_pred)

        # MLflow logging
        self._log_to_mlflow(metrics)

        return metrics

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates macro, weighted, and overall accuracy/F1/ROC-AUC metrics."""
        accuracy = float(np.mean(y_true == y_pred))

        try:
            macro_roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            weighted_roc_auc = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
        except Exception as e:
            logger.warning("Failed to calculate ROC-AUC score for NLP: %s", e)
            macro_roc_auc = 0.0
            weighted_roc_auc = 0.0

        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        metrics = {
            "test_accuracy": accuracy,
            "macro_roc_auc": macro_roc_auc,
            "weighted_roc_auc": weighted_roc_auc,
            "macro_f1": report_dict["macro avg"]["f1-score"],
            "macro_precision": report_dict["macro avg"]["precision"],
            "macro_recall": report_dict["macro avg"]["recall"],
            "weighted_f1": report_dict["weighted avg"]["f1-score"],
            "weighted_precision": report_dict["weighted avg"]["precision"],
            "weighted_recall": report_dict["weighted avg"]["recall"],
            "class_metrics": {},
        }

        for idx, name in enumerate(self.class_names):
            str_idx = str(idx)
            if str_idx in report_dict:
                metrics["class_metrics"][name] = {
                    "precision": report_dict[str_idx]["precision"],
                    "recall": report_dict[str_idx]["recall"],
                    "f1-score": report_dict[str_idx]["f1-score"],
                    "support": int(report_dict[str_idx]["support"]),
                }
            else:
                metrics["class_metrics"][name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1-score": 0.0,
                    "support": 0,
                }

        return metrics

    def _generate_plots(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Generates confusion matrix and ROC curves plots for NLP."""
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_prob.argmax(axis=1))

        plt.figure(figsize=(10, 8))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im)
        plt.title("NLP Symptom Classifier Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(self.reports_dir / "nlp_confusion_matrix.png", dpi=150)
        plt.close()

        # ROC Curves (OVR)
        plt.figure(figsize=(10, 8))
        plotted = 0
        for i, class_name in enumerate(self.class_names):
            if np.sum(y_true == i) > 0 and plotted < 8:
                fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.4f})")
                plotted += 1
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("NLP Symptom Classifier ROC Curves (Selected Classes)")
        plt.legend(loc="lower right", fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.reports_dir / "nlp_roc_curves.png", dpi=150)
        plt.close()
        logger.info("Saved NLP evaluation plots to: %s", self.reports_dir)

    def _write_reports(
        self, metrics: Dict[str, Any], y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """Exports metrics.json, classification_report.csv, and model_evaluation_report.md for NLP."""
        json_path = self.reports_dir / "NLP_Metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        csv_path = self.reports_dir / "NLP_Classification_Report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
            for name, m in metrics["class_metrics"].items():
                writer.writerow(
                    [
                        name,
                        f"{m['precision']:.4f}",
                        f"{m['recall']:.4f}",
                        f"{m['f1-score']:.4f}",
                        m["support"],
                    ]
                )

        md_path = self.reports_dir / "NLP_Evaluation_Report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Model Evaluation Report — NLP Symptom Classifier\n\n")
            f.write(f"*   **Total Test Accuracy:** `{metrics['test_accuracy']:.6f}`\n")
            f.write(f"*   **Macro ROC-AUC:** `{metrics['macro_roc_auc']:.6f}`\n")
            f.write(f"*   **Macro F1-Score:** `{metrics['macro_f1']:.6f}`\n")
            f.write(f"*   **Weighted F1-Score:** `{metrics['weighted_f1']:.6f}`\n\n")

    def _log_to_mlflow(self, metrics: Dict[str, Any]) -> None:
        """Logs metrics and report files to MLflow."""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        active_run = mlflow.active_run()
        if active_run:
            self._mlflow_log_all(metrics)
        else:
            with mlflow.start_run(run_name="nlp_evaluation") as run:
                logger.info("Started new MLflow run for NLP evaluation: %s", run.info.run_id)
                self._mlflow_log_all(metrics)

    def _mlflow_log_all(self, metrics: Dict[str, Any]) -> None:
        """Logs evaluation outcomes inside active run context."""
        mlflow.log_metrics(
            {
                "nlp_test_accuracy": metrics["test_accuracy"],
                "nlp_test_macro_roc_auc": metrics["macro_roc_auc"],
                "nlp_test_macro_f1": metrics["macro_f1"],
                "nlp_test_weighted_f1": metrics["weighted_f1"],
            }
        )

        mlflow.log_artifact(
            str(self.reports_dir / "nlp_confusion_matrix.png"), "nlp_evaluation_plots"
        )
        mlflow.log_artifact(str(self.reports_dir / "nlp_roc_curves.png"), "nlp_evaluation_plots")
        mlflow.log_artifact(
            str(self.reports_dir / "NLP_Evaluation_Report.md"), "nlp_evaluation_reports"
        )
        mlflow.log_artifact(str(self.reports_dir / "NLP_Metrics.json"), "nlp_evaluation_reports")
