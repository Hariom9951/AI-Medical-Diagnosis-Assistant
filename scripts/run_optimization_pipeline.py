import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Subset
from transformers import DistilBertTokenizer

from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ImageClassifierEvaluator, NLPClassifierEvaluator
from src.components.model_trainer import ImageClassifierConfig, ImageClassifierTrainer
from src.components.nlp_model_trainer import (
    NLPClassifierConfig,
    NLPClassifierTrainer,
    NLPTextDataset,
    SymptomDataPreprocessor,
)
from src.components.pytorch_dataset import create_pytorch_dataloader
from src.components.report_generator import PDFReportGenerator
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


def compute_class_weights(labels: list[int], num_classes: int) -> list[float]:
    """Helper to calculate class weights for mitigating dataset imbalance."""
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Avoid divide-by-zero
    weights = [float(total / (num_classes * max(count, 1))) for count in counts]
    return weights


def main():
    logger.info("Initializing optimization pipeline...")

    # Paths
    image_config_path = Path("configs/training_config.yaml")
    nlp_config_path = Path("configs/nlp_training_config.yaml")
    reports_dir = Path("docs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cpu = device.type == "cpu"

    # ── PART 1: Optimizing Image Classification Model ─────────────────────────
    logger.info("--- STAGE 1: Image Classification Model Optimization ---")

    # Create dataloaders
    img_transform_config_path = Path("configs/transformation_config.yaml")
    img_transformation = DataTransformation(img_transform_config_path)
    img_loaders = img_transformation.create_dataloaders()

    # Compute image class weights
    train_img_labels = img_loaders["splits"]["train_img_lbl"]
    image_classes = len(img_loaders["splits"]["image_mapping"])
    image_weights = compute_class_weights(train_img_labels, image_classes)

    logger.info("Class weights computed for Image classifier: %s", image_weights)

    # Check for CPU training subset to run tests fast
    if is_cpu:
        logger.info("CPU detected. Subsampling image dataset to speed up training runs...")
        train_img_loader = create_pytorch_dataloader(
            Subset(img_loaders["train_img_loader"].dataset, list(range(200))),
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        val_img_loader = create_pytorch_dataloader(
            Subset(img_loaders["val_img_loader"].dataset, list(range(50))),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        test_img_loader = create_pytorch_dataloader(
            Subset(img_loaders["test_img_loader"].dataset, list(range(50))),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
    else:
        train_img_loader = img_loaders["train_img_loader"]
        val_img_loader = img_loaders["val_img_loader"]
        test_img_loader = img_loaders["test_img_loader"]

    # Initialise trainer with weighted loss and run training
    image_trainer = ImageClassifierTrainer(image_config_path, class_weights=image_weights)

    image_train_start = time.time()
    # If on CPU, limit training to 2 epochs for verification, else train full configured epochs
    max_epochs = 2 if is_cpu else None
    image_history = image_trainer.train(
        train_loader=train_img_loader, val_loader=val_img_loader, max_epochs=max_epochs
    )
    image_train_time = time.time() - image_train_start

    # Plot accuracy and loss curves for image model
    epochs = list(range(1, len(image_history) + 1))
    train_loss = [h["train_loss"] for h in image_history]
    val_loss = [h["val_loss"] for h in image_history]
    train_acc = [h["train_acc"] for h in image_history]
    val_acc = [h["val_acc"] for h in image_history]

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss")
    ax1.plot(epochs, val_loss, "r-s", label="Val Loss")
    ax1.set_title("Image Model Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc, "b-o", label="Train Acc")
    ax2.plot(epochs, val_acc, "r-s", label="Val Acc")
    ax2.set_title("Image Model Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    image_curves_path = reports_dir / "image_training_curves.png"
    plt.savefig(image_curves_path)
    plt.close()

    # Find best epoch index (1-based)
    best_image_epoch = int(np.argmax(val_acc) + 1)

    # ── PART 2: Optimizing DistilBERT NLP Model ────────────────────────────────
    logger.info("--- STAGE 2: DistilBERT NLP Model Optimization ---")

    nlp_config = NLPClassifierConfig.from_yaml(nlp_config_path)
    df = pd.read_csv(nlp_config.validated_symptoms_csv)
    symptom_texts, labels, disease_mapping = SymptomDataPreprocessor.preprocess_df(
        df=df, disease_mapping_file=nlp_config.disease_mapping_file
    )
    nlp_num_classes = len(disease_mapping)

    # Split symptom dataset
    from sklearn.model_selection import train_test_split

    temp_ratio = nlp_config.val_split + nlp_config.test_split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        symptom_texts, labels, test_size=temp_ratio, random_state=nlp_config.random_state
    )
    val_test_ratio = nlp_config.val_split / temp_ratio
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=1.0 - val_test_ratio,
        random_state=nlp_config.random_state,
    )

    # CPU sub-sampling to run fast
    if is_cpu:
        logger.info("CPU detected. Subsampling NLP datasets to speed up training runs...")
        train_texts = train_texts[:100]
        train_labels = train_labels[:100]
        val_texts = val_texts[:30]
        val_labels = val_labels[:30]
        test_texts = test_texts[:30]
        test_labels = test_labels[:30]

    nlp_class_weights = compute_class_weights(train_labels, nlp_num_classes)
    logger.info("Class weights computed for NLP: %s", nlp_class_weights)

    tokenizer = DistilBertTokenizer.from_pretrained(nlp_config.tokenizer_name)
    train_dataset = NLPTextDataset(train_texts, train_labels, tokenizer, nlp_config.max_length)
    val_dataset = NLPTextDataset(val_texts, val_labels, tokenizer, nlp_config.max_length)
    test_dataset = NLPTextDataset(test_texts, test_labels, tokenizer, nlp_config.max_length)

    # Grid Search Hyperparameter Tuning
    best_val_acc = 0.0
    best_lr = nlp_config.learning_rate
    best_bs = nlp_config.batch_size

    logger.info("Starting NLP hyperparameter tuning grid search...")
    # Limit search space on CPU to reduce time
    tuning_lrs = nlp_config.tune_learning_rates[:2] if is_cpu else nlp_config.tune_learning_rates
    tuning_bss = nlp_config.tune_batch_sizes[:2] if is_cpu else nlp_config.tune_batch_sizes

    for lr in tuning_lrs:
        for bs in tuning_bss:
            logger.info("Tuning run with LR=%s, Batch Size=%d", lr, bs)

            # Temporary config modification
            with open(nlp_config_path, "r", encoding="utf-8") as f:
                temp_cfg = yaml.safe_load(f)
            temp_cfg["learning_rate"] = lr
            temp_cfg["batch_size"] = bs

            temp_config_file = nlp_config_path.parent / "temp_nlp_config.yaml"
            with open(temp_config_file, "w", encoding="utf-8") as f:
                yaml.dump(temp_cfg, f)

            temp_train_loader = create_pytorch_dataloader(
                train_dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            temp_val_loader = create_pytorch_dataloader(
                val_dataset,
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )

            temp_trainer = NLPClassifierTrainer(
                temp_config_file,
                num_classes=nlp_num_classes,
                class_weights=nlp_class_weights,
                num_training_steps=len(temp_train_loader) * nlp_config.tuning_epochs,
            )

            # Run fast tuning epochs
            temp_history = temp_trainer.train(
                train_loader=temp_train_loader,
                val_loader=temp_val_loader,
                max_epochs=nlp_config.tuning_epochs,
            )

            run_val_acc = max([h["val_acc"] for h in temp_history])
            logger.info("Tuning validation accuracy: %.4f", run_val_acc)

            if run_val_acc > best_val_acc:
                best_val_acc = run_val_acc
                best_lr = lr
                best_bs = bs

            # Clean up temp file
            if temp_config_file.exists():
                temp_config_file.unlink()

    logger.info(
        "Best NLP Parameters found: Learning Rate = %s, Batch Size = %d (Val Accuracy = %.4f)",
        best_lr,
        best_bs,
        best_val_acc,
    )

    # Final NLP Training with Best Params
    with open(nlp_config_path, "r", encoding="utf-8") as f:
        final_cfg = yaml.safe_load(f)
    final_cfg["learning_rate"] = best_lr
    final_cfg["batch_size"] = best_bs

    with open(nlp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(final_cfg, f)

    final_train_loader = create_pytorch_dataloader(
        train_dataset,
        batch_size=best_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    final_val_loader = create_pytorch_dataloader(
        val_dataset,
        batch_size=best_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    final_test_loader = create_pytorch_dataloader(
        test_dataset,
        batch_size=best_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    max_nlp_epochs = 2 if is_cpu else nlp_config.epochs
    num_training_steps = len(final_train_loader) * max_nlp_epochs

    nlp_trainer = NLPClassifierTrainer(
        nlp_config_path,
        num_classes=nlp_num_classes,
        class_weights=nlp_class_weights,
        num_training_steps=num_training_steps,
    )

    nlp_train_start = time.time()
    nlp_history = nlp_trainer.train(
        train_loader=final_train_loader, val_loader=final_val_loader, max_epochs=max_nlp_epochs
    )
    nlp_train_time = time.time() - nlp_train_start

    # Plot accuracy and loss curves for NLP model
    epochs = list(range(1, len(nlp_history) + 1))
    train_loss = [h["train_loss"] for h in nlp_history]
    val_loss = [h["val_loss"] for h in nlp_history]
    train_acc = [h["train_acc"] for h in nlp_history]
    val_acc = [h["val_acc"] for h in nlp_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss")
    ax1.plot(epochs, val_loss, "r-s", label="Val Loss")
    ax1.set_title("NLP Model Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc, "b-o", label="Train Acc")
    ax2.plot(epochs, val_acc, "r-s", label="Val Acc")
    ax2.set_title("NLP Model Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    nlp_curves_path = reports_dir / "nlp_training_curves.png"
    plt.savefig(nlp_curves_path)
    plt.close()

    best_nlp_epoch = int(np.argmax(val_acc) + 1)

    # ── STAGE 3: Run Model Evaluators ──────────────────────────────────────────
    logger.info("--- STAGE 3: Centralized Model Evaluations ---")

    image_evaluator = ImageClassifierEvaluator(image_config_path, reports_dir=reports_dir)
    image_test_metrics = image_evaluator.evaluate(test_img_loader)

    nlp_evaluator = NLPClassifierEvaluator(nlp_config_path, reports_dir=reports_dir)
    nlp_test_metrics = nlp_evaluator.evaluate(final_test_loader)

    # Fetch active mlflow run or last run IDs
    image_run_id = "N/A"
    nlp_run_id = "N/A"

    try:
        # Search last image run
        mlflow.set_tracking_uri(image_trainer.config.mlflow_tracking_uri)
        image_exp = mlflow.get_experiment_by_name(image_trainer.config.mlflow_experiment_name)
        if image_exp:
            runs = mlflow.search_runs(
                experiment_ids=[image_exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                image_run_id = runs.iloc[0]["run_id"]

        # Search last NLP run
        mlflow.set_tracking_uri(nlp_trainer.config.mlflow_tracking_uri)
        nlp_exp = mlflow.get_experiment_by_name(nlp_trainer.config.mlflow_experiment_name)
        if nlp_exp:
            runs = mlflow.search_runs(
                experiment_ids=[nlp_exp.experiment_id], order_by=["start_time DESC"], max_results=1
            )
            if not runs.empty:
                nlp_run_id = runs.iloc[0]["run_id"]
    except Exception as e:
        logger.warning("Failed to retrieve MLflow run IDs: %s", e)

    # ── STAGE 4: Generating PDF Reports ───────────────────────────────────────
    logger.info("--- STAGE 4: Generating PDF Reports ---")

    image_hyperparams = {
        "optimizer": image_trainer.config.optimizer,
        "learning_rate": image_trainer.config.learning_rate,
        "weight_decay": image_trainer.config.weight_decay,
        "batch_size": image_trainer.config.batch_size,
        "epochs": image_trainer.config.epochs,
        "scheduler": image_trainer.config.scheduler,
        "use_amp": image_trainer.config.use_amp,
        "max_grad_norm": image_trainer.config.max_grad_norm,
    }

    nlp_hyperparams = {
        "optimizer": nlp_trainer.config.optimizer,
        "learning_rate": nlp_trainer.config.learning_rate,
        "weight_decay": nlp_trainer.config.weight_decay,
        "batch_size": nlp_trainer.config.batch_size,
        "epochs": nlp_trainer.config.epochs,
        "scheduler": nlp_trainer.config.scheduler,
        "warmup_ratio": nlp_trainer.config.warmup_ratio,
        "max_grad_norm": nlp_trainer.config.max_grad_norm,
    }

    report_gen = PDFReportGenerator(reports_dir=reports_dir)

    # 1. Training Report
    report_gen.generate_training_report(
        "Training_Report.pdf",
        image_metrics=image_test_metrics,
        nlp_metrics=nlp_test_metrics,
        image_hyperparams=image_hyperparams,
        nlp_hyperparams=nlp_hyperparams,
        image_curves_path=image_curves_path,
        nlp_curves_path=nlp_curves_path,
    )

    # 2. Evaluation Report
    report_gen.generate_evaluation_report(
        "Evaluation_Report.pdf",
        image_metrics=image_test_metrics,
        nlp_metrics=nlp_test_metrics,
        image_cm_path=reports_dir / "confusion_matrix.png",
        nlp_cm_path=reports_dir / "nlp_confusion_matrix.png",
        image_roc_path=reports_dir / "roc_curves.png",
        nlp_roc_path=reports_dir / "nlp_roc_curves.png",
    )

    # 3. Model Comparison Report
    report_gen.generate_comparison_report(
        "Model_Comparison_Report.pdf",
        image_metrics=image_test_metrics,
        nlp_metrics=nlp_test_metrics,
        image_run_id=image_run_id,
        nlp_run_id=nlp_run_id,
        image_time=image_train_time,
        nlp_time=nlp_train_time,
        image_best_epoch=best_image_epoch,
        nlp_best_epoch=best_nlp_epoch,
    )

    print("\n" + "=" * 60)
    print("FINAL SYSTEM ENHANCEMENT SUMMARY")
    print("=" * 60)
    print(f"1. Final Image Model Test Accuracy: {image_test_metrics.get('test_accuracy', 0.0):.4f}")
    print(f"2. Final NLP Model Test Accuracy:   {nlp_test_metrics.get('test_accuracy', 0.0):.4f}")
    print("\n3. Saved Checkpoints Locations:")
    print(f"  - Image Model: {image_trainer.config.best_model_path}")
    print(f"  - NLP Model:   {nlp_trainer.config.best_model_path}")
    print("\n4. MLflow Experiments Run details:")
    print(
        f"  - Image Experiment: {image_trainer.config.mlflow_experiment_name} | Run ID: {image_run_id}"
    )
    print(
        f"  - NLP Experiment:   {nlp_trainer.config.mlflow_experiment_name}   | Run ID: {nlp_run_id}"
    )
    print("\n5. Reports Directory:")
    print(f"  - PDFs saved in: {reports_dir.absolute()}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
