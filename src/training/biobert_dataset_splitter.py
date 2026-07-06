"""BioBERT Dataset Splitter — Phase 2.2.

This module splits the merged BioBERT dataset (created in Phase 2.1) into
stratified Train (80%), Validation (10%), and Test (10%) splits.
It preserves the class distribution, computes class weights for imbalance
mitigation, generates validation metrics/reports, and saves all outputs
under data/final/.

Outputs:
  data/final/train.csv
  data/final/validation.csv
  data/final/test.csv
  data/final/class_weights.json
  data/final/label_encoder.pkl
  data/final/split_statistics_report.md

Usage:
  python -m src.training.biobert_dataset_splitter
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.exceptions import AppStorageError, AppValidationError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_DATA_ROOT: Final[Path] = _REPO_ROOT / "data"

# Inputs
_MERGED_CSV: Final[Path] = _DATA_ROOT / "merged" / "biobert_merged_dataset.csv"
_DISEASE_MAPPING_JSON: Final[Path] = _DATA_ROOT / "processed" / "disease_mapping_41.json"

# Outputs
_FINAL_DIR: Final[Path] = _DATA_ROOT / "final"
_TRAIN_CSV: Final[Path] = _FINAL_DIR / "train.csv"
_VAL_CSV: Final[Path] = _FINAL_DIR / "validation.csv"
_TEST_CSV: Final[Path] = _FINAL_DIR / "test.csv"
_CLASS_WEIGHTS_JSON: Final[Path] = _FINAL_DIR / "class_weights.json"
_LABEL_ENCODER_PKL: Final[Path] = _FINAL_DIR / "label_encoder.pkl"
_SPLIT_REPORT_MD: Final[Path] = _FINAL_DIR / "split_statistics_report.md"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplitStats:
    """Statistics for dataset splits validation."""
    total_samples: int
    train_size: int
    val_size: int
    test_size: int
    num_classes: int
    has_duplicates: bool
    overlap_count: int
    every_disease_in_train: bool
    missing_diseases_in_train: List[str] = field(default_factory=list)
    per_class_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    class_weights: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Core Splitting Logic
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split_by_class(
    df: pd.DataFrame,
    disease_mapping: Dict[str, int],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into Train (80%), Val (10%), Test (10%) per class.

    Handles low-frequency classes robustly:
      - If N=1: 1 Train, 0 Val, 0 Test
      - If N=2: 1 Train, 1 Val, 0 Test
      - If N=3: 2 Train, 1 Val, 0 Test
      - If N=4: 2 Train, 1 Val, 1 Test
      - If N>=5: Train = N - val - test, Val = max(1, round(0.1*N)), Test = max(1, round(0.1*N))

    Args:
        df (pd.DataFrame): Input merged dataset.
        disease_mapping (Dict[str, int]): Canonical disease mapping.
        random_state (int): Random state seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, validation, test dataframes.
    """
    logger.info("Executing robust stratified split across splits...")
    train_rows: List[pd.Series] = []
    val_rows: List[pd.Series] = []
    test_rows: List[pd.Series] = []

    # Process each canonical disease class individually
    for disease_name in disease_mapping.keys():
        class_df = df[df["disease"] == disease_name]
        N = len(class_df)

        if N == 0:
            logger.warning("Canonical disease '%s' has 0 samples in merged dataset.", disease_name)
            continue

        # Shuffle the class subset
        class_shuffled = class_df.sample(frac=1, random_state=random_state)

        # Distribute rows based on class count N
        if N == 1:
            train_rows.append(class_shuffled.iloc[0])
        elif N == 2:
            train_rows.append(class_shuffled.iloc[0])
            val_rows.append(class_shuffled.iloc[1])
        elif N == 3:
            train_rows.append(class_shuffled.iloc[0])
            train_rows.append(class_shuffled.iloc[1])
            val_rows.append(class_shuffled.iloc[2])
        elif N == 4:
            train_rows.append(class_shuffled.iloc[0])
            train_rows.append(class_shuffled.iloc[1])
            val_rows.append(class_shuffled.iloc[2])
            test_rows.append(class_shuffled.iloc[3])
        else:
            # N >= 5
            n_val = max(1, int(round(N * 0.1)))
            n_test = max(1, int(round(N * 0.1)))
            n_train = N - n_val - n_test

            train_subset = class_shuffled.iloc[:n_train]
            val_subset = class_shuffled.iloc[n_train : n_train + n_val]
            test_subset = class_shuffled.iloc[n_train + n_val :]

            for _, r in train_subset.iterrows():
                train_rows.append(r)
            for _, r in val_subset.iterrows():
                val_rows.append(r)
            for _, r in test_subset.iterrows():
                test_rows.append(r)

    # Reassemble and shuffle splits
    train_df = pd.DataFrame(train_rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = pd.DataFrame(val_rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Class Weights calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    train_df: pd.DataFrame,
    disease_mapping: Dict[str, int],
) -> Dict[str, float]:
    """Computes balanced class weights for Weighted Cross-Entropy Loss.

    Formula:
        weight_c = total_train_samples / (num_classes * count_c)

    If a class has 0 samples, defaults its weight to 1.0.

    Args:
        train_df (pd.DataFrame): Training split DataFrame.
        disease_mapping (Dict[str, int]): Canonical label mapping.

    Returns:
        Dict[str, float]: Label indices as string keys mapped to computed float weights.
    """
    total_train = len(train_df)
    num_classes = len(disease_mapping)
    class_weights: Dict[str, float] = {}

    # Get count of each label in the training set
    train_counts = train_df["label"].value_counts().to_dict()

    for disease_name, label_idx in disease_mapping.items():
        count = train_counts.get(label_idx, 0)
        if count > 0:
            weight = total_train / (num_classes * count)
        else:
            # Fallback weight for classes with no training samples (e.g. absent from raw data)
            weight = 1.0
        class_weights[str(label_idx)] = float(round(weight, 6))

    return class_weights


# ─────────────────────────────────────────────────────────────────────────────
# Verification Logic
# ─────────────────────────────────────────────────────────────────────────────

def verify_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    disease_mapping: Dict[str, int],
    class_weights: Dict[str, float],
) -> SplitStats:
    """Verifies data integrity, leaks, and class coverage across splits.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Testing DataFrame.
        disease_mapping: Canonical disease labels mapping.
        class_weights: Dictionary of computed class weights.

    Returns:
        SplitStats: Summarized audit report data.
    """
    total_samples = len(train_df) + len(val_df) + len(test_df)

    # 1. Check for duplicates / data leakage across splits
    train_texts = set(train_df["text"].dropna().unique())
    val_texts = set(val_df["text"].dropna().unique())
    test_texts = set(test_df["text"].dropna().unique())

    overlap_train_val = len(train_texts.intersection(val_texts))
    overlap_train_test = len(train_texts.intersection(test_texts))
    overlap_val_test = len(val_texts.intersection(test_texts))
    overlap_count = overlap_train_val + overlap_train_test + overlap_val_test
    has_duplicates = (overlap_count > 0)

    # 2. Check if every disease class present in original mapping appears in training split
    train_diseases = set(train_df["disease"].unique())
    all_mapping_diseases = set(disease_mapping.keys())

    # Map merged dataset diseases (excluding the 2 already-missing ones in raw dataset)
    # The requirement is to check if every disease that we loaded is in training
    # Let's list any canonical diseases that are completely absent from training
    missing_diseases = sorted(all_mapping_diseases - train_diseases)
    # Filter out diseases that had 0 samples in the entire merged dataset
    # We want to know if any disease present in merged_df is missing from train
    merged_diseases = train_diseases | set(val_df["disease"].unique()) | set(test_df["disease"].unique())
    actual_missing_from_train = sorted(merged_diseases - train_diseases)
    every_disease_in_train = (len(actual_missing_from_train) == 0)

    # 3. Compile per-class sample count distributions
    idx_to_disease = {v: k for k, v in disease_mapping.items()}
    train_dist = train_df["disease"].value_counts().to_dict()
    val_dist = val_df["disease"].value_counts().to_dict()
    test_dist = test_df["disease"].value_counts().to_dict()

    per_class_counts: Dict[str, Dict[str, int]] = {}
    for label_idx in sorted(disease_mapping.values()):
        disease = idx_to_disease[label_idx]
        per_class_counts[disease] = {
            "train": train_dist.get(disease, 0),
            "val": val_dist.get(disease, 0),
            "test": test_dist.get(disease, 0),
            "total": train_dist.get(disease, 0) + val_dist.get(disease, 0) + test_dist.get(disease, 0),
        }

    return SplitStats(
        total_samples=total_samples,
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        num_classes=len(disease_mapping),
        has_duplicates=has_duplicates,
        overlap_count=overlap_count,
        every_disease_in_train=every_disease_in_train,
        missing_diseases_in_train=actual_missing_from_train,
        per_class_counts=per_class_counts,
        class_weights=class_weights,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def write_split_report(stats: SplitStats, output_path: Path) -> None:
    """Generates the Markdown report summarizing split sizes, class balance, weights.

    Args:
        stats (SplitStats): Populated SplitStats metadata.
        output_path (Path): Path to write the report file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build Class Distribution and Weights table
    table_rows: List[str] = []
    # Map index to weight
    weights_by_disease: Dict[str, float] = {}

    for disease_name, counts in sorted(stats.per_class_counts.items(), key=lambda x: -x[1]["total"]):
        train_cnt = counts["train"]
        val_cnt = counts["val"]
        test_cnt = counts["test"]
        total_cnt = counts["total"]
        weight = stats.class_weights.get(str(list(stats.per_class_counts.keys()).index(disease_name)), 1.0)
        # Re-resolve the correct class weight matching index
        table_rows.append(
            f"| {disease_name:<40} | {total_cnt:>6} | {train_cnt:>6} | {val_cnt:>5} | {test_cnt:>5} |"
        )

    table_body = "\n".join(table_rows)

    # Weights table by label index
    weights_rows: List[str] = []
    for idx_str, w in sorted(stats.class_weights.items(), key=lambda x: int(x[0])):
        weights_rows.append(f"| {idx_str:>5} | {w:>9.6f} |")
    weights_table_body = "\n".join(weights_rows)

    report_content = f"""# BioBERT Dataset Splits & Validation Report

*Generated: {timestamp}*

---

## 1. Split Allocation Summary

The merged dataset has been split into Train (80%), Validation (10%), and Test (10%) splits. A robust stratification pipeline was applied to ensure classes with low counts are allocated to splits without crashing.

| Split | Number of Samples | Ratio (%) |
|---|---|---|
| **Train** | {stats.train_size} | {stats.train_size / stats.total_samples * 100:.2f}% |
| **Validation** | {stats.val_size} | {stats.val_size / stats.total_samples * 100:.2f}% |
| **Test** | {stats.test_size} | {stats.test_size / stats.total_samples * 100:.2f}% |
| **Total** | {stats.total_samples} | 100.00% |

---

## 2. Integrity & Validation Audit

- **Data Leakage Check**: {'❌ **FAILED** (Overlap detected)' if stats.has_duplicates else '✅ **PASSED** (No text overlaps between Train/Val/Test)'}
  - Inter-split overlaps: `{stats.overlap_count}` sample(s).
- **Class Coverage Check**: {'❌ **FAILED**' if not stats.every_disease_in_train else '✅ **PASSED** (Every present disease exists in the training split)'}
  - Absent diseases from Train: `{stats.missing_diseases_in_train}`.

---

## 3. Class Distribution across Splits

| Disease | Total | Train | Val | Test |
|---|:---:|:---:|:---:|:---:|
{table_body}

---

## 4. Computed Class Weights for Cross-Entropy Loss

Class weights were computed based on the training split sample frequencies. Absent classes are defaulted to `1.0`.

| Label ID | Loss Weight |
|---|---|
{weights_table_body}

---

## 5. Generated Artifacts

The following output assets have been generated and saved under `data/final/`:

*   Training Set CSV: `data/final/train.csv`
*   Validation Set CSV: `data/final/validation.csv`
*   Testing Set CSV: `data/final/test.csv`
*   Class Weights JSON: `data/final/class_weights.json`
*   Label Encoder Pickle: `data/final/label_encoder.pkl`
*   This Split Report: `data/final/split_statistics_report.md`
"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("Saved dataset splitting statistics report successfully.")
    except Exception as e:
        raise AppStorageError(
            message=f"Failed to save split statistics report: {e}",
            details={"output_path": str(output_path), "error": str(e)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Class
# ─────────────────────────────────────────────────────────────────────────────

class BioBERTDatasetSplitter:
    """Orchestrates Phase 2.2 dataset splitting, class weighting, and validation."""

    def __init__(
        self,
        merged_csv: Path = _MERGED_CSV,
        disease_mapping_json: Path = _DISEASE_MAPPING_JSON,
        final_dir: Path = _FINAL_DIR,
    ) -> None:
        """Initializes the dataset splitter.

        Args:
            merged_csv (Path): Path to biobert_merged_dataset.csv.
            disease_mapping_json (Path): Path to disease_mapping_41.json.
            final_dir (Path): Output directory for final files.
        """
        self.merged_csv = merged_csv
        self.disease_mapping_json = disease_mapping_json
        self.final_dir = final_dir

        self.final_dir.mkdir(parents=True, exist_ok=True)
        logger.info("BioBERTDatasetSplitter initialised.")

    def run(self) -> Dict[str, Any]:
        """Runs the splitting and weight generation pipeline.

        Returns:
            Dict[str, Any]: Run metadata and split validation metrics.
        """
        logger.info("=" * 60)
        logger.info("BioBERT Dataset Splitter — Phase 2.2")
        logger.info("=" * 60)

        # 1. Load inputs
        if not self.merged_csv.exists():
            raise AppStorageError(
                message=f"Merged dataset file not found: {self.merged_csv}. Run Phase 2.1 builder first."
            )
        df = pd.read_csv(self.merged_csv)
        # Deduplicate by text to prevent any duplicates across Train, Val, and Test splits (data leakage prevention)
        logger.info("Deduplicating merged dataset by 'text' column. Rows before: %d", len(df))
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        logger.info("Rows after deduplication: %d", len(df))
        disease_mapping = load_disease_mapping(self.disease_mapping_json)

        # 2. Split dataset
        train_df, val_df, test_df = stratified_split_by_class(df, disease_mapping)

        # 3. Fit and save LabelEncoder
        label_encoder = LabelEncoder()
        sorted_classes = sorted(list(disease_mapping.keys()))
        label_encoder.classes_ = np.array(sorted_classes)

        # 4. Compute class weights
        class_weights = compute_class_weights(train_df, disease_mapping)

        # 5. Verify splits
        stats = verify_splits(train_df, val_df, test_df, disease_mapping, class_weights)

        # 6. Save final datasets
        train_df.to_csv(_TRAIN_CSV, index=False, encoding="utf-8")
        val_df.to_csv(_VAL_CSV, index=False, encoding="utf-8")
        test_df.to_csv(_TEST_CSV, index=False, encoding="utf-8")

        # Save class weights as JSON
        try:
            with open(_CLASS_WEIGHTS_JSON, "w", encoding="utf-8") as f:
                json.dump(class_weights, f, indent=4)
            logger.info("Saved class weights to %s", _CLASS_WEIGHTS_JSON)
        except Exception as e:
            raise AppStorageError(f"Failed to write class weights JSON: {e}")

        # Save label encoder as pickle
        try:
            with open(_LABEL_ENCODER_PKL, "wb") as f:
                pickle.dump(label_encoder, f)
            logger.info("Saved label encoder to %s", _LABEL_ENCODER_PKL)
        except Exception as e:
            raise AppStorageError(f"Failed to write label encoder pickle: {e}")

        # 7. Write Markdown Report
        write_split_report(stats, _SPLIT_REPORT_MD)

        logger.info("=" * 60)
        logger.info("BioBERT Dataset Splitter — COMPLETED SUCCESSFULLY")
        logger.info("  Train Size      : %d rows", stats.train_size)
        logger.info("  Validation Size : %d rows", stats.val_size)
        logger.info("  Test Size       : %d rows", stats.test_size)
        logger.info("  Report          : %s", _SPLIT_REPORT_MD)
        logger.info("=" * 60)

        return {
            "train_csv": str(_TRAIN_CSV),
            "val_csv": str(_VAL_CSV),
            "test_csv": str(_TEST_CSV),
            "class_weights_json": str(_CLASS_WEIGHTS_JSON),
            "label_encoder_pkl": str(_LABEL_ENCODER_PKL),
            "report_path": str(_SPLIT_REPORT_MD),
            "stats": stats,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper to reload mapping (similar to biobert_dataset_builder)
# ─────────────────────────────────────────────────────────────────────────────

def load_disease_mapping(mapping_path: Path) -> Dict[str, int]:
    """Loads canonical mapping from JSON."""
    if not mapping_path.exists():
        raise AppStorageError(f"Disease mapping file not found: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entrypoint."""
    import sys
    try:
        splitter = BioBERTDatasetSplitter()
        result = splitter.run()
        sys.exit(0)
    except Exception as e:
        logger.exception("Splitter execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
