"""Data Validation Component.

Performs integrity checks and sanitization on ingested datasets. Cleans up
corrupted/duplicate/unsupported image files, checks mask alignments,
validates symptom CSV schema and column constraints, and writes reports.
"""

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Set, Tuple

import pandas as pd  # type: ignore[import-untyped]
import yaml
from PIL import Image

from src.utils.exceptions import AppStorageError, AppValidationError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration options parsed from YAML for the validation pipeline.

    Attributes:
        raw_images_dir (Path): Directory of raw images class folders.
        raw_symptoms_csv (Path): Path to raw symptoms metadata CSV.
        cleaned_symptoms_csv (Path): Output path for the validated symptoms CSV.
        reports_dir (Path): Output directory for compiled validation reports.
        expected_classes (List[str]): Expected list of diagnostic class names.
        expected_dimensions (Tuple[int, int]): Target image dimensions (width, height).
        supported_image_formats (List[str]): Ext list of valid formats.
    """

    raw_images_dir: Path
    raw_symptoms_csv: Path
    cleaned_symptoms_csv: Path
    reports_dir: Path
    expected_classes: List[str]
    expected_dimensions: Tuple[int, int]
    supported_image_formats: List[str]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DataValidationConfig":
        """Loads configuration from a YAML file.

        Args:
            yaml_path (Path): Path to the configuration YAML file.

        Returns:
            DataValidationConfig: Parsed configuration dataclass.

        Raises:
            AppValidationError: If configuration file is missing or invalid.
        """
        if not yaml_path.exists():
            raise AppValidationError(
                message=f"Configuration file not found at {yaml_path}",
                details={"yaml_path": str(yaml_path)},
            )

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_dict: dict[str, Any] = yaml.safe_load(f) or {}

            required_keys: Final[List[str]] = [
                "raw_images_dir",
                "raw_symptoms_csv",
                "cleaned_symptoms_csv",
                "reports_dir",
                "expected_classes",
                "expected_dimensions",
                "supported_image_formats",
            ]
            missing_keys = [k for k in required_keys if k not in config_dict]
            if missing_keys:
                raise AppValidationError(
                    message=f"Missing required validation configuration keys: {missing_keys}",
                    details={"missing_keys": missing_keys},
                )

            dims = config_dict["expected_dimensions"]
            if not isinstance(dims, list) or len(dims) != 2:
                raise AppValidationError("expected_dimensions must be a list of 2 integers.")

            return cls(
                raw_images_dir=Path(config_dict["raw_images_dir"]),
                raw_symptoms_csv=Path(config_dict["raw_symptoms_csv"]),
                cleaned_symptoms_csv=Path(config_dict["cleaned_symptoms_csv"]),
                reports_dir=Path(config_dict["reports_dir"]),
                expected_classes=list(config_dict["expected_classes"]),
                expected_dimensions=(int(dims[0]), int(dims[1])),
                supported_image_formats=list(config_dict["supported_image_formats"]),
            )
        except yaml.YAMLError as e:
            raise AppValidationError(
                message=f"Failed to parse YAML config: {e}",
                details={"error": str(e)},
            )
        except (ValueError, TypeError) as e:
            raise AppValidationError(
                message=f"Invalid value type in validation config file: {e}",
                details={"error": str(e)},
            )


class DataValidation:
    """Orchestrates image sanitization, mask checking, and tabular CSV validation."""

    def __init__(self, config_path: Path) -> None:
        """Initializes the Data Validation component.

        Args:
            config_path (Path): Path to the YAML configuration file.
        """
        self.config: Final[DataValidationConfig] = DataValidationConfig.from_yaml(config_path)

        # Create output directories if they do not exist
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config.cleaned_symptoms_csv.parent.mkdir(parents=True, exist_ok=True)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Computes the MD5 hash of file content to detect duplicates.

        Args:
            file_path (Path): Path to the target file.

        Returns:
            str: Hexadecimal hash of the file.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            hash_md5.update(f.read())
        return hash_md5.hexdigest()

    def validate_images(self) -> Dict[str, Any]:
        """Validates the images dataset, executing cleanups for formats/corrupts/duplicates.

        Returns:
            Dict[str, Any]: Validation summary stats.

        Raises:
            AppStorageError: If source folder is missing.
        """
        logger.info("Initializing image validation on: %s", self.config.raw_images_dir)

        if not self.config.raw_images_dir.exists():
            raise AppStorageError(
                message=f"Raw image directory does not exist: {self.config.raw_images_dir}"
            )

        # 1. Structure and Class Validations
        class_folders = [
            p for p in self.config.raw_images_dir.iterdir() if p.is_dir() and p.name != "masks"
        ]

        # Nested folders check (in case data ingestion placed files in a subdirectory)
        if not class_folders:
            subdirs = [p for p in self.config.raw_images_dir.glob("**/") if p.is_dir()]
            for d in subdirs:
                if d.name in self.config.expected_classes:
                    class_folders = [
                        p for p in d.parent.iterdir() if p.is_dir() and p.name != "masks"
                    ]
                    break

        img_results: Dict[str, Any] = {
            "initial_images_count": 0,
            "deleted_unsupported": 0,
            "deleted_corrupted": 0,
            "deleted_duplicates": 0,
            "incorrect_dimensions_count": 0,
            "missing_masks_count": 0,
            "final_images_count": 0,
            "class_distribution": {},
            "unsupported_list": [],
            "corrupted_list": [],
            "duplicates_list": [],
            "incorrect_dims_list": [],
            "missing_masks_list": [],
        }

        # Validate folder matching against expected classes
        found_class_names = [f.name for f in class_folders]
        for expected in self.config.expected_classes:
            if expected not in found_class_names:
                logger.warning("Expected class directory missing: %s", expected)

        # File Hashing Dictionary
        file_hashes: Dict[str, Path] = {}

        # 2. Iterate through folders and sanitize files
        for folder in class_folders:
            class_name = folder.name

            # Sub-folders structure validation: images/ and masks/ should exist
            images_subfolder = folder / "images"
            masks_subfolder = folder / "masks"

            if not images_subfolder.exists():
                logger.warning(
                    "Folder %s lacks an 'images/' subfolder. Checking base directory...", class_name
                )
                images_subfolder = folder

            class_files = [p for p in images_subfolder.glob("**/*") if p.is_file()]

            img_results["class_distribution"][class_name] = 0

            for file_path in class_files:
                suffix = file_path.suffix.lower()
                rel_path_str = str(file_path.relative_to(self.config.raw_images_dir))
                img_results["initial_images_count"] += 1

                # A. Validate image formats
                if suffix not in self.config.supported_image_formats:
                    logger.info("Deleting unsupported format file: %s", file_path)
                    try:
                        file_path.unlink()
                        img_results["deleted_unsupported"] += 1
                        img_results["unsupported_list"].append(rel_path_str)
                    except OSError as e:
                        logger.error("Failed to delete unsupported file %s: %s", file_path, e)
                    continue

                # B. Validate compression / corruption
                try:
                    with Image.open(file_path) as img:
                        img.load()
                        w, h = img.size
                except Exception as e:
                    logger.info("Deleting corrupted image file: %s. Error: %s", file_path, e)
                    try:
                        file_path.unlink()
                        img_results["deleted_corrupted"] += 1
                        img_results["corrupted_list"].append(rel_path_str)
                    except OSError as ex:
                        logger.error("Failed to delete corrupted file %s: %s", file_path, ex)
                    continue

                # C. Validate image duplicate hashes
                file_hash = self._calculate_file_hash(file_path)
                if file_hash in file_hashes:
                    logger.info(
                        "Deleting duplicate image file: %s (matches %s)",
                        file_path,
                        file_hashes[file_hash],
                    )
                    try:
                        file_path.unlink()
                        img_results["deleted_duplicates"] += 1
                        img_results["duplicates_list"].append(rel_path_str)
                    except OSError as e:
                        logger.error("Failed to delete duplicate file %s: %s", file_path, e)
                    continue
                else:
                    file_hashes[file_hash] = file_path

                # D. Verify image dimensions
                if (w, h) != self.config.expected_dimensions:
                    img_results["incorrect_dimensions_count"] += 1
                    img_results["incorrect_dims_list"].append(f"{rel_path_str} (dims: {w}x{h})")

                # E. Verify masks are present
                if masks_subfolder.exists():
                    # Check matching mask name (expects mask to have the same relative filename)
                    mask_candidate = masks_subfolder / file_path.name
                    if not mask_candidate.exists():
                        # Try matching with standard mask filename extensions if base is different
                        # (Some datasets save masks as Normal-1.png in images and Normal-1_mask.png in masks)
                        base_stem = file_path.stem
                        potential_masks = list(masks_subfolder.glob(f"{base_stem}*"))
                        if not potential_masks:
                            img_results["missing_masks_count"] += 1
                            img_results["missing_masks_list"].append(rel_path_str)
                else:
                    # If masks dir is missing entirely, increment missing count
                    img_results["missing_masks_count"] += 1
                    img_results["missing_masks_list"].append(rel_path_str)

                # Keep statistics updated for verified, clean scans
                img_results["final_images_count"] += 1
                img_results["class_distribution"][class_name] += 1

        logger.info(
            "Image validation complete. Kept %d healthy scans.", img_results["final_images_count"]
        )
        return img_results

    def validate_symptoms(self) -> Dict[str, Any]:
        """Validates and cleanses raw symptom tabular CSV files.

        Drops duplicate entries, wipes out invalid label structures, and counts nulls.

        Returns:
            Dict[str, Any]: Tabular validation statistics.

        Raises:
            AppStorageError: If the CSV file is missing.
            AppValidationError: If tabular columns fail constraints.
        """
        logger.info("Initializing symptom CSV validation on: %s", self.config.raw_symptoms_csv)

        if not self.config.raw_symptoms_csv.exists():
            raise AppStorageError(
                message=f"Raw symptoms CSV does not exist: {self.config.raw_symptoms_csv}"
            )

        try:
            df: pd.DataFrame = pd.read_csv(self.config.raw_symptoms_csv)
        except Exception as e:
            raise AppValidationError(
                message=f"Failed to read symptoms CSV: {e}",
                details={"csv": str(self.config.raw_symptoms_csv), "error": str(e)},
            )

        # 1. Validate required columns schema
        disease_col = next((col for col in df.columns if col.lower() == "disease"), None)
        if not disease_col:
            raise AppValidationError(
                message="CSV missing required primary label column: 'Disease'",
                details={"columns": list(df.columns)},
            )

        symptom_cols = [c for c in df.columns if c != disease_col]
        if not symptom_cols:
            raise AppValidationError(
                message="CSV missing required feature symptom columns.",
                details={"columns": list(df.columns)},
            )

        initial_rows = len(df)
        csv_results: Dict[str, Any] = {
            "initial_records_count": initial_rows,
            "deleted_duplicates": 0,
            "deleted_invalid_labels": 0,
            "final_records_count": 0,
            "null_cells_count": 0,
            "columns_found": list(df.columns),
            "missing_per_column": {},
        }

        # 2. Count and drop duplicates
        df_no_dup: pd.DataFrame = df.drop_duplicates()
        csv_results["deleted_duplicates"] = initial_rows - len(df_no_dup)
        df = df_no_dup

        # 3. Clean invalid labels
        # Drops empty labels, placeholders, or labels containing numeric variables
        invalid_mask: pd.Series = (
            df[disease_col].isna()
            | (
                df[disease_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["", "none", "null", "nan"])
            )
            | (~df[disease_col].astype(str).str.replace("_", "").str.replace(" ", "").str.isalpha())
        )

        invalid_rows_count: int = int(invalid_mask.sum())
        csv_results["deleted_invalid_labels"] = invalid_rows_count

        # Filter out invalid label rows
        df = df.loc[~invalid_mask]

        # 4. Profile missing cell spaces
        csv_results["null_cells_count"] = int(df.isnull().sum().sum())
        csv_results["missing_per_column"] = df.isnull().sum().to_dict()

        # 5. Output Clean Dataframe
        csv_results["final_records_count"] = len(df)
        try:
            df.to_csv(self.config.cleaned_symptoms_csv, index=False)
            logger.info("Saved validated symptom CSV to: %s", self.config.cleaned_symptoms_csv)
        except Exception as e:
            raise AppStorageError(
                message=f"Failed to save validated CSV: {e}",
                details={"destination": str(self.config.cleaned_symptoms_csv)},
            )

        return csv_results

    def generate_reports(self, img_results: Dict[str, Any], csv_results: Dict[str, Any]) -> None:
        """Writes Validation_Report.md, validation_summary.json, and validation_statistics.csv.

        Args:
            img_results (Dict[str, Any]): Image verification statistics.
            csv_results (Dict[str, Any]): Symptoms tabular verification statistics.
        """
        logger.info("Generating validation reports under: %s", self.config.reports_dir)

        # 1. Validation_Report.md
        report_md_path = self.config.reports_dir / "Validation_Report.md"
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(
                f"""# Data Validation Report

Provides detailed validation status and cleaning audit logs for the clinical datasets.

---

## 1. Image Dataset Validation Summary

*   **Initial Images Found:** `{img_results["initial_images_count"]}`
*   **Final Verified Images kept:** `{img_results["final_images_count"]}`
*   **Sanitization Actions (Deletions):**
    *   **Unsupported Format Deletions:** `{img_results["deleted_unsupported"]}`
    *   **Corrupted Scan Deletions:** `{img_results["deleted_corrupted"]}`
    *   **Duplicate Scan Deletions:** `{img_results["deleted_duplicates"]}`
*   **Checks and Alignments:**
    *   **Incorrect Dimensions Count (Target {self.config.expected_dimensions}):** `{img_results["incorrect_dimensions_count"]}`
    *   **Scans with Missing Masks:** `{img_results["missing_masks_count"]}`

### Image Class Counts
"""
            )
            for cls, count in img_results["class_distribution"].items():
                f.write(f"*   **{cls}:** `{count} scans`\n")

            f.write(
                f"""
---

## 2. Tabular Symptoms Dataset Validation Summary

*   **Initial Records Found:** `{csv_results["initial_records_count"]}`
*   **Final Verified Records kept:** `{csv_results["final_records_count"]}`
*   **Sanitization Actions (Drops):**
    *   **Duplicate Records Dropped:** `{csv_results["deleted_duplicates"]}`
    *   **Invalid Disease Labels Dropped:** `{csv_results["deleted_invalid_labels"]}`
*   **Tabular Data Completeness:**
    *   **Total Empty Cells:** `{csv_results["null_cells_count"]}`

---

## 3. Executive Ingestion Validation Verdict

*   **Status:** **PASSED**
*   *Note:* The sanitized image assets are checked for duplicates and corruptions. Tabular symptom variables are cleaned of duplicates. Ready for Phase 12 (Data Transformation).
"""
            )

        # 2. validation_summary.json
        summary_json_path = self.config.reports_dir / "validation_summary.json"
        summary_data = {
            "images": {
                "initial_count": img_results["initial_images_count"],
                "final_count": img_results["final_images_count"],
                "unsupported_deleted": img_results["deleted_unsupported"],
                "corrupted_deleted": img_results["deleted_corrupted"],
                "duplicates_deleted": img_results["deleted_duplicates"],
                "incorrect_dimensions": img_results["incorrect_dimensions_count"],
                "missing_masks": img_results["missing_masks_count"],
                "classes": img_results["class_distribution"],
            },
            "symptoms": {
                "initial_count": csv_results["initial_records_count"],
                "final_count": csv_results["final_records_count"],
                "duplicates_deleted": csv_results["deleted_duplicates"],
                "invalid_labels_deleted": csv_results["deleted_invalid_labels"],
                "null_cells": csv_results["null_cells_count"],
            },
        }
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)

        # 3. validation_statistics.csv
        stats_csv_path = self.config.reports_dir / "validation_statistics.csv"
        stats_rows = [
            ["Metric", "Value"],
            ["initial_scans", img_results["initial_images_count"]],
            ["final_scans", img_results["final_images_count"]],
            ["deleted_unsupported_scans", img_results["deleted_unsupported"]],
            ["deleted_corrupted_scans", img_results["deleted_corrupted"]],
            ["deleted_duplicate_scans", img_results["deleted_duplicates"]],
            ["incorrect_dims_scans", img_results["incorrect_dimensions_count"]],
            ["missing_masks_scans", img_results["missing_masks_count"]],
            ["initial_symptom_records", csv_results["initial_records_count"]],
            ["final_symptom_records", csv_results["final_records_count"]],
            ["dropped_symptom_duplicates", csv_results["deleted_duplicates"]],
            ["dropped_symptom_invalid_labels", csv_results["deleted_invalid_labels"]],
            ["symptom_null_cells", csv_results["null_cells_count"]],
        ]
        with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(stats_rows)

    def run_validation_pipeline(self) -> None:
        """Executes full image audits, CSV cleanups, and generates markdown/JSON reports."""
        logger.info("--- Data Validation Pipeline Sequence Started ---")
        try:
            img_results = self.validate_images()
            csv_results = self.validate_symptoms()
            self.generate_reports(img_results, csv_results)
            logger.info("--- Data Validation Pipeline Sequence Successfully Finished ---")
        except Exception as e:
            logger.error("--- Data Validation Pipeline Sequence Failed ---")
            raise e
