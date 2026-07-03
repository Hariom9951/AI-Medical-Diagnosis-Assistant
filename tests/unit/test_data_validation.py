"""Unit Tests for Data Validation Component.

Verifies configuration parsers, image sanitization rules (in-place deletes),
mask checks, symptom CSV cleaning rules, and output reports verification.
"""

import csv
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml
from PIL import Image

from src.components.data_validation import DataValidation, DataValidationConfig
from src.utils.exceptions import AppStorageError, AppValidationError


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary folder directory."""
    return tmp_path


@pytest.fixture
def mock_validation_config_yaml(temp_dir: Path) -> Path:
    """Fixture creating a valid validation config YAML file pointing to temp_dir."""
    config_data = {
        "raw_images_dir": str(temp_dir / "raw_images"),
        "raw_symptoms_csv": str(temp_dir / "dataset.csv"),
        "cleaned_symptoms_csv": str(temp_dir / "cleaned_dataset.csv"),
        "reports_dir": str(temp_dir / "reports"),
        "expected_classes": ["COVID", "Normal"],
        "expected_dimensions": [100, 100],
        "supported_image_formats": [".png", ".jpg"],
    }
    config_path = temp_dir / "validation_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return config_path


def test_validation_config_load_success(mock_validation_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that DataValidationConfig loads config parameters correctly."""
    config = DataValidationConfig.from_yaml(mock_validation_config_yaml)
    assert config.raw_images_dir == temp_dir / "raw_images"
    assert config.raw_symptoms_csv == temp_dir / "dataset.csv"
    assert config.cleaned_symptoms_csv == temp_dir / "cleaned_dataset.csv"
    assert config.reports_dir == temp_dir / "reports"
    assert config.expected_classes == ["COVID", "Normal"]
    assert config.expected_dimensions == (100, 100)
    assert config.supported_image_formats == [".png", ".jpg"]


def test_validation_config_load_missing_keys(temp_dir: Path) -> None:
    """Verifies config loading raises AppValidationError if required keys are missing."""
    invalid_data = {
        "raw_images_dir": str(temp_dir / "raw_images"),
        # missing expected_classes and other elements
    }
    config_path = temp_dir / "invalid_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(invalid_data, f)

    with pytest.raises(AppValidationError) as excinfo:
        DataValidationConfig.from_yaml(config_path)
    assert "Missing required validation" in str(excinfo.value)


def test_validate_images_success(mock_validation_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies image validation sanitizes raw files in-place and checks masks."""
    raw_images_dir = temp_dir / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    # Setup directories
    classes = ["COVID", "Normal"]
    for cls in classes:
        (raw_images_dir / cls / "images").mkdir(parents=True, exist_ok=True)
        (raw_images_dir / cls / "masks").mkdir(parents=True, exist_ok=True)

    # 1. Clean valid image in Normal class
    normal_img = raw_images_dir / "Normal" / "images" / "normal_01.png"
    img1 = Image.new("L", (100, 100), color=128)
    img1.save(normal_img)

    # Create corresponding mask for it
    normal_mask = raw_images_dir / "Normal" / "masks" / "normal_01.png"
    img1.save(normal_mask)

    # 2. Duplicate image in Normal class (exact same pixels and hash)
    normal_dup = raw_images_dir / "Normal" / "images" / "normal_dup.png"
    img1.save(normal_dup)

    # 3. Valid image in COVID class with incorrect dimensions (50x50 px)
    covid_wrong_dims = raw_images_dir / "COVID" / "images" / "covid_dims.png"
    img2 = Image.new("L", (50, 50), color=50)
    img2.save(covid_wrong_dims)

    # (No mask for this COVID image to check missing masks detection)

    # 4. Corrupted image file (plain text renamed as png)
    corrupt_scan = raw_images_dir / "COVID" / "images" / "corrupt_01.png"
    corrupt_scan.write_text("corrupted metadata text data")

    # 5. Unsupported file format (e.g. .txt file in images folder)
    txt_scan = raw_images_dir / "Normal" / "images" / "readme.txt"
    txt_scan.write_text("plain documentation text file")

    # Run Data Validation
    validator = DataValidation(mock_validation_config_yaml)
    stats = validator.validate_images()

    assert stats["initial_images_count"] == 5
    assert stats["final_images_count"] == 2  # normal_01 and covid_dims are kept
    assert stats["deleted_unsupported"] == 1
    assert stats["deleted_corrupted"] == 1
    assert stats["deleted_duplicates"] == 1

    # Verified dimension check warnings
    assert stats["incorrect_dimensions_count"] == 1
    assert "covid_dims.png" in stats["incorrect_dims_list"][0]

    # Verified missing masks check warnings (covid_dims.png lacks a mask)
    assert stats["missing_masks_count"] == 1
    assert "covid_dims.png" in stats["missing_masks_list"][0]

    # Verify that deleted files are actually removed from disk
    assert not txt_scan.exists()
    assert not corrupt_scan.exists()
    assert not normal_dup.exists()
    assert normal_img.exists()
    assert covid_wrong_dims.exists()


def test_validate_symptoms_success(mock_validation_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies symptoms CSV cleaning drops duplicates, strips invalid labels, and counts nulls."""
    csv_path = temp_dir / "dataset.csv"
    cleaned_csv_path = temp_dir / "cleaned_dataset.csv"

    # Row 1: Valid Allergy
    # Row 2: Duplicate of row 1
    # Row 3: Invalid label (contains numbers)
    # Row 4: Valid GERD with missing Symptom_2
    data = [
        ["Disease", "Symptom_1", "Symptom_2", "Symptom_3"],
        ["Allergy", "sneezing", "itching", "tearing"],
        ["Allergy", "sneezing", "itching", "tearing"],  # Duplicate
        ["Allergy123", "sneezing", "itching", "tearing"],  # Invalid Label
        ["GERD", "cough", "", "heartburn"]  # Missing Cell
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    validator = DataValidation(mock_validation_config_yaml)
    stats = validator.validate_symptoms()

    assert stats["initial_records_count"] == 4
    assert stats["deleted_duplicates"] == 1
    assert stats["deleted_invalid_labels"] == 1
    assert stats["final_records_count"] == 2
    assert stats["null_cells_count"] == 1
    assert stats["missing_per_column"]["Symptom_2"] == 1

    # Verify cleaned file exists
    assert cleaned_csv_path.exists()
    df_cleaned = pd.read_csv(cleaned_csv_path)
    assert len(df_cleaned) == 2
    assert "Allergy123" not in df_cleaned["Disease"].values


def test_validate_symptoms_missing_disease(mock_validation_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies validation raises AppValidationError if the primary label column is missing."""
    csv_path = temp_dir / "dataset.csv"
    data = [
        ["Symptom_1", "Symptom_2"],
        ["cough", "fever"]
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    validator = DataValidation(mock_validation_config_yaml)
    with pytest.raises(AppValidationError) as excinfo:
        validator.validate_symptoms()
    assert "missing required primary label column" in str(excinfo.value).lower()


def test_generate_reports(mock_validation_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that all three reports are generated successfully."""
    validator = DataValidation(mock_validation_config_yaml)

    img_results = {
        "initial_images_count": 10,
        "final_images_count": 8,
        "deleted_unsupported": 1,
        "deleted_corrupted": 0,
        "deleted_duplicates": 1,
        "incorrect_dimensions_count": 0,
        "missing_masks_count": 2,
        "class_distribution": {"COVID": 4, "Normal": 4}
    }

    csv_results = {
        "initial_records_count": 100,
        "final_records_count": 95,
        "deleted_duplicates": 4,
        "deleted_invalid_labels": 1,
        "null_cells_count": 10,
        "missing_per_column": {"Symptom_4": 10}
    }

    validator.generate_reports(img_results, csv_results)

    reports_dir = temp_dir / "reports"
    assert (reports_dir / "Validation_Report.md").exists()
    assert (reports_dir / "validation_summary.json").exists()
    assert (reports_dir / "validation_statistics.csv").exists()

    # Content checks on validation statistics CSV
    stats_csv = reports_dir / "validation_statistics.csv"
    with open(stats_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    assert rows[0] == ["Metric", "Value"]
    assert ["initial_scans", "10"] in rows
    assert ["dropped_symptom_duplicates", "4"] in rows
