"""Unit Tests for Dataset Verification & EDA Component.

Tests configuration parsing, image dataset verification stats, tabular profiling
for symptoms, plot exports, and Markdown report compiles using mock filesystem fixtures.
"""

import csv
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml
from PIL import Image

from src.components.dataset_verification_eda import DatasetVerificationEDA, EDAConfig
from src.utils.exceptions import AppStorageError, AppValidationError


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary folder directory."""
    return tmp_path


@pytest.fixture
def mock_eda_config_yaml(temp_dir: Path) -> Path:
    """Fixture creating a valid EDA config YAML file pointing to temp_dir."""
    config_data = {
        "raw_images_dir": str(temp_dir / "raw_images"),
        "raw_symptoms_csv": str(temp_dir / "dataset.csv"),
        "reports_dir": str(temp_dir / "reports"),
        "plots_dir": str(temp_dir / "plots"),
        "supported_image_formats": [".png", ".jpg"],
    }
    config_path = temp_dir / "eda_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return config_path


def test_eda_config_load_success(mock_eda_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that EDAConfig loads config parameters correctly."""
    config = EDAConfig.from_yaml(mock_eda_config_yaml)
    assert config.raw_images_dir == temp_dir / "raw_images"
    assert config.raw_symptoms_csv == temp_dir / "dataset.csv"
    assert config.reports_dir == temp_dir / "reports"
    assert config.plots_dir == temp_dir / "plots"
    assert config.supported_image_formats == [".png", ".jpg"]


def test_eda_config_load_missing_keys(temp_dir: Path) -> None:
    """Verifies EDAConfig throws AppValidationError on missing properties."""
    invalid_data = {
        "raw_images_dir": str(temp_dir / "raw_images"),
        # missing raw_symptoms_csv and other output dirs
    }
    config_path = temp_dir / "invalid_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(invalid_data, f)

    with pytest.raises(AppValidationError) as excinfo:
        EDAConfig.from_yaml(config_path)
    assert "Missing required configuration keys" in str(excinfo.value)


def test_verify_image_dataset(mock_eda_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies image audits count folders, corrupted, duplicate, and resolution metrics."""
    raw_images_dir = temp_dir / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Class Directories
    classes = ["COVID", "Normal"]
    for cls in classes:
        (raw_images_dir / cls).mkdir(parents=True, exist_ok=True)

    # Write a clean PNG image to Normal
    normal_img_path = raw_images_dir / "Normal" / "normal_01.png"
    img = Image.new("L", (100, 100), color=128)
    img.save(normal_img_path)

    # Write a duplicate PNG image to Normal (exact same content)
    dup_img_path = raw_images_dir / "Normal" / "normal_01_dup.png"
    img.save(dup_img_path)

    # Write a clean PNG image to COVID with different dimensions (50x80)
    covid_img_path = raw_images_dir / "COVID" / "covid_01.png"
    img_covid = Image.new("L", (50, 80), color=50)
    img_covid.save(covid_img_path)

    # Write a corrupted image file (just plain text named as .png)
    corrupt_path = raw_images_dir / "COVID" / "corrupt_01.png"
    corrupt_path.write_text("this is not an image file")

    # Write an unsupported format file (txt format)
    txt_path = raw_images_dir / "Normal" / "readme.txt"
    txt_path.write_text("documentation instructions")

    # Create an empty subdirectory to test empty folder detection
    empty_subfolder = raw_images_dir / "Normal" / "empty_dir"
    empty_subfolder.mkdir(parents=True, exist_ok=True)

    # Run verification pipeline
    verification = DatasetVerificationEDA(mock_eda_config_yaml)
    stats = verification.verify_image_dataset()

    assert stats["total_images"] == 4  # Normal (2), COVID (2) - txt is excluded
    assert stats["classes"]["Normal"] == 2
    assert stats["classes"]["COVID"] == 2

    # Check corrupted count
    assert len(stats["corrupted"]) == 1
    assert "corrupt_01.png" in stats["corrupted"][0]

    # Check duplicates count
    assert len(stats["duplicates"]) == 1  # 1 duplicate hash list

    # Check unsupported format count
    assert len(stats["unsupported_files"]) == 1
    assert "readme.txt" in stats["unsupported_files"][0]

    # Check empty folder count
    assert len(stats["empty_folders"]) == 1
    assert "empty_dir" in stats["empty_folders"][0]

    # Check resolution statistics (only includes successfully opened images: 3 files)
    assert len(stats["widths"]) == 3
    assert 100 in stats["widths"]
    assert 50 in stats["widths"]
    assert 80 in stats["heights"]

    # Verify distribution plots exist
    plots_dir = temp_dir / "plots"
    assert (plots_dir / "image_class_distribution.png").exists()
    assert (plots_dir / "image_resolutions_histogram.png").exists()

    # Verify random samples copied (up to 5 per class)
    assert (plots_dir / "samples" / "Normal").exists()
    assert (plots_dir / "samples" / "COVID").exists()
    assert len(stats["exported_samples"]["Normal"]) == 2  # we only had 2 valid image candidates


def test_verify_image_dataset_missing_folder(mock_eda_config_yaml: Path) -> None:
    """Verifies that verification raises AppStorageError if source folder is missing."""
    verification = DatasetVerificationEDA(mock_eda_config_yaml)
    with pytest.raises(AppStorageError) as excinfo:
        verification.verify_image_dataset()
    assert "Raw image directory does not exist" in str(excinfo.value)


def test_verify_symptom_dataset(mock_eda_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies symptom CSV profiling reports row/col dims, duplicate count, and missing counts."""
    csv_path = temp_dir / "dataset.csv"
    
    # Generate mock symptoms metadata CSV
    # Missing values: 'age' missing on row 2, 'gender' missing on row 3
    # Row 4 is duplicate of Row 1
    data = [
        ["Disease", "Symptom_1", "Symptom_2", "age", "gender"],
        ["Allergy", "sneezing", "itching", "22", "M"],
        ["Allergy", "sneezing", "itching", "", "F"],
        ["GERD", "cough", "chest_pain", "45", ""],
        ["Allergy", "sneezing", "itching", "22", "M"],  # duplicate
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    verification = DatasetVerificationEDA(mock_eda_config_yaml)
    stats = verification.verify_symptom_dataset()

    assert stats["row_count"] == 4
    assert stats["col_count"] == 5
    assert stats["duplicate_rows"] == 1
    assert stats["missing_values"]["age"] == 1
    assert stats["missing_values"]["gender"] == 1
    assert stats["empty_records"] == 0

    # Verify generated plots
    plots_dir = temp_dir / "plots"
    assert (plots_dir / "symptom_missingness_heatmap.png").exists()
    assert (plots_dir / "disease_frequency.png").exists()
    assert (plots_dir / "symptom_frequency.png").exists()


def test_verify_symptom_dataset_missing_file(mock_eda_config_yaml: Path) -> None:
    """Verifies profiling raises AppStorageError if symptoms file is missing."""
    verification = DatasetVerificationEDA(mock_eda_config_yaml)
    with pytest.raises(AppStorageError) as excinfo:
        verification.verify_symptom_dataset()
    assert "Raw symptoms CSV does not exist" in str(excinfo.value)


def test_generate_reports(mock_eda_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that all three reports (Verification, EDA, Summary) are written to docs."""
    verification = DatasetVerificationEDA(mock_eda_config_yaml)

    # Prepare mock stats dictionaries
    img_stats = {
        "total_images": 10,
        "classes": {"COVID": 5, "Normal": 5},
        "corrupted": ["corrupt_01.png"],
        "duplicates": {"hash1": ["file1.png", "file2.png"]},
        "empty_folders": [],
        "unsupported_files": [],
        "avg_size_bytes": 10240.0,
        "widths": [100, 100],
        "heights": [100, 100],
        "aspect_ratios": [1.0, 1.0]
    }

    symptom_stats = {
        "row_count": 100,
        "col_count": 5,
        "duplicate_rows": 2,
        "invalid_labels": [],
        "empty_records": 0,
        "missing_values": {"age": 5, "gender": 2},
        "top_symptoms": {"itching": 50, "cough": 30}
    }

    verification.generate_reports(img_stats, symptom_stats)

    reports_dir = temp_dir / "reports"
    assert (reports_dir / "Dataset_Verification_Report.md").exists()
    assert (reports_dir / "EDA_Report.md").exists()
    assert (reports_dir / "Summary_Report.md").exists()

    # Read content checks
    ver_content = (reports_dir / "Dataset_Verification_Report.md").read_text()
    assert "**Corrupted Images Count:** `1`" in ver_content
    assert "**Duplicate Rows Count:** `2`" in ver_content

    eda_content = (reports_dir / "EDA_Report.md").read_text()
    assert "**Average File Size:** `10.00 KB`" in eda_content
    assert "**itching:** `50 occurrences`" in eda_content
