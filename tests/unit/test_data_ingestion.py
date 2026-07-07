"""Unit Tests for Kaggle Data Ingestion Component.

Tests configuration loading, reliable mock-based downloads using the Kaggle SDK,
retry logic, archive decompression, and path traversal security boundary validations.
"""

import io
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Mock kaggle module if not installed to prevent import/patch errors in CI
try:
    import kaggle  # type: ignore[import-not-found,import-untyped]
except ImportError:
    kaggle_mock = MagicMock()
    sys.modules["kaggle"] = kaggle_mock
    sys.modules["kaggle.api"] = kaggle_mock.api
    sys.modules["kaggle.api.kaggle_api_extended"] = kaggle_mock.api.kaggle_api_extended

    class KaggleApi:
        def authenticate(self) -> None:
            pass

        def dataset_download_files(self, *args, **kwargs) -> None:
            pass

    kaggle_mock.api.kaggle_api_extended.KaggleApi = KaggleApi

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.utils.exceptions import AppStorageError, AppValidationError


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for test storage."""
    return tmp_path


@pytest.fixture
def valid_config_yaml(temp_dir: Path) -> Path:
    """Fixture providing a path to a valid YAML config file."""
    config_data = {
        "image_dataset_slug": "username/image-dataset",
        "symptom_dataset_slug": "username/symptom-dataset",
        "download_dir": str(temp_dir / "downloads"),
        "extract_dir": str(temp_dir / "raw"),
        "processed_dir": str(temp_dir / "processed"),
        "interim_dir": str(temp_dir / "interim"),
        "max_retries": 2,
        "backoff_factor": 0.1,
        "chunk_size": 1024,
    }
    config_path = temp_dir / "ingestion_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return config_path


def test_config_load_success(valid_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that DataIngestionConfig loads values correctly from YAML."""
    config = DataIngestionConfig.from_yaml(valid_config_yaml)
    assert config.image_dataset_slug == "username/image-dataset"
    assert config.symptom_dataset_slug == "username/symptom-dataset"
    assert config.download_dir == temp_dir / "downloads"
    assert config.extract_dir == temp_dir / "raw"
    assert config.processed_dir == temp_dir / "processed"
    assert config.interim_dir == temp_dir / "interim"
    assert config.max_retries == 2
    assert config.backoff_factor == 0.1
    assert config.chunk_size == 1024


def test_config_load_missing_keys(temp_dir: Path) -> None:
    """Verifies config loading raises AppValidationError if required keys are missing."""
    invalid_data = {
        "image_dataset_slug": "username/image-dataset",
        # missing downloads and other folders
    }
    config_path = temp_dir / "invalid_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(invalid_data, f)

    with pytest.raises(AppValidationError) as excinfo:
        DataIngestionConfig.from_yaml(config_path)
    assert "Missing required configuration keys" in str(excinfo.value)


def test_config_load_file_not_found() -> None:
    """Verifies config loading raises AppValidationError when file does not exist."""
    non_existent = Path("non_existent_config.yaml")
    with pytest.raises(AppValidationError) as excinfo:
        DataIngestionConfig.from_yaml(non_existent)
    assert "Configuration file not found" in str(excinfo.value)


@patch("kaggle.api.kaggle_api_extended.KaggleApi.authenticate")
@patch("kaggle.api.kaggle_api_extended.KaggleApi.dataset_download_files")
def test_download_kaggle_dataset_success(
    mock_download: MagicMock, mock_auth: MagicMock, valid_config_yaml: Path, temp_dir: Path
) -> None:
    """Verifies successful download writes the mock zip target file."""

    # Write empty target file on mock call to simulate successful SDK download
    def download_side_effect(*args: Any, **kwargs: Any) -> None:
        target = temp_dir / "downloads" / "image-dataset.zip"
        target.write_bytes(b"zipfile_mock_bytes")

    mock_download.side_effect = download_side_effect

    ingestion = DataIngestion(valid_config_yaml)
    zip_path = ingestion.download_kaggle_dataset("username/image-dataset")

    expected_path = temp_dir / "downloads" / "image-dataset.zip"
    assert zip_path == expected_path
    assert zip_path.exists()
    assert zip_path.read_bytes() == b"zipfile_mock_bytes"
    mock_auth.assert_called_once()
    mock_download.assert_called_once_with(
        dataset="username/image-dataset", path=str(temp_dir / "downloads"), unzip=False, quiet=False
    )


@patch("kaggle.api.kaggle_api_extended.KaggleApi.authenticate")
@patch("kaggle.api.kaggle_api_extended.KaggleApi.dataset_download_files")
def test_download_kaggle_dataset_idempotence(
    mock_download: MagicMock, mock_auth: MagicMock, valid_config_yaml: Path, temp_dir: Path
) -> None:
    """Verifies that downloading is skipped if the local target zip already exists."""
    downloads_dir = temp_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    target_zip = downloads_dir / "image-dataset.zip"
    target_zip.write_bytes(b"existing_mock_zip")

    ingestion = DataIngestion(valid_config_yaml)
    zip_path = ingestion.download_kaggle_dataset("username/image-dataset")

    assert zip_path == target_zip
    mock_auth.assert_not_called()
    mock_download.assert_not_called()


@patch("kaggle.api.kaggle_api_extended.KaggleApi.authenticate")
def test_download_kaggle_dataset_auth_failure(
    mock_auth: MagicMock, valid_config_yaml: Path
) -> None:
    """Verifies AppStorageError is raised if Kaggle API authentication fails."""
    mock_auth.side_effect = Exception("Credentials rejected")

    ingestion = DataIngestion(valid_config_yaml)
    with pytest.raises(AppStorageError) as excinfo:
        ingestion.download_kaggle_dataset("username/image-dataset")

    assert "Kaggle API authentication failed" in str(excinfo.value)


@patch("kaggle.api.kaggle_api_extended.KaggleApi.authenticate")
@patch("kaggle.api.kaggle_api_extended.KaggleApi.dataset_download_files")
@patch("time.sleep")
def test_download_kaggle_dataset_retries_and_succeeds(
    mock_sleep: MagicMock,
    mock_download: MagicMock,
    mock_auth: MagicMock,
    valid_config_yaml: Path,
    temp_dir: Path,
) -> None:
    """Verifies download retries on SDK failures and succeeds if a retry passes."""
    call_count = 0

    def download_side_effect(*args: Any, **kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Kaggle Rate Limit / Timeout")
        # Write zip on second call
        target = temp_dir / "downloads" / "image-dataset.zip"
        target.write_bytes(b"recovered_zip_bytes")

    mock_download.side_effect = download_side_effect

    ingestion = DataIngestion(valid_config_yaml)
    zip_path = ingestion.download_kaggle_dataset("username/image-dataset")

    assert zip_path.read_bytes() == b"recovered_zip_bytes"
    assert mock_download.call_count == 2
    mock_sleep.assert_called_once()


@patch("kaggle.api.kaggle_api_extended.KaggleApi.authenticate")
@patch("kaggle.api.kaggle_api_extended.KaggleApi.dataset_download_files")
@patch("time.sleep")
def test_download_kaggle_dataset_all_retries_fail(
    mock_sleep: MagicMock, mock_download: MagicMock, mock_auth: MagicMock, valid_config_yaml: Path
) -> None:
    """Verifies AppStorageError is raised when all download attempts fail."""
    mock_download.side_effect = Exception("Kaggle Connection Error")

    ingestion = DataIngestion(valid_config_yaml)
    with pytest.raises(AppStorageError) as excinfo:
        ingestion.download_kaggle_dataset("username/image-dataset")

    assert "Failed to download dataset" in str(excinfo.value)
    assert mock_download.call_count == 2


def test_extract_zip_success(valid_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that zip files are correctly extracted to raw subdirectory."""
    ingestion = DataIngestion(valid_config_yaml)

    # Create dummy zip bytes
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w") as zf:
        zf.writestr("test_file.txt", "zip_contents")
    zip_bytes.seek(0)

    zip_path = temp_dir / "downloads" / "dataset.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(zip_bytes.read())

    extracted_dir = ingestion.extract_file(zip_path, "sub_dataset")

    assert extracted_dir == temp_dir / "raw" / "sub_dataset"
    assert (extracted_dir / "test_file.txt").read_text() == "zip_contents"


def test_extract_tar_success(valid_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that tar files are correctly extracted to raw subdirectory."""
    ingestion = DataIngestion(valid_config_yaml)

    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tf:
        tarinfo = tarfile.TarInfo(name="test_file.txt")
        tarinfo.size = len(b"tar_contents")
        tf.addfile(tarinfo, io.BytesIO(b"tar_contents"))
    tar_bytes.seek(0)

    tar_path = temp_dir / "downloads" / "dataset.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tar_path.write_bytes(tar_bytes.read())

    extracted_dir = ingestion.extract_file(tar_path, "sub_tar")
    assert (extracted_dir / "test_file.txt").read_text() == "tar_contents"


def test_extract_unsupported_format(valid_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies extracting an unsupported format raises AppValidationError."""
    ingestion = DataIngestion(valid_config_yaml)
    txt_file = temp_dir / "downloads" / "dataset.txt"
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    txt_file.write_text("plain text")

    with pytest.raises(AppValidationError) as excinfo:
        ingestion.extract_file(txt_file, "sub_txt")
    assert "Unsupported archive file format" in str(excinfo.value)


def test_extraction_cleanup_on_failure(valid_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that extracting a corrupt file deletes any partial extraction directories."""
    ingestion = DataIngestion(valid_config_yaml)

    corrupt_zip = temp_dir / "downloads" / "dataset.zip"
    corrupt_zip.parent.mkdir(parents=True, exist_ok=True)
    corrupt_zip.write_bytes(b"corrupted zip bytes")

    with pytest.raises(AppStorageError) as excinfo:
        ingestion.extract_file(corrupt_zip, "sub_corrupt")

    assert "Failed to extract file" in str(excinfo.value)
    # Target directory should have been cleaned up and not exist
    assert not (temp_dir / "raw" / "sub_corrupt").exists()


def test_zip_slip_prevention(valid_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that directory traversal names (Zip Slip) raise AppValidationError."""
    ingestion = DataIngestion(valid_config_yaml)

    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w") as zf:
        zf.writestr("../malicious_file.txt", "malicious_content")
    zip_bytes.seek(0)

    malicious_zip = temp_dir / "downloads" / "malicious.zip"
    malicious_zip.parent.mkdir(parents=True, exist_ok=True)
    malicious_zip.write_bytes(zip_bytes.read())

    with pytest.raises(AppValidationError) as excinfo:
        ingestion.extract_file(malicious_zip, "sub_malicious")

    assert "Malicious zip member detected" in str(excinfo.value)
    assert not (temp_dir / "raw" / "sub_malicious").exists()


@patch.object(DataIngestion, "download_kaggle_dataset")
@patch.object(DataIngestion, "extract_file")
def test_initiate_data_ingestion(
    mock_extract: MagicMock, mock_download: MagicMock, valid_config_yaml: Path
) -> None:
    """Verifies correct execution flow of initiate_data_ingestion for multiple datasets."""
    mock_download.side_effect = [
        Path("downloads/image-dataset.zip"),
        Path("downloads/symptom-dataset.zip"),
    ]
    mock_extract.side_effect = [
        Path("raw/covid19-radiography-database"),
        Path("raw/disease-symptom-description-dataset"),
    ]

    ingestion = DataIngestion(valid_config_yaml)
    paths = ingestion.initiate_data_ingestion()

    assert paths["image_raw_dir"] == Path("raw/covid19-radiography-database")
    assert paths["symptom_raw_dir"] == Path("raw/disease-symptom-description-dataset")
    assert mock_download.call_count == 2
    assert mock_extract.call_count == 2
