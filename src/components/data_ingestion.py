"""Data Ingestion Component using Kaggle API.

Provides functionality to download remote datasets from Kaggle, verify their presence,
and extract them to raw data locations, with exponential backoff retries, logging,
and cleanup mechanisms.
"""

import os
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final

import kaggle  # type: ignore[import-untyped]
import yaml

from src.utils.exceptions import AppStorageError, AppValidationError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


@dataclass(frozen=True)
class DataIngestionConfig:
    """Ingestion pipeline configuration values parsed from YAML.

    Attributes:
        image_dataset_slug (str): Kaggle dataset slug for medical images.
        symptom_dataset_slug (str): Kaggle dataset slug for symptoms metadata.
        download_dir (Path): Local target directory to save downloaded archives.
        extract_dir (Path): Target directory to extract archive contents.
        processed_dir (Path): Target directory for downstream processed/validated data.
        interim_dir (Path): Target directory for intermediate data transforms.
        max_retries (int): Maximum number of retry attempts for network/API failures.
        backoff_factor (float): Multiplier for exponential backoff delay during retries.
        chunk_size (int): Size in bytes for generic file operations.
    """

    image_dataset_slug: str
    symptom_dataset_slug: str
    download_dir: Path
    extract_dir: Path
    processed_dir: Path
    interim_dir: Path
    max_retries: int
    backoff_factor: float
    chunk_size: int

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DataIngestionConfig":
        """Loads configuration from a YAML file.

        Args:
            yaml_path (Path): Path to the configuration YAML file.

        Returns:
            DataIngestionConfig: Parsed configuration dataclass.

        Raises:
            AppValidationError: If configuration file is missing or has invalid format.
        """
        if not yaml_path.exists():
            raise AppValidationError(
                message=f"Configuration file not found at {yaml_path}",
                details={"yaml_path": str(yaml_path)},
            )

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_dict: dict[str, Any] = yaml.safe_load(f) or {}

            # Perform basic validation of required keys
            required_keys: Final[list[str]] = [
                "image_dataset_slug",
                "symptom_dataset_slug",
                "download_dir",
                "extract_dir",
                "processed_dir",
                "interim_dir",
            ]
            missing_keys = [k for k in required_keys if k not in config_dict]
            if missing_keys:
                raise AppValidationError(
                    message=f"Missing required configuration keys: {missing_keys}",
                    details={"missing_keys": missing_keys},
                )

            return cls(
                image_dataset_slug=config_dict["image_dataset_slug"],
                symptom_dataset_slug=config_dict["symptom_dataset_slug"],
                download_dir=Path(config_dict["download_dir"]),
                extract_dir=Path(config_dict["extract_dir"]),
                processed_dir=Path(config_dict["processed_dir"]),
                interim_dir=Path(config_dict["interim_dir"]),
                max_retries=int(config_dict.get("max_retries", 3)),
                backoff_factor=float(config_dict.get("backoff_factor", 2.0)),
                chunk_size=int(config_dict.get("chunk_size", 1024 * 1024)),
            )
        except yaml.YAMLError as e:
            raise AppValidationError(
                message=f"Failed to parse YAML config: {e}",
                details={"error": str(e)},
            )
        except (ValueError, TypeError) as e:
            raise AppValidationError(
                message=f"Invalid value type in config file: {e}",
                details={"error": str(e)},
            )


class DataIngestion:
    """Orchestrates Kaggle API downloads and extraction of clinical datasets."""

    def __init__(self, config_path: Path) -> None:
        """Initializes the Data Ingestion pipeline and creates directory structures.

        Args:
            config_path (Path): Path to the YAML configuration file.
        """
        self.config: Final[DataIngestionConfig] = DataIngestionConfig.from_yaml(config_path)

        # Create target directories if they do not exist
        self.config.download_dir.mkdir(parents=True, exist_ok=True)
        self.config.extract_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config.interim_dir.mkdir(parents=True, exist_ok=True)

    def download_kaggle_dataset(self, dataset_slug: str) -> Path:
        """Downloads a dataset from Kaggle using the API with retries and backoff.

        Args:
            dataset_slug (str): Kaggle dataset slug (e.g. 'username/dataset-name').

        Returns:
            Path: Path to the downloaded compressed archive.

        Raises:
            AppStorageError: If the download fails consistently after retries.
        """
        dataset_name = dataset_slug.split("/")[-1]
        target_zip = self.config.download_dir / f"{dataset_name}.zip"

        logger.info("Initializing Kaggle download for slug: %s", dataset_slug)

        # Idempotency Check: If file already exists, skip downloading
        if target_zip.exists():
            logger.info(
                "Archive %s already exists in downloads. Skipping download.", target_zip.name
            )
            return target_zip

        # Perform authentication first
        try:
            kaggle.api.authenticate()
        except Exception as e:
            raise AppStorageError(
                message="Kaggle API authentication failed. Verify that kaggle.json is correctly configured.",
                details={"error": str(e)},
            )

        # Download with exponential backoff retry loop
        last_exception: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info("Download attempt %d/%d...", attempt, self.config.max_retries)

                # Fetch dataset files using Kaggle API client
                kaggle.api.dataset_download_files(
                    dataset=dataset_slug,
                    path=str(self.config.download_dir),
                    unzip=False,
                    quiet=False,
                )

                if not target_zip.exists():
                    raise FileNotFoundError(
                        f"Expected archive download target file {target_zip} not found."
                    )

                logger.info("Kaggle download completed successfully: %s", target_zip.name)
                break
            except Exception as e:
                logger.warning("Attempt %d failed. Error: %s", attempt, e)
                last_exception = e
                # Remove partial file if it exists
                if target_zip.exists():
                    try:
                        target_zip.unlink()
                    except OSError:
                        pass

                if attempt < self.config.max_retries:
                    sleep_time = self.config.backoff_factor**attempt
                    logger.info("Sleeping for %.1f seconds before retrying...", sleep_time)
                    time.sleep(sleep_time)
        else:
            raise AppStorageError(
                message=f"Failed to download dataset {dataset_slug} from Kaggle after {self.config.max_retries} attempts.",
                details={"slug": dataset_slug, "last_error": str(last_exception)},
            )

        return target_zip

    def extract_file(self, archive_path: Path, sub_dir_name: str) -> Path:
        """Extracts a compressed archive to a specific subdirectory in raw data location.

        Performs full directory cleanups if extraction fails mid-way to keep system clean.

        Args:
            archive_path (Path): Path to the compressed archive file.
            sub_dir_name (str): Directory name where contents should be unpacked.

        Returns:
            Path: Path to the target extraction directory.

        Raises:
            AppValidationError: If archive member paths fail directory traversal checks.
            AppStorageError: If decompression or filesystem writes fail.
        """
        target_dir = self.config.extract_dir / sub_dir_name
        logger.info("Initializing extraction: %s to %s", archive_path.name, target_dir)

        # Clear target directory first
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    # Zip Slip directory traversal security prevention checks
                    for zip_member in zip_ref.infolist():
                        abs_target_path = Path(os.path.abspath(target_dir / zip_member.filename))
                        abs_extract_dir = Path(os.path.abspath(target_dir))
                        if not abs_target_path.is_relative_to(abs_extract_dir):
                            raise AppValidationError(
                                message="Malicious zip member detected with directory traversal: "
                                + zip_member.filename
                            )
                    zip_ref.extractall(target_dir)

            elif archive_path.suffix in [".tar", ".gz", ".tgz", ".bz2", ".xz"]:
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    # Prevent directory traversal attacks
                    for tar_member in tar_ref.getmembers():
                        abs_target_path = Path(os.path.abspath(target_dir / tar_member.name))
                        abs_extract_dir = Path(os.path.abspath(target_dir))
                        if not abs_target_path.is_relative_to(abs_extract_dir):
                            raise AppValidationError(
                                message="Malicious tar member detected with directory traversal: "
                                + tar_member.name
                            )
                    tar_ref.extractall(target_dir)
            else:
                raise AppValidationError(
                    message=f"Unsupported archive file format: {archive_path.suffix}",
                    details={"file": str(archive_path)},
                )

            logger.info("Extraction completed successfully to: %s", target_dir)
            return target_dir

        except AppValidationError as e:
            logger.error(
                "Validation failed during extraction. Initiating pipeline cleanup. Error: %s", e
            )
            if target_dir.exists():
                shutil.rmtree(target_dir)
            raise e
        except Exception as e:
            logger.error("Extraction failed. Initiating pipeline cleanup. Error: %s", e)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            raise AppStorageError(
                message=f"Failed to extract file {archive_path}: {e}",
                details={"archive_path": str(archive_path), "error": str(e)},
            )

    def initiate_data_ingestion(self) -> Dict[str, Path]:
        """Orchestrates Kaggle API downloads and extraction stages for all datasets.

        Returns:
            Dict[str, Path]: Dict containing mapping of extracted dataset paths.
        """
        logger.info("--- Kaggle Data Ingestion Sequence Started ---")
        try:
            # 1. Download and extract the image dataset
            img_archive = self.download_kaggle_dataset(self.config.image_dataset_slug)
            img_extracted = self.extract_file(img_archive, "covid19-radiography-database")

            # 2. Download and extract the symptoms dataset
            symptom_archive = self.download_kaggle_dataset(self.config.symptom_dataset_slug)
            symptom_extracted = self.extract_file(
                symptom_archive, "disease-symptom-description-dataset"
            )

            logger.info("--- Kaggle Data Ingestion Sequence Successfully Finished ---")
            return {"image_raw_dir": img_extracted, "symptom_raw_dir": symptom_extracted}
        except Exception as e:
            logger.error("--- Kaggle Data Ingestion Sequence Failed ---")
            raise e
