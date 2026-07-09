"""Production checkpoint downloader for the AI Medical Diagnosis Assistant.

Handles downloading, SHA256 checksum validation, caching, retries, and network error handling.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import hf_hub_download

from src.utils.exceptions import AppStorageError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

# Expected SHA256 checksums for all trained model files.
# Ensures integrity and prevents corruption.
EXPECTED_SHA256: Dict[str, str] = {
    "image/best_model.pth": "affa284a444d924edb9426a0bf6fb26d41f4d0d96c3744e9b3cc705db39b37b3",
    "image/checkpoint_epoch_050.pth": "affa284a444d924edb9426a0bf6fb26d41f4d0d96c3744e9b3cc705db39b37b3",
    "nlp/best_model.pt": "39903c7f536ac200003f8a87ca8c0af2847eb2af66839da7620eca1a8d76fb15",
    "nlp/tokenizer.json": "9355eae89d401cee6b1f7c9acaf4791191e3b22c918e5f616b6baea13b66e748",
    "nlp/tokenizer_config.json": "9f0c2c65a70ea18113ffa2e717103e7210ea826e7acc3442b69b346686b55a48",
    "nlp/label_encoder.pkl": "ae00d6ca392a2605c539285a00b896358c6f98b5876743c86256b0f5b615c50c",
    "nlp/model_metadata.json": "38f984eca04d3d629f96aa1f7d5cc501acad04057e1b870a08be01f610ae8262",
    "nlp/temperature_scaler.json": "0e75867a68a7fd12ec20dcbb4ac9bb44ce5cc9bc1b069c9c3b066fa03865e689",
    "nlp/clinical_explanations.json": "5246003b14661544a1aaa93525ca911afc7bde02bea4c2982999451d166015c1",
}


class ModelDownloader:
    """Production-grade downloader for trained models and configurations.

    Verifies SHA256 checksums, caches files locally, handles retries, and env variables.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initializes the ModelDownloader.

        Args:
            repo_id: The Hugging Face Model Repository ID.
                     Resolves to HF_MODEL_REPO_ID environment variable or defaults to
                     'Hariom9951/AI-Medical-Diagnosis-Models'.
            token: The HF API Token. Resolves to HF_TOKEN environment variable.
            max_retries: Maximum number of download attempts before failing.
            backoff_factor: Exponential backoff factor for retries.
        """
        self.repo_id = (
            repo_id or os.getenv("HF_MODEL_REPO_ID") or "Hariom9951/AI-Medical-Diagnosis-Models"
        )
        self.token = token or os.getenv("HF_TOKEN")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Log environment status
        self._detect_environment()

    def _detect_environment(self) -> None:
        """Detects and logs the runtime environment configuration."""
        if os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"):
            env = "Hugging Face Space"
        elif os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER"):
            env = "Docker Container"
        elif os.getenv("RENDER"):
            env = "Render"
        elif os.name == "nt":
            env = "Local Windows"
        else:
            env = "Local/Cloud VM"

        logger.info("Downloader Initialized | Environment: %s | Repo: %s", env, self.repo_id)

    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Calculates the SHA256 checksum of a file.

        Args:
            file_path: Path to the target file.

        Returns:
            str: Hex digest of SHA256.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verifies if the SHA256 checksum of a file matches the expected value.

        Args:
            file_path: Path to check.
            expected_sha256: Expected hex digest of SHA256.

        Returns:
            bool: True if matched, False otherwise.
        """
        if not file_path.exists():
            return False
        try:
            actual = self.calculate_sha256(file_path)
            matched = actual == expected_sha256
            if not matched:
                logger.warning(
                    "Checksum mismatch for %s. Expected: %s, Actual: %s",
                    file_path.name,
                    expected_sha256,
                    actual,
                )
            return matched
        except Exception as e:
            logger.error("Error calculating SHA256 checksum for %s: %s", file_path, e)
            return False

    def download_file(self, repo_path: str, local_path: str | Path) -> Path:
        """Downloads a missing or invalid file from HF Model Repository.

        Args:
            repo_path: Relative path in HF repo (e.g. 'image/best_model.pth').
            local_path: Absolute or project-relative destination file path.

        Returns:
            Path: The resolved destination path of the validated file.
        """
        dest_path = Path(local_path)
        if not dest_path.is_absolute():
            dest_path = self.project_root / dest_path

        expected_sha256 = EXPECTED_SHA256.get(repo_path)

        # 1. Check if the file is already cached and valid
        if dest_path.exists() and dest_path.stat().st_size > 0:
            if expected_sha256:
                if self.verify_checksum(dest_path, expected_sha256):
                    logger.info("Cached and verified file is valid: %s", dest_path)
                    return dest_path
                else:
                    logger.info("Cached file %s failed validation. Overwriting...", dest_path)
            else:
                logger.info(
                    "File %s exists (no verification checksum defined). Skipping...", dest_path
                )
                return dest_path

        # 2. Check if offline mode is set via environment variable
        is_offline = (
            os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_DATASETS_OFFLINE") == "1"
        )
        if is_offline:
            if dest_path.exists() and dest_path.stat().st_size > 0:
                logger.warning(
                    "Offline mode active: using existing cached file %s without verification.",
                    dest_path,
                )
                return dest_path
            else:
                raise AppStorageError(
                    message=(
                        f"Offline mode active and file is missing at: {dest_path}. "
                        "Ensure the model cache is prepopulated."
                    ),
                    details={"local_path": str(dest_path)},
                )

        # 3. Perform download with retries and validation
        logger.info(
            "Downloading '%s' from repository '%s' to '%s'...",
            repo_path,
            self.repo_id,
            dest_path,
        )

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Delete existing file to prevent permission or cached symlink conflicts
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except Exception:
                        pass

                # Perform Hub download
                downloaded_file = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=repo_path,
                    repo_type="model",
                    local_dir=str(self.project_root),
                    token=self.token,
                )
                downloaded_path = Path(downloaded_file)

                # Ensure it is placed at the exact expected local path
                if downloaded_path.resolve() != dest_path.resolve():
                    import shutil

                    shutil.move(str(downloaded_path), str(dest_path))

                # Verify downloaded file checksum
                if expected_sha256:
                    if not self.verify_checksum(dest_path, expected_sha256):
                        raise AppStorageError(
                            message=f"Downloaded file {dest_path.name} failed SHA256 verification.",
                            details={"expected": expected_sha256},
                        )

                logger.info(
                    "Successfully downloaded and verified %s (Size: %d bytes)",
                    dest_path.name,
                    dest_path.stat().st_size,
                )
                return dest_path

            except Exception as e:
                last_error = e
                wait_time = self.backoff_factor**attempt
                logger.warning(
                    "Download attempt %d/%d for '%s' failed: %s. Retrying in %.1fs...",
                    attempt,
                    self.max_retries,
                    repo_path,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)

        raise AppStorageError(
            message=f"Failed to download and verify model file '{repo_path}' after {self.max_retries} attempts.",
            details={"repo_id": self.repo_id, "repo_path": repo_path, "error": str(last_error)},
        )
