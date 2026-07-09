"""Integration tests for the ModelDownloader service.

Tests cover:
- First startup: files are downloaded and SHA256-verified
- Second startup: cached valid files are skipped (no re-download)
- Offline mode: raises AppStorageError if file is missing
- Checksum mismatch: corrupted cached file triggers re-download
- CI mock mode: all tests are mocked when CI_MOCK_DOWNLOADER=true

Run with:
    pytest tests/integration/test_model_downloader.py -v
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.utils.downloader import EXPECTED_SHA256, ModelDownloader
from src.utils.exceptions import AppStorageError

# ─── Helpers ──────────────────────────────────────────────────────────────────

IS_CI_MOCK = os.getenv("CI_MOCK_DOWNLOADER", "false").lower() == "true"


def make_fake_file(path: Path, content: bytes = b"fake model content") -> Path:
    """Creates a fake file at the given path with the given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_project(tmp_path: Path) -> Generator[Path, None, None]:
    """Provides a temporary project root with checkpoint directories."""
    (tmp_path / "artifacts" / "checkpoints").mkdir(parents=True)
    (tmp_path / "artifacts" / "checkpoints_nlp").mkdir(parents=True)
    yield tmp_path


@pytest.fixture
def downloader(tmp_project: Path) -> ModelDownloader:
    """Returns a ModelDownloader with the project root pointing at tmp_project."""
    dl = ModelDownloader(
        repo_id="Hariom9951/AI-Medical-Diagnosis-Models",
        max_retries=1,
        backoff_factor=0.0,
    )
    dl.project_root = tmp_project
    return dl


# ─── Unit-level tests (always run, use mocks) ─────────────────────────────────


class TestModelDownloaderInit:
    """Tests for ModelDownloader initialization."""

    def test_default_repo_id(self) -> None:
        """Default repo_id must point at the production model repository."""
        dl = ModelDownloader()
        assert dl.repo_id == "Hariom9951/AI-Medical-Diagnosis-Models"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HF_MODEL_REPO_ID env variable overrides the default repo_id."""
        monkeypatch.setenv("HF_MODEL_REPO_ID", "myorg/my-custom-models")
        dl = ModelDownloader()
        assert dl.repo_id == "myorg/my-custom-models"

    def test_explicit_repo_id_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit repo_id argument takes precedence over env variable."""
        monkeypatch.setenv("HF_MODEL_REPO_ID", "myorg/my-custom-models")
        dl = ModelDownloader(repo_id="explicit/repo")
        assert dl.repo_id == "explicit/repo"


class TestSHA256Verification:
    """Tests for the SHA256 checksum verification logic."""

    def test_correct_checksum(self, tmp_path: Path) -> None:
        content = b"test model data"
        f = make_fake_file(tmp_path / "model.pth", content)
        expected = sha256_of(content)
        dl = ModelDownloader()
        assert dl.verify_checksum(f, expected) is True

    def test_wrong_checksum(self, tmp_path: Path) -> None:
        content = b"test model data"
        f = make_fake_file(tmp_path / "model.pth", content)
        dl = ModelDownloader()
        assert dl.verify_checksum(f, "deadbeef" * 8) is False

    def test_missing_file(self, tmp_path: Path) -> None:
        dl = ModelDownloader()
        assert dl.verify_checksum(tmp_path / "nonexistent.pth", "abc123") is False


class TestCacheHitSkipsDownload:
    """Tests that a valid cached file is not re-downloaded."""

    def test_cached_file_skips_download(self, tmp_project: Path, downloader: ModelDownloader) -> None:
        """If a file exists and SHA256 matches, hf_hub_download is never called."""
        repo_path = "nlp/tokenizer_config.json"
        local_path = tmp_project / "artifacts" / "checkpoints_nlp" / "tokenizer_config.json"

        # Write a file with the correct checksum
        expected_sha256 = EXPECTED_SHA256[repo_path]
        # We don't have the real file so we patch the checksum verifier to return True
        make_fake_file(local_path, b"placeholder")

        with patch.object(downloader, "verify_checksum", return_value=True) as mock_verify:
            with patch("src.utils.downloader.hf_hub_download") as mock_dl:
                result = downloader.download_file(repo_path, local_path)
                assert result == local_path
                mock_dl.assert_not_called()
                mock_verify.assert_called_once()


class TestOfflineMode:
    """Tests for offline mode behaviour."""

    def test_offline_mode_raises_if_file_missing(
        self, tmp_project: Path, downloader: ModelDownloader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If offline mode is active and the file doesn't exist, raise AppStorageError."""
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        repo_path = "image/best_model.pth"
        local_path = tmp_project / "artifacts" / "checkpoints" / "best_model.pth"
        # Ensure file does NOT exist
        if local_path.exists():
            local_path.unlink()

        with pytest.raises(AppStorageError, match="Offline mode active"):
            downloader.download_file(repo_path, local_path)


class TestDownloadWithMock:
    """Tests the download path using a mocked hf_hub_download."""

    def test_successful_download_and_verify(
        self, tmp_project: Path, downloader: ModelDownloader
    ) -> None:
        """Simulates a successful download: creates the file, then verifies it."""
        repo_path = "nlp/model_metadata.json"
        local_path = tmp_project / "artifacts" / "checkpoints_nlp" / "model_metadata.json"
        fake_content = b'{"model": "test"}'

        def fake_hf_download(**kwargs: object) -> str:
            # Simulate hf_hub_download placing the file at local_dir/repo_path
            dest = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(fake_content)
            return str(dest)

        with patch("src.utils.downloader.hf_hub_download", side_effect=fake_hf_download):
            # Patch verify_checksum to return True (no real checksum to match)
            with patch.object(downloader, "verify_checksum", return_value=True):
                result = downloader.download_file(repo_path, local_path)
                assert result.exists()

    def test_failed_download_raises_after_retries(
        self, tmp_project: Path, downloader: ModelDownloader
    ) -> None:
        """If all download attempts fail, AppStorageError is raised."""
        repo_path = "image/best_model.pth"
        local_path = tmp_project / "artifacts" / "checkpoints" / "best_model.pth"

        with patch(
            "src.utils.downloader.hf_hub_download",
            side_effect=ConnectionError("Network error"),
        ):
            with pytest.raises(AppStorageError, match="Failed to download"):
                downloader.download_file(repo_path, local_path)


# ─── Integration tests (require real HF access; skipped in CI mock mode) ──────


@pytest.mark.skipif(IS_CI_MOCK, reason="CI_MOCK_DOWNLOADER=true: skipping real network tests")
@pytest.mark.integration
class TestRealDownloadFlow:
    """Real integration tests that exercise actual HF Hub downloads.

    These tests are skipped when CI_MOCK_DOWNLOADER=true.
    They require a valid HF_TOKEN and network access.
    """

    def test_first_start_downloads_metadata_file(self, tmp_project: Path) -> None:
        """On first start, a small metadata file is downloaded and verified."""
        repo_path = "nlp/model_metadata.json"
        local_path = tmp_project / "artifacts" / "checkpoints_nlp" / "model_metadata.json"

        # Ensure file is not cached
        if local_path.exists():
            local_path.unlink()

        downloader = ModelDownloader()
        downloader.project_root = tmp_project

        result = downloader.download_file(repo_path, local_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_second_start_uses_cache(self, tmp_project: Path) -> None:
        """On second start, a previously downloaded valid file is not re-fetched."""
        repo_path = "nlp/temperature_scaler.json"
        local_path = tmp_project / "artifacts" / "checkpoints_nlp" / "temperature_scaler.json"

        downloader = ModelDownloader()
        downloader.project_root = tmp_project

        # First download
        downloader.download_file(repo_path, local_path)
        assert local_path.exists()

        # Second call — should use cache without calling hf_hub_download
        with patch("src.utils.downloader.hf_hub_download") as mock_dl:
            downloader.download_file(repo_path, local_path)
            mock_dl.assert_not_called()

    def test_corrupted_cache_triggers_redownload(self, tmp_project: Path) -> None:
        """A file with a mismatched SHA256 should be overwritten with a fresh download."""
        repo_path = "nlp/temperature_scaler.json"
        local_path = tmp_project / "artifacts" / "checkpoints_nlp" / "temperature_scaler.json"

        # Write a corrupted file
        make_fake_file(local_path, b"this is corrupted data that will not match sha256")

        downloader = ModelDownloader()
        downloader.project_root = tmp_project

        # Should detect mismatch and re-download
        result = downloader.download_file(repo_path, local_path)
        assert result.exists()
        # After re-download, checksum should pass
        expected = EXPECTED_SHA256.get(repo_path, "")
        if expected:
            assert downloader.verify_checksum(result, expected)
