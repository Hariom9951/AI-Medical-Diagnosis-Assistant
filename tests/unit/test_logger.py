"""Unit Tests for Logging Utility.

Asserts file creation, console writing, rotating limits, and decorator exceptions logging.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.utils.logger import AppLogger, log_exception


@pytest.fixture(autouse=True)
def reset_logger_configuration() -> None:
    """Fixture to reset AppLogger state between tests to verify clean setups."""
    AppLogger._configured = False
    # Reset root logger handlers
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()


def test_logger_fallback_configuration(tmp_path: Path) -> None:
    """Verifies that the fallback logging creates a rotating file correctly."""
    log_file = tmp_path / "test_logs" / "app.log"

    # Configure using fallback params (JSON config path does not exist)
    AppLogger.configure(
        config_path=tmp_path / "missing_config.json",
        default_level=logging.INFO,
        default_file=log_file,
    )

    logger = AppLogger.get_logger("test_namespace")
    logger.info("Test fallback logs file creation")

    # Assert file exists and is populated
    assert log_file.exists()
    assert log_file.stat().st_size > 0
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "[INFO] test_namespace: Test fallback logs file creation" in content


def test_logger_file_rotation(tmp_path: Path) -> None:
    """Verifies that the rotating file handler splits files when size limits are reached."""
    log_file = tmp_path / "rotation_logs" / "app.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Reset root logger levels
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Instantiate RotatingFileHandler with small maxBytes (100 bytes)
    formatter = logging.Formatter("%(message)s")
    handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=100,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Write logs to trigger rotation
    # Writing a 40-character log multiple times should overflow 100 bytes
    for i in range(5):
        root.info(f"Log message number {i} - padding padding padding")

    # Force write close to release file descriptors
    handler.close()

    # Verify that rotation files are created
    assert log_file.exists()
    rotated_file = log_file.parent / "app.log.1"
    assert rotated_file.exists()
    assert rotated_file.stat().st_size > 0


def test_log_exception_decorator() -> None:
    """Verifies that the @log_exception decorator captures and logs traceback exceptions."""
    mock_logger = MagicMock()

    @log_exception(mock_logger)
    def division_by_zero() -> float:
        raise ZeroDivisionError("division by zero")

    with pytest.raises(ZeroDivisionError):
        division_by_zero()

    # Verify logging exception interface was called
    mock_logger.exception.assert_called_once()
    args, kwargs = mock_logger.exception.call_args
    assert "Unhandled exception captured in function" in args[0]
    assert "division_by_zero" in args[1]
