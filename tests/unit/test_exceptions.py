"""Unit Tests for Custom Exceptions.

Asserts properties mappings, string serializations, and subclasses type assertions.
"""

from src.utils.error_codes import ErrorCode
from src.utils.exceptions import (
    AppException,
    AppValidationError,
    AppDatabaseError,
    AppInferenceError,
    AppImageProcessingError,
    AppStorageError,
    AppReportError,
)


def test_base_exception_instantiation() -> None:
    """Verifies that the base AppException captures and formats metadata correctly."""
    ex = AppException(
        message="System failure details",
        error_code=ErrorCode.SYSTEM_ERROR,
        status_code=503,
        details={"disk_space": "1%"},
    )

    assert ex.message == "System failure details"
    assert ex.error_code == ErrorCode.SYSTEM_ERROR
    assert ex.status_code == 503
    assert ex.details == {"disk_space": "1%"}

    # Assert string representation formats parameters
    str_ex = str(ex)
    assert "[SYSTEM_ERROR]" in str_ex
    assert "(Status 503)" in str_ex
    assert "System failure details" in str_ex
    assert "disk_space" in str_ex


def test_validation_exception_defaults() -> None:
    """Verifies AppValidationError presets HTTP status 400 and validation error code."""
    ex = AppValidationError(
        message="Invalid image dimensions",
        details={"dimension": "must be square"},
    )

    assert ex.message == "Invalid image dimensions"
    assert ex.error_code == ErrorCode.VALIDATION_ERROR
    assert ex.status_code == 400
    assert ex.details == {"dimension": "must be square"}


def test_database_exception_defaults() -> None:
    """Verifies AppDatabaseError presets HTTP status 500 and database error code."""
    ex = AppDatabaseError(
        message="Connection pool exhausted",
    )

    assert ex.message == "Connection pool exhausted"
    assert ex.error_code == ErrorCode.DATABASE_ERROR
    assert ex.status_code == 500
    assert ex.details == {}


def test_other_subclass_defaults() -> None:
    """Verifies remaining exception subclasses assign their respective ErrorCodes and statuses."""
    # 1. Inference exception
    inf_ex = AppInferenceError("Model path not loaded")
    assert inf_ex.error_code == ErrorCode.INFERENCE_ERROR
    assert inf_ex.status_code == 500

    # 2. Image Processing exception
    img_ex = AppImageProcessingError("Augmentation failed")
    assert img_ex.error_code == ErrorCode.IMAGE_PROCESSING_ERROR
    assert img_ex.status_code == 400

    # 3. Storage exception
    store_ex = AppStorageError("S3 credentials rejected")
    assert store_ex.error_code == ErrorCode.STORAGE_ERROR
    assert store_ex.status_code == 500

    # 4. Report exception
    rep_ex = AppReportError("PDF buffer write block")
    assert rep_ex.error_code == ErrorCode.REPORT_ERROR
    assert rep_ex.status_code == 500
