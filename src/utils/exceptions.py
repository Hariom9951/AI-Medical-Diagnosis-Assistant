"""Custom Application Exceptions.

Defines the base AppException and concrete sub-exceptions mapped to standard HTTP codes.
Allows clean boundary mapping (e.g., inside FastAPI middleware controllers).
"""

from typing import Any, Final
from src.utils.error_codes import ErrorCode


class AppException(Exception):
    """Base exception for all domain and infrastructure operations."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SYSTEM_ERROR,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the base application exception.

        Args:
            message (str): User-friendly display message.
            error_code (ErrorCode): Structured error classification code.
            status_code (int): Corresponding HTTP status code.
            details (dict[str, Any] | None): Optional dictionary tracking raw debugging contexts.
        """
        super().__init__(message)
        self.message: Final[str] = message
        self.error_code: Final[ErrorCode] = error_code
        self.status_code: Final[int] = status_code
        self.details: Final[dict[str, Any]] = details or {}

    def __str__(self) -> str:
        return f"[{self.error_code}] (Status {self.status_code}): {self.message} | Details: {self.details}"


class AppValidationError(AppException):
    """Raised when data validations or input structures fail checks."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details,
        )


class AppDatabaseError(AppException):
    """Raised when query executions or connection limits to PostgreSQL fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=500,
            details=details,
        )


class AppInferenceError(AppException):
    """Raised when ML models, tensor formatting, or Triton client calls fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.INFERENCE_ERROR,
            status_code=500,
            details=details,
        )


class AppImageProcessingError(AppException):
    """Raised when scan resizing, OpenCV format changes, or Albumentations fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.IMAGE_PROCESSING_ERROR,
            status_code=400,
            details=details,
        )


class AppStorageError(AppException):
    """Raised when raw files or Grad-CAM heatmaps fail to persist to cloud storage."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.STORAGE_ERROR,
            status_code=500,
            details=details,
        )


class AppReportError(AppException):
    """Raised when clinical PDF report compilation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.REPORT_ERROR,
            status_code=500,
            details=details,
        )
