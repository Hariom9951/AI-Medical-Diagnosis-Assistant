"""Error Codes.

Defines the ErrorCode Enum to classify system, business, and model exceptions
consistently across all boundary adapters.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Machine-readable error codes for the diagnosis assistant service."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    IMAGE_PROCESSING_ERROR = "IMAGE_PROCESSING_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    REPORT_ERROR = "REPORT_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"

    def __str__(self) -> str:
        return self.value
