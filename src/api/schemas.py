"""Pydantic schemas for FastAPI request/response validation.

All API input validation and response serialization models are defined here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared / Base Schemas
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Standard error detail wrapper."""

    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Unified error envelope returned by all exception handlers."""

    success: bool = False
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Root / Health / Version
# ---------------------------------------------------------------------------


class ProjectInfoResponse(BaseModel):
    """Response schema for GET /"""

    success: bool = True
    project: str
    description: str
    version: str
    author: str
    endpoints: List[str]


class HealthResponse(BaseModel):
    """Response schema for GET /health"""

    success: bool = True
    status: str
    image_model_loaded: bool
    nlp_model_loaded: bool
    device: str


class CheckpointInfo(BaseModel):
    """Nested checkpoint metadata."""

    checkpoint_file: str
    epoch: Any
    metrics: Dict[str, Any]


class VersionResponse(BaseModel):
    """Response schema for GET /version"""

    success: bool = True
    app_version: str
    image_model: CheckpointInfo
    nlp_model: CheckpointInfo


# ---------------------------------------------------------------------------
# Image Prediction
# ---------------------------------------------------------------------------


class TopPrediction(BaseModel):
    """Single entry in a top-N predictions list."""

    rank: int
    disease: str
    confidence: float


class ImagePredictionResponse(BaseModel):
    """Response schema for POST /predict/image"""

    success: bool = True
    predicted_disease: str
    confidence: float
    top_predictions: List[TopPrediction]
    gradcam_image_path: Optional[str] = None
    gradcam_available: bool = False


# ---------------------------------------------------------------------------
# Symptom / NLP Prediction
# ---------------------------------------------------------------------------


class SymptomRequest(BaseModel):
    """Request body schema for POST /predict/symptoms"""

    symptoms: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Free-text description of patient symptoms.",
        examples=["fever, cough, sore throat, headache and body pain"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=41,
        description="Number of top predictions to return (1–41).",
    )

    @field_validator("symptoms")
    @classmethod
    def symptoms_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symptoms must not be empty or whitespace only.")
        return v


class ClinicalExplanation(BaseModel):
    """Structured clinical advice and warning details for prediction explanations."""

    specialist: str
    tests: List[str]
    emergency_signs: str
    home_care: str
    lifestyle: str
    similar_diseases: List[str]


class SymptomPredictionResponse(BaseModel):
    """Response schema for POST /predict/symptoms"""

    success: bool = True
    predicted_disease: str
    confidence: float
    top_predictions: List[TopPrediction]
    preprocessed_text: str
    clinical_explanation: Optional[ClinicalExplanation] = None

