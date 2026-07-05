"""FastAPI route definitions.

All endpoint handlers are collected here and registered on the main app
via an APIRouter so that main.py stays slim.
"""

from __future__ import annotations

import logging
from typing import List

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ImagePredictionResponse,
    ProjectInfoResponse,
    SymptomPredictionResponse,
    SymptomRequest,
    TopPrediction,
    VersionResponse,
    CheckpointInfo,
)
from src.api.utils import (
    extract_checkpoint_info,
    get_image_pipeline,
    get_nlp_pipeline,
    is_image_pipeline_loaded,
    is_nlp_pipeline_loaded,
    read_upload_as_pil,
    run_gradcam,
)
from src.utils.exceptions import AppInferenceError, AppValidationError

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------

APP_VERSION = "1.0.0"
PROJECT_NAME = "AI Medical Diagnosis Assistant"
DESCRIPTION = (
    "Production-ready REST API for automated medical image and symptom-based "
    "disease classification using EfficientNet-B0 and DistilBERT models."
)
AUTHOR = "Hariom Sharma"

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"}
MAX_IMAGE_SIZE_MB = 10


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=ProjectInfoResponse,
    summary="Project information",
    tags=["Info"],
)
async def root() -> ProjectInfoResponse:
    """Returns high-level project information and available API endpoints."""
    return ProjectInfoResponse(
        project=PROJECT_NAME,
        description=DESCRIPTION,
        version=APP_VERSION,
        author=AUTHOR,
        endpoints=[
            "GET  /           — Project information",
            "GET  /health     — Application health status",
            "GET  /version    — Model versions and checkpoint details",
            "POST /predict/image    — Chest X-ray disease classification",
            "POST /predict/symptoms — Free-text symptom-based diagnosis",
            "GET  /docs       — Interactive Swagger UI",
            "GET  /redoc      — ReDoc API documentation",
        ],
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    tags=["Info"],
)
async def health_check() -> HealthResponse:
    """Returns the current health status of the application and model load state."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_loaded = is_image_pipeline_loaded()
    nlp_loaded = is_nlp_pipeline_loaded()

    overall_status = "healthy" if (image_loaded and nlp_loaded) else "degraded"

    return HealthResponse(
        status=overall_status,
        image_model_loaded=image_loaded,
        nlp_model_loaded=nlp_loaded,
        device=device,
    )


# ---------------------------------------------------------------------------
# GET /version
# ---------------------------------------------------------------------------


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Model and application version information",
    tags=["Info"],
)
async def version_info() -> VersionResponse:
    """Returns model checkpoint information and the application version string."""
    try:
        img_pipeline = get_image_pipeline()
        img_info = extract_checkpoint_info(img_pipeline, pipeline_type="image")
    except Exception as exc:
        logger.error("Failed to load image pipeline for version info: %s", exc)
        img_info = {"checkpoint_file": "unavailable", "epoch": "N/A", "metrics": {}}

    try:
        nlp_pipeline = get_nlp_pipeline()
        nlp_info = extract_checkpoint_info(nlp_pipeline, pipeline_type="nlp")
    except Exception as exc:
        logger.error("Failed to load NLP pipeline for version info: %s", exc)
        nlp_info = {"checkpoint_file": "unavailable", "epoch": "N/A", "metrics": {}}

    return VersionResponse(
        app_version=APP_VERSION,
        image_model=CheckpointInfo(**img_info),
        nlp_model=CheckpointInfo(**nlp_info),
    )


# ---------------------------------------------------------------------------
# POST /predict/image
# ---------------------------------------------------------------------------


@router.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    summary="Chest X-ray disease classification",
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or request"},
        500: {"model": ErrorResponse, "description": "Internal inference error"},
    },
)
async def predict_image(
    file: UploadFile = File(
        ...,
        description="Chest X-ray image file (JPEG / PNG / BMP / TIFF, max 10 MB).",
    ),
) -> ImagePredictionResponse:
    """Accepts a chest X-ray image upload and returns disease classification results.

    The endpoint uses the pre-loaded EfficientNet-B0 model and optionally
    generates a Grad-CAM attention map for the predicted class.

    **Supported formats:** JPEG, PNG, BMP, TIFF  
    **Max file size:** 10 MB
    """
    # ── Validate content type ─────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{content_type}'. "
                   f"Allowed: {sorted(ALLOWED_IMAGE_TYPES)}",
        )

    # ── Read and size-check the file ──────────────────────────────────────
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size {size_mb:.1f} MB exceeds maximum allowed {MAX_IMAGE_SIZE_MB} MB.",
        )

    # ── Decode image ──────────────────────────────────────────────────────
    try:
        pil_img, img_np = read_upload_as_pil(file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # ── Run inference ─────────────────────────────────────────────────────
    try:
        pipeline = get_image_pipeline()
        result = pipeline.predict(img_np)
    except AppValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        )
    except AppInferenceError as exc:
        logger.error("Image inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc.message}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during image inference.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {exc}",
        )

    # ── Build top-3 predictions from class_probabilities ─────────────────
    class_probs: dict = result.get("class_probabilities", {})
    sorted_preds = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    top_3: List[TopPrediction] = [
        TopPrediction(rank=i + 1, disease=d, confidence=round(c, 6))
        for i, (d, c) in enumerate(sorted_preds[:3])
    ]

    # ── Derive predicted class index for Grad-CAM ─────────────────────────
    predicted_disease: str = result["predicted_disease"]
    pred_idx = pipeline.CLASSES.index(predicted_disease) if predicted_disease in pipeline.CLASSES else None

    # ── Grad-CAM (best-effort, non-blocking) ──────────────────────────────
    gradcam_path: str | None = None
    if pred_idx is not None:
        gradcam_path = run_gradcam(
            image_pipeline=pipeline,
            image_np=img_np,
            original_pil=pil_img,
            pred_class_idx=pred_idx,
        )

    return ImagePredictionResponse(
        predicted_disease=predicted_disease,
        confidence=round(result["confidence"], 6),
        top_predictions=top_3,
        gradcam_image_path=gradcam_path,
        gradcam_available=gradcam_path is not None,
    )


# ---------------------------------------------------------------------------
# POST /predict/symptoms
# ---------------------------------------------------------------------------


@router.post(
    "/predict/symptoms",
    response_model=SymptomPredictionResponse,
    summary="Free-text symptom-based disease diagnosis",
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid symptom text"},
        500: {"model": ErrorResponse, "description": "Internal inference error"},
    },
)
async def predict_symptoms(body: SymptomRequest) -> SymptomPredictionResponse:
    """Accepts a free-text symptom description and returns disease predictions.

    The endpoint uses the pre-loaded DistilBERT symptom classifier trained on
    41 disease classes.

    **Example input:**  
    `"I have fever, cough, sore throat, headache and body pain."`
    """
    try:
        pipeline = get_nlp_pipeline()
        result = pipeline.predict(body.symptoms, top_k=body.top_k)
    except AppValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        )
    except AppInferenceError as exc:
        logger.error("NLP inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc.message}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during NLP inference.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {exc}",
        )

    top_preds: List[TopPrediction] = [
        TopPrediction(
            rank=p["rank"],
            disease=p["disease"],
            confidence=p["confidence"],
        )
        for p in result.get("top_predictions", [])
    ]

    return SymptomPredictionResponse(
        predicted_disease=result["predicted_disease"],
        confidence=result["confidence"],
        top_predictions=top_preds,
        preprocessed_text=result.get("preprocessed_text", ""),
    )
