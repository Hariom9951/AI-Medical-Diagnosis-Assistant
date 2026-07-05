"""FastAPI application entry point.

Instantiates the FastAPI app, registers exception handlers, mounts the
router, and pre-loads both inference pipelines at startup so that the first
prediction request is not slow.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import APP_VERSION, DESCRIPTION, PROJECT_NAME, router
from src.api.schemas import ErrorDetail, ErrorResponse
from src.api.utils import get_image_pipeline, get_nlp_pipeline
from src.utils.exceptions import AppException, AppInferenceError, AppValidationError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-loads both inference pipelines when the server starts so that model
    weights are in memory before the first HTTP request arrives."""
    logger.info("=== AI Medical Diagnosis Assistant API starting up ===")

    # Pre-load Image model
    try:
        get_image_pipeline()
        logger.info("[OK] ImageInferencePipeline loaded successfully.")
    except Exception as exc:
        logger.error("[ERR] Failed to load ImageInferencePipeline: %s", exc)

    # Pre-load NLP model
    try:
        get_nlp_pipeline()
        logger.info("[OK] NLPInferencePipeline loaded successfully.")
    except Exception as exc:
        logger.error("[ERR] Failed to load NLPInferencePipeline: %s", exc)

    logger.info("=== API startup complete. Ready to serve requests. ===")
    yield
    logger.info("=== AI Medical Diagnosis Assistant API shutting down. ===")


# ---------------------------------------------------------------------------
# App instantiation
# ---------------------------------------------------------------------------


app = FastAPI(
    title=PROJECT_NAME,
    description=DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "Hariom Sharma",
        "url": "https://github.com/Hariom9951",
    },
    license_info={
        "name": "MIT",
    },
)


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs each incoming request with method, path, status, and elapsed time."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(AppValidationError)
async def validation_error_handler(request: Request, exc: AppValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=ErrorDetail(
                error_code=exc.error_code.value,
                message=exc.message,
                details=exc.details or None,
            )
        ).model_dump(),
    )


@app.exception_handler(AppInferenceError)
async def inference_error_handler(request: Request, exc: AppInferenceError):
    logger.error("AppInferenceError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                error_code=exc.error_code.value,
                message=exc.message,
                details=exc.details or None,
            )
        ).model_dump(),
    )


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    logger.error("AppException on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                error_code=exc.error_code.value,
                message=exc.message,
                details=exc.details or None,
            )
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                error_code="SYSTEM_ERROR",
                message="An unexpected internal error occurred. Please try again later.",
                details={"detail": str(exc)},
            )
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Include router
# ---------------------------------------------------------------------------


app.include_router(router)
