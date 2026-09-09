from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pathoverse.config import get_settings
from pathoverse.api.routes import (
    analysis,
    benchmarks,
    health,
    models,
    retrieval,
    slides,
)


settings = get_settings()


# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("pathoverse.api")


# ==========================================================
# APPLICATION LIFECYCLE
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown lifecycle.

    Heavy models are intentionally not loaded here.
    Model loading remains lazy so that the API can start
    quickly and individual services can control inference
    resources.
    """

    logger.info(
        "Starting %s v%s [%s]",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )

    logger.info(
        "Project root: %s",
        settings.project_root,
    )

    logger.info(
        "Data root: %s",
        settings.data_root,
    )

    yield

    logger.info(
        "Shutting down %s",
        settings.app_name,
    )


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title=settings.app_name,
    description=(
        "Multimodal Foundation Model Platform for "
        "Whole-Slide Pathology Analysis & Evaluation"
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_method_list,
    allow_headers=settings.cors_header_list,
)


# ==========================================================
# GLOBAL VALIDATION ERROR HANDLER
# ==========================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """
    Return a consistent JSON response for invalid requests.
    """

    logger.warning(
        "Request validation failed: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": exc.errors(),
            }
        },
    )


# ==========================================================
# GLOBAL UNHANDLED EXCEPTION HANDLER
# ==========================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    """
    Prevent internal implementation details from leaking
    through the API.

    Full exception information remains in server logs.
    Clients receive a safe generic error response.
    """

    logger.exception(
        "Unhandled exception during %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": (
                    "An unexpected internal error occurred."
                ),
            }
        },
    )


# ==========================================================
# ROUTERS
# ==========================================================

app.include_router(
    health.router,
    prefix="/api/health",
    tags=["Health"],
)

app.include_router(
    slides.router,
    prefix="/api/slides",
    tags=["Slides"],
)

app.include_router(
    models.router,
    prefix="/api/models",
    tags=["Models"],
)

app.include_router(
    retrieval.router,
    prefix="/api/retrieval",
    tags=["Retrieval"],
)

app.include_router(
    analysis.router,
    prefix="/api/analysis",
    tags=["Analysis"],
)

app.include_router(
    benchmarks.router,
    prefix="/api/benchmarks",
    tags=["Benchmarks"],
)


# ==========================================================
# ROOT ENDPOINT
# ==========================================================

@app.get(
    "/",
    tags=["Health"],
    summary="API root",
)
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
    }