from __future__ import annotations

import logging

from fastapi import APIRouter

from pathoverse.config import get_settings


router = APIRouter()

settings = get_settings()

logger = logging.getLogger("pathoverse.api.health")


# ==========================================================
# LIVENESS
# ==========================================================

@router.get(
    "",
    summary="Health check",
)
def health_check():
    """
    Basic health check.

    Indicates that the API process is alive.
    """

    return {
        "status": "healthy",
        "service": "pathoverse-api",
        "version": settings.app_version,
        "environment": settings.environment,
    }


# ==========================================================
# LIVENESS ALIAS
# ==========================================================

@router.get(
    "/live",
    summary="Liveness check",
)
def liveness_check():
    """
    Liveness endpoint used by deployment/orchestration
    systems to determine whether the API process is alive.
    """

    return {
        "status": "alive",
    }


# ==========================================================
# READINESS
# ==========================================================

@router.get(
    "/ready",
    summary="Readiness check",
)
def readiness_check():
    """
    Readiness endpoint.

    Checks whether the core filesystem locations required
    by the platform are available.
    """

    required_paths = {
        "project_root": settings.project_root,
        "data_root": settings.data_root,
        "metadata_root": settings.metadata_root,
        "processed_root": settings.processed_root,
        "results_root": settings.results_root,
    }

    checks = {
        name: path.exists()
        for name, path in required_paths.items()
    }

    ready = all(checks.values())

    if not ready:
        logger.warning(
            "PathoVerse readiness check failed: %s",
            checks,
        )

    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }