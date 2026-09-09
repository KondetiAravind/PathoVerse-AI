from datetime import datetime, timezone

from fastapi import APIRouter

from pathoverse.api.dependencies import get_project_root


router = APIRouter()


@router.get("")
def health_check():
    project_root = get_project_root()

    return {
        "status": "healthy",
        "service": "pathoverse-api",
        "project": "PathoVerse AI",
        "project_root_exists": project_root.exists(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
def readiness_check():
    return {
        "status": "ready",
        "api": True,
    }
