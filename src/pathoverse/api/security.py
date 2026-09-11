from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


PUBLIC_PATHS = {
    "/",
    "/api/health",
    "/api/health/live",
    "/api/health/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
}


bearer_scheme = HTTPBearer(
    auto_error=False,
)


def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
) -> None:
    """Verify the configured PathoVerse Bearer API key."""

    if request.url.path in PUBLIC_PATHS:
        return

    from pathoverse.config import get_settings

    settings = get_settings()

    if settings.environment in {"development", "test"}:
        return

    expected_key = settings.api_key.strip()

    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(
        credentials.credentials,
        expected_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
) -> None:
    """FastAPI dependency for protected PathoVerse API endpoints."""

    verify_api_key(request, credentials)