from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pathoverse.config import get_settings

from pathoverse.api.security import require_api_key

from pathoverse.api.routes.health import router as health_router
from pathoverse.api.routes.models import router as models_router
from pathoverse.api.routes.slides import router as slides_router
from pathoverse.api.routes.analysis import router as analysis_router
from pathoverse.api.routes.retrieval import router as retrieval_router
from pathoverse.api.routes.benchmarks import router as benchmarks_router


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle.

    Heavy foundation models are loaded lazily by the model service,
    so application startup remains lightweight.
    """
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "PathoVerse AI — Multimodal Foundation Model Platform "
        "for Whole-Slide Pathology Analysis & Evaluation"
    ),
    lifespan=lifespan,
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# ==========================================================
# PUBLIC ROUTES
# ==========================================================

app.include_router(
    health_router,
    prefix="/api/health",
    tags=["Health"],
)


# ==========================================================
# PROTECTED ROUTES
# ==========================================================

app.include_router(
    models_router,
    prefix="/api/models",
    tags=["Models"],
    dependencies=[Depends(require_api_key)],
)

app.include_router(
    slides_router,
    prefix="/api/slides",
    tags=["Slides"],
    dependencies=[Depends(require_api_key)],
)

app.include_router(
    analysis_router,
    prefix="/api/analysis",
    tags=["Analysis"],
    dependencies=[Depends(require_api_key)],
)

app.include_router(
    retrieval_router,
    prefix="/api/retrieval",
    tags=["Retrieval"],
    dependencies=[Depends(require_api_key)],
)

app.include_router(
    benchmarks_router,
    prefix="/api/benchmarks",
    tags=["Benchmarks"],
    dependencies=[Depends(require_api_key)],
)


# ==========================================================
# ROOT
# ==========================================================

@app.get("/", tags=["Root"])
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
    }