from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


app = FastAPI(
    title=settings.app_name,
    description=(
        "Multimodal Foundation Model Platform for "
        "Whole-Slide Pathology Analysis & Evaluation"
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/", tags=["Health"])
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }