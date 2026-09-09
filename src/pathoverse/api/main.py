from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pathoverse.api.routes import (
    analysis,
    benchmarks,
    health,
    models,
    retrieval,
    slides,
)


app = FastAPI(
    title="PathoVerse AI",
    description=(
        "Multimodal Foundation Model Platform for "
        "Whole-Slide Pathology Analysis & Evaluation"
    ),
    version="0.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
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
        "name": "PathoVerse AI",
        "version": app.version,
        "status": "running",
    }
