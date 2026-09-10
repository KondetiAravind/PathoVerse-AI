FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# =========================================================
# System dependencies
# =========================================================

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libopenslide0 \
    libopenslide-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# =========================================================
# Python environment
# =========================================================

RUN python3 --version

RUN python3 -m venv /opt/pathoverse-venv

ENV PATH="/opt/pathoverse-venv/bin:${PATH}"

RUN python --version

RUN python -m pip install --upgrade pip setuptools wheel

# =========================================================
# Application
# =========================================================

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts

# =========================================================
# Existing external research repositories
# =========================================================

COPY external/prov-gigapath /app/external/prov-gigapath
COPY external/CONCH /app/external/CONCH

# =========================================================
# Install PathoVerse
# =========================================================

RUN python -m pip install -e .
RUN python -m pip install "uvicorn[standard]"

# =========================================================
# Install GigaPath
# =========================================================

RUN if [ -f /app/external/prov-gigapath/requirements.txt ]; then \
        python -m pip install \
        -r /app/external/prov-gigapath/requirements.txt; \
    fi

# =========================================================
# Install CONCH
# CONCH uses pyproject.toml, not requirements.txt
# =========================================================

RUN python -m pip install /app/external/CONCH

# =========================================================
# Runtime directories
# =========================================================

RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/metadata \
    /app/results \
    /app/logs \
    /app/data/huggingface

# =========================================================
# Non-root user
# =========================================================

RUN useradd \
    --create-home \
    --shell /bin/bash \
    pathoverse

RUN chown -R pathoverse:pathoverse /app
RUN chown -R pathoverse:pathoverse /opt/pathoverse-venv

USER pathoverse

# =========================================================
# Runtime configuration
# =========================================================

ENV HF_HOME=/app/data/huggingface

ENV PATHOVERSE_ENV=production
ENV PATHOVERSE_HOST=0.0.0.0
ENV PATHOVERSE_PORT=8000
ENV PATHOVERSE_LOG_LEVEL=INFO

EXPOSE 8000

# =========================================================
# Health check
# =========================================================

HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# =========================================================
# Start FastAPI
# =========================================================

CMD ["python", "-m", "uvicorn", "pathoverse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]