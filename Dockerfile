FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

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

RUN python3 --version

RUN python3 -m venv /opt/pathoverse-venv

ENV PATH="/opt/pathoverse-venv/bin:${PATH}"

RUN python --version

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY external/prov-gigapath /app/external/prov-gigapath
COPY external/CONCH /app/external/CONCH

RUN python -m pip install -e .

RUN python -m pip install /app/external/prov-gigapath
RUN python -m pip install /app/external/CONCH

RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/metadata \
    /app/results \
    /app/logs \
    /app/data/huggingface

RUN useradd \
    --create-home \
    --shell /bin/bash \
    pathoverse

RUN chown -R pathoverse:pathoverse /app
RUN chown -R pathoverse:pathoverse /opt/pathoverse-venv

USER pathoverse

ENV HF_HOME=/app/data/huggingface

ENV PATHOVERSE_ENV=production
ENV PATHOVERSE_HOST=0.0.0.0
ENV PATHOVERSE_PORT=8000
ENV PATHOVERSE_LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "pathoverse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]