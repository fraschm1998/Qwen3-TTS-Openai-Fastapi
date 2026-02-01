# Qwen3-TTS OpenAI-Compatible API Server
# Multi-stage Dockerfile optimized for GPU/CUDA and CPU deployments
#
# Dependencies are resolved from pyproject.toml (single source of truth).
# CUDA torch is installed first from the PyTorch index; subsequent
# pip install from pyproject.toml sees it as already satisfied.

# =============================================================================
# Stage 1: Base runtime image with system dependencies
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# NVIDIA Container Runtime (required for PyTorch CUDA detection)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Builder — compile flash-attn (needs CUDA dev headers + GCC)
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    git \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja packaging

WORKDIR /build
COPY pyproject.toml README.md ./

# 1. Install CUDA torch first (from PyTorch index, not PyPI)
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install all deps from pyproject.toml — pip sees torch already satisfied
#    Use onnxruntime-gpu instead of onnxruntime for CUDA support
RUN pip install --no-cache-dir onnxruntime-gpu && \
    pip install --no-cache-dir ".[api]"

# 3. Compile flash-attention (requires CUDA dev tools)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# =============================================================================
# Stage 3: Production image (official backend)
# =============================================================================
FROM base AS production

WORKDIR /app

# Copy pre-built venv from builder (includes CUDA torch + flash-attn)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

# Register package without re-resolving dependencies
RUN pip install --no-cache-dir --no-deps -e .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /tmp/numba_cache \
    && chown -R appuser:appuser /app /tmp/numba_cache
USER appuser

ENV HOST=0.0.0.0
ENV PORT=8880
ENV WORKERS=1
ENV PYTHONPATH=/app
ENV TTS_BACKEND=official

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["python", "-m", "api.main"]

# =============================================================================
# Stage 4: vLLM builder
# =============================================================================
FROM base AS vllm-builder

WORKDIR /build
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir vllm>=0.4.0

RUN pip install --no-cache-dir onnxruntime-gpu && \
    pip install --no-cache-dir ".[api]"

# flash-attn optional for vLLM (may conflict)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# =============================================================================
# Stage 5: vLLM production image
# =============================================================================
FROM base AS vllm-production

WORKDIR /app

COPY --from=vllm-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

RUN pip install --no-cache-dir --no-deps -e .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /tmp/numba_cache \
    && chown -R appuser:appuser /app /tmp/numba_cache
USER appuser

ENV HOST=0.0.0.0
ENV PORT=8880
ENV WORKERS=1
ENV PYTHONPATH=/app
ENV TTS_BACKEND=vllm_omni

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["python", "-m", "api.main"]

# =============================================================================
# CPU-only variant
# =============================================================================
FROM python:3.11-slim AS cpu-base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./

# Install CPU torch, then resolve everything from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[api]"

COPY . .

RUN pip install --no-cache-dir --no-deps -e .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /tmp/numba_cache \
    && chown -R appuser:appuser /app /tmp/numba_cache
USER appuser

ENV HOST=0.0.0.0
ENV PORT=8880
ENV WORKERS=1
ENV PYTHONPATH=/app

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["python", "-m", "api.main"]
