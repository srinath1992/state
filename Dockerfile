# Simplified Docker build for H100 systems - focusing on basics first
# syntax=docker/dockerfile:1.4

FROM ubuntu:22.04

# Build arguments for optimization
ARG BUILDKIT_INLINE_CACHE=1
ARG MAX_JOBS=16

# Environment variables for build speed and reproducibility
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MAKEFLAGS="-j${MAX_JOBS}"

# Install system dependencies with BuildKit cache mounts for speed
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    git \
    python3 \
    python3-dev \
    python3-pip

# Install uv with cache mount for faster repeated builds
RUN --mount=type=cache,target=/root/.cache \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Create app directory
WORKDIR /app

# Install State using uv (fastest Python package manager)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install arc-state

# Copy configuration files
COPY examples/ /app/examples/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD state --help || exit 1

# Set entrypoint
ENTRYPOINT ["state"]
CMD ["--help"]