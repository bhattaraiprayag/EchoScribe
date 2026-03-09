# Multi-stage Dockerfile for EchoScribe using uv
# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment explicitly
RUN uv venv /app/.venv

# Copy dependency metadata for reproducible cacheable installs
COPY pyproject.toml uv.lock /app/

# Install dependencies from standards-compliant metadata for portability.
# `--no-sources` avoids requiring custom source indexes during container builds.
# This layer rebuilds whenever uv.lock or pyproject.toml changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/backend
WORKDIR /app

# Copy application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Create models_cache directory and set permissions
RUN mkdir -p /app/models_cache && \
    chmod +x /usr/local/bin/docker-entrypoint.sh && \
    chown -R appuser:appuser /app

# Expose port
EXPOSE 8000

# Health check (httpx is installed in the main group)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/config', timeout=5)" || exit 1

# Fix mounted-volume ownership before launching the non-root app process
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
