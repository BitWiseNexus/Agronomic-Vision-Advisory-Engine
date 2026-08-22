# ---------------------------------------------------------------------------
# API image. Python 3.12 because PyTorch publishes no 3.14 wheels yet.
# Torch comes from the CPU wheel index — the CUDA build is ~2.5 GB and this
# service is CPU-only inference.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the HF model cache on a volume so the container does not re-download
    # bge-small on every restart.
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

WORKDIR /app

# libgl/libglib are needed by OpenCV even in the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Source last, so a code change does not invalidate the dependency layer.
COPY backend/ ./backend/
COPY data/knowledge/ ./data/knowledge/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/backend

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models /app/storage /app/models/checkpoints \
    && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
