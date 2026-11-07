# ==============================================================================
# Production Dockerfile for Industrial Inspection Multimodal AI Agent
# ==============================================================================

FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    APP_DEVICE=auto

# Install system dependencies needed for OpenCV headless & utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CPU wheel by default for portable deployment; override for CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, documentation, and frontend
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY project_context/ ./project_context/
COPY docs/ ./docs/
COPY runs/ ./runs/

# Copy fitted models and seed knowledge base (preserve pre-trained PaDiM models)
COPY data/index/ ./data/index/
COPY data/kb/ ./data/kb/

# Create runtime directories for uploads and heatmaps
RUN mkdir -p data/uploads data/heatmaps

EXPOSE 8000

# Healthcheck probe hitting /api/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Production server entrypoint
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

