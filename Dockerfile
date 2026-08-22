# Hugging Face Spaces expects the app on port 7860.
# Build is intentionally torch-free: fastembed uses ONNX Runtime, which keeps
# this image near ~400MB instead of ~2.5GB and starts in seconds on 2 vCPU.

FROM python:3.12-slim

# HF Spaces runs as a non-root user; matching it locally avoids
# "works on my machine, permission-denied on the Space" surprises.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# System deps: faiss-cpu needs libgomp (OpenMP runtime).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, so code edits don't invalidate the pip layer.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser data/ ./data/

USER appuser

# Pre-download the embedding model into the image. Without this, the first
# request after every cold start pays a ~30s model download and the p100
# latency number becomes meaningless.
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

# Single worker on purpose: the FAISS index and ONNX session are held in
# process memory. Multiple workers would duplicate both for no throughput
# gain at this corpus size.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
