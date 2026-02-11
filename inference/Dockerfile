# CapCheck AI Detection Service - CPU Only
# ViT-Base (86M params) runs efficiently on CPU (~25-100ms inference)
# Image size: ~2GB (no CUDA dependencies needed)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY predict.py .
COPY server.py .

# Model cache directory (mounted as Fly Volume)
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

# Pre-download both models at build time (from HuggingFace source of truth)
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='capcheck/ai-image-detection')"
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='capcheck/ai-human-generated-image-detection')"

EXPOSE 5000

# Use Gunicorn for production - worker isolation prevents container crash on segfault
# Single worker (ViT model is memory-heavy), 120s timeout for large images
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "--graceful-timeout", "30", "server:app"]
