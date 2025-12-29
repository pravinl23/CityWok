# CityWok API - Production Dockerfile
# Audio-only identification service

FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    zstd \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/uploads

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Expose API port
EXPOSE 8000

# Health check (increased start period for database download)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/test || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}
