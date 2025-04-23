FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy local repo into container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    websockets \
    numpy \
    torchaudio \
    soundfile \
    yaml \
    aiortc \
    aiohttp \
    pydantic

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for FastAPI
EXPOSE 8000

# Health check (optional)
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Run FastAPI app
CMD ["uvicorn", "voice_api:app", "--host", "0.0.0.0", "--port", "8000"]
