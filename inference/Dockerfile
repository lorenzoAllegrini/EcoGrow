FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV ECOGROW_EMBEDDINGS_DIR=/app/artifacts \
    ECOGROW_CLIP_MODEL_NAME=ViT-B-32 \
    ECOGROW_CLIP_PRETRAINED=laion2b_s34b_b79k \
    ECOGROW_SEGMENTATION=1

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "inference.inference_service:app"]
