FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

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

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV ECOGROW_EMBEDDINGS_DIR=/app/artifacts \
    ECOGROW_CLIP_MODEL_NAME=ViT-B-32 \
    ECOGROW_CLIP_PRETRAINED=laion2b_s34b_b79k

EXPOSE 8080

CMD ["uvicorn", "ecoGrow.inference_service:app", "--host", "0.0.0.0", "--port", "8080"]
