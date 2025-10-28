import io
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import open_clip
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from image_segmentator import black_bg_composite, crop_to_alpha_bbox, segment_plant_rgba
from utils import make_segment_fn


class EmbeddingRecord(Dict):
    """Typed alias for the embedding payload."""


def _load_embeddings(root: Path, device: torch.device) -> Dict[str, EmbeddingRecord]:
    if not root.is_dir():
        raise RuntimeError(
            f"Embeddings directory '{root}' not found. Set ECOGROW_EMBEDDINGS_DIR to a valid path."
        )

    records: Dict[str, EmbeddingRecord] = {}
    index_path = root / "index.json"
    index_data = None
    if index_path.is_file():
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

    candidates = sorted(root.glob("*.pt"))
    if not candidates:
        raise RuntimeError(f"No *.pt embedding files found in '{root}'.")

    for path in candidates:
        payload = torch.load(path, map_location="cpu")
        family = payload.get("family")
        if not family:
            raise ValueError(f"Embedding file '{path.name}' missing 'family' field.")
        text_features = payload.get("text_features")
        classes = payload.get("classes")
        temperature = float(payload.get("temperature", 0.07))
        if text_features is None or classes is None:
            raise ValueError(f"Embedding file '{path.name}' is incomplete.")

        text_features = F.normalize(text_features, dim=-1).to(device)
        records[family] = {
            "family": family,
            "classes": list(classes),
            "text_features": text_features,
            "temperature": temperature,
            "source_file": path.name,
        }

    if index_data and isinstance(index_data, dict):
        records["_index"] = index_data

    return records


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


def _prepare_segment_fn() -> Optional[Callable]:
    enabled = os.environ.get("ECOGROW_SEGMENTATION", "1")
    if enabled.lower() in {"0", "false", "no"}:
        return None
    return make_segment_fn(
        segment_plant_rgba,
        crop_to_alpha_bbox,
        black_bg_composite,
        pad=12,
    )


MODEL_NAME = os.environ.get("ECOGROW_CLIP_MODEL_NAME", "ViT-B-32")
PRETRAINED_TAG = os.environ.get("ECOGROW_CLIP_PRETRAINED", "laion2b_s34b_b79k")
EMBEDDINGS_DIR = Path(os.environ.get("ECOGROW_EMBEDDINGS_DIR", "artifacts")).expanduser()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_MODEL, _, PREPROCESS = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED_TAG
)
CLIP_MODEL = CLIP_MODEL.to(DEVICE).eval()
for param in CLIP_MODEL.parameters():
    param.requires_grad_(False)

SEGMENT_FN = _prepare_segment_fn()
EMBEDDINGS = _load_embeddings(EMBEDDINGS_DIR, DEVICE)

app = FastAPI(title="EcoGrow CLIP Inference Service")


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    family: Optional[str] = Query(default=None, description="Limit inference to a specific family."),
    unknown_threshold: float = Query(
        default=0.5, ge=0.0, le=1.0, description="Threshold below which the prediction is marked unknown."
    ),
) -> Dict[str, object]:
    file_bytes = await image.read()
    pil_image = _read_image(file_bytes)
    if SEGMENT_FN is not None:
        pil_image = SEGMENT_FN(pil_image)

    image_tensor = PREPROCESS(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = CLIP_MODEL.encode_image(image_tensor)
    image_features = F.normalize(image_features, dim=-1)

    families_to_eval: List[str]
    if family:
        if family not in EMBEDDINGS:
            raise HTTPException(status_code=404, detail=f"Family '{family}' not available.")
        families_to_eval = [family]
    else:
        families_to_eval = [fam for fam in EMBEDDINGS.keys() if not fam.startswith("_")]

    if not families_to_eval:
        raise HTTPException(status_code=500, detail="No embeddings available for inference.")

    family_predictions = []
    for fam in families_to_eval:
        record = EMBEDDINGS[fam]
        text_features = record["text_features"]
        logits = image_features @ text_features.t()
        probs = (logits / record["temperature"]).softmax(dim=-1).squeeze(0)

        top_prob, top_idx = probs.max(dim=0)
        top_label = record["classes"][top_idx.item()]
        if top_prob.item() < unknown_threshold:
            display_label = "unknown"
        else:
            display_label = top_label

        per_class = [
            {"label": label, "probability": float(probs[i].item())}
            for i, label in enumerate(record["classes"])
        ]
        per_class.sort(key=lambda item: item["probability"], reverse=True)

        family_predictions.append(
            {
                "family": fam,
                "prediction": display_label,
                "raw_label": top_label,
                "probability": float(top_prob.item()),
                "classes": per_class,
            }
        )

    family_predictions.sort(key=lambda item: item["probability"], reverse=True)
    temperature_map = {
        fam: float(EMBEDDINGS[fam]["temperature"]) for fam in families_to_eval
    }
    return {
        "top_prediction": family_predictions[0],
        "predictions": family_predictions,
        "model": {
            "clip_model": MODEL_NAME,
            "pretrained": PRETRAINED_TAG,
            "temperatures": temperature_map,
        },
    }
