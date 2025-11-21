from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests
import torch
from flask import Flask, jsonify, request
from PIL import Image

from ecogrow.data.plant_data import make_segment_fn
from ecogrow.models.checkpoint_cache import ensure_mobileclip_checkpoint
from ecogrow.models.open_clip_wrapper import init_open_clip, freeze_open_clip_backbone, DiseaseClipDetector
from ecogrow.preprocessing.image_segmentator import (
    black_bg_composite,
    crop_to_alpha_bbox,
    segment_plant_rgba,
)
# Note: training engine is not required at runtime for inference

DEFAULT_UNKNOWN_THRESHOLD = float(os.getenv("ECOGROW_UNKNOWN_THRESHOLD", "0.5"))


def _resolve_model_name() -> str:
    return os.getenv("ECOGROW_CLIP_MODEL_NAME", "MobileCLIP-S2")


def _resolve_pretrained_tag(model_name: str) -> str:
    env_override = os.getenv("ECOGROW_CLIP_PRETRAINED")
    if env_override:
        return env_override
    if model_name.startswith("MobileCLIP"):
        return ensure_mobileclip_checkpoint(model_name=model_name)
    return "laion2b_s34b_b79k"


MODEL_NAME = _resolve_model_name()
PRETRAINED_TAG = _resolve_pretrained_tag(MODEL_NAME)
EMBEDDINGS_DIR = Path(os.getenv("ECOGROW_EMBEDDINGS_DIR", "artifacts/embeddings")).expanduser()
SEGMENTATION_ENABLED = os.getenv("ECOGROW_SEGMENTATION", "1").lower() not in {"0", "false", "no"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


## Disease detectors are provided by ecogrow.models.open_clip_wrapper


def _build_segmenter():
    if not SEGMENTATION_ENABLED:
        return None
    return make_segment_fn(
        segment_plant_rgba,
        crop_to_alpha_bbox,
        black_bg_composite,
        pad=12,
    )


def _load_detectors(
    root: Path,
    clip_model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    device: torch.device,
) -> Dict[str, DiseaseClipDetector]:
    if not root.is_dir():
        raise RuntimeError(
            f"Embeddings directory '{root}' not found. "
            "Set ECOGROW_EMBEDDINGS_DIR to a folder containing *.pt artifacts."
        )

    detectors: Dict[str, DiseaseClipDetector] = {}
    for path in sorted(root.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        detector_name = (
            payload.get("detector")
            or payload.get("family")
            or path.stem
        )
        classes = payload.get("classes")
        text_features = payload.get("text_features")
        temperature = float(payload.get("temperature", 0.07))
        if classes is None or text_features is None:
            raise ValueError(f"Embedding file '{path.name}' missing classes or text_features.")

        tensor_features = torch.as_tensor(text_features)
        detectors[detector_name] = DiseaseClipDetector(
            classes=list(classes),
            temperature=temperature,
            clip_model=clip_model,
            text_encoder=text_encoder,
            device=device,
            text_features=tensor_features,
            detector_id=detector_name,
        )

    if not detectors:
        raise RuntimeError(f"No *.pt embedding files found in '{root}'.")
    return detectors


class DiseaseInferenceService:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained_tag: str = PRETRAINED_TAG,
        embeddings_dir: Path = EMBEDDINGS_DIR,
        device: Optional[torch.device] = None,
        default_unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
        enable_segmentation: bool = SEGMENTATION_ENABLED,
    ) -> None:
        self.device = device or DEVICE
        self.model_name = model_name
        self.pretrained_tag = pretrained_tag
        (
            self.clip_model,
            self.preprocess,
            _,
            self.text_encoder,
        ) = init_open_clip(
            model_name=model_name,
            pretrained_tag=pretrained_tag,
            device=self.device,
        )
        freeze_open_clip_backbone(self.clip_model)

        self.segment_fn = _build_segmenter() if enable_segmentation else None
        self.embeddings_dir = Path(embeddings_dir)
        self.detectors = _load_detectors(
            self.embeddings_dir,
            self.clip_model,
            self.text_encoder,
            self.device,
        )
        self.default_unknown_threshold = float(default_unknown_threshold)

    def list_families(self) -> List[str]:
        return sorted(self.detectors.keys())

    def classes_for(self, family: str) -> List[str]:
        if family not in self.detectors:
            raise KeyError(f"Family '{family}' not available.")
        return list(self.detectors[family].classes)

    def reload_embeddings(self) -> int:
        self.detectors = _load_detectors(
            self.embeddings_dir,
            self.clip_model,
            self.text_encoder,
            self.device,
        )
        return len(self.detectors)

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        processed = image.convert("RGB")
        if self.segment_fn is not None:
            processed = self.segment_fn(processed)
        return self.preprocess(processed).unsqueeze(0)

    def _run(
        self,
        image: Image.Image,
        *,
        family: Optional[str],
        unknown_threshold: Optional[float],
    ) -> Dict[str, object]:
        if not self.detectors:
            raise RuntimeError("No family detectors available.")

        thr = self.default_unknown_threshold if unknown_threshold is None else float(unknown_threshold)
        if family:
            if family not in self.detectors:
                raise KeyError(f"Family '{family}' not available.")
            targets = [self.detectors[family]]
        else:
            targets = list(self.detectors.values())

        with torch.no_grad():
            tensor = self._prepare_tensor(image)
            preds = [det.predict(tensor, unknown_threshold=thr) for det in targets]

        preds.sort(key=lambda item: item["probability"], reverse=True)
        return {
            "top_prediction": preds[0],
            "predictions": preds,
            "model": {
                "clip_model": self.model_name,
                "pretrained": self.pretrained_tag,
                "device": str(self.device),
                "temperatures": {
                    (det.detector_id or f"detector_{idx}"): det.temperature
                    for idx, det in enumerate(targets)
                },
            },
        }

    def predict_from_bytes(
        self,
        data: bytes,
        *,
        family: Optional[str] = None,
        unknown_threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return self._run(image, family=family, unknown_threshold=unknown_threshold)

    def predict_from_url(
        self,
        url: str,
        *,
        timeout: float = 4.0,
        family: Optional[str] = None,
        unknown_threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return self.predict_from_bytes(resp.content, family=family, unknown_threshold=unknown_threshold)


_SERVICE: Optional[DiseaseInferenceService] = None


def get_disease_inference_service() -> DiseaseInferenceService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = DiseaseInferenceService()
    return _SERVICE


def _parse_unknown_threshold(raw_value) -> float:
    if raw_value is None:
        return DEFAULT_UNKNOWN_THRESHOLD
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        raise ValueError("unknown_threshold must be a float between 0 and 1.") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError("unknown_threshold must be between 0.0 and 1.0.")
    return value


def _pick_param(body: Optional[dict], name: str, default=None):
    if name in request.values:
        return request.values.get(name)
    if isinstance(body, dict) and name in body:
        return body.get(name, default)
    return default


@app.post("/api/detect_disease")
def detect_disease():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file in request."}), 400

    file_bytes = request.files["image"].read()

    service = get_disease_inference_service()

    body = request.get_json(silent=True) if request.is_json else None
    try:
        unknown_threshold = _parse_unknown_threshold(
            _pick_param(body, "unknown_threshold", service.default_unknown_threshold)
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    family = _pick_param(body, "family")
    if isinstance(family, str):
        family = family.strip() or None

    try:
        result = service.predict_from_bytes(
            file_bytes,
            family=family,
            unknown_threshold=unknown_threshold,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    response = {"status": "success", "data": result}
    if family:
        response["data"]["requested_family"] = family
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
