from __future__ import annotations

import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - defensive guard for optional dep
    raise ImportError(
        "huggingface-hub is required to download MobileCLIP checkpoints. "
        "Install it via `pip install huggingface-hub`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_MOBILECLIP_SPECS = {
    "MobileCLIP-S1": {
        "repo_id": "apple/MobileCLIP-S1-OpenCLIP",
        "filename": "open_clip_pytorch_model.bin",
    },
    "MobileCLIP-S2": {"repo_id": "pcuenq/MobileCLIP-S2", "filename": "mobileclip_s2.pt"},
}


def _model_cache_root() -> Path:
    """Return the directory used to persist downloaded checkpoints."""
    cache_root = Path(
        os.environ.get("ECOGROW_MODEL_CACHE", PROJECT_ROOT / "artifacts" / "pretrained")
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def ensure_mobileclip_checkpoint(model_name: str = "MobileCLIP-S2") -> str:
    """
    Ensure that the requested MobileCLIP checkpoint is available locally and return its path.
    """
    env_override = os.environ.get("ECOGROW_CLIP_PRETRAINED")
    if env_override:
        return env_override

    try:
        spec = _MOBILECLIP_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"No checkpoint metadata defined for '{model_name}'.") from exc

    cache_dir = _model_cache_root() / model_name.lower()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / spec["filename"]
    if target_path.exists():
        return str(target_path)

    downloaded_path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    return str(downloaded_path)


__all__ = ["ensure_mobileclip_checkpoint"]
