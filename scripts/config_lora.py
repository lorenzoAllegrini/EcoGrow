import os
from pathlib import Path

import torch


def _find_adapter_file(run_dir: Path, rel_path: str) -> Path | None:
    """Resolve the adapter path from the value stored in `lora_adapter_path`."""
    candidate = (run_dir / rel_path).resolve()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        # Try to pick a single .pt file inside the directory
        pt_files = sorted(candidate.glob("*.pt"))
        if pt_files:
            return pt_files[0]
    return None


def convert_run(run_dir: Path) -> None:
    """Convert all detector payloads in a single experiment run."""
    detectors_dir = run_dir / "detectors"
    if not detectors_dir.is_dir():
        return

    for det_path in sorted(detectors_dir.glob("*.pt")):
        # Detector payload may contain custom objects (e.g. PEFT enums); disable
        # weights_only safety since these files are local/trusted.
        payload = torch.load(det_path, map_location="cpu", weights_only=False)

        # Already converted
        if "lora_adapter" in payload:
            continue

        rel = payload.get("lora_adapter_path")
        if not rel:
            # No LoRA info to migrate
            continue

        adapter_path = _find_adapter_file(run_dir, rel)
        if adapter_path is None or not adapter_path.is_file():
            print(f"[WARN] Adapter file not found for {det_path} (path={rel!r})")
            continue

        # Adapter payload was saved by our training code and may contain
        # custom PEFT types; use weights_only=False for compatibility.
        adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=False)

        # Embed adapter payload into detector and drop path reference
        payload["lora_adapter"] = adapter_payload
        payload.pop("lora_adapter_path", None)

        torch.save(payload, det_path)
        print(f"[OK] Updated detector with embedded LoRA: {det_path}")


def main() -> None:
    # Default: ../experiments relative to this script
    script_dir = Path(__file__).resolve().parent
    experiments_root = (script_dir.parent / "experiments").resolve()

    if not experiments_root.is_dir():
        raise SystemExit(f"Experiments directory not found: {experiments_root}")

    for entry in sorted(experiments_root.iterdir()):
        if entry.is_dir():
            convert_run(entry)


if __name__ == "__main__":
    main()
