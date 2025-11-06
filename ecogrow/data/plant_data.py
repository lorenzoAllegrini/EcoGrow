"""Dataset utilities for EcoGrow projects."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_SPECIES_TO_FAMILY: Dict[str, str] = {
    "Money_Plant": "Araceae",
    "Snake_Plant": "Asparagaceae",
    "Spider_Plant": "Asparagaceae",
}

DEFAULT_MAPPING: Dict[str, str] = DEFAULT_SPECIES_TO_FAMILY

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _segment_pipeline(
    img_rgb: Image.Image,
    segment_plant_rgba,
    crop_to_alpha_bbox,
    black_bg_composite,
    pad: int,
) -> Image.Image:
    rgba = segment_plant_rgba(img_rgb)
    rgba = crop_to_alpha_bbox(rgba, pad=pad)
    return black_bg_composite(rgba)


def make_segment_fn(
    segment_plant_rgba, crop_to_alpha_bbox, black_bg_composite, pad: int = 12
) -> Callable[[Image.Image], Image.Image]:
    """Convenience builder that composes segmentation utilities into a callable."""

    return partial(
        _segment_pipeline,
        segment_plant_rgba=segment_plant_rgba,
        crop_to_alpha_bbox=crop_to_alpha_bbox,
        black_bg_composite=black_bg_composite,
        pad=pad,
    )


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class PlantData(Dataset):

    def __init__(
        self,
        dataset_root: str | Path,
        family_id: str,
        train: bool = True,
        split: Optional[str] = None,
        segment_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
        transform: Optional[Callable[[Image.Image], object]] = None,
        split_name: Optional[str] = None,
    ) -> None:
        self.root = Path(dataset_root).expanduser().resolve()
        self.family_id = family_id
        self.segment_fn = segment_fn
        self.transform = transform

        if split is not None and split_name is not None:
            raise ValueError("Passa solo uno tra 'split' e 'split_name'.")

        if split is not None:
            resolved_split = split
        else:
            resolved_split = split_name or ("train" if train else "test")

        self.split = resolved_split
        self._prepare_samples()

    # ------------------------------------------------------------------
    # torch Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[object, int]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.segment_fn is not None:
            image = self.segment_fn(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_samples(self) -> None:
        family_dir = self._resolve_family_dir()
        class_to_idx: Dict[str, int] = {}
        samples: List[Sample] = []

        for disease_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            disease_name = disease_dir.name
            idx = class_to_idx.setdefault(disease_name, len(class_to_idx))
            for image_path in sorted(disease_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                samples.append(Sample(path=image_path, label=idx))

        if not samples:
            raise RuntimeError(
                f"Nessuna immagine trovata per la famiglia '{self.family_id}' "
                f"nel split '{self.split}'."
            )

        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    def make_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        *,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """Create a DataLoader with sane defaults for vision datasets."""

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def _resolve_family_dir(self) -> Path:
        split_dir = self.root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split '{self.split}' non presente in '{self.root}'."
            )

        lookup = {p.name.lower(): p for p in split_dir.iterdir() if p.is_dir()}
        key = self.family_id.lower()
        if key not in lookup:
            raise FileNotFoundError(
                f"La famiglia '{self.family_id}' non esiste nello split '{self.split}'. "
                f"Disponibili: {sorted(lookup)}"
            )
        return lookup[key]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def classes(self) -> List[str]:
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

    def label_maps(self) -> Dict[str, Dict[int, str] | Dict[str, int]]:
        return {
            "idx2label": dict(self.idx_to_class),
            "label2idx": dict(self.class_to_idx),
        }

def roboflow_format(
    init_root: str,
    final_root: str,
    species_to_family: Optional[Dict[str, str]] = None,
    *,
    overwrite: bool = False,
) -> None:
    """Convert a Roboflow folder export into the layout expected by EcoGrow.

    All source splits (train/valid/test) are collapsed into the single split `train`.
    """

    src_root = Path(init_root).expanduser().resolve()
    dst_root = Path(final_root).expanduser().resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Sorgente '{src_root}' non trovata.")
    dst_root.mkdir(parents=True, exist_ok=True)

    mapping = species_to_family or DEFAULT_MAPPING

    def resolve_family_and_disease(name: str) -> Tuple[str, str, str]:
        clean = name.replace("-", "_").strip("_")
        lower = clean.lower()
        for species, family in mapping.items():
            key = species.lower()
            if lower.startswith(key):
                remaining = clean[len(species):].strip("_")
                if not remaining and "_" in name:
                    remaining = name.split("_", 1)[1]
                disease = remaining or "healthy"
                return family.lower(), species, disease
        parts = clean.split("_", 1)
        if len(parts) == 2:
            return parts[0], name.split("_", 1)[0], parts[1] or "unknown"
        return clean, name, "unknown"

    for split_dir in src_root.iterdir():
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name.lower()
        target_split = "train" if split_name in {"train", "valid"} else "test"
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            family_name, species_name, disease_name = resolve_family_and_disease(
                class_dir.name
            )
            # Map train/valid into the train split; everything else goes to test so
            # downstream code retains a simple train/test structure.
            dest_dir = (
                dst_root / target_split / family_name / disease_name
            )
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in class_dir.iterdir():
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                dest_path = dest_dir / img_path.name
                if dest_path.exists():
                    if not overwrite:
                        continue
                    dest_path.unlink()
                shutil.copy2(img_path, dest_path)


__all__ = [
    "PlantData",
    "make_segment_fn",
    "roboflow_format",
    "DEFAULT_MAPPING",
    "DEFAULT_SPECIES_TO_FAMILY",
]
