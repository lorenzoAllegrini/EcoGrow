"""Dataset utilities for EcoGrow projects."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

DEFAULT_SPECIES = {
    "chrysanthemum",
    "hibiscus",
    "money_plant",
    "rose",
    "turmeric",
    "snake_plant",
    "spider_plant",
    "aloe",
    "cactus",
}

DEFAULT_DISEASES = {
    "bacterial_leaf_spot",
    "septoria_leaf_spot",
    "blight",
    "necrosis",
    "scorch",
    "bacterial_wilt",
    "chlorosis",
    "manganese_toxicity",
    "black_spot",
    "downy_mildew",
    "mosaic_virus",
    "powdery_mildew",
    "rust",
    "yellow_mosaic_virus",
    "aphid_infestation",
    "blotch",
    "leaf_necrosis",
    "leaf_spot",
    "healthy",
    "anthracnose",
    "leaf_withering",
    "fungal_leaf_spot",
    "leaf_tip_necrosis",
    "sunburn",
    "insect_damage",
    "dactylopius_opuntia"
}

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
    original_class_name: str


def _slugify(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_").lower()

class PlantData(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        family_id: Optional[str] = None,
        *,
        families: Optional[Sequence[str]] = None,
        train: bool = True,
        split: Optional[str] = None,
        segment_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
        transform: Optional[Callable[[Image.Image], object]] = None,
        split_name: Optional[str] = None,
    ) -> None:
        
        self.root = Path(dataset_root).expanduser().resolve()
        self.segment_fn = segment_fn
        self.transform = transform

        if family_id is not None and families is not None:
            raise ValueError("Specificare solo 'family_id' oppure 'families', non entrambi.")

        if families is not None:
            if not families:
                raise ValueError("'families' non può essere vuoto.")
            self._family_filter = tuple(dict.fromkeys(families))
            # Manteniamo family_id per compatibilità con il vecchio attributo pubblico
            self.family_id = self._family_filter[0] if len(self._family_filter) == 1 else None
        elif family_id is not None:
            self._family_filter = (family_id,)
            self.family_id = family_id
        else:
            # Nessun filtro: usa tutte le famiglie disponibili nello split
            self._family_filter = None
            self.family_id = None

        if split is not None and split_name is not None:
            raise ValueError("Passa solo uno tra 'split' e 'split_name'.")

        if split is not None:
            resolved_split = split
        else:
            resolved_split = split_name or ("train" if train else "test")

        self.split = resolved_split
        self.families: List[str] = []
        self.original_class_counts: Dict[str, int] = {}
        self.class_counts: Dict[int, int] = {}
        self._prepare_samples()

        counts = torch.tensor(
            [self.class_counts[idx] for idx in range(len(self.class_counts))],
            dtype=torch.float32,
        )
        freq = counts / counts.sum()
        self.log_priors = torch.log(freq + 1e-8)

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

    def _prepare_samples(self) -> None:
        family_dirs = self._resolve_family_dirs()
        self.families = [name for name, _ in family_dirs]

        class_to_idx: Dict[str, int] = {}
        samples: List[Sample] = []
        multi_family = len(self.families) > 1

        for family_display_name, family_dir in family_dirs:
            family_slug = self.resolve_name(family_display_name, DEFAULT_SPECIES)
            for disease_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                disease_slug = self.resolve_name(disease_dir.name, DEFAULT_DISEASES)
                label_name = (
                    disease_dir.name
                    if not multi_family
                    else f"{family_display_name}/{disease_dir.name}"
                )
                idx = class_to_idx.setdefault(label_name, len(class_to_idx))

                valid_images = [
                    p
                    for p in sorted(disease_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]

                key = f"{family_slug}:{disease_slug}"
                self.original_class_counts[key] = (
                    self.original_class_counts.get(key, 0) + len(valid_images)
                )
                self.class_counts[idx] = self.class_counts.get(idx, 0) + len(valid_images)

                for image_path in valid_images:
                    samples.append(
                        Sample(path=image_path, label=idx, original_class_name=key)
                    )

        if not samples:
            target = (
                f"le famiglie {', '.join(self.families)}"
                if self.families
                else "tutte le famiglie"
            )
            raise RuntimeError(
                f"Nessuna immagine trovata per {target} nello split '{self.split}'."
            )

        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    def resolve_name(self, original_name: str, name_set: Set[str]) -> str:
        normalized = _slugify(original_name)
        for name in name_set:
            if name in normalized:
                return name
        return normalized
    
    def make_dataloader(
        self,
        batch_size: int,
        weighted: bool = True,
        shuffle: bool = True,
        *,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        gamma: float = 0.4
    ) -> DataLoader:
        """Create a DataLoader with sane defaults for vision datasets."""
        sampler = None
        if weighted:
            weights = []
            for sample in self.samples:
                count = self.original_class_counts.get(sample.original_class_name, 1)
                weights.append(1.0 / (count ** gamma))
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            sampler = WeightedRandomSampler(weights_tensor, len(weights_tensor), replacement=True)
            shuffle = False
        
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )


    def _resolve_family_dirs(self) -> List[Tuple[str, Path]]:
        split_dir = self.root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split '{self.split}' non presente in '{self.root}'."
            )

        lookup = {p.name.lower(): p for p in split_dir.iterdir() if p.is_dir()}
        if not lookup:
            raise FileNotFoundError(
                f"Nessuna famiglia trovata nello split '{self.split}'."
            )

        if self._family_filter is None:
            return sorted((path.name, path) for path in lookup.values())

        resolved: List[Tuple[str, Path]] = []
        for family_name in self._family_filter:
            key = family_name.lower()
            if key not in lookup:
                raise FileNotFoundError(
                    f"La famiglia '{family_name}' non esiste nello split '{self.split}'. "
                    f"Disponibili: {sorted(lookup)}"
                )
            path = lookup[key]
            resolved.append((path.name, path))
        return resolved

    @property
    def classes(self) -> List[str]:
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

    def label_maps(self) -> Dict[str, Dict[int, str] | Dict[str, int]]:
        return {
            "idx2label": dict(self.idx_to_class),
            "label2idx": dict(self.class_to_idx),
        }


__all__ = [
    "PlantData",
    "make_segment_fn",
    "DEFAULT_SPECIES",
    "DEFAULT_DISEASES",
]
