"""Dataset utilities for EcoGrow projects."""

from __future__ import annotations
import torch
import shutil,random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Set
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
import math 

SPECIES_MAPPING = {
    "chrysanthemum": "chrysanthemum",
    "hibiscus": "hibiscus",
    "money_plant": "money_plant",
    "rose": "rose",
    "turmeric": "turmeric",
    "snake_plant": "snake_plant",
    "spider_plant": "spider_plant",
    "aloe": "aloe",
    "cactus": "cactus",
}

DISEASE_MAPPING = {
    "bacterial_leaf_spot": "bacterial_leaf_spot",
    "bacterial_wilt": "bacterial_wilt",

    "septoria_leaf_spot": "septoria_leaf_spot",

    "black_spot": "black_spot",

    "anthracnose": "anthracnose",

    "blotch": "necrotic_fungal_lesion",

    "leaf_spot": "generic_fungal_leaf_spot",
    "fungal_leaf_spot": "generic_fungal_leaf_spot",


    "blight": "blight",

    "rust": "rust",

    "necrosis": "foliar_necrosis",
    "leaf_necrosis": "foliar_necrosis",
    "leaf_tip_necrosis": "foliar_necrosis",

    "downy_mildew": "downy_mildew",
    "powdery_mildew": "powdery_mildew",

    "mosaic_virus": "mosaic_virus",
    "yellow_mosaic_virus": "yellow_mosaic_virus",

    "aphid_infestation": "aphids",  
    "dactylopius_opuntia": "cochineal",   
    "insect_damage": "generic_insect_damage",  

    "chlorosis": "chlorosis",                  
    "manganese_toxicity": "manganese_toxicity",  

    "scorch": "sun_scorch",
    "sunburn": "sun_scorch",

    "leaf_withering": "leaf_withering",

    "healthy": "healthy",
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
        self.original_classes_count: Dict[str, int] = {}
        self.class_counts: Dict[str, int] = {}
        self._prepare_samples()
 
        counts = torch.tensor(
            [self.class_counts[idx] for idx in sorted(self.class_counts.keys())],
            dtype=torch.float
        )

        freq = counts / counts.sum()     
        log_prior = torch.log(freq + 1e-8) 

        self.log_priors = log_prior     

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
        species_dirs = self._resolve_family_dirs()
        self.families = [name for name, _ in species_dirs]
        class_to_idx: Dict[str, int] = {}
        samples: List[Sample] = []

        for species_original_name, species_dir in species_dirs:
            species_name = self.resolve_name(species_original_name, SPECIES_MAPPING)
            for disease_dir in sorted(p for p in species_dir.iterdir() if p.is_dir()):
                disease_name = self.resolve_name(disease_dir.name, DISEASE_MAPPING)
                idx = class_to_idx.setdefault(disease_name, len(class_to_idx))

                valid_images = [
                    p for p in sorted(disease_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]

                key = f"{species_name}_{disease_name}"
                self.original_classes_count.setdefault(key, len(valid_images))
                self.class_counts[idx] = self.class_counts.get(idx, 0) + len(valid_images)
                for image_path in valid_images:
                    samples.append(
                        Sample(
                            path=image_path,
                            label=idx,
                            original_class_name=key,
                        )
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

    def resolve_name(self, original_name: str, mapping: Dict[str, str]) -> str:
        original_name = original_name.replace("-", "_").replace(" ", "_").lower()
        for name in mapping.keys():
            if name in original_name:
                return mapping[name]
        raise ValueError("name not in name_set")
    
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
            weights = [1.0/(self.original_classes_count[sample.original_class_name] ** gamma) for sample in self.samples]
            weights = torch.tensor(weights, dtype=torch.float)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
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
    "SPECIES_MAPPING",
    "DISEASES_MAPPING",
]
