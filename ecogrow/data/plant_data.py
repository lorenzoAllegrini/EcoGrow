"""Dataset utilities for EcoGrow projects."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Set

from PIL import Image
from torch.utils.data import DataLoader, Dataset
# 🌿 SPECIE
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

# 🦠 MALATTIE / CONDIZIONI
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

from pathlib import Path
from typing import Set, Tuple
import shutil, random, os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def roboflow_format(
    init_root: str,
    final_root: str,
    default_species: Set[str] = DEFAULT_SPECIES,
    default_diseases: Set[str] = DEFAULT_DISEASES,
    *,
    overwrite: bool = False,
    test_ratio: float = 0.20,
    seed: int = 1337,
) -> None:
    """Convert a Roboflow folder export (con o senza split) nel layout EcoGrow.

    Supporta due tipologie di dataset di input:
    1) Con split: dentro `init_root` ci sono cartelle `train/valid/val/test` e,
       dentro ciascuna, cartelle di classe con nome combinato pianta+malattia
       (es. `Money_Plant_Bacterial_wilt_disease`) che contengono direttamente le immagini.
       Mapping: train+valid/val -> train ; test -> test.
    2) Senza split: due varianti gestite automaticamente:
       2a) Flat: cartelle di classe direttamente in root (nome combinato pianta+malattia)
           che contengono le immagini.
       2b) Annidato: cartelle per pianta in root, e dentro ciascuna cartelle per malattia
           (es. `Rose/Black_Spot`, `Rose/Healthy`), con le immagini all'interno.
           In questo caso viene creato uno split deterministico 80/20 per classe (configurabile).
    """

    src_root = Path(init_root).expanduser().resolve()
    dst_root = Path(final_root).expanduser().resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Sorgente '{src_root}' non trovata.")
    dst_root.mkdir(parents=True, exist_ok=True)

    random_gen = random.Random(seed)

    def resolve_family_and_disease(name: str) -> Tuple[str, str]:
        clean = name.replace("-", "_").replace(" ", "_").strip("_")
        lower = clean.lower()
        s = None
        d = None
        for specie in default_species:
            if specie in lower:
                s = specie
        for disease in default_diseases:
            if disease in lower:
                d = disease
        if not s:
            raise ValueError(f"Specie non supportata in '{lower}'")
        if not d:
            raise ValueError(f"Malattia/condizione non supportata in '{lower}'")
        return s, d

    def iter_images(dir_path: Path):
        for p in dir_path.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p

    def dir_has_images(dir_path: Path) -> bool:
        return any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in dir_path.iterdir()
        )

    def safe_copy(src: Path, dst: Path):
        """Copia gestendo collisioni di nome se overwrite=False."""
        target = dst
        if target.exists():
            if overwrite:
                target.unlink()
            else:
                stem, ext = target.stem, target.suffix
                k = 1
                while target.exists():
                    target = target.with_name(f"{stem}_{k}{ext}")
                    k += 1
        shutil.copy2(src, target)

    # Rileva se lo schema ha split o no
    split_names = {"train", "valid", "val", "test"}
    has_splits = any(
        d.is_dir() and d.name.lower() in split_names
        for d in src_root.iterdir()
    )

    if has_splits:
        # Schema con split alla Roboflow
        for split_dir in src_root.iterdir():
            if not split_dir.is_dir():
                continue
            name_l = split_dir.name.lower()
            if name_l not in split_names:
                # cartella extra: ignora
                continue
            target_split = "train" if name_l in {"train", "valid", "val"} else "test"

            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                specie_name, disease_name = resolve_family_and_disease(class_dir.name)
                dest_dir = dst_root / target_split / specie_name / disease_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                for img_path in iter_images(class_dir):
                    dest_path = dest_dir / img_path.name
                    safe_copy(img_path, dest_path)
    else:
        # Nessuno split: gestisce sia layout flat (classi direttamente in root)
        # sia layout annidato (pianta/ -> malattia/ -> immagini). In entrambi i casi
        # crea split 80/20 per classe in modo deterministico.

        for top_dir in src_root.iterdir():
            if not top_dir.is_dir():
                continue

            if dir_has_images(top_dir):
                # Layout flat: la cartella è già una classe combinata pianta+malattia
                specie_name, disease_name = resolve_family_and_disease(top_dir.name)
                imgs = list(iter_images(top_dir))
                imgs.sort(key=lambda p: p.name)
                random_gen.shuffle(imgs)

                n_total = len(imgs)
                n_test = int(round(n_total * test_ratio))
                test_set = set(imgs[:n_test])
                train_set = imgs[n_test:]

                for img_path in train_set:
                    dest_dir = dst_root / "train" / specie_name / disease_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / img_path.name
                    safe_copy(img_path, dest_path)

                for img_path in test_set:
                    dest_dir = dst_root / "test" / specie_name / disease_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / img_path.name
                    safe_copy(img_path, dest_path)
            else:
                # Possibile layout annidato: dentro `top_dir` ci sono sottocartelle per malattia
                for disease_dir in top_dir.iterdir():
                    if not disease_dir.is_dir():
                        continue
                    if not dir_has_images(disease_dir):
                        # Sottocartella non valida (ulteriore nesting o vuota): salta
                        continue
                    # Risolvi specie+malattia usando i nomi combinati plant+disease
                    combined_name = f"{top_dir.name}_{disease_dir.name}"
                    specie_name, disease_name = resolve_family_and_disease(combined_name)

                    imgs = list(iter_images(disease_dir))
                    imgs.sort(key=lambda p: p.name)
                    random_gen.shuffle(imgs)

                    n_total = len(imgs)
                    n_test = int(round(n_total * test_ratio))
                    test_set = set(imgs[:n_test])
                    train_set = imgs[n_test:]

                    for img_path in train_set:
                        dest_dir = dst_root / "train" / specie_name / disease_name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / img_path.name
                        safe_copy(img_path, dest_path)

                    for img_path in test_set:
                        dest_dir = dst_root / "test" / specie_name / disease_name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / img_path.name
                        safe_copy(img_path, dest_path)

__all__ = [
    "PlantData",
    "make_segment_fn",
    "roboflow_format",
    "DEFAULT_MAPPING",
    "DEFAULT_SPECIES_TO_FAMILY",
]
