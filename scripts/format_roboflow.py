#!/usr/bin/env python3
"""Utility script to reshape a Roboflow export into the EcoGrow layout.

Adds creation of a validation split (default 0.2) in addition to train/test.
Supports Roboflow exports both with and without existing splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Set, Tuple
import shutil
import random



PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from ecogrow.data.plant_data import (  # noqa: E402
    DEFAULT_SPECIES,
    DEFAULT_DISEASES,
)

# Local image extensions for file detection during formatting
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def roboflow_format(
    init_root: str,
    final_root: str,
    default_species: Set[str] = DEFAULT_SPECIES,
    default_diseases: Set[str] = DEFAULT_DISEASES,
    *,
    overwrite: bool = False,
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
    seed: int = 1337,
) -> None:
    """Convert a Roboflow folder export (con o senza split) nel layout EcoGrow.

    Crea sempre tre split in output: `train/`, `val/`, `test/`.

    - Con split in input (cartelle train/valid|val/test):
      - Mappa `train` -> train, `valid|val` -> val, `test` -> test.
      - Se `valid/val` mancano, ricava `val` da `train` con `val_ratio` per classe.
    - Senza split in input (flat o annidato):
      - Effettua split per classe: prima `test` con `test_ratio`, poi `val` con `val_ratio` dal rimanente.

    Args:
        init_root: percorso sorgente (Roboflow export).
        final_root: cartella destinazione formattata per EcoGrow.
        default_species: insiemi di specie riconosciute nel nome classe.
        default_diseases: insiemi di malattie/condizioni riconosciute nel nome classe.
        overwrite: se True sovrascrive file con lo stesso nome in destinazione.
        test_ratio: porzione destinata a test quando si genera lo split.
        val_ratio: porzione del rimanente destinata a validation quando si genera lo split.
        seed: seme per randomizzazione deterministica.
    """

    src_root = Path(init_root).expanduser().resolve()
    dst_root = Path(final_root).expanduser().resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Sorgente '{src_root}' non trovata.")
    dst_root.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(seed)

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

    split_names = {"train", "valid", "val", "test"}
    has_splits = any(d.is_dir() and d.name.lower() in split_names for d in src_root.iterdir())

    if has_splits:
        has_val_folder = any(
            d.is_dir() and d.name.lower() in {"valid", "val"} for d in src_root.iterdir()
        )

        # 1) Copy train/val/test if present, keeping val separate
        for split_dir in src_root.iterdir():
            if not split_dir.is_dir():
                continue
            name_l = split_dir.name.lower()
            if name_l not in split_names:
                continue
            if name_l in {"valid", "val"}:
                target_split = "val"
            else:
                target_split = name_l  # 'train' or 'test'

            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                specie_name, disease_name = resolve_family_and_disease(class_dir.name)
                dest_dir = dst_root / target_split / specie_name / disease_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img_path in iter_images(class_dir):
                    safe_copy(img_path, dest_dir / img_path.name)

        # 2) If no explicit val in input, split part of train into val per-class
        if not has_val_folder:
            # Build a per-class list of files under output train, then move a portion to val
            train_root = dst_root / "train"
            if train_root.is_dir():
                for specie_dir in train_root.iterdir():
                    if not specie_dir.is_dir():
                        continue
                    for disease_dir in specie_dir.iterdir():
                        if not disease_dir.is_dir():
                            continue
                        imgs = [p for p in disease_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                        imgs.sort(key=lambda p: p.name)
                        rnd.shuffle(imgs)
                        n_total = len(imgs)
                        n_val = int(round(n_total * val_ratio))
                        val_subset = imgs[:n_val]
                        val_dest = dst_root / "val" / specie_dir.name / disease_dir.name
                        val_dest.mkdir(parents=True, exist_ok=True)
                        for p in val_subset:
                            safe_copy(p, val_dest / p.name)
                            if not overwrite:
                                # Keep original under train if not overwriting; if overwriting, move semantics
                                continue
                            # If overwriting, remove from train to avoid duplication
                            try:
                                p.unlink()
                            except Exception:
                                pass
    else:
        # No input split: create train/val/test by splitting per-class
        for top_dir in src_root.iterdir():
            if not top_dir.is_dir():
                continue

            if dir_has_images(top_dir):
                specie_name, disease_name = resolve_family_and_disease(top_dir.name)
                imgs = list(iter_images(top_dir))
                imgs.sort(key=lambda p: p.name)
                rnd.shuffle(imgs)

                n_total = len(imgs)
                n_test = int(round(n_total * test_ratio))
                rem = n_total - n_test
                n_val = int(round(rem * val_ratio))

                test_set = set(imgs[:n_test])
                val_set = set(imgs[n_test : n_test + n_val])
                train_set = imgs[n_test + n_val :]

                for img_path in train_set:
                    dest_dir = dst_root / "train" / specie_name / disease_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    safe_copy(img_path, dest_dir / img_path.name)
                for img_path in val_set:
                    dest_dir = dst_root / "val" / specie_name / disease_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    safe_copy(img_path, dest_dir / img_path.name)
                for img_path in test_set:
                    dest_dir = dst_root / "test" / specie_name / disease_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    safe_copy(img_path, dest_dir / img_path.name)
            else:
                # Nested layout: species/ -> disease/ -> images
                for disease_dir in top_dir.iterdir():
                    if not disease_dir.is_dir():
                        continue
                    if not dir_has_images(disease_dir):
                        continue
                    combined_name = f"{top_dir.name}_{disease_dir.name}"
                    specie_name, disease_name = resolve_family_and_disease(combined_name)

                    imgs = list(iter_images(disease_dir))
                    imgs.sort(key=lambda p: p.name)
                    rnd.shuffle(imgs)

                    n_total = len(imgs)
                    n_test = int(round(n_total * test_ratio))
                    rem = n_total - n_test
                    n_val = int(round(rem * val_ratio))

                    test_set = set(imgs[:n_test])
                    val_set = set(imgs[n_test : n_test + n_val])
                    train_set = imgs[n_test + n_val :]

                    for img_path in train_set:
                        dest_dir = dst_root / "train" / specie_name / disease_name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        safe_copy(img_path, dest_dir / img_path.name)
                    for img_path in val_set:
                        dest_dir = dst_root / "val" / specie_name / disease_name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        safe_copy(img_path, dest_dir / img_path.name)
                    for img_path in test_set:
                        dest_dir = dst_root / "test" / specie_name / disease_name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        safe_copy(img_path, dest_dir / img_path.name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formatta un export Roboflow nella struttura dataset di EcoGrow."
    )
    parser.add_argument(
        "--source",
        default=str("Cactus_dataset"),
        help="Percorso della directory Roboflow da cui leggere i dati (default: ./roboflow_data).",
    )
    parser.add_argument(
        "--dest",
        default=str(PROJECT_ROOT / "datasets"),
        help="Percorso della directory di destinazione (default: ./datasets).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive eventuali file esistenti nella destinazione.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Quota per il test split quando generato (default: 0.20).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Quota per il validation split ricavato dal rimanente (default: 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seme per la randomizzazione deterministica.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    roboflow_format(
        init_root=args.source,
        final_root=args.dest,
        overwrite=args.overwrite,
        test_ratio=float(args.test_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    print(f"Dataset formattato in '{args.dest}'.")


if __name__ == "__main__":
    main()
