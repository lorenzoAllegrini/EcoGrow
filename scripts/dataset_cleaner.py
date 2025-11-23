#!/usr/bin/env python3
"""Utility script to clean EcoGrow datasets by removing duplicates and re-splitting data.

The workflow mirrors the script shared by the user:

1. Collects every image from the raw `train/`, `val/`, `test/` directories into a single
   `all/` directory while preserving the relative tree.
2. Removes perfect duplicates across the aggregated `all/` directory by hashing files.
3. Recreates clean train/val/test splits from the deduplicated pool, preserving class paths.
4. Verifies that no duplicate hashes leak across the new splits.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from math import ceil, isclose
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Any
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch
import torch.nn.functional as F

# riduci il numero di thread per limitare conflitti/lanci simultanei di OpenMP
torch.set_num_threads(1)
import faiss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

"""model = torch.hub.load(
    'facebookresearch/dinov2', 
    'dinov2_vits14'   # questa è la variante ViT-S/14
).to(device)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
    ),
])

index = faiss.IndexFlatIP(emb.shape[1])     # cosine similarity se normalizzati
index.add(emb)

k = 5
D, I = index.search(emb, k)  # D: similarità, I: indici
Ora hai:

I[i] = immagini più simili a immagine i

D[i] = quanto sono simili

"""
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class DatasetCleaner:
    def __init__(
        self,
        base_dir: str = "datasets",
        all_dir: str = "all",
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: str = "test",
        source_splits: Tuple[str, str, str] = ("train", "val", "test"),
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1, 
        emb_size: int = 384,
        threshold: float = 0.9,
        splits: int = 1,
        seed: int = 42,
    ) -> None:
        
        self.base_dir = base_dir = Path(str(PROJECT_ROOT / base_dir)).expanduser().resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory '{base_dir}' not found.")
        self.all_dir = Path(str(self.base_dir / all_dir)).expanduser().resolve()
        base_train = Path(str(self.base_dir / train_dir)).expanduser().resolve()
        base_val = Path(str(self.base_dir / val_dir)).expanduser().resolve()
        self.test_dir = Path(str(self.base_dir / test_dir)).expanduser().resolve()
        self.image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        self.source_splits = source_splits
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        if splits < 1:
            raise ValueError("'splits' must be >= 1.")
        self.splits = int(splits)
        self.seed = int(seed)
        self.emb_size = emb_size
        self.threshold = threshold

        self.train_dirs = self._expand_split_dirs(base_train, self.splits)
        self.val_dirs = self._expand_split_dirs(base_val, self.splits)
        # Backwards compatibility for single split references
        self.train_dir = self.train_dirs[0]
        self.val_dir = self.val_dirs[0]

        self.faiss_index = None

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load(
            'facebookresearch/dinov2', 
            'dinov2_vits14'   
        ).to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.collect_all_images_and_embed()
        self.remove_duplicates()
        self.original_image_groups = self.build_similarity_clusters()

        self.path_to_group = {}
        for gid, paths in self.original_image_groups.items():
            for p in paths:
                self.path_to_group[Path(p)] = gid

        

    def collect_all_images_and_embed(self) -> None:
        self.all_dir.mkdir(parents=True, exist_ok=True)
        print("\nCollecting all images into ALL/ preserving subfolders...")

        faiss_index = faiss.IndexFlatIP(self.emb_size)
        self.index_paths = []
        self.embeddings = []  

        lower_exts = tuple(ext.lower() for ext in self.image_extensions)
        for split in self.source_splits:
            split_path = self.base_dir / split
            if not split_path.exists():
                continue

            walker = tqdm(os.walk(split_path), desc=f"Walking {split}", unit="dir")
            for root, _, files in walker:
                root_path = Path(root)
                rel_root = root_path.relative_to(split_path)
                dst_root = self.all_dir / rel_root if rel_root != Path(".") else self.all_dir
                dst_root.mkdir(parents=True, exist_ok=True)

                for fname in files:
                    if not fname.lower().endswith(lower_exts):
                        continue
                    src_path = root_path / fname

                    img = Image.open(src_path).convert("RGB")
                    emb = self.embed_image(img)          # [1, emb_size]
                    emb_np = emb.detach().cpu().float().numpy()  # (1, D)
                    faiss_index.add(emb_np)

                    self.embeddings.append(emb_np) 

                    dst_path = dst_root / fname
                    DatasetCleaner._copy_with_suffix(src_path, dst_path)
                    self.index_paths.append(dst_path)

        # mappa path → indice nell'index
        self.embeddings = np.vstack(self.embeddings).astype("float32")  # (N, D) 
        self.path_to_index = {p: i for i, p in enumerate(self.index_paths)}
        self.faiss_index = faiss_index
        print("All images copied to ALL/ with preserved structure.\n")
    
    @staticmethod
    def _expand_split_dirs(base_dir: Path, splits: int) -> List[Path]:
        if splits <= 1:
            return [base_dir]
        return [
            base_dir.parent / f"{base_dir.name}_{idx+1}"
            for idx in range(splits)
        ]

    def build_similarity_clusters(self) -> Dict[int, List[Path]]:

        if self.faiss_index is None or not hasattr(self, "embeddings"):
            raise RuntimeError("FAISS index / embeddings non inizializzati")

        embeddings = self.embeddings  # (N, D)
        N = embeddings.shape[0]

        D, I = self.faiss_index.search(embeddings, N)  # D, I: (N, N)

        adj = [[] for _ in range(N)]

        for i in range(N):
            sims = D[i]
            idxs = I[i]
            for j, sim in zip(idxs, sims):
                if j == i:
                    continue  # salta se stesso
                if sim < self.threshold:
                    # I risultati sono ordinati per similarità decrescente:
                    # appena scendiamo sotto la soglia possiamo interrompere.
                    break
                adj[i].append(j)
                adj[j].append(i)  # grafo non diretto

        visited = [False] * N
        clusters: Dict[int, List[Path]] = {}
        cluster_id = 0

        for i in range(N):
            if visited[i]:
                continue
            # Avvia un nuovo cluster
            queue = [i]
            visited[i] = True
            current_paths: List[Path] = []

            while queue:
                u = queue.pop()
                current_paths.append(self.index_paths[u])
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)

            clusters[cluster_id] = current_paths
            cluster_id += 1

        return clusters


    def remove_duplicates(self) -> int:
        """Remove files with duplicate hashes across the aggregated ALL/ directory."""
        print("Removing exact duplicates (MD5) across ALL/ ...")
        lower_exts = tuple(ext.lower() for ext in self.image_extensions)
        hash_map: Dict[str, Path] = {}
        removed = 0

        for leaf in tqdm(self._iter_leaf_dirs(lower_exts), desc="Scanning leaf classes", unit="class"):
            class_path = self.all_dir / leaf
            leaf_label = "." if str(leaf) == "." else str(leaf)
            for fname in tqdm(sorted(os.listdir(class_path)), desc=f"Files in {leaf_label}", leave=False):
                if not fname.lower().endswith(lower_exts):
                    continue
                fpath = class_path / fname
                if not fpath.is_file():
                    continue
                try:
                    digest = self.file_hash(fpath)
                except Exception as exc:  # pragma: no cover - informational
                    print(f"Warning: cannot hash {fpath}: {exc}")
                    continue

                if digest in hash_map:
                    fpath.unlink()
                    removed += 1
                else:
                    hash_map[digest] = fpath

        print(f"Duplicate removal complete. Removed: {removed} files.\n")
        return removed
    

    def create_splits(self) -> None:
        """Create clean train/val/test splits from the deduplicated ALL directory,
            leakage safe
        """
        print(
            f"Creating clean train/val/test splits (cluster-safe) with {self.splits} "
            f"train/val split(s)..."
        )

        for target in list(self.train_dirs) + list(self.val_dirs):
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        lower_exts = tuple(ext.lower() for ext in self.image_extensions)

        path_to_group: Dict[Path, int] = {}
        for gid, paths in self.original_image_groups.items():
            for p in paths:
                path_to_group[Path(p)] = gid

        for leaf in tqdm(self._iter_leaf_dirs(), desc="Splitting leaf classes", unit="class"):
            class_src = self.all_dir / leaf
            images = [
                class_src / fname
                for fname in os.listdir(class_src)
                if fname.lower().endswith(lower_exts) and (class_src / fname).is_file()
            ]
            if not images:
                continue

            group_to_images: Dict[int, List[Path]] = {}
            next_fake_gid = -1 
            for img in images:
                gid = path_to_group.get(img)
                if gid is None:
                    gid = next_fake_gid
                    next_fake_gid -= 1
                group_to_images.setdefault(gid, []).append(img)

            group_ids = list(group_to_images.keys())
            per_split_assignments: List[Tuple[List[int], List[int]]] = []

            if len(group_ids) < 3 or (self.val_ratio + self.test_ratio) == 0:
                test_groups: List[int] = []
                per_split_assignments = [(group_ids, []) for _ in range(self.splits)]
            else:
                if self.test_ratio > 0:
                    trainval_groups, test_groups = self._simple_train_test_split(
                        group_ids,
                        test_size=self.test_ratio,
                        seed=self.seed,
                    )
                else:
                    trainval_groups, test_groups = group_ids, []

                total_tv = self.train_ratio + self.val_ratio
                if total_tv <= 0 or not trainval_groups:
                    per_split_assignments = [(trainval_groups, []) for _ in range(self.splits)]
                else:
                    val_fraction = self.val_ratio / total_tv if total_tv > 0 else 0.0
                    for split_idx in range(self.splits):
                        current_seed = self.seed + split_idx + 1
                        if val_fraction <= 0 or len(trainval_groups) < 2:
                            train_groups = trainval_groups
                            val_groups = []
                        elif val_fraction >= 1.0:
                            train_groups = []
                            val_groups = trainval_groups
                        else:
                            train_groups, val_groups = self._simple_train_test_split(
                                trainval_groups,
                                test_size=val_fraction,
                                seed=current_seed,
                            )
                        per_split_assignments.append((train_groups, val_groups))

            for split_idx, (train_root, val_root) in enumerate(zip(self.train_dirs, self.val_dirs)):
                train_groups, val_groups = per_split_assignments[split_idx]
                train_imgs = [img for gid in train_groups for img in group_to_images[gid]]
                val_imgs = [img for gid in val_groups for img in group_to_images[gid]]
                self._copy_images(train_imgs, train_root, self.all_dir)
                self._copy_images(val_imgs, val_root, self.all_dir)

            test_imgs = [img for gid in test_groups for img in group_to_images[gid]]
            self._copy_images(test_imgs, self.test_dir, self.all_dir)

        print("Splits created successfully (cluster-safe)!\n")


    def file_hash(self, path: Path) -> str:
        """Return the MD5 hash of a file, reading it in chunks."""
        hasher = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        img = self.transform(image).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        with torch.no_grad():
            emb = self.model(img)          # [1, emb_size]
            emb = F.normalize(emb, dim=-1) # normalizzato per cosine
        return emb
    
    def _iter_leaf_dirs(self, lower_exts: Tuple[str, ...] | None = None) -> List[Path]:
        """Return every directory relative to all_dir that contains at least one image."""
        lower_exts = lower_exts or tuple(ext.lower() for ext in self.image_extensions)
        leaf_dirs: List[Path] = []
        for root, _, files in os.walk(self.all_dir):
            if any(fname.lower().endswith(lower_exts) for fname in files):
                rel_root = Path(root).relative_to(self.all_dir)
                leaf_dirs.append(rel_root)
        return sorted(leaf_dirs, key=lambda p: str(p))

    @staticmethod
    def _simple_train_test_split(
        items: Sequence[Path],
        *,
        test_size: float,
        seed: int = 42,
    ) -> Tuple[List[Path], List[Path]]:
        """Minimal train/test splitter to avoid depending on scikit-learn."""
        if not 0 < test_size <= 1:
            raise ValueError("test_size must be within (0, 1].")

        samples = list(items)
        if not samples:
            return [], []

        rng = random.Random(seed)
        rng.shuffle(samples)
        n_samples = len(samples)
        n_test = max(1, min(n_samples, ceil(n_samples * test_size)))
        test_subset = samples[:n_test]
        train_subset = samples[n_test:]
        return train_subset, test_subset

    @staticmethod
    def _copy_with_suffix(src_path: Path, dst_path: Path) -> None:
        """Copy file to `dst_path`, appending suffixes when collisions occur."""
        target = dst_path
        if target.exists():
            stem, ext = target.stem, target.suffix
            count = 1
            while target.exists():
                target = target.with_name(f"{stem}_{count}{ext}")
                count += 1
        shutil.copy2(src_path, target)

    def _copy_images(self, images: Iterable[Path], target_root: Path, all_root: Path) -> None:
        """Copy images to the target split while preserving their relative subfolders."""
        for img in images:
            rel_subdir = img.parent.relative_to(all_root)
            dst_dir = target_root / rel_subdir
            dst_dir.mkdir(parents=True, exist_ok=True)
            DatasetCleaner._copy_with_suffix(img, dst_dir / img.name)


    def verify_no_leakage(self) -> bool:
        """Ensure that no duplicate hashes exist across the generated split directories."""
        print("Final check: no duplicates across splits...")
        lower_exts = tuple(ext.lower() for ext in self.image_extensions)
        seen: Dict[str, Path] = {}

        split_roots: Dict[str, Path] = {}
        for idx, split_dir in enumerate(self.train_dirs, start=1):
            split_name = "train" if self.splits == 1 else f"train_{idx}"
            split_roots[split_name] = split_dir
        for idx, split_dir in enumerate(self.val_dirs, start=1):
            split_name = "val" if self.splits == 1 else f"val_{idx}"
            split_roots[split_name] = split_dir
        split_roots["test"] = self.test_dir

        for split_name, split_root in split_roots.items():
            if not split_root.exists():
                continue
            for root, _, files in os.walk(split_root):
                for fname in files:
                    if not fname.lower().endswith(lower_exts):
                        continue
                    fpath = Path(root) / fname
                    try:
                        digest = self.file_hash(fpath)
                    except Exception as exc:  # pragma: no cover - informational
                        print(f"Warning hashing {fpath}: {exc}")
                        continue
                    if digest in seen:
                        print("LEAKAGE FOUND:")
                        print(f"- {seen[digest]}")
                        print(f"- {fpath}")
                        return False
                    seen[digest] = fpath

        print("No duplicates across datasets. Dataset is clean!\n")
        return True



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format and deduplicate EcoGrow datasets exported from Roboflow."
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT / "datasets"),
        help="Root directory containing the original train/val/test folders.",
    )
    parser.add_argument(
        "--all-dir",
        default=None,
        help="Directory that will aggregate all images before deduplication (default: <base>/all).",
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Destination directory for the cleaned train split (default: <base>/train_clean).",
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        help="Destination directory for the cleaned val split (default: <base>/val_clean).",
    )
    parser.add_argument(
        "--test-dir",
        default= None,
        help="Destination directory for the cleaned test split (default: <base>/test_clean).",
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Input split folders to scan inside base_dir (default: train val test).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Portion of samples per class for the train split when recreating splits.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Portion of samples per class for the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Portion of samples per class for the test split.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=3,
        help="Number of train/val splits to create (>=1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic splitting.",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=DEFAULT_IMAGE_EXTENSIONS,
        help="Image file extensions to consider. Defaults to common formats.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_cleaner = DatasetCleaner(
        base_dir=args.base_dir,
        all_dir=args.all_dir or "all",
        train_dir=args.train_dir or "train_clean",
        val_dir=args.val_dir or "val_clean",
        test_dir=args.test_dir or "test_clean",
        source_splits=tuple(args.source_splits),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        splits=args.splits,
        seed=args.seed,
        threshold=0.9,  # usa la soglia predefinita o passa via CLI se la aggiungi
    )
    dataset_cleaner.create_splits()
    dataset_cleaner.verify_no_leakage()


if __name__ == "__main__":
    main()
