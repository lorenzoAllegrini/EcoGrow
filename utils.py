import os
from typing import List, Tuple, Dict, Optional, Union, Sequence, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from functools import partial
from collections import defaultdict
from torchvision.transforms import Compose



class PlantSamplesDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]], preprocess, segment_fn=None):
        """
        items: [(condition, img_path), ...]
        preprocess: trasformazione (es. CLIP preprocess)
        segment_fn: opzionale, PIL.Image -> PIL.Image (RGB)
        """
        self.items = items
        self.preprocess = preprocess
        self.segment_fn = segment_fn

        labels = sorted({cond for cond, _ in items})
        self.label2idx = {c: i for i, c in enumerate(labels)}
        self.idx2label = {i: c for c, i in self.label2idx.items()}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cond, img_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.segment_fn is not None:
            img = self.segment_fn(img)
        x = self.preprocess(img)
        y = self.label2idx[cond]
        return x, torch.tensor(y, dtype=torch.long)


# ---------- DataModule unico ----------
class PlantDataModule:
    """
    Costruisce DataLoader per split (train/valid/test), per pianta o combinati.

    Esempio:
        dm = PlantDataModule(dataset_path, preprocess, segment_fn=seg_fn, train_transforms=train_aug)
        loader = dm.get_train()                  # tutte le piante combinate
        loader_mp = dm.get_train("Money_Plant")  # solo Money_Plant
        idx2label = dm.get_label_map("Money_Plant", split="train")["idx2label"]
    """
    def __init__(self,
                 dataset_path: str,
                 preprocess,
                 segment_fn=None,
                 batch_size: int = 16,
                 shuffle_train: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 splits: Tuple[str, ...] = ("train", "valid", "test"),
                 train_transforms: Optional[Union[Sequence, Callable]] = None):
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.segment_fn = segment_fn
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.splits = splits
        self.train_transforms = train_transforms

        # Dizionari:
        #  - per split -> { plant_name: [(cond, path), ...] }
        #  - per split -> { plant_name: (loader, idx2label, label2idx) }
        self._grouped_index: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        self._loaders_per_plant: Dict[str, Dict[str, Tuple[DataLoader, Dict[int, str], Dict[str, int]]]] = {}
        self._combined_loaders: Dict[str, Tuple[DataLoader, Dict[int, str], Dict[str, int]]] = {}

        # Costruisci tutto
        self._build_all()

    # ---------- API pubblica ----------
    def get_train(self, plant_name: Optional[str] = None, shots: Optional[int] = None, per_class: Optional[bool] = True) -> DataLoader:
        return self._get_loader(split="train", plant_name=plant_name, shots=shots, per_class=per_class)

    def get_val(self, plant_name: Optional[str] = None, shots: Optional[int] = None) -> DataLoader:
        split = "valid" if "valid" in self._grouped_index else "val" if "val" in self._grouped_index else "test"
        return self._get_loader(split=split, plant_name=plant_name, shots=shots)

    def get_test(self, plant_name: Optional[str] = None) -> DataLoader:
        return self._get_loader(split="test", plant_name=plant_name)

    def get_label_map(self, plant_name: Optional[str], split: str = "train") -> Dict[str, Dict]:
        """
        Ritorna i mapping per una pianta o per il combinato dello split.
        """
        if plant_name is None:
            _, idx2label, label2idx = self._combined_loaders[split]
            return {"idx2label": idx2label, "label2idx": label2idx}
        else:
            loader, idx2label, label2idx = self._loaders_per_plant[split][plant_name]
            return {"idx2label": idx2label, "label2idx": label2idx}

    # ---------- Interni ----------
    def _build_all(self):
        for split in self.splits:
            if not os.path.isdir(os.path.join(self.dataset_path, split)):
                continue
            grouped_index = self._build_grouped_index(split)
            self._grouped_index[split] = grouped_index
            self._loaders_per_plant[split] = self._make_loaders_per_plant(grouped_index, split)
            self._combined_loaders[split] = self._make_combined_loader(self._loaders_per_plant[split], split)

    def _make_subset_loader(self, loader, shots: int, per_class: bool = True):
        """
        Crea un nuovo DataLoader che usa solo 'shots' esempi dal dataset originale.
        Se per_class=True, prende 'shots' esempi per ciascuna classe.
        """
        ds = loader.dataset
        bs = loader.batch_size
        nw = loader.num_workers
        pm = loader.pin_memory

        if shots is None or shots <= 0:
            return loader  # niente subset

        if not per_class:
            # primi 'shots' esempi nel dataset
            idx = list(range(min(shots, len(ds))))
        else:
            # k-shot per classe (richiede __getitem__ -> (x, y))
            buckets = defaultdict(list)
            idx = []
            for i in range(len(ds)):
                # estrai l'etichetta senza caricare tutte le trasformazioni pesanti
                x, y = ds[i]           # se è costoso, valuta un indice leggero in ds
                y_i = int(y)
                if len(buckets[y_i]) < shots:
                    buckets[y_i].append(i)
                    idx.append(i)
                # se tutte le classi raggiungono k, fermati
                # (stima del #classi: max label visto + 1; oppure quando nessuna cresce)
                if all(len(b) >= shots for b in buckets.values()):
                    break
            # opzionale: ordina per stabilità
            idx.sort()

        subset = Subset(ds, idx)
        small_loader = DataLoader(
            subset,
            batch_size=bs,
            shuffle=True,          # in debug: no shuffle; metti True se preferisci
            num_workers=nw,
            pin_memory=pm,
        )
        return small_loader

    def _get_loader(self, split: str, plant_name: Optional[str], shots: Optional[int] = None, per_class: bool = True):
        if split not in self._combined_loaders:
            raise ValueError(f"Split '{split}' not available in dataset located at {self.dataset_path}")

        if plant_name is None:
            loader, _, _ = self._combined_loaders[split]
            return self._make_subset_loader(loader, shots, per_class)

        per_plant = self._loaders_per_plant.get(split, {})
        if plant_name not in per_plant:
            available = sorted(per_plant.keys())
            raise ValueError(f"Plant '{plant_name}' not available for split '{split}'. Available plants: {available}")

        loader, _, _ = per_plant[plant_name]
        return self._make_subset_loader(loader, shots, per_class)

    def _build_grouped_index(self, split: str) -> Dict[str, List[Tuple[str, str]]]:
        split_dir = os.path.join(self.dataset_path, split)
        grouped: Dict[str, List[Tuple[str, str]]] = {}
        for folder in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            parts = folder.split("_")
            if len(parts) > 2:
                plant_name = "_".join(parts[:2])
                condition  = "_".join(parts[2:])
            else:
                plant_name = parts[0]
                condition  = parts[1] if len(parts) > 1 else "Unknown"

            for fname in os.listdir(folder_path):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                img_path = os.path.join(folder_path, fname)
                grouped.setdefault(plant_name, []).append((condition, img_path))
        return grouped

    def _make_loaders_per_plant(self, grouped_index: Dict[str, List[Tuple[str, str]]], split: str):
        out = {}
        # shuffle True solo sul train
        do_shuffle = self.shuffle_train if split == "train" else False
        for plant, items in grouped_index.items():
            preprocess = self._get_preprocess_for_split(split)
            ds = PlantSamplesDataset(items, preprocess, segment_fn=self.segment_fn)
            loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=do_shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            out[plant] = (loader, ds.idx2label, ds.label2idx)
        return out

    def _make_combined_loader(self, per_plant: Dict[str, Tuple[DataLoader, Dict[int, str], Dict[str, int]]], split: str):
        """
        Crea un DataLoader combinato con tutte le piante.
        Unifica i label space per split, preservando le classi per pianta senza collisioni
        (prefissa le label con il nome pianta).
        """
        # Costruisci un dataset combinato “on-the-fly”
        datasets = []
        global_labels = set()
        for plant, (loader, idx2label, label2idx) in per_plant.items():
            # Ricrea il dataset (prendiamo dagli loader i dataset originali)
            ds: PlantSamplesDataset = loader.dataset
            # Rimappa le etichette in uno spazio globale: "PlantName|Condition"
            items_relabel = []
            for cond, img_path in ds.items:
                gcond = f"{plant}|{cond}"
                items_relabel.append((gcond, img_path))
                global_labels.add(gcond)
            datasets.append(PlantSamplesDataset(items_relabel, ds.preprocess, segment_fn=ds.segment_fn))

        if not datasets:
            # dataset vuoto per quello split
            preprocess = self._get_preprocess_for_split(split)
            empty = PlantSamplesDataset([], preprocess, segment_fn=self.segment_fn)
            loader = DataLoader(empty, batch_size=self.batch_size)
            return loader, {}, {}

        combo = ConcatDataset(datasets)

        # Per costruire idx2label/label2idx globali, usiamo l’unione ordinata
        global_labels = sorted(global_labels)
        label2idx = {c: i for i, c in enumerate(global_labels)}
        idx2label = {i: c for c, i in label2idx.items()}

        # Wrapper per applicare preprocess/segment e mapping globale
        class _Wrapper(Dataset):
            def __init__(self, concat_ds, label2idx):
                self.concat_ds = concat_ds      # ConcatDataset di PlantSamplesDataset
                self.label2idx = label2idx
            def __len__(self):
                return len(self.concat_ds)
            def __getitem__(self, idx):
                # Estrae (x, y_local) ma ignora y_local: ricalcola globalmente
                # Recupera la coppia originale cond/path per rietichettare
                sub_idx = idx
                for ds in self.concat_ds.datasets:
                    if sub_idx < len(ds):
                        cond, img_path = ds.items[sub_idx]
                        # ds.items qui contiene (gcond, path)
                        break
                    sub_idx -= len(ds)
                # Ricrea tensore con lo stesso preprocess/segment
                img = Image.open(img_path).convert("RGB")
                if ds.segment_fn is not None:
                    img = ds.segment_fn(img)
                x = ds.preprocess(img)
                y = self.label2idx[cond]
                return x, torch.tensor(y, dtype=torch.long)

        wrapped = _Wrapper(combo, label2idx)
        loader = DataLoader(
            wrapped,
            batch_size=self.batch_size,
            shuffle=(self.shuffle_train if split == "train" else False),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader, idx2label, label2idx

    def _get_preprocess_for_split(self, split: str):
        if split == "train" and self.train_transforms is not None:
            extra = self.train_transforms
            if isinstance(extra, (list, tuple)):
                extra = Compose(list(extra))
            return Compose([extra, self.preprocess])
        return self.preprocess

def _segment_pipeline(
    img_rgb: Image.Image,
    segment_plant_rgba,
    crop_to_alpha_bbox,
    black_bg_composite,
    pad: int,
):
    rgba = segment_plant_rgba(img_rgb)
    rgba = crop_to_alpha_bbox(rgba, pad=pad)
    return black_bg_composite(rgba)

# ---------- (opzionale) builder per segment_fn con le tue util ----------
def make_segment_fn(segment_plant_rgba, crop_to_alpha_bbox, black_bg_composite, pad=12):
    return partial(
        _segment_pipeline,
        segment_plant_rgba=segment_plant_rgba,
        crop_to_alpha_bbox=crop_to_alpha_bbox,
        black_bg_composite=black_bg_composite,
        pad=pad,
    )