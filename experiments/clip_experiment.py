import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, Namespace, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from roboflow import Roboflow
from torch.utils.data import DataLoader

from ecogrow.models.open_clip_wrapper import OpenClipWrapper, TextEncoderOpenCLIP
from ecogrow.preprocessing.image_segmentator import (
    black_bg_composite,
    crop_to_alpha_bbox,
    segment_plant_rgba,
)
from ecogrow.training.prompt_learners import PromptLearnerOpenCLIP
from ecogrow.training.trainers import PromptTuningTrainer
from ecogrow.data.plant_data import (
    DEFAULT_SPECIES_TO_FAMILY,
    PlantData,
    make_segment_fn,
)
from experiments.utils import build_prompt_learner


def export_family_embeddings(
    out_path: Path,
    family_name: str,
    class_order: Iterable[str],
    prompt_learner: PromptLearnerOpenCLIP,
    text_encoder: TextEncoderOpenCLIP,
    temperature: float,
) -> Dict:
    with torch.no_grad():
        prompts_embeds, tokenized_prompts = prompt_learner()
        text_features = text_encoder(prompts_embeds, tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)

    payload = {
        "family": family_name,
        "classes": list(class_order),
        "text_features": text_features.cpu(),
        "temperature": float(temperature),
    }
    torch.save(payload, out_path)
    return payload


@dataclass(frozen=True)
class Config:
    dataset_path: Path
    embeddings_dir: Optional[Path]
    prompts_config: Path


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="EcoGrow CLIP prompt tuning experiment")
    parser.add_argument(
        "--dataset-path",
        default="Indoor-Plant-disease-dataset-1",
        help="Percorso alla directory del dataset (es. data/Indoor-Plant-disease-dataset-1)",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=None,
        help="Directory in cui salvare gli embedding generati (opzionale)",
    )
    parser.add_argument(
        "--prompts-config",
        default="prompts.json",
        help="File JSON con la configurazione dei prompt",
    )
    args: Namespace = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path '{dataset_path}' non esiste o non è una directory.")

    embeddings_dir = None
    if args.embeddings_dir:
        embeddings_dir = Path(args.embeddings_dir).expanduser().resolve()
        embeddings_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = Path(args.prompts_config).expanduser().resolve()
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompts config '{prompts_path}' non esiste.")

    return Config(
        dataset_path=dataset_path,
        embeddings_dir=embeddings_dir,
        prompts_config=prompts_path,
    )


def main():
    config = _parse_args()
    dataset_path_str = str(config.dataset_path)
    embeddings_dir = config.embeddings_dir
    prompts_path = config.prompts_config

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model_name = "ViT-B-32"
    clip_wrapper = OpenClipWrapper(
        model_name=model_name,
        pretrained_tag=os.environ.get("ECOGROW_CLIP_PRETRAINED", "laion2b_s34b_b79k"),
        device=device_str,
    )

    segment_fn = make_segment_fn(
        segment_plant_rgba,
        crop_to_alpha_bbox,
        black_bg_composite,
        pad=12,
    )

    train_augmentations = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    ])

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompt_config = json.load(f)

    species_prompts = prompt_config.get("species", {})
    species_to_family = prompt_config.get("species_to_family", DEFAULT_SPECIES_TO_FAMILY)
    family_definitions = prompt_config.get("families", {})

    if not family_definitions:
        fallback: Dict[str, Dict[str, Iterable[str]]] = {}
        for species_name, spec_data in species_prompts.items():
            fam_name = species_to_family.get(species_name, species_name)
            entry = fallback.setdefault(fam_name, {"species": set(), "diseases": set()})
            entry["species"].add(species_name)
            entry["diseases"].update(spec_data.get("diseases", {}).keys())
        family_definitions = {
            fam: {
                "species": sorted(entry["species"]),
                "diseases": sorted(entry["diseases"]),
            }
            for fam, entry in fallback.items()
        }

    clip_wrapper.freeze_backbone()
    epochs = 10
    results: Dict[str, Dict[str, object]] = {}
    index_payload = {
        "families": {},
        "metadata": {"model": model_name, "pretrained": clip_wrapper.pretrained_tag},
    }

    train_transform = T.Compose([train_augmentations, clip_wrapper.preprocess])
    eval_transform = clip_wrapper.preprocess

    def build_dataset(
        family: str,
        split_candidates: Sequence[str],
        transform,
        *,
        shuffle: bool,
    ) -> Tuple[Optional[PlantData], Optional[DataLoader]]:
        for split_name in split_candidates:
            try:
                dataset = PlantData(
                    dataset_root=dataset_path_str,
                    family=family,
                    split=split_name,
                    transform=transform,
                    segment_fn=segment_fn,
                )
                loader = dataset.make_dataloader(
                    batch_size=16,
                    shuffle=shuffle,
                    num_workers=4,
                    pin_memory=True,
                )
                return dataset, loader
            except FileNotFoundError:
                continue
        return None, None

    for family_name, family_info in family_definitions.items():
        train_dataset, train_loader = build_dataset(
            family_name,
            ("train",),
            train_transform,
            shuffle=True,
        )
        if train_dataset is None or train_loader is None:
            print(f"[WARN] Split 'train' non trovato per la famiglia '{family_name}', salto.")
            continue

        val_dataset, val_loader = build_dataset(
            family_name,
            ("valid", "val", "validation"),
            eval_transform,
            shuffle=False,
        )
        test_dataset, test_loader = build_dataset(
            family_name,
            ("test",),
            eval_transform,
            shuffle=False,
        )

        class_order = train_dataset.classes
        family_diseases = set(family_info.get("diseases", []))
        diseases_lower = {d.lower(): d for d in family_diseases}
        undefined_conditions = [
            cls for cls in class_order if cls.lower() not in diseases_lower
        ]
        if undefined_conditions:
            raise KeyError(
                f"Family '{family_name}' is missing definitions for diseases: {undefined_conditions}"
            )
        prompts_per_class = []
        mapped_species = [species for species, fam in species_to_family.items() if fam == family_name]
        family_species = list(dict.fromkeys(family_info.get("species", []) + mapped_species))
        for cls in class_order:
            aggregated_prompts = []
            for species_name in family_species:
                species_diseases = species_prompts.get(species_name, {}).get("diseases", {})
                config_key = diseases_lower.get(cls.lower(), cls)
                prompts_for_class = (
                    species_diseases.get(config_key)
                    or species_diseases.get(cls)
                    or next(
                        (
                            prompts
                            for key, prompts in species_diseases.items()
                            if key.lower() == cls.lower()
                        ),
                        [],
                    )
                )
                aggregated_prompts.extend(prompts_for_class)
            if not aggregated_prompts:
                raise KeyError(
                    f"Missing textual prompts for class '{cls}' in family '{family_name}'."
                )
            prompts_per_class.append(aggregated_prompts)

        classnames = [c.replace("_", " ") for c in class_order]
        prompt_learner = build_prompt_learner(
            classnames=classnames,
            clip_wrapper=clip_wrapper,
            device=device,
            prompts_per_class=prompts_per_class,
            model_name=model_name,
            n_ctx=16,
            ctx_init=None,
            class_token_position="end",
            alpha_seed=0.3,
        )

        trainer = PromptTuningTrainer(
            clip_wrapper,
            device,
            prompt_learner,
            temperature=0.07,
        )

        optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=5e-3, weight_decay=0.0)
        history = trainer.fit(
            optimizer,
            train_loader,
            epochs=epochs,
            grad_clip=None,
            log_fn=lambda msg, family=family_name: print(f"[{family}] {msg}"),
        )

        plant_results = {
            "history": history,
            "label_map": train_dataset.label_maps(),
            "class_order": class_order,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset is not None else 0,
            "test_samples": len(test_dataset) if test_dataset is not None else 0,
        }
        if test_loader is not None and hasattr(trainer, "evaluate"):
            plant_results["test"] = trainer.evaluate(prompt_learner, test_loader)

        if embeddings_dir is not None:
            family_file = embeddings_dir / f"{family_name}.pt"
            export_family_embeddings(
                family_file,
                family_name,
                class_order,
                prompt_learner,
                clip_wrapper.text_encoder,
                temperature=trainer.temperature,
            )
            index_payload["families"][family_name] = {
                "file": family_file.name,
                "classes": class_order,
            }

        results[family_name] = plant_results

    for family_name, plant_results in results.items():
        if "test" in plant_results:
            test_metrics = plant_results["test"]
            print(
                f"[{family_name}] test loss {test_metrics.loss:.4f} "
                f"acc {test_metrics.accuracy:.3f}"
            )

    if embeddings_dir is not None and index_payload["families"]:
        index_path = embeddings_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_payload, f, indent=2)

    return results

if __name__ == "__main__":
    main()
