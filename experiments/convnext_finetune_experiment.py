import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecogrow.benchmark.ecogrow_benchmark import EcogrowBenchmark
from ecogrow.data.plant_data import PlantData, make_segment_fn
from ecogrow.models.open_clip_wrapper import ConvNextDetector
from ecogrow.preprocessing.image_segmentator import (
    black_bg_composite,
    crop_to_alpha_bbox,
    segment_plant_rgba,
)
from ecogrow.training.trainers import ConvNextFineTuneEngine


class SplitTransforms:
    def __init__(self, train_transform, eval_transform):
        self._train = train_transform
        self._eval = eval_transform

    def for_split(self, split: str):
        if split == "train":
            return self._train
        return self._eval


@dataclass(frozen=True)
class ConvNextConfig:
    dataset_path: Path
    exp_dir: Path
    run_id: str
    model_name: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    drop_rate: float
    train_backbone: bool
    use_pretrained: bool
    detector_name: str


def _parse_args() -> ConvNextConfig:
    parser = argparse.ArgumentParser(description="ConvNeXt fine-tuning experiment for EcoGrow.")
    parser.add_argument("--dataset-path", default="datasets", help="Path to the EcoGrow dataset root.")
    parser.add_argument(
        "--exp-dir",
        default="artifacts",
        help="Directory where run artifacts are stored (default: artifacts).",
    )
    parser.add_argument("--run-id", default=None, help="Identifier for this run; defaults to convnext_<model>.")
    parser.add_argument("--model-name", default="convnext_small", help="Name of the timm ConvNeXt variant.")
    parser.add_argument("--image-size", type=int, default=224, help="Input size for ConvNeXt transforms.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Drop rate applied inside the ConvNeXt head.")
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="If set, allows the ConvNeXt backbone to update; otherwise only the classifier head trains.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable loading ImageNet-pretrained weights for the ConvNeXt model.",
    )
    parser.add_argument(
        "--detector-name",
        default="global",
        help="Human-readable name for the detector, used in logs and outputs.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"convnext_{args.model_name}"

    return ConvNextConfig(
        dataset_path=dataset_path,
        exp_dir=exp_dir,
        run_id=run_id,
        model_name=args.model_name,
        image_size=max(32, int(args.image_size)),
        batch_size=max(1, int(args.batch_size)),
        epochs=max(1, int(args.epochs)),
        lr=float(args.lr),
        drop_rate=max(0.0, float(args.dropout)),
        train_backbone=bool(args.train_backbone),
        use_pretrained=not bool(args.no_pretrained),
        detector_name=args.detector_name,
    )


def _build_transforms(image_size: int) -> SplitTransforms:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return SplitTransforms(train_transform, eval_transform)


def run_convnext_experiment() -> Dict[str, Dict[str, object]]:
    config = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms_spec = _build_transforms(config.image_size)

    segment_fn = make_segment_fn(
        segment_plant_rgba,
        crop_to_alpha_bbox,
        black_bg_composite,
        pad=12,
    )

    preview_dataset = PlantData(
        dataset_root=config.dataset_path,
        split="train",
        transform=None,
        segment_fn=segment_fn,
    )
    classnames = preview_dataset.classes

    detector = ConvNextDetector(
        classes=classnames,
        pretrained=config.use_pretrained,
        device=device,
        preprocess=transforms_spec,
        train_backbone=config.train_backbone,
        drop_rate=config.drop_rate,
        detector_id=config.detector_name,
        model_name=config.model_name,
    )

    trainer = ConvNextFineTuneEngine(detector=detector)

    benchmark = EcogrowBenchmark(
        run_id=config.run_id,
        exp_dir=str(config.exp_dir),
        data_root=str(config.dataset_path),
    )

    fit_args = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "log_fn": lambda msg: print(f"[{config.detector_name}] {msg}"),
        "patience_before_stopping": 1,
    }

    result = benchmark.run(
        trainer=trainer,
        segment_fn=None,
        fit_predictor_args=fit_args,
    )

    run_dir = Path(benchmark.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    class_names_path = run_dir / "convnext_class_names.json"
    print(class_names_path)
    with open(class_names_path, "w", encoding="utf-8") as fp:
        json.dump(classnames, fp, indent=2)

    torch.save(detector.model.state_dict(), run_dir / f"{config.model_name}_final.pth")

    # Persist detector metadata for future inference reconstruction
    detector_dir = run_dir / "detectors"
    detector_dir.mkdir(parents=True, exist_ok=True)
    detector_payload = {
        "version": 1,
        "detector_type": "convnext",
        "name": config.detector_name,
        "classes": list(classnames),
        "model_name": config.model_name,
        "dropout": float(config.drop_rate),
        "model_state_dict": {
            k: v.detach().cpu() for k, v in detector.model.state_dict().items()
        },
    }
    detector_path = detector_dir / f"{config.detector_name}.pt"
    torch.save(detector_payload, detector_path)

    eval_metrics = result.get("eval_metrics")
    test_metrics = result.get("test_metrics")
    summary_row = {
        "detector": config.detector_name,
        "model_name": config.model_name,
        "train_samples": result["train_samples"],
        "eval_samples": result["eval_samples"],
        "test_samples": result["test_samples"],
        "eval_loss": eval_metrics["loss"] if eval_metrics else None,
        "eval_f1": eval_metrics["f1"] if eval_metrics else None,
        "test_loss": test_metrics["loss"] if test_metrics else None,
        "test_f1": test_metrics["f1"] if test_metrics else None,
    }

    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    return {"global": result}


if __name__ == "__main__":
    run_convnext_experiment()
