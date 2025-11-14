import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecogrow.benchmark.ecogrow_benchmark import EcogrowBenchmark
from ecogrow.models.open_clip_wrapper import init_open_clip, FamilyAdaptedClipDetector
from ecogrow.models.mobile_clip import init_mobile_clip
from ecogrow.preprocessing.image_segmentator import (
    black_bg_composite,
    crop_to_alpha_bbox,
    segment_plant_rgba,
)
from ecogrow.training.trainers import ClipFineTuneEngine
from ecogrow.data.plant_data import PlantData, make_segment_fn


@dataclass(frozen=True)
class Config:
    dataset_path: Path
    prompts_config: Path
    run_id: str
    exp_dir: Path
    epochs: int
    batch_size: int
    perc_eval: float
    lr: float
    classifier_dropout: float


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="EcoGrow CLIP/Mobile fine-tuning experiment")
    parser.add_argument(
        "--backend",
        choices=["openclip", "mobile"],
        default="openclip",
        help="Seleziona il backend del modello immagine (openclip o mobile)",
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets",
        help="Percorso della directory del dataset (es. data/Indoor-Plant-disease-dataset-1)",
    )
    parser.add_argument(
        "--prompts-config",
        default="experiments/prompts.json",
        help="File JSON con la configurazione delle famiglie e relative classi",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Identificativo della run; se non specificato viene derivato dal file di prompt.",
    )
    parser.add_argument(
        "--exp-dir",
        default="experiments",
        help="Directory principale dove salvare i risultati.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Numero di epoche di fine-tuning per ciascuna famiglia.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size utilizzata durante il fine-tuning.",
    )
    parser.add_argument(
        "--perc-eval",
        type=float,
        default=0.2,
        help="Frazione del training da usare come validation (0 disabilita lo split).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate per l'ottimizzatore AdamW.",
    )
    parser.add_argument(
        "--classifier-dropout",
        type=float,
        default=0.1,
        help="Dropout applicato prima del classificatore lineare.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path '{dataset_path}' non esiste o non è una directory.")

    prompts_path = Path(args.prompts_config).expanduser().resolve()
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompts config '{prompts_path}' non esiste.")

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"clip_finetune_{prompts_path.stem}"

    return Config(
        dataset_path=dataset_path,
        prompts_config=prompts_path,
        run_id=run_id,
        exp_dir=exp_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        perc_eval=max(0.0, float(args.perc_eval)),
        lr=args.lr,
        classifier_dropout=max(0.0, float(args.classifier_dropout)),
    )


def _clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main() -> Dict[str, Dict[str, object]]:
    config = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = os.environ.get("ECOGROW_BACKEND", args.backend)
    text_encoder = None
    if backend == "openclip":
        model_name = "ViT-B-32"
        pretrained_tag = os.environ.get("ECOGROW_CLIP_PRETRAINED", "laion2b_s34b_b79k")
        clip_model, preprocess, _, text_encoder = init_open_clip(
            model_name=model_name,
            pretrained_tag=pretrained_tag,
            device=device,
        )
    else:
        # Mobile backend: only image encoder is used; fine-tune with linear head
        model_name = "mobilenet_v3_large"
        clip_model, preprocess, _, _ = init_mobile_clip(
            model_name=model_name,
            pretrained=True,
            device=device,
            embed_dim=512,
        )

    base_state = _clone_state_dict(clip_model)

    benchmark = EcogrowBenchmark(
        run_id=config.run_id,
        exp_dir=str(config.exp_dir),
        data_root=str(config.dataset_path),
    )

    segment_fn = make_segment_fn(
        segment_plant_rgba,
        crop_to_alpha_bbox,
        black_bg_composite,
        pad=12,
    )
    with open(config.prompts_config, "r", encoding="utf-8") as f:
        prompt_config = json.load(f)

    family_results: Dict[str, Dict[str, object]] = {}
    summary_rows: List[Dict[str, object]] = []

    num_families = len(prompt_config)
    for i, (family_name, family_cfg) in enumerate(prompt_config.items(), start=1):
        print(f"[{i}/{num_families}] Family: {family_name}")

        clip_model.load_state_dict(base_state, strict=True)
        clip_model.to(device)

        classnames = [disease for disease in family_cfg.keys()]
        family_detector = FamilyAdaptedClipDetector(
            name=family_name,
            classes=classnames,
            clip_model=clip_model,
            preprocess=preprocess,
            device=device,
            feature_dropout=config.classifier_dropout,
            text_encoder=text_encoder,
        )
        trainer = ClipFineTuneEngine(
            family_detector=family_detector,
        )

        fit_args = {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "log_fn": lambda msg, fam=family_name: print(f"[{fam}] {msg}"),
        }

        result = benchmark.run(
            family_id=family_name,
            trainer=trainer,
            segment_fn=segment_fn,
            perc_eval=config.perc_eval,
            fit_predictor_args=fit_args,
        )

        test_dataset = PlantData(
            dataset_root=config.dataset_path,
            family_id=family_name,
            splits=("test",),
            segment_fn=segment_fn,
            transform=preprocess,
        ).get_split("test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
        result["test_samples"] = len(test_dataset)

        test_loss = 0.0
        test_total = 0
        cm = None
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                logits = trainer.logits(xb)
            preds = logits.argmax(dim=-1)
            test_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            test_total += yb.size(0)
            C = logits.size(-1)
            idx = (yb.view(-1) * C + preds.view(-1)).to(torch.long).cpu()
            counts = torch.bincount(idx, minlength=C * C).view(C, C)
            if cm is None:
                cm = counts
            else:
                cm += counts

        if cm is None:
            test_f1 = 0.0
        else:
            tp = torch.diag(cm).to(torch.float32)
            fp = cm.sum(dim=0).to(torch.float32) - tp
            fn = cm.sum(dim=1).to(torch.float32) - tp
            prec = tp / torch.clamp(tp + fp, min=1.0)
            rec = tp / torch.clamp(tp + fn, min=1.0)
            f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-12)
            test_f1 = float(f1.mean().item())

        test_metrics = {
            "loss": test_loss / max(test_total, 1),
            "f1": test_f1,
        }
        result["test_metrics"] = test_metrics

        family_results[family_name] = result

        eval_metrics = result.get("eval_metrics")
        summary_rows.append(
            {
                "family_id": family_name,
                "train_samples": result["train_samples"],
                "eval_samples": result["eval_samples"],
                "test_samples": result["test_samples"],
                "eval_loss": eval_metrics["loss"] if eval_metrics else None,
                "eval_f1": eval_metrics["f1"] if eval_metrics else None,
                "test_loss": test_metrics["loss"],
                "test_f1": test_metrics["f1"],
                "temperature": result.get("temperature"),
            }
        )

    if summary_rows:
        csv_path = Path(benchmark.run_dir) / "results.csv"
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Results saved to {csv_path}")

        test_f1s = [row["test_f1"] for row in summary_rows if row["test_f1"] is not None]
        if test_f1s:
            avg_f1 = sum(test_f1s) / len(test_f1s)
            print(f"Average test F1: {avg_f1:.3f}")

    return family_results


if __name__ == "__main__":
    main()
