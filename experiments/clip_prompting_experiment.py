import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

# Avoid numba cache issues when running via Poetry.
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")

# Ensure project root is on PYTHONPATH when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecogrow.benchmark.ecogrow_benchmark import EcogrowBenchmark
from ecogrow.models.open_clip_wrapper import init_open_clip, freeze_open_clip_backbone, DiseaseClipDetector
from ecogrow.models.checkpoint_cache import ensure_mobileclip_checkpoint
from ecogrow.preprocessing.image_segmentator import (
    black_bg_composite,
    crop_to_alpha_bbox,
    segment_plant_rgba,
)
from ecogrow.training.prompt_learners import ClipPromptLearner
from ecogrow.training.trainers import ClipPromptEngine
from ecogrow.data.plant_data import PlantData, make_segment_fn
from experiments.utils import compute_init_ctx, export_family_embeddings



@dataclass(frozen=True)
class Config:
    dataset_path: Path
    embeddings_dir: Optional[Path]
    prompts_config: Path
    run_id: str
    exp_dir: Path
    epochs: int
    batch_size: int
    temperature: float
    perc_eval: float
    lr: float


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="EcoGrow CLIP benchmark experiment")
    parser.add_argument(
        "--dataset-path",
        default="datasets",
        help="Percorso alla directory del dataset (es. data/Indoor-Plant-disease-dataset-1)",
    )
    parser.add_argument(
        "--embeddings-dir",
        default="experiment/clip_prompts",
        help="Directory in cui salvare gli embedding generati (opzionale)",
    )
    parser.add_argument(
        "--prompts-config",
        default="experiments/prompts_2.json",
        help="File JSON con la configurazione dei prompt",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Identificativo della run; se non specificato viene derivato dal nome del file di prompt.",
    )
    parser.add_argument(
        "--exp-dir",
        default="experiments",
        help="Directory principale dove salvare i risultati del benchmark.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Numero di epoche di fine-tuning per ciascuna famiglia.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size utilizzata durante il fine-tuning.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperatura da usare durante la similarità immagine-testo.",
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
        default=5e-3,
        help="Learning rate del prompt tuner (AdamW).",
    )
    args = parser.parse_args()

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

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"clip_{prompts_path.stem}"

    return Config(
        dataset_path=dataset_path,
        embeddings_dir=embeddings_dir,
        prompts_config=prompts_path,
        run_id=run_id,
        exp_dir=exp_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        perc_eval=max(0.0, float(args.perc_eval)),
        lr=args.lr,
    )


def main() -> Dict[str, Dict[str, object]]:
    config = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "MobileCLIP-S2"
    pretrained_tag = ensure_mobileclip_checkpoint(model_name=model_name)

    clip_model, preprocess, tokenizer, text_encoder = init_open_clip(
        model_name=model_name,
        pretrained_tag=pretrained_tag,
        device=device,
    )
    freeze_open_clip_backbone(clip_model)

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
    index_entries: List[Dict[str, object]] = []
    for i, (family_name, family_cfg) in enumerate(prompt_config.items(), start=1):
        print(f"[{i}/{num_families}] Family: {family_name}")

        class_prompts = [prompts[1] for _, prompts in family_cfg.items()]
        classnames = [disease for disease in family_cfg.keys()]

        ctx_init = compute_init_ctx(
            n_ctx=16,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            class_prompts=class_prompts,
        )

        prompt_learner = ClipPromptLearner(
            classnames=classnames,
            text_encoder=text_encoder,
            ctx_vectors=ctx_init,
            model_name=model_name,
        ).to(device)

        disease_detector = DiseaseClipDetector(
            classes=classnames,
            temperature=config.temperature,
            clip_model=clip_model,
            text_encoder=text_encoder,
            device=device,
            prompt_learner=prompt_learner,
            detector_id=family_name,
        )

        trainer = ClipPromptEngine(
            detector=disease_detector,
            prompt_learner=prompt_learner,
            device=device,
            preprocess=preprocess
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
        test_epoch = trainer.eval(test_loader)
        test_metrics = {
            "loss": test_epoch.loss,
            "f1": test_epoch.f1,
        }
        result["test_metrics"] = test_metrics

        if config.embeddings_dir is not None:
            family_file = config.embeddings_dir / f"{family_name}.pt"
            exported = export_family_embeddings(
                family_file,
                family_name,
                classnames,
                prompt_learner,
                text_encoder,
                temperature=family_detector.temperature,
            )
            index_entries.append(
                {
                    "family": family_name,
                    "classes": list(exported["classes"]),
                    "file": family_file.name,
                    "temperature": exported["temperature"],
                }
            )

        family_results[family_name] = result

        test_metrics = result["test_metrics"]
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
                "temperature": result["temperature"],
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

        test_f1s = [
            row["test_f1"] for row in summary_rows if row["test_f1"] is not None
        ]
        if test_f1s:
            avg_f1 = sum(test_f1s) / len(test_f1s)
            print(f"Average test F1: {avg_f1:.3f}")

    if config.embeddings_dir is not None and index_entries:
        index_payload = {
            "clip_model": model_name,
            "pretrained": pretrained_tag,
            "families": index_entries,
        }
        index_path = config.embeddings_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_payload, f, indent=2)
        print(f"Embeddings index saved to {index_path}")

    return family_results


if __name__ == "__main__":
    main()
