import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
import torch
import torch.nn.functional as F
from peft import LoraConfig, LoraModel
from experiments.utils import save_lora_adapter

os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from disease_detection.benchmark.ecogrow_benchmark import EcogrowBenchmark
from disease_detection.models.model_wrappers import init_open_clip, ClipClassifierDetector
from disease_detection.preprocessing.image_segmentator import (
    black_bg_composite,
    crop_to_alpha_bbox,
    segment_plant_rgba,
)
from disease_detection.training.trainers import ClipFineTuneEngine
from disease_detection.data.plant_data import make_segment_fn, DISEASE_MAPPING
from disease_detection.models.checkpoint_cache import ensure_mobileclip_checkpoint

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
    use_segmentation: bool
    num_workers: int
    num_splits: int
    segmenter: str


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="EcoGrow CLIP fine-tuning experiment")
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
        default="artifacts",
        help="Directory principale dove salvare i risultati (default: artifacts).",
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
        default=32,
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
        default=5e-4,
        help="Learning rate per l'ottimizzatore AdamW.",
    )
    parser.add_argument(
        "--classifier-dropout",
        type=float,
        default=0.1,
        help="Dropout applicato prima del classificatore lineare.",
    )
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disabilita la segmentazione della pianta per massimizzare la velocità.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Numero di worker per il DataLoader (0 = solo main process).",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=3,
        help="Numero di split di cross validation (usa 1 per disabilitare la CV).",
    )
    parser.add_argument(
        "--segmenter",
        choices=["current", "convnext"],
        default="current",
        help="Seleziona la pipeline di segmentazione da applicare (usa --no-segmentation per disabilitare).",
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

    run_id = args.run_id or f"clip_lora_finetuning_{prompts_path.stem}"

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
        use_segmentation=not bool(args.no_segmentation),
        num_workers=max(0, int(args.num_workers)),
        num_splits=max(1, int(args.num_splits)),
        segmenter=str(args.segmenter),
    )


def _resolve_segment_fn(
    use_segmentation: bool, segmenter: str
) -> Optional[Callable]:
    if not use_segmentation:
        return None
    if segmenter == "convnext":
        # Allinea alla pipeline usata nell'esperimento ConvNeXt.
        return make_segment_fn(
            segment_plant_rgba,
            crop_to_alpha_bbox,
            black_bg_composite,
            pad=12,
        )
    # Pipeline di segmentazione predefinita attuale.
    return make_segment_fn(
        segment_plant_rgba,
        crop_to_alpha_bbox,
        black_bg_composite,
        pad=12,
    )


def main() -> Dict[str, Dict[str, object]]:
    config = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "MobileCLIP-S1"
    pretrained_tag = ensure_mobileclip_checkpoint(model_name=model_name)
    clip_model, preprocess, tokenizer, text_encoder = init_open_clip(
        model_name=model_name,
        pretrained_tag=pretrained_tag,
        device=device,
    )

    candidate_targets = [
        "token_mixer.qkv",
        "token_mixer.proj",
    ]

    submodule_names = [name for name, _ in clip_model.visual.named_modules()]
    filtered_targets = [t for t in candidate_targets if any(t in n for n in submodule_names)]
    if not filtered_targets:
        fallback = ["attn.qkv", "attn.proj"]
        filtered_targets = [t for t in fallback if any(t in n for n in submodule_names)]

    print(f"[LoRA] target_modules candidates matched: {filtered_targets if filtered_targets else 'NONE'}")

    base_visual = clip_model.visual
    for p in base_visual.parameters():
        p.requires_grad_(False)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=filtered_targets if filtered_targets else candidate_targets,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    # Wrap the visual encoder with LoRA adapters
    clip_model.visual = LoraModel(base_visual, lora_config, adapter_name="default")

    # Sanity logs: adapters inserted and trainable parameters
    num_adapters = sum(1 for m in clip_model.visual.modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B"))
    num_lora_params = sum(p.numel() for n, p in clip_model.visual.named_parameters() if p.requires_grad and "lora_" in n)
    total_trainable_visual = sum(p.numel() for p in clip_model.visual.parameters() if p.requires_grad)
    print(f"[LoRA] adapters inserted: {num_adapters}")
    print(f"[LoRA] trainable LoRA params: {num_lora_params}")
    print(f"[LoRA] total trainable params in visual: {total_trainable_visual}")

    benchmark = EcogrowBenchmark(
        run_id=config.run_id,
        exp_dir=str(config.exp_dir),
        data_root=str(config.dataset_path),
    )

    segment_fn = _resolve_segment_fn(
        config.use_segmentation, config.segmenter
    )
    with open(config.prompts_config, "r", encoding="utf-8") as f:
        prompt_config = json.load(f)

    if not prompt_config:
        raise ValueError("Prompts config must define at least one entry.")

    clip_model.to(device)

    detector_label = "global"
    clip_detector = ClipClassifierDetector(
        classes=DISEASE_MAPPING.values(),
        clip_model=clip_model,
        preprocess=preprocess,
        device=device,
        feature_dropout=config.classifier_dropout,
        train_backbone=True,
        text_encoder=text_encoder,
        detector_id=detector_label,
    )
    trainer = ClipFineTuneEngine(
        clip_detector=clip_detector,
    )

    split_indices = (
        list(range(1, config.num_splits + 1)) if config.num_splits > 1 else None
    )

    fit_args = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "log_fn": lambda msg: print(f"[GLOBAL] {msg}"),
        "num_workers": config.num_workers,
        "patience_before_stopping": 1,
    }

    result = benchmark.run(
        trainer=trainer,
        segment_fn=segment_fn,
        perc_eval=None,
        fit_predictor_args=fit_args,
        split_indices=split_indices,
    )

    # Collect LoRA adapter weights for reuse (embedded in payload)
    trainable_param_names = {
        name for name, p in clip_model.visual.named_parameters() if p.requires_grad
    }
    full_state = clip_model.visual.state_dict()
    lora_state = {
        k: v.detach().cpu() for k, v in full_state.items() if k in trainable_param_names
    }
    try:
        lora_config_dict = lora_config.to_dict()  # type: ignore[assignment]
    except AttributeError:
        lora_config_dict = dict(lora_config.__dict__)

    with torch.no_grad():
        prompt_texts = [
            f"a close-up photo of a plant showing {cls.replace('_', ' ')}"
            for cls in clip_detector.classes
        ]
        tokens = tokenizer(prompt_texts).to(device)
        text_features = clip_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1).cpu()

    # Persist detector metadata for future inference reconstruction
    detector_dir = Path(benchmark.run_dir) / "detectors"
    detector_dir.mkdir(parents=True, exist_ok=True)
    detector_payload = {
        "version": 1,
        "detector_type": "clip_classifier",
        "detector_id": detector_label,
        "classes": list(clip_detector.classes),
        "temperature": float(clip_detector.temperature),
        "dropout": float(config.classifier_dropout),
        "model_name": model_name,
        "pretrained_tag": pretrained_tag,
        "lora_adapter": {
            "peft_type": "LORA",
            "config": lora_config_dict,
            "state_dict": lora_state,
        },
        "text_features": text_features,
        "model_state_dict": {
            k: v.detach().cpu() for k, v in clip_detector.classifier.state_dict().items()
        },
    }
    detector_path = detector_dir / f"{detector_label}.pt"
    torch.save(detector_payload, detector_path)
    print(f"[DETECTOR] metadata saved to {detector_path}")
    eval_metrics = result.get("eval_metrics")
    test_metrics = result.get("test_metrics")
    summary_row = {
        "detector": detector_label,
        "train_samples": result["train_samples"],
        "eval_samples": result["eval_samples"],
        "test_samples": result["test_samples"],
        "eval_loss": eval_metrics["loss"] if eval_metrics else None,
        "eval_f1": eval_metrics["f1"] if eval_metrics else None,
        "test_loss": test_metrics["loss"] if test_metrics else None,
        "test_f1": test_metrics["f1"] if test_metrics else None,
        "temperature": result.get("temperature"),
    }

    csv_path = Path(benchmark.run_dir) / "results.csv"
    fieldnames = list(summary_row.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)
    print(f"Results saved to {csv_path}")

    return {"global": result}


if __name__ == "__main__":
    main()
