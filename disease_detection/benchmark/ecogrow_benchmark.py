from __future__ import annotations

import copy
import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from disease_detection.data.plant_data import PlantData
from disease_detection.training.trainers import EpochMetrics


@dataclass
class FitParams:
    batch_size: int = 16
    epochs: int = 10
    optimizer: Any = None
    lr: float = 5e-3
    scheduler: Any = None
    grad_clip: Optional[float] = None
    log_fn: Callable[..., None] = logging.info
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    patience_before_stopping: Optional[int] = None
    min_delta: Optional[float] = None
    restore_best: bool = False


class EcogrowBenchmark:
    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        data_root: str = "datasets",
    ):
        """Initializes a new Eco benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
            data_root (str): The root directory of the NASA dataset.
        """
        self.run_id = run_id
        self.exp_dir = os.path.abspath(exp_dir)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.run_dir = os.path.join(self.exp_dir, self.run_id)
        self.data_root: str = data_root
        self.all_results: List[Dict[str, Any]] = []

    def load_data(
        self,
        *,
        transform=None,
        segment_fn=None,
        split_index: Optional[int] = None,
    ) -> Tuple[PlantData, PlantData]:
        """Load dataset splits for the global model."""

        train_transform = self._resolve_transform(transform, "train")
        val_transform = self._resolve_transform(transform, "val")

        train_data = PlantData(
            split="train",
            dataset_root=self.data_root,
            transform=train_transform,
            segment_fn=segment_fn,
            split_index=split_index,
        )

        val_data = PlantData(
            split="val",
            dataset_root=self.data_root,
            transform=val_transform,
            segment_fn=segment_fn,
            split_index=split_index,
        )

        return train_data, val_data

    @staticmethod
    def _resolve_transform(transform_spec, split: str):
        if transform_spec is None:
            return None
        if hasattr(transform_spec, "for_split"):
            return transform_spec.for_split(split)
        if isinstance(transform_spec, dict):
            if split in transform_spec:
                return transform_spec[split]
            if "default" in transform_spec:
                return transform_spec["default"]
            # fallback to first available transform
            return next(iter(transform_spec.values()))
        return transform_spec

    def _prepare_fit_params(
        self, fit_predictor_args: Optional[Dict[str, Any]]
    ) -> FitParams:
        fit_args = dict(fit_predictor_args or {})
        params = FitParams(
            batch_size=fit_args.pop("batch_size", FitParams.batch_size),
            epochs=fit_args.pop("epochs", FitParams.epochs),
            optimizer=fit_args.pop("optimizer", FitParams.optimizer),
            lr=fit_args.pop("lr", FitParams.lr),
            scheduler=fit_args.pop("scheduler", FitParams.scheduler),
            grad_clip=fit_args.pop("grad_clip", FitParams.grad_clip),
            log_fn=fit_args.pop("log_fn", FitParams.log_fn),
            num_workers=fit_args.pop("num_workers", FitParams.num_workers),
            pin_memory=fit_args.pop("pin_memory", FitParams.pin_memory),
            drop_last=fit_args.pop("drop_last", FitParams.drop_last),
            patience_before_stopping=fit_args.pop(
                "patience_before_stopping", FitParams.patience_before_stopping
            ),
            min_delta=fit_args.pop("min_delta", FitParams.min_delta),
            restore_best=fit_args.pop("restore_best", FitParams.restore_best),
        )

        if fit_args:
            unknown = ", ".join(sorted(fit_args.keys()))
            raise ValueError(f"Unsupported fit arguments: {unknown}")

        return params

    def run_validation(
        self,
        *,
        trainer,
        segment_fn: Any,
        perc_eval: Optional[float] = None,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
        split_index: Optional[int] = None,
        preprocess: Any = None
    ) -> Dict[str, Any]:
        """Run training/evaluation for the global detector.

        Args:
            trainer: pre-configured trainer compatible with the benchmark interface.
            segment_fn: preprocessing function applied before ``trainer.preprocess``.
            perc_eval: deprecated, kept for backward compatibility (ignored).
            fit_predictor_args: arguments controlling optimisation.
            split_index: optional index to pick specific train/val splits (e.g., train_2/val_2).

        Returns:
            Dict[str, Any]: collected metrics and bookkeeping for the run.
        """
        os.makedirs(self.run_dir, exist_ok=True)
        fit_params = self._prepare_fit_params(fit_predictor_args)

        train_data, val_data = self.load_data(
            transform=preprocess,
            segment_fn=segment_fn,
            split_index=split_index,
        )

        train_loader = train_data.make_dataloader(
            batch_size=fit_params.batch_size,
            weighted=True,
            num_workers=fit_params.num_workers,
            pin_memory=fit_params.pin_memory,
            drop_last=fit_params.drop_last,
        )

        eval_loader = (
            DataLoader(
                val_data,
                batch_size=fit_params.batch_size,
                shuffle=False,
            )
            if val_data is not None and len(val_data) > 0
            else None
        )

        optimizer = fit_params.optimizer
        if optimizer is None:
            optimizer = torch.optim.AdamW(trainer.parameters(), lr=fit_params.lr)

        logging.info(
            "Starting fine-tuning | epochs=%d batch_size=%d",
            fit_params.epochs,
            fit_params.batch_size,
        )

        history = trainer.fit(
            optimizer=optimizer,
            train_loader=train_loader,
            epochs=fit_params.epochs,
            scheduler=fit_params.scheduler,
            grad_clip=fit_params.grad_clip,
            log_fn=fit_params.log_fn,
            eval_loader=eval_loader,
            patience_before_stopping=fit_params.patience_before_stopping,
            min_delta=fit_params.min_delta,
            restore_best=fit_params.restore_best,
        )
        eval_metrics: Optional[EpochMetrics] = None
        if eval_loader is not None and hasattr(trainer, "eval"):
            eval_metrics = trainer.eval(eval_loader)

        results: Dict[str, Any] = {
            "train_history": history,
            "train_samples": len(train_loader.dataset),
            "eval_samples": len(val_data) if val_data is not None else 0,
            "temperature": getattr(trainer, "temperature", None),
            "detector_name": getattr(trainer, "detector_name", None),
        }
        if split_index is not None:
            results["split_index"] = split_index

        if eval_metrics is not None:
            results["eval_metrics"] = {
                "loss": eval_metrics.loss,
                "f1": eval_metrics.f1,
            }

        self.all_results.append(results)
        return results

    def _discover_crossval_split_indices(self) -> List[Optional[int]]:
        """Return available split indices; falls back to default train/val if none found."""
        train_indices: set[int] = set()
        val_indices: set[int] = set()
        if not os.path.isdir(self.data_root):
            return []

        with os.scandir(self.data_root) as entries:
            for entry in entries:
                if not entry.is_dir():
                    continue
                name = entry.name
                train_match = re.match(r"^train(?:_clean)?_(\d+)$", name)
                if train_match:
                    train_indices.add(int(train_match.group(1)))
                val_match = re.match(r"^val(?:_clean)?_(\d+)$", name)
                if val_match:
                    val_indices.add(int(val_match.group(1)))

        indexed = sorted(train_indices & val_indices)
        if indexed:
            return indexed

        # Fallback: if plain train/val (or train_clean/val_clean) exist, use a single split (None index).
        root = Path(self.data_root)
        has_clean = (root / "train_clean").is_dir() and (root / "val_clean").is_dir()
        has_plain = (root / "train").is_dir() and (root / "val").is_dir()
        if has_clean or has_plain:
            return [None]

        return []

    def run(
        self,
        *,
        trainer,
        segment_fn: Any,
        perc_eval: Optional[float] = None,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
        split_indices: Optional[Sequence[int]] = None,
        trainer_factory: Optional[Callable[[], Any]] = None,
    ) -> Dict[str, Any]:
        """Run the benchmark across multiple train/val splits created by ``format_roboflow.py``.

        Validation metrics are averaged across splits, then the model is retrained on
        train+val before a final evaluation on the test split.
        """

        indices = (
            list(split_indices)
            if split_indices is not None
            else self._discover_crossval_split_indices()
        )

        if not indices:
            raise RuntimeError(
                "No cross-validation splits found. "
                "Ensure train_<i>/val_<i> or train_clean_<i>/val_clean_<i> exist "
                "under the dataset root (run scripts/format_roboflow.py with 'splits' > 1)."
            )

        original_run_dir = self.run_dir
        split_results: Dict[Optional[int], Dict[str, Any]] = {}
        preprocess = trainer.preprocess

        for split_idx in indices:
            logging.info("Starting cross-validation split %s", split_idx)
            self.run_dir = os.path.join(original_run_dir, f"split_{split_idx}")

            current_trainer = (
                trainer_factory()
                if trainer_factory is not None
                else copy.deepcopy(trainer)
            )

            split_results[split_idx] = self.run_validation(
                trainer=current_trainer,
                segment_fn=segment_fn,
                perc_eval=perc_eval,
                fit_predictor_args=fit_predictor_args,
                split_index=split_idx,
                preprocess=preprocess
            )

            self.run_dir = original_run_dir

        eval_metrics = [
            res["eval_metrics"]
            for res in split_results.values()
            if "eval_metrics" in res
        ]
        averaged_eval: Optional[Dict[str, float]] = None
        if eval_metrics:
            averaged_eval = {
                key: sum(m[key] for m in eval_metrics) / len(eval_metrics)
                for key in eval_metrics[0].keys()
            }

        avg_eval_samples = (
            int(
                sum(res.get("eval_samples", 0) for res in split_results.values())
                / len(split_results)
            )
            if split_results
            else 0
        )

        self.run_dir = original_run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        fit_params = self._prepare_fit_params(fit_predictor_args)

        final_trainer = (
            trainer_factory()
            if trainer_factory is not None
            else copy.deepcopy(trainer)
        )

        final_preprocess = getattr(final_trainer, "preprocess", preprocess)
        train_data, val_data = self.load_data(
            transform=final_preprocess,
            segment_fn=segment_fn,
            split_index=indices[0],
        )

        full_train_data = copy.deepcopy(train_data)
        full_train_data.extend_with(val_data)

        train_loader = full_train_data.make_dataloader(
            batch_size=fit_params.batch_size,
            weighted=True,
            num_workers=fit_params.num_workers,
            pin_memory=fit_params.pin_memory,
            drop_last=fit_params.drop_last,
        )

        optimizer = fit_params.optimizer
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                final_trainer.parameters(), lr=fit_params.lr
            )

        logging.info(
            "Retraining on train+val before testing | epochs=%d batch_size=%d",
            fit_params.epochs,
            fit_params.batch_size,
        )

        train_history = final_trainer.fit(
            optimizer=optimizer,
            train_loader=train_loader,
            epochs=fit_params.epochs,
            scheduler=fit_params.scheduler,
            grad_clip=fit_params.grad_clip,
            log_fn=fit_params.log_fn,
            eval_loader=None,
            patience_before_stopping=fit_params.patience_before_stopping,
            min_delta=fit_params.min_delta,
            restore_best=fit_params.restore_best,
        )

        test_transform = self._resolve_transform(final_preprocess, "test")
        test_data = PlantData(
            split="test",
            dataset_root=self.data_root,
            transform=test_transform,
            segment_fn=segment_fn,
        )

        test_loader = (
            DataLoader(
                test_data,
                batch_size=fit_params.batch_size,
                shuffle=False,
            )
            if test_data is not None and len(test_data) > 0
            else None
        )

        test_metrics: Optional[EpochMetrics] = None
        if test_loader is not None and hasattr(final_trainer, "eval"):
            test_metrics = final_trainer.eval(test_loader)

        result: Dict[str, Any] = {
            "train_history": train_history,
            "train_samples": len(train_loader.dataset),
            "eval_samples": avg_eval_samples,
            "temperature": getattr(final_trainer, "temperature", None),
            "detector_name": getattr(final_trainer, "detector_name", None),
            "split_results": split_results,
            "test_samples": len(test_data),
        }

        if averaged_eval is not None:
            result["eval_metrics"] = averaged_eval

        if test_metrics is not None:
            result["test_metrics"] = {
                "loss": test_metrics.loss,
                "f1": test_metrics.f1,
            }

        self.all_results.append(result)
        return result
