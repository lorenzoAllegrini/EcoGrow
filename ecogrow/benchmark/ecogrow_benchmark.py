from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from ecogrow.data.plant_data import PlantData
from ecogrow.training.trainers import EpochMetrics

class EcogrowBenchmark:
    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        data_root: str = "data/Indoor-Plant-disease-dataset-1",
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
    ) -> Tuple[PlantData, PlantData, Optional[PlantData]]:
        """Load dataset splits for the global model."""

        train_transform = self._resolve_transform(transform, "train")
        val_transform = self._resolve_transform(transform, "val")
        test_transform = self._resolve_transform(transform, "test")

        train_data = PlantData(
            split="train",
            dataset_root=self.data_root,
            transform=train_transform,
            segment_fn=segment_fn,
        )

        val_data = PlantData(
            split="val",
            dataset_root=self.data_root,
            transform=val_transform,
            segment_fn=segment_fn,
        )

        try:
            test_data = PlantData(
                split="test",
                dataset_root=self.data_root,
                transform=test_transform,
                segment_fn=segment_fn,
            )
        except FileNotFoundError as exc:
            logging.warning("Test split unavailable: %s", exc)
            test_data = None

        return train_data, val_data, test_data

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
    
    def run(
        self,
        *,
        trainer,
        segment_fn: Any,
        perc_eval: Optional[float] = None,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run training/evaluation for the global detector.

        Args:
            trainer: pre-configured trainer compatible with the benchmark interface.
            segment_fn: preprocessing function applied before ``trainer.preprocess``.
            perc_eval: deprecated, kept for backward compatibility (ignored).
            fit_predictor_args: arguments controlling optimisation.

        Returns:
            Dict[str, Any]: collected metrics and bookkeeping for the run.
        """
        os.makedirs(self.run_dir, exist_ok=True)
        fit_args = dict(fit_predictor_args or {})

        preprocess = trainer.preprocess

        batch_size = fit_args.pop("batch_size", 16)
        epochs = fit_args.pop("epochs", 10)
        optimizer = fit_args.pop("optimizer", None)
        lr = fit_args.pop("lr", 5e-3)
        scheduler = fit_args.pop("scheduler", None)
        grad_clip = fit_args.pop("grad_clip", None)
        log_fn = fit_args.pop("log_fn", logging.info)
        num_workers = fit_args.pop("num_workers", 0)
        pin_memory = fit_args.pop("pin_memory", False)
        drop_last = fit_args.pop("drop_last", False)
        patience_before_stopping = fit_args.pop("patience_before_stopping", None)
        min_delta = fit_args.pop("min_delta", None)
        restore_best = fit_args.pop("restore_best", False)

        if fit_args:
            unknown = ", ".join(sorted(fit_args.keys()))
            raise ValueError(f"Unsupported fit arguments: {unknown}")

        if perc_eval not in (None, 0):
            logging.info(
                "'perc_eval' is deprecated and ignored; using the explicit validation split."
            )

        train_data, val_data, test_data = self.load_data(
            transform=preprocess,
            segment_fn=segment_fn,
        )

        train_loader = train_data.make_dataloader(
            batch_size=batch_size,
            weighted=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        
        eval_loader = (
            DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False,
            )
            if val_data is not None and len(val_data) > 0
            else None
        )

        test_loader = (
            DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
            )
            if test_data is not None and len(test_data) > 0
            else None
        )

        if optimizer is None:
            optimizer = torch.optim.AdamW(trainer.parameters(), lr=lr)

        logging.info(
            "Starting fine-tuning | epochs=%d batch_size=%d",
            epochs,
            batch_size,
        )

        history = trainer.fit(
            optimizer=optimizer,
            train_loader=train_loader,
            epochs=epochs,
            scheduler=scheduler,
            grad_clip=grad_clip,
            log_fn=log_fn,
            eval_loader=eval_loader,
            patience_before_stopping=patience_before_stopping,
            min_delta=min_delta,
            restore_best=restore_best,
        )
        eval_metrics: Optional[EpochMetrics] = None
        if eval_loader is not None and hasattr(trainer, "eval"):
            eval_metrics = trainer.eval(eval_loader)

        test_metrics: Optional[EpochMetrics] = None
        if test_loader is not None and hasattr(trainer, "eval"):
            test_metrics = trainer.eval(test_loader)

        results: Dict[str, Any] = {
            "train_history": history,
            "train_samples": len(train_loader.dataset),
            "eval_samples": len(val_data) if val_data is not None else 0,
            "test_samples": len(test_data) if test_data is not None else 0,
            "temperature": getattr(trainer, "temperature", None),
            "detector_name": getattr(trainer, "detector_name", None),
        }

        if eval_metrics is not None:
            results["eval_metrics"] = {
                "loss": eval_metrics.loss,
                "f1": eval_metrics.f1,
            }
        if test_metrics is not None:
            results["test_metrics"] = {
                "loss": test_metrics.loss,
                "f1": test_metrics.f1,
            }

        self.all_results.append(results)
        return results
