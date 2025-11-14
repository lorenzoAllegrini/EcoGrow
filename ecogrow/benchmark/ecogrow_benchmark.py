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
        family_id: Optional[str] = None,
        families: Optional[Sequence[str]] = None,
        transform=None,
        segment_fn=None,
    ) -> Tuple[PlantData, PlantData, PlantData]:
        """Load dataset splits, optionally filtering by family/families.

        Args:
            family_id: single-family identifier (mutually exclusive with ``families``).
            families: optional sequence of families to keep; ``None`` keeps every family.
            transform: optional transform to apply when fetching samples.
            segment_fn: optional preprocessing callable used before the transform.

        Returns:
            Tuple[PlantData, PlantData, PlantData]: train/val/test datasets.
        """

        if family_id is not None and families is not None:
            raise ValueError("Specify only 'family_id' or 'families', not both.")

        train_data = PlantData(
            split="train",
            dataset_root=self.data_root,
            transform=transform,
            segment_fn=segment_fn,
            families=families,
            family_id=family_id  
        )

        val_data = PlantData(
            split="val",
            dataset_root=self.data_root,
            transform=transform,
            segment_fn=segment_fn,
            families=families,
            family_id=family_id 
        )

        return train_data, val_data
    
    def run(
        self,
        *,
        trainer,
        segment_fn: Any,
        family_id: Optional[str] = None,
        families: Optional[Sequence[str]] = None,
        perc_eval: Optional[float] = None,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run training/evaluation on one or more families (or globally).

        Args:
            trainer: pre-configured trainer compatible with the benchmark interface.
            segment_fn: preprocessing function applied before ``trainer.preprocess``.
            family_id: optional single family identifier (legacy API).
            families: optional sequence of families to include (``None`` = all).
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

        if fit_args:
            unknown = ", ".join(sorted(fit_args.keys()))
            raise ValueError(f"Unsupported fit arguments: {unknown}")

        if perc_eval not in (None, 0):
            logging.info(
                "'perc_eval' is deprecated and ignored; using the explicit validation split."
            )

        train_data, val_data = self.load_data(
            family_id=family_id,
            families=families,
            transform=preprocess,
            segment_fn=segment_fn,
        )

        train_loader = train_data.make_dataloader(
            batch_size=batch_size,
            weighted=True,
        )
        
        eval_loader = (
            DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False,
            )
            if len(val_data) > 0
            else None
        )

        if optimizer is None:
            optimizer = torch.optim.AdamW(trainer.parameters(), lr=lr)

        if family_id:
            target_label = family_id
        elif families:
            target_label = ",".join(families)
        else:
            target_label = "all"

        logging.info(
            "Starting fine-tuning for '%s' | epochs=%d batch_size=%d",
            target_label,
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
        )
        eval_metrics: Optional[EpochMetrics] = None
        if eval_loader is not None and hasattr(trainer, "eval"):
            eval_metrics = trainer.eval(eval_loader)

        results: Dict[str, Any] = {
            "family_id": family_id,
            "families": list(families) if families is not None else None,
            "train_history": history,
            "train_samples": len(train_loader.dataset),
            "eval_samples": len(val_data),
            "temperature": getattr(trainer, "temperature", None),
        }

        if eval_metrics is not None:
            results["eval_metrics"] = {
                "loss": eval_metrics.loss,
                "f1": eval_metrics.f1,
            }

        self.all_results.append(results)
        return results
