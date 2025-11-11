from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Subset

from ecogrow.data.plant_data import PlantData
from ecogrow.training.trainers import ClipPromptEngine, EpochMetrics

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
     
    def load_channel(
        self,
        family_id: str,
        *,
        transform=None,
        segment_fn=None,
    ) -> Tuple[PlantData, PlantData]:
        """Load the training and testing datasets for a given plant.

        Args:
            family_id (str): the ID of the family of palnts to be used
            transform: optional transform to apply when fetching samples.
            segment_fn: optional preprocessing callable used before the transform.

        Returns:
            Tuple[PlantData, PlantData]: training and testing datasets
        """
        train_data = PlantData(
            dataset_root=self.data_root,
            family_id=family_id,
            split="train",
            transform=transform,
            segment_fn=segment_fn,
        )

        test_data = PlantData(
            dataset_root=self.data_root,
            family_id=family_id,
            split="test",
            transform=transform,
            segment_fn=segment_fn,
        )

        return train_data, test_data
    
    def run(
        self,
        family_id: str,
        trainer: ClipPromptEngine,
        segment_fn: Any,
        perc_eval: Optional[float] = 0.2,
        fit_predictor_args: Optional[Dict[str, Any]] = None,    
    ) -> Dict[str, Any]:
        """Runs the benchmark for a given family.

        Args:
            family_id (str): the ID of the channel to be used
            trainer (ClipPromptEngine): pre-configured trainer to be tuned.
            fit_predictor_args (Optional[Dict[str, Any]]): arguments controlling optimisation.
            perc_eval (Optional[float]): the percentage of the training data to be used for evaluation

        Returns:
            Dict[str, Any]: collected metrics and bookkeeping for the run.
        """
        os.makedirs(self.run_dir, exist_ok=True)

        fit_args = dict(fit_predictor_args or {})

        prompt_learner = trainer.prompt_learner
        preprocess = trainer.preprocess_fn

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

        train_data, test_data = self.load_channel(
            family_id,
            transform=preprocess,
            segment_fn=segment_fn,
        )

        if perc_eval is not None and perc_eval > 0:
            indices = np.arange(len(train_data))
            np.random.shuffle(indices)
            eval_size = max(int(len(train_data) * perc_eval), 1)
            eval_dataset = Subset(train_data, indices[:eval_size])
            train_data = Subset(train_data, indices[eval_size:])
        else:
            eval_dataset = None

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        eval_loader = (
            DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
            )
            if eval_dataset is not None
            else None
        )

        if optimizer is None:
            optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=lr)

        logging.info(
            "Starting fine-tuning for family '%s' | epochs=%d batch_size=%d",
            family_id,
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
        if eval_loader is not None:
            eval_total = 0
            eval_correct = 0
            eval_loss = 0.0

            for xb, yb in eval_loader:
                xb = xb.to(trainer.device)
                yb = yb.to(trainer.device)
                with torch.no_grad():
                    logits = trainer.detector.logits(xb)
                pred_idx = logits.argmax(dim=-1)

                eval_loss += F.cross_entropy(logits, yb, reduction="sum").item()
                eval_correct += (pred_idx == yb).sum().item()
                eval_total += yb.size(0)

            eval_metrics = EpochMetrics(
                loss=eval_loss / max(eval_total, 1),
                accuracy=eval_correct / max(eval_total, 1),
            )

        results: Dict[str, Any] = {
            "family_id": family_id,
            "train_history": history,
            "train_samples": len(train_loader.dataset),
            "eval_samples": len(eval_loader.dataset) if eval_loader is not None else 0,
            "test_samples": len(test_data),
            "temperature": trainer.temperature,
        }

        if eval_metrics is not None:
            results["eval_metrics"] = {
                "loss": eval_metrics.loss,
                "accuracy": eval_metrics.accuracy,
            }

        self.all_results.append(results)
        return results
