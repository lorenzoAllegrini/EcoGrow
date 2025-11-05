from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd
import logging

from torch.utils.data import DataLoader, Subset

from ecogrow.data.plant_data import PlantData
from ecogrow.models.open_clip_wrapper import OpenClipWrapper
from ecogrow.training.prompt_learners import PromptLearnerOpenCLIP
from ecogrow.training.trainers import PromptTuningTrainer

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
        self.run_id = self.run_id
        self.exp_dir = exp_dir
        self.data_root: str = data_root
        self.all_results: List[Dict[str, Any]] = []
     
    def load_channel(self, family_id: str) -> Tuple[PlantData, PlantData]:
        """Load the training and testing datasets for a given plant.

        Args:
            family_id (str): the ID of the family of palnts to be used

        Returns:
            Tuple[NASA, NASA]: training and testing datasets
        """
        train_data = PlantData(
            dataset_root=self.data_root,
            family=family_id,
            split="train",
        )

        test_data = PlantData(
            dataset_root=self.data_root,
            family=family_id,
            split="test",
        )

        return train_data, test_data
    
    def run(
        self,
        family_id: str,
        model : OpenClipWrapper,
        prompt_learner: PromptLearnerOpenCLIP,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
        perc_eval: Optional[float] = 0.2,
    ):
        """Runs the benchmark for a given family.

        Args:
            family_id (str): the ID of the channel to be used
            fit_predictor_args (Optional[Dict[str, Any]]): additional arguments for the predictor's fit method
            perc_eval (Optional[float]): the percentage of the training data to be used for evaluation
        """
        
        train_channel, test_channel = self.load_channel(
            family_id,
        )
        os.makedirs(self.run_dir, exist_ok=True)

        results: Dict[str, Any] = {"family_id": family_id}
        train_history = None

        if fit_predictor_args is not None:
            logging.info(f"Fitting the predictor for family {family_id}...")
            # Training the predictor
            batch_size = fit_predictor_args.pop("batch_size", 16)
            eval_data = None
            if perc_eval is not None:
                # Split the training data into training and evaluation sets
                indices = np.arange(len(train_channel))
                np.random.shuffle(indices)
                eval_size = int(len(train_channel) * perc_eval)
                eval_channel = Subset(train_channel, indices[:eval_size])
                train_channel = Subset(train_channel, indices[eval_size:])
            train_loader = DataLoader(
                train_channel,
                batch_size=batch_size,
                shuffle=True,
            )
            eval_loader = (
                DataLoader(
                    eval_channel,
                    batch_size=batch_size,
                    shuffle=False,
                )
                if eval_channel is not None
                else None
            )
            train_history = model.fit(
                train_loader=train_loader,
                valid_loader=eval_loader,
                **fit_predictor_args,
            )
         
