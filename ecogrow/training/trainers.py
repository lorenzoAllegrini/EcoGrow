"""Training utilities for prompt learning with OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from ecogrow.models.open_clip_wrapper import OpenClipWrapper
from .prompt_learners import PromptLearnerOpenCLIP

@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


class PromptTuningTrainer:
    def __init__(
        self,
        clip_wrapper: OpenClipWrapper,
        device: torch.device,
        prompt_learner: PromptLearnerOpenCLIP,
        temperature: float = 0.01,
        use_amp: bool = True,
    ) -> None:
        
        self.clip_wrapper = clip_wrapper
        self.clip_model = clip_wrapper.model
        self.text_encoder = clip_wrapper.text_encoder
        self.prompt_learner = prompt_learner
        self.device = device
        self.temperature = temperature
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def fit(
        self,
        optimizer: torch.optim.Optimizer,
        train_loader,
        epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None,
        log_fn: Optional[Callable[[str], None]] = print,
    ) -> List[Dict[str, Optional[EpochMetrics]]]:
        history: List[Dict[str, Optional[EpochMetrics]]] = []

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_train_epoch(
                optimizer, train_loader, grad_clip
            )

            if scheduler is not None:
                scheduler.step()

           
            if log_fn is not None:
                msg = (
                    f"epoch {epoch}/{epochs} | "
                    f"train loss {train_metrics.loss:.4f} acc {train_metrics.accuracy:.3f}"
                )
                log_fn(msg)

            history.append({"train": train_metrics})

        return history

    def _run_train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader,
        grad_clip: Optional[float],
    ) -> EpochMetrics:
        
        self.prompt_learner.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            optimizer.zero_grad()

            # CLIP immagine congelato
            with torch.no_grad():
                image_features = self.clip_model.encode_image(xb)
            image_features = F.normalize(image_features, dim=-1)

            # FORWARD testo (senza autocast)
            prompts_embeds, tokenized_prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts_embeds, tokenized_prompts)
            text_features = F.normalize(text_features, dim=-1)

            logits = (image_features @ text_features.t()) / self.temperature
            loss = F.cross_entropy(logits, yb)

            # BACKWARD classico
            loss.backward()

            # clipping opzionale
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.prompt_learner.parameters(), grad_clip)

            # STEP ottimizzatore
            optimizer.step()
    
            total_loss += loss.detach().item() * xb.size(0)
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            total += xb.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return EpochMetrics(loss=avg_loss, accuracy=acc)
