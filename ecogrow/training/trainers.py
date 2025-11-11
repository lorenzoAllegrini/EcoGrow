"""Training utilities for prompt learning with OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ecogrow.models.open_clip_wrapper import FamilyDetector
from .prompt_learners import ClipPromptLearner

@dataclass
class EpochMetrics:
    loss: float
    accuracy: float



def snapshot_params(model):
    # copia leggera dei tensori dei pesi per calcolare le differenze dopo lo step
    return {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }


def grad_report(model, prefix=""):
    lines = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            lines.append(f"[FROZEN] {prefix}{n}")
            continue
        g = p.grad
        if g is None:
            lines.append(f"[NO-GRAD] {prefix}{n}")
        else:
            lines.append(f"[GRAD]    {prefix}{n}: ||g||={g.data.norm().item():.4e}")
    return "\n".join(lines)


def delta_report(model, before, prefix=""):
    # mostra quanto è cambiato ogni parametro dopo optimizer.step()
    lines = []
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n not in before:
                lines.append(f"[NEW]  {prefix}{n} (aggiunto dopo lo snapshot)")
                continue
            if not p.requires_grad:
                lines.append(f"[FROZEN]{prefix}{n}")
                continue
            delta = (p - before[n]).norm().item()
            lines.append(f"[Δ]     {prefix}{n}: ||Δ||={delta:.4e}")
    return "\n".join(lines)


class ClipPromptEngine:
    def __init__(
        self,
        clip_model: torch.nn.Module,
        text_encoder: torch.nn.Module,
        preprocess,
        device: torch.device,
        *,
        class_names: Sequence[str],
        family_name: str,
        prompt_learner: ClipPromptLearner,
        temperature: float = 0.01,
    ) -> None:
        self.prompt_learner = prompt_learner
        self.device = device
        self.temperature = temperature
        self.preprocess_fn = preprocess

        self.detector = FamilyDetector(
            name=family_name,
            classes=class_names,
            temperature=temperature,
            source=None,
            clip_model=clip_model,
            text_encoder=text_encoder,
            preprocess=preprocess,
            device=device,
            prompt_learner=prompt_learner,
        )

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

        prompt_module = self.prompt_learner
        prompt_module.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for step, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            optimizer.zero_grad()

            logits = self.detector.logits(xb, require_grad=True)
            loss = F.cross_entropy(logits, yb)

            # ============ BACKWARD ============
            loss.backward()

            # Failsafe: se gradienti hanno NaN/Inf, salta lo step
            bad_grad = False
            for p in prompt_module.parameters():
                if p.grad is None:
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    bad_grad = True
                    break
            if bad_grad:
                print(f"[WARN] NaN/Inf gradient at step {step} — skipping optimizer.step()")
                optimizer.zero_grad(set_to_none=True)
                continue

            # ============ STEP ============
            optimizer.step()

            # Metriche batch
            bs = xb.size(0)
            total_loss += loss.detach().item() * bs
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            total += bs

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return EpochMetrics(loss=avg_loss, accuracy=acc)

