"""Training utilities for prompt learning with OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecogrow.models.open_clip_wrapper import FamilyClipDetector, FamilyAdaptedClipDetector
from .prompt_learners import ClipPromptLearner

from peft import LoraConfig, get_peft_model

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
        device: torch.device,
        *,
        prompt_learner: ClipPromptLearner,
        family_detector:FamilyClipDetector,
        preprocess: Any,
    ) -> None:
        self.prompt_learner = prompt_learner
        self.device = device
        self.preprocess = preprocess
        self.detector = family_detector

    def parameters(self):
        """Expose trainable parameters so the benchmark can build optimizers generically."""
        return self.prompt_learner.parameters()

    def logits(self, images: torch.Tensor, *, require_grad: bool = False) -> torch.Tensor:
        """Shared helper so evaluation code does not need direct access to the detector."""
        return self.detector.logits(images, require_grad=require_grad)

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
            prompts, _ = prompt_module()

            xb = xb.to(self.device)
            yb = yb.to(self.device)

            optimizer.zero_grad()

            logits = self.detector.logits(xb, require_grad=True, prompts_embeds=prompts)
            loss = F.cross_entropy(logits, yb)

            loss.backward()

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
    
    def eval(self, eval_loader):
        eval_total = 0
        eval_correct = 0
        eval_loss = 0.0
        prompts, _ = self.prompt_learner()
        for xb, yb in eval_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            with torch.no_grad():
                logits = self.detector.logits(xb, prompts_embeds=prompts)
            pred_idx = logits.argmax(dim=-1)

            eval_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            eval_correct += (pred_idx == yb).sum().item()
            eval_total += yb.size(0)

        eval_metrics = EpochMetrics(
            loss=eval_loss / max(eval_total, 1),
            accuracy=eval_correct / max(eval_total, 1),
        ) 




class ClipFineTuneEngine:
    """Fine-tunes the CLIP image encoder plus a linear classifier per family."""

    def __init__(
        self,
        *,
        family_detector: FamilyAdaptedClipDetector,
    ) -> None:
        self.detector = family_detector
        self.device = family_detector.device
        self.preprocess = family_detector.preprocess
        self.family_name = family_detector.name
        self.temperature = family_detector.temperature

    def parameters(self):
        """Expose trainable parameters for optimizer construction."""
        return self.detector.parameters()

    def logits(self, images: torch.Tensor, *, require_grad: bool = False) -> torch.Tensor:
        """Proxy to detector logits for evaluation loops."""
        return self.detector.logits(images, require_grad=require_grad)

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
                log_fn(
                    f"epoch {epoch}/{epochs} | train loss {train_metrics.loss:.4f} acc {train_metrics.accuracy:.3f}"
                )

            history.append({"train": train_metrics})
            print(history)
        return history

    def _run_train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader,
        grad_clip: Optional[float],
    ) -> EpochMetrics:
        total_loss = 0.0
        correct = 0
        total = 0
        trainable_params = list(self.parameters())

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            optimizer.zero_grad()

            logits = self.detector.logits(xb, require_grad=True)
            loss = F.cross_entropy(logits, yb)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            bad_grad = False
            for p in trainable_params:
                if p.grad is None:
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    bad_grad = True
                    break
            if bad_grad:
                print("[WARN] NaN/Inf gradient — skipping optimizer.step()")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            bs = xb.size(0)
            total_loss += loss.detach().item() * bs
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            total += bs

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return EpochMetrics(loss=avg_loss, accuracy=acc)
