"""Training utilities for prompt learning with OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecogrow.models.open_clip_wrapper import DiseaseClipDetector, FamilyAdaptedClipDetector
from .prompt_learners import ClipPromptLearner

from peft import LoraConfig, get_peft_model

@dataclass
class EpochMetrics:
    loss: float
    f1: float



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
        detector: DiseaseClipDetector,
        preprocess: Any,
    ) -> None:
        self.prompt_learner = prompt_learner
        self.device = device
        self.preprocess = preprocess
        self.detector = detector

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
                    f"train loss {train_metrics.loss:.4f} f1 {train_metrics.f1:.3f}"
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
        cm = None  # confusion matrix [C,C] rows=true, cols=pred

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
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                C = logits.size(-1)
                idx = (yb.view(-1) * C + preds.view(-1)).to(torch.long).cpu()
                counts = torch.bincount(idx, minlength=C * C).view(C, C)
                if cm is None:
                    cm = counts
                else:
                    cm += counts

        denom = int(cm.sum().item()) if cm is not None else 1
        avg_loss = total_loss / max(denom, 1)
        if cm is None:
            f1_macro = 0.0
        else:
            tp = torch.diag(cm).to(torch.float32)
            fp = cm.sum(dim=0).to(torch.float32) - tp
            fn = cm.sum(dim=1).to(torch.float32) - tp
            prec = tp / torch.clamp(tp + fp, min=1.0)
            rec = tp / torch.clamp(tp + fn, min=1.0)
            f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-12)
            f1_macro = float(f1.mean().item())
        return EpochMetrics(loss=avg_loss, f1=f1_macro)
    
    def eval(self, eval_loader):
        eval_loss = 0.0
        cm = None
        prompts, _ = self.prompt_learner()
        for xb, yb in eval_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            with torch.no_grad():
                logits = self.detector.logits(xb, prompts_embeds=prompts)
            pred_idx = logits.argmax(dim=-1)

            eval_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            C = logits.size(-1)
            idx = (yb.view(-1) * C + pred_idx.view(-1)).to(torch.long).cpu()
            counts = torch.bincount(idx, minlength=C * C).view(C, C)
            if cm is None:
                cm = counts
            else:
                cm += counts

        denom = int(cm.sum().item()) if cm is not None else 1
        if cm is None:
            f1_macro = 0.0
        else:
            tp = torch.diag(cm).to(torch.float32)
            fp = cm.sum(dim=0).to(torch.float32) - tp
            fn = cm.sum(dim=1).to(torch.float32) - tp
            prec = tp / torch.clamp(tp + fp, min=1.0)
            rec = tp / torch.clamp(tp + fn, min=1.0)
            f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-12)
            f1_macro = float(f1.mean().item())

        eval_metrics = EpochMetrics(
            loss=eval_loss / max(denom, 1),
            f1=f1_macro,
        ) 
        return eval_metrics




class ClipFineTuneEngine:
    """Fine-tunes the CLIP image encoder plus a linear classifier per family."""

    supports_log_priors = True

    def __init__(
        self,
        *,
        family_detector: FamilyAdaptedClipDetector,
        prompt_learner: Optional[ClipPromptLearner] = None,
    ) -> None:
        self.detector = family_detector
        self.device = family_detector.device
        self.preprocess = family_detector.preprocess
        self.family_name = family_detector.name
        self.temperature = family_detector.temperature
        self.prompt_learner = prompt_learner
        self._log_prior_bias: Optional[torch.Tensor] = None
        self._prior_tau: float = 1.0

    def parameters(self):
        """Expose trainable parameters for optimizer construction."""
        return self.detector.parameters()

    def logits(self, images: torch.Tensor, *, require_grad: bool = False) -> torch.Tensor:
        """Proxy to detector logits for evaluation loops."""
        logits = self.detector.logits(images, require_grad=require_grad)
        if self._log_prior_bias is not None:
            logits = logits - self._prior_tau * self._log_prior_bias
        return logits

    def fit(
        self,
        optimizer: torch.optim.Optimizer,
        train_loader,
        epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None,
        log_fn: Optional[Callable[[str], None]] = print,
        log_priors = None,
        tau: float = 1.0
    ) -> List[Dict[str, Optional[EpochMetrics]]]:
        history: List[Dict[str, Optional[EpochMetrics]]] = []

        prior_bias = None
        if log_priors is not None:
            prior_bias = log_priors.to(self.device)
            if prior_bias.numel() != len(self.detector.classes):
                raise ValueError(
                    "log_priors size does not match detector classes "
                    f"({prior_bias.numel()} vs {len(self.detector.classes)})"
                )
            self._log_prior_bias = prior_bias
            self._prior_tau = tau
        else:
            self._log_prior_bias = None

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_train_epoch(
                optimizer,
                train_loader,
                grad_clip,
                log_priors=prior_bias,
                tau=tau,
            )

            if scheduler is not None:
                scheduler.step()

            if log_fn is not None:
                log_fn(
                    f"epoch {epoch}/{epochs} | train loss {train_metrics.loss:.4f} f1 {train_metrics.f1:.3f}"
                )

            history.append({"train": train_metrics})
        return history

    def _run_train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader,
        grad_clip: Optional[float],
        log_priors = None,
        tau: float = 1.0
    ) -> EpochMetrics:
        total_loss = 0.0
        cm = None
        trainable_params = list(self.parameters())
        if self.prompt_learner is not None:
            self.prompt_learner.train()
        for xb, yb in loader:

            if self.prompt_learner:
                prompts, _ = self.prompt_learner()
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            optimizer.zero_grad()
            if self.prompt_learner: 
                logits = self.detector.logits(xb, require_grad=True, prompts_embeds=prompts) 
            else:
                logits = self.detector.logits(xb, require_grad=True)
            
            if log_priors is not None:
                logits = logits - tau * log_priors

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
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                C = logits.size(-1)
                idx = (yb.view(-1) * C + preds.view(-1)).to(torch.long).cpu()
                counts = torch.bincount(idx, minlength=C * C).view(C, C)
                if cm is None:
                    cm = counts
                else:
                    cm += counts

        denom = int(cm.sum().item()) if cm is not None else 1
        avg_loss = total_loss / max(denom, 1)
        if cm is None:
            f1_macro = 0.0
        else:
            tp = torch.diag(cm).to(torch.float32)
            fp = cm.sum(dim=0).to(torch.float32) - tp
            fn = cm.sum(dim=1).to(torch.float32) - tp
            prec = tp / torch.clamp(tp + fp, min=1.0)
            rec = tp / torch.clamp(tp + fn, min=1.0)
            f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-12)
            f1_macro = float(f1.mean().item())
        return EpochMetrics(loss=avg_loss, f1=f1_macro)
    
    def eval(self, eval_loader):
        eval_loss = 0.0
        cm = None
        bias = self._log_prior_bias
        for xb, yb in eval_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            with torch.no_grad():
                logits = self.detector.logits(xb)
                if bias is not None:
                    logits = logits - self._prior_tau * bias
            pred_idx = logits.argmax(dim=-1)

            eval_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            C = logits.size(-1)
            idx = (yb.view(-1) * C + pred_idx.view(-1)).to(torch.long).cpu()
            counts = torch.bincount(idx, minlength=C * C).view(C, C)
            if cm is None:
                cm = counts
            else:
                cm += counts

        denom = int(cm.sum().item()) if cm is not None else 1
        if cm is None:
            f1_macro = 0.0
        else:
            tp = torch.diag(cm).to(torch.float32)
            fp = cm.sum(dim=0).to(torch.float32) - tp
            fn = cm.sum(dim=1).to(torch.float32) - tp
            prec = tp / torch.clamp(tp + fp, min=1.0)
            rec = tp / torch.clamp(tp + fn, min=1.0)
            f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-12)
            f1_macro = float(f1.mean().item())

        eval_metrics = EpochMetrics(
            loss=eval_loss / max(denom, 1),
            f1=f1_macro,
        ) 
        return eval_metrics
