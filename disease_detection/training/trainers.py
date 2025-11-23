"""Training utilities for prompt learning with OpenCLIP."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from disease_detection.models.model_wrappers import DiseaseClipDetector, ClipClassifierDetector
from .prompt_learners import ClipPromptLearner

from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

@dataclass
class EpochMetrics:
    loss: float
    f1: float


class _EarlyStoppingController:
    def __init__(
        self,
        patience: Optional[int],
        min_delta: Optional[float],
        restore_best: bool,
        snapshot_fn: Callable[[], Any],
        restore_fn: Callable[[Any], None],
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta if min_delta is not None else 0.0
        self.restore_best = restore_best
        self.snapshot_fn = snapshot_fn
        self.restore_fn = restore_fn
        self.best_loss = float("inf")
        self.epochs_since_improvement = 0
        self.best_state = snapshot_fn() if restore_best else None

    def update(self, loss: float) -> bool:
        improved = loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = loss
            self.epochs_since_improvement = 0
            if self.restore_best and self.best_state is not None:
                self.best_state = self.snapshot_fn()
            return False
        if self.patience is None:
            return False
        self.epochs_since_improvement += 1
        return self.epochs_since_improvement >= self.patience

    def restore(self) -> None:
        if self.restore_best and self.best_state is not None:
            self.restore_fn(self.best_state)



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
        eval_loader=None,
        patience_before_stopping: Optional[int] = None,
        min_delta: Optional[float] = None,
        restore_best: bool = False,
    ) -> List[Dict[str, Optional[EpochMetrics]]]:
        history: List[Dict[str, Optional[EpochMetrics]]] = []
        should_track = (patience_before_stopping is not None) or restore_best
        early_stopper: Optional[_EarlyStoppingController] = None
        if should_track:
            def _snapshot():
                return copy.deepcopy(self.prompt_learner.state_dict())

            def _restore(state):
                self.prompt_learner.load_state_dict(state)

            early_stopper = _EarlyStoppingController(
                patience_before_stopping,
                min_delta,
                restore_best,
                _snapshot,
                _restore,
            )

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_train_epoch(
                optimizer,
                train_loader,
                grad_clip,
                epoch=epoch,
                epochs=epochs,
            )

            eval_metrics: Optional[EpochMetrics] = None
            if eval_loader is not None and early_stopper is not None:
                eval_metrics = self.eval(eval_loader)

            if scheduler is not None:
                scheduler.step()

            if log_fn is not None:
                msg = (
                    f"epoch {epoch}/{epochs} | "
                    f"train loss {train_metrics.loss:.4f} f1 {train_metrics.f1:.3f}"
                )
                if eval_metrics is not None:
                    msg += (
                        f" | eval loss {eval_metrics.loss:.4f}"
                        f" f1 {eval_metrics.f1:.3f}"
                    )
                log_fn(msg)

            history.append({"train": train_metrics})

            if early_stopper is not None:
                monitored = eval_metrics if eval_metrics is not None else train_metrics
                if early_stopper.update(monitored.loss):
                    if log_fn is not None:
                        log_fn(
                            f"Early stopping triggered at epoch {epoch}"
                            f" (patience={patience_before_stopping})"
                        )
                    break

        if early_stopper is not None:
            early_stopper.restore()
        return history

    def _run_train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader,
        grad_clip: Optional[float],
        *,
        epoch: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> EpochMetrics:

        prompt_module = self.prompt_learner
        prompt_module.train()
        total_loss = 0.0
        cm = None  # confusion matrix [C,C] rows=true, cols=pred

        desc = "prompt epoch"
        if epoch is not None and epochs is not None:
            desc = f"prompt epoch {epoch}/{epochs}"
        progress_bar = tqdm(loader, desc=desc, leave=False)

        for step, (xb, yb) in enumerate(progress_bar, start=1):
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
        clip_detector: ClipClassifierDetector,
        prompt_learner: Optional[ClipPromptLearner] = None,
    ) -> None:
        self.detector = clip_detector
        self.device = clip_detector.device
        self.preprocess = clip_detector.preprocess
        self.detector_name = getattr(
            clip_detector, "detector_id", clip_detector.__class__.__name__
        )
        self.temperature = clip_detector.temperature
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
        tau: float = 1.0,
        eval_loader=None,
        patience_before_stopping: Optional[int] = None,
        min_delta: Optional[float] = None,
        restore_best: bool = False,
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

        should_track = (patience_before_stopping is not None) or restore_best
        early_stopper: Optional[_EarlyStoppingController] = None
        if should_track:
            def _snapshot_state():
                state = {"detector": copy.deepcopy(self.detector.state_dict())}
                if self.prompt_learner is not None:
                    state["prompt"] = copy.deepcopy(self.prompt_learner.state_dict())
                return state

            def _restore_state(state):
                self.detector.load_state_dict(state["detector"])
                if self.prompt_learner is not None and "prompt" in state:
                    self.prompt_learner.load_state_dict(state["prompt"])

            early_stopper = _EarlyStoppingController(
                patience_before_stopping,
                min_delta,
                restore_best,
                _snapshot_state,
                _restore_state,
            )

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_train_epoch(
                optimizer,
                train_loader,
                grad_clip,
                log_priors=prior_bias,
                tau=tau,
                epoch=epoch,
                epochs=epochs,
            )

            eval_metrics: Optional[EpochMetrics] = None
            if eval_loader is not None and early_stopper is not None:
                eval_metrics = self.eval(eval_loader)

            if scheduler is not None:
                scheduler.step()

            if log_fn is not None:
                msg = (
                    f"epoch {epoch}/{epochs} | "
                    f"train loss {train_metrics.loss:.4f} f1 {train_metrics.f1:.3f}"
                )
                if eval_metrics is not None:
                    msg += (
                        f" | eval loss {eval_metrics.loss:.4f}"
                        f" f1 {eval_metrics.f1:.3f}"
                    )
                log_fn(msg)

            history.append({"train": train_metrics})

            if early_stopper is not None:
                monitored = eval_metrics if eval_metrics is not None else train_metrics
                if early_stopper.update(monitored.loss):
                    if log_fn is not None:
                        log_fn(
                            f"Early stopping triggered at epoch {epoch}"
                            f" (patience={patience_before_stopping})"
                        )
                    break

        if early_stopper is not None:
            early_stopper.restore()
        return history

    def _run_train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader,
        grad_clip: Optional[float],
        log_priors = None,
        tau: float = 1.0,
        *,
        epoch: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> EpochMetrics:
        total_loss = 0.0
        cm = None
        trainable_params = list(self.parameters())
        if self.prompt_learner is not None:
            self.prompt_learner.train()
        desc = f"{self.detector_name} finetune"
        if epoch is not None and epochs is not None:
            desc = f"{self.detector_name} finetune {epoch}/{epochs}"

        progress_bar = tqdm(loader, desc=desc, leave=False)

        for xb, yb in progress_bar:

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


class ConvNextFineTuneEngine:
    """Trainer for ConvNeXt-style detectors using the EcogrowBenchmark interface."""

    supports_log_priors = False

    def __init__(self, *, detector) -> None:
        self.detector = detector
        self.device = detector.device
        self.preprocess = detector.preprocess
        self.detector_name = getattr(
            detector, "detector_id", detector.__class__.__name__
        )

    def parameters(self):
        return self.detector.parameters()

    def logits(self, images: torch.Tensor, *, require_grad: bool = False) -> torch.Tensor:
        return self.detector.logits(images, require_grad=require_grad)

    def fit(
        self,
        optimizer: torch.optim.Optimizer,
        train_loader,
        epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None,
        log_fn: Optional[Callable[[str], None]] = print,
        eval_loader=None,
        patience_before_stopping: Optional[int] = None,
        min_delta: Optional[float] = None,
        restore_best: bool = False,
    ) -> List[Dict[str, Optional[EpochMetrics]]]:
        history: List[Dict[str, Optional[EpochMetrics]]] = []

        should_track = (patience_before_stopping is not None) or restore_best
        early_stopper: Optional[_EarlyStoppingController] = None
        if should_track:

            def _snapshot_state():
                return copy.deepcopy(self.detector.state_dict())

            def _restore_state(state):
                self.detector.load_state_dict(state)

            early_stopper = _EarlyStoppingController(
                patience_before_stopping,
                min_delta,
                restore_best,
                _snapshot_state,
                _restore_state,
            )

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_train_epoch(
                optimizer,
                train_loader,
                grad_clip,
                epoch=epoch,
                epochs=epochs,
            )

            eval_metrics: Optional[EpochMetrics] = None
            if eval_loader is not None and early_stopper is not None:
                eval_metrics = self.eval(eval_loader)

            if scheduler is not None:
                scheduler.step()

            if log_fn is not None:
                msg = (
                    f"epoch {epoch}/{epochs} | "
                    f"train loss {train_metrics.loss:.4f} f1 {train_metrics.f1:.3f}"
                )
                if eval_metrics is not None:
                    msg += (
                        f" | eval loss {eval_metrics.loss:.4f}"
                        f" f1 {eval_metrics.f1:.3f}"
                    )
                log_fn(msg)

            history.append({"train": train_metrics})

            if early_stopper is not None:
                monitored = eval_metrics if eval_metrics is not None else train_metrics
                if early_stopper.update(monitored.loss):
                    if log_fn is not None:
                        log_fn(
                            f"Early stopping triggered at epoch {epoch}"
                            f" (patience={patience_before_stopping})"
                        )
                    break

        if early_stopper is not None:
            early_stopper.restore()
        return history

    def _run_train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader,
        grad_clip: Optional[float],
        *,
        epoch: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> EpochMetrics:
        total_loss = 0.0
        cm = None
        trainable_params = list(self.parameters())
        desc = f"{self.detector_name} convnext finetune"
        if epoch is not None and epochs is not None:
            desc = f"{self.detector_name} convnext finetune {epoch}/{epochs}"

        progress_bar = tqdm(loader, desc=desc, leave=False)

        for xb, yb in progress_bar:
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
        for xb, yb in eval_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            with torch.no_grad():
                logits = self.detector.logits(xb, require_grad=False)
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
