"""Training utilities for prompt learning with OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ecogrow.models.open_clip_wrapper import OpenClipWrapper
from .prompt_learners import PromptLearnerOpenCLIP

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

        for step, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            optimizer.zero_grad()

            # ============ FORWARD (immagini congelate) ============
            with torch.no_grad():
                image_features = self.clip_model.encode_image(xb)
            image_features = F.normalize(image_features, dim=-1)

            prompts_embeds, tokenized_prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts_embeds, tokenized_prompts)
            text_features = F.normalize(text_features, dim=-1)

            logits = (image_features @ text_features.t()) / self.temperature
            loss = F.cross_entropy(logits, yb)

            # ============ BACKWARD ============
            loss.backward()

            # Failsafe: se gradienti hanno NaN/Inf, salta lo step
            bad_grad = False
            for p in self.prompt_learner.parameters():
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


    def predict(
        self,
        images: torch.Tensor,
        *,
        class_names: Optional[Sequence[str]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, torch.Tensor | List[str]]:
        """
        Esegue la predizione per un batch di immagini usando i prompt appresi.

        Args:
            images: tensore batch in formato compatibile con la preprocess pipeline di CLIP.
            class_names: etichette opzionali (stessa lunghezza delle classi) da includere nell'output.
            temperature: override opzionale della temperatura usata per normalizzare i logits.

        Returns:
            Dizionario con logits, probabilità, indici predetti e, se fornito, le etichette corrispondenti.
        """
        was_training = self.prompt_learner.training
        self.prompt_learner.eval()

        with torch.no_grad():
            prompts_embeds, tokenized_prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts_embeds, tokenized_prompts)
            text_features = F.normalize(text_features, dim=-1)

            images = images.to(self.device)
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            temp = temperature if temperature is not None else self.temperature
            logits = (image_features @ text_features.t()) / temp
            probs = logits.softmax(dim=-1)

        if was_training:
            self.prompt_learner.train()

        pred_indices = probs.argmax(dim=-1)
        result: Dict[str, torch.Tensor | List[str]] = {
            "logits": logits,
            "probs": probs,
            "pred_indices": pred_indices,
        }
        if class_names is not None:
            result["pred_labels"] = [class_names[i] for i in pred_indices.cpu().tolist()]

        return result
