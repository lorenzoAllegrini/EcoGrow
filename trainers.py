"""Training utilities for prompt learning with OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


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
        clip_model,
        text_encoder,
        device: torch.device,
        temperature: float = 0.01,
        use_amp: bool = True,
    ) -> None:
        self.clip_model = clip_model
        self.text_encoder = text_encoder
        self.device = device
        self.temperature = temperature
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def fit(
        self,
        prompt_learner,
        optimizer: torch.optim.Optimizer,
        train_loader,
        epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        val_loader=None,
        grad_clip: Optional[float] = None,
        log_fn: Optional[Callable[[str], None]] = print,
    ) -> List[Dict[str, Optional[EpochMetrics]]]:
        history: List[Dict[str, Optional[EpochMetrics]]] = []

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_train_epoch(
                prompt_learner, optimizer, train_loader, grad_clip
            )

            if scheduler is not None:
                scheduler.step()

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(prompt_learner, val_loader)

            if log_fn is not None:
                msg = (
                    f"epoch {epoch}/{epochs} | "
                    f"train loss {train_metrics.loss:.4f} acc {train_metrics.accuracy:.3f}"
                )
                if val_metrics is not None:
                    msg += (
                        f" | val loss {val_metrics.loss:.4f} "
                        f"acc {val_metrics.accuracy:.3f}"
                    )
                log_fn(msg)

            history.append({"train": train_metrics, "val": val_metrics})

        return history

    def _run_train_epoch(
        self,
        prompt_learner,
        optimizer: torch.optim.Optimizer,
        loader,
        grad_clip: Optional[float],
    ) -> EpochMetrics:
        prompt_learner.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # opzionale: stampa quali parametri sono addestrabili
        
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            optimizer.zero_grad()

            # CLIP immagine congelato
            with torch.no_grad():
                image_features = self.clip_model.encode_image(xb)
            image_features = F.normalize(image_features, dim=-1)

            # snapshot prima dell'update (per i tuoi report)
            before = snapshot_params(prompt_learner)
            # FORWARD testo (senza autocast)
            prompts_embeds, tokenized_prompts = prompt_learner()
            text_features = self.text_encoder(prompts_embeds, tokenized_prompts)
            text_features = F.normalize(text_features, dim=-1)

            logits = (image_features @ text_features.t()) / self.temperature
            loss = F.cross_entropy(logits, yb)
            print(loss)
            # deve essere True: altrimenti ctx non riceve grad per costruzione

            # grad vero su ctx (senza toccare optimizer)
            g = torch.autograd.grad(loss, prompt_learner.ctx, retain_graph=True, allow_unused=True)[0]

            # BACKWARD classico
            loss.backward()

            # clipping opzionale
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), grad_clip)

            # report gradienti (ora sono reali, non scalati)
            #print(grad_report(prompt_learner, prefix="ctx/"))

            # STEP ottimizzatore
            optimizer.step()

            # report delta parametri
            #print(delta_report(prompt_learner, before, prefix="ctx/"))
            print(prompt_learner.ctx)
            total_loss += loss.detach().item() * xb.size(0)
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            total += xb.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return EpochMetrics(loss=avg_loss, accuracy=acc)

    def evaluate(self, prompt_learner, loader) -> EpochMetrics:
        prompt_learner.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            prompts_embeds, tokenized_prompts = prompt_learner()
            text_features = self.text_encoder(prompts_embeds, tokenized_prompts)
            text_features = F.normalize(text_features, dim=-1)

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                image_features = self.clip_model.encode_image(xb)
                image_features = F.normalize(image_features, dim=-1)
                logits = (image_features @ text_features.t()) / self.temperature
                loss = F.cross_entropy(logits, yb)

                total_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(dim=-1) == yb).sum().item()
                total += xb.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return EpochMetrics(loss=avg_loss, accuracy=acc)
