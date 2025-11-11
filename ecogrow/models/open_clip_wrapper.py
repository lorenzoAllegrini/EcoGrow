from __future__ import annotations

from itertools import chain
from typing import Callable, Tuple, List, Optional, Dict, Sequence, Iterable

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_open_clip(
    model_name: str = "ViT-B-32",
    pretrained_tag: str = "laion2b_s34b_b79k",
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Callable, Callable, "TextEncoderOpenCLIP"]:
    """
    Factory function that returns the raw OpenCLIP components required across the project.
    """
    device = torch.device(device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_tag,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model.float()
    model = model.to(device).eval()

    text_encoder = TextEncoderOpenCLIP(model).float().to(device).eval()
    return model, preprocess, tokenizer, text_encoder


def freeze_open_clip_backbone(model: nn.Module) -> None:
    """Utility to freeze every parameter of the OpenCLIP backbone."""
    for param in model.parameters():
        param.requires_grad_(False)


class TextEncoderOpenCLIP(nn.Module):
    """
    Esegue il ramo testuale di open_clip a partire da EMBEDDING già costruiti:
    prompts_embeds: [C, L, D] oppure [B, C, L, D]
    tokenized_prompts: [C, L] (gli stessi token usati per calcolare EOT position)
    Ritorna: [C, D] oppure [B, C, D] (feature normalizzate prima della projection)
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding  # [L, D]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection  # [D_out, D]
        self.dtype = next(clip_model.parameters()).dtype

    def _encode(self, x, tokenized_prompts):
        # x: [C, L, D]
        C, L, D = x.shape
        pos = self.positional_embedding[:L, :].to(x.dtype)
        x = x + pos.unsqueeze(0)
        x = self.transformer(x)  # [L,C,D] se il tuo transformer vuole [L,C,D], altrimenti regola i permute
        x = x.permute(1, 0, 2) if x.shape[0] == L else x  # mantieni [C,L,D]
        x = self.ln_final(x)

        # --- EOT SHIFT FIX ---
        # L0 = lunghezza dei token originali (senza ctx)
        L0 = tokenized_prompts.shape[-1]
        shift = L - L0  # = n_ctx, in tutti i posizionamenti ('end','middle','front')
        eot_idx = tokenized_prompts.argmax(dim=-1) + shift  # [C]

        feats = x[torch.arange(C), eot_idx]  # [C, D]
        feats = feats @ self.text_projection.T
        return feats

    def forward(self, prompts_embeds, tokenized_prompts):
        if prompts_embeds.dim() == 3:  # [C, L, D]
            return self._encode(prompts_embeds, tokenized_prompts)
        elif prompts_embeds.dim() == 4:  # [B, C, L, D]
            B, C, L, D = prompts_embeds.shape
            outs = []
            for b in range(B):
                o = self._encode(prompts_embeds[b], tokenized_prompts)  # [C, D_out]
                outs.append(o)
            return torch.stack(outs, dim=0)  # [B, C, D_out]
        else:
            raise ValueError("prompts_embeds must be [C,L,D] or [B,C,L,D]")


class FamilyClipDetector:
    """Shared CLIP inference helper for a specific family of classes."""

    def __init__(
        self,
        name: str,
        classes: Sequence[str],
        temperature: float,
        source: Optional[str],
        *,
        clip_model: nn.Module,
        text_encoder: nn.Module,
        preprocess,
        device: torch.device,
        text_features: Optional[torch.Tensor] = None,
        prompt_learner: Optional[nn.Module] = None,
    ) -> None:
        if text_features is None and prompt_learner is None:
            raise ValueError("Provide either text_features or a prompt_learner.")

        self.name = name
        self.classes = list(classes)
        self.temperature = float(temperature)
        self.source = source

        self.clip_model = clip_model
        self.text_encoder = text_encoder
        self.preprocess = preprocess
        self.device = device
        self.prompt_learner = prompt_learner

        self._static_text_features: Optional[torch.Tensor] = None
        if text_features is not None:
            feats = torch.as_tensor(text_features).to(self.device)
            self._static_text_features = F.normalize(feats, dim=-1)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        feats = self.clip_model.encode_image(images)
        return F.normalize(feats, dim=-1)

    def text_features(self, *, with_grad: bool = False) -> torch.Tensor:
        if self._static_text_features is not None:
            return self._static_text_features
        if self.prompt_learner is None:
            raise RuntimeError("FamilyDetector has no prompt learner or static text features.")
        if with_grad:
            prompts_embeds, tokenized_prompts = self.prompt_learner()
            feats = self.text_encoder(prompts_embeds, tokenized_prompts)
            return F.normalize(feats, dim=-1)
        with torch.no_grad():
            prompts_embeds, tokenized_prompts = self.prompt_learner()
            feats = self.text_encoder(prompts_embeds, tokenized_prompts)
            feats = F.normalize(feats, dim=-1)
        return feats

    def encode_text_from_embeddings(
        self,
        prompts_embeds: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        """Encode text features directly from pre-built prompt embeddings.

        Args:
            prompts_embeds: shape [C, L, D] or [B, C, L, D]
            tokenized_prompts: shape [C, L]
            with_grad: whether to retain gradients through the text encoder.
        Returns:
            Normalized text features [C, D] or [B, C, D].
        """
        if with_grad:
            feats = self.text_encoder(prompts_embeds, tokenized_prompts)
            return F.normalize(feats, dim=-1)
        with torch.no_grad():
            feats = self.text_encoder(prompts_embeds, tokenized_prompts)
            feats = F.normalize(feats, dim=-1)
        return feats

    def update_embeddings(
        self,
        *,
        text_features: Optional[torch.Tensor] = None,
        prompts_embeds: Optional[torch.Tensor] = None,
        tokenized_prompts: Optional[torch.Tensor] = None,
    ) -> None:
        """Update cached text features used by the detector.

        Provide either precomputed `text_features` or a pair of
        (`prompts_embeds`, `tokenized_prompts`) to recompute and cache
        normalized features. Passing no arguments clears the cache and
        reverts to using the linked prompt_learner at runtime.
        """
        if text_features is not None:
            feats = torch.as_tensor(text_features).to(self.device)
            self._static_text_features = F.normalize(feats, dim=-1)
            return

        if prompts_embeds is not None and tokenized_prompts is not None:
            feats = self.encode_text_from_embeddings(prompts_embeds, tokenized_prompts, with_grad=False)
            # ensure cached as [C, D]; if batched, take average (or first) — prefer first for determinism
            if feats.dim() == 3:  # [B, C, D]
                feats = feats[0]
            self._static_text_features = feats
            return

        # Clear cache if no args
        self._static_text_features = None

    def logits(
        self,
        images: torch.Tensor,
        *,
        require_grad: bool = False,
        prompts_embeds: Optional[torch.Tensor] = None,
        tokenized_prompts: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits, optionally overriding text features.

        Precedence of text representation:
        1) `text_features` if provided
        2) (`prompts_embeds`, `tokenized_prompts`) if provided
        3) internal state: cached features or prompt_learner
        """
        with torch.no_grad():
            image_features = self.encode_images(images)

        if text_features is not None:
            feats = torch.as_tensor(text_features).to(self.device)
            txt = F.normalize(feats, dim=-1)
        elif prompts_embeds is not None and tokenized_prompts is not None:
            txt = self.encode_text_from_embeddings(
                prompts_embeds,
                tokenized_prompts,
                with_grad=require_grad,
            )
            if txt.dim() == 3:  # [B, C, D] -> use first for single set of class feats
                txt = txt[0]
        else:
            txt = self.text_features(with_grad=require_grad)

        return (image_features @ txt.t()) / self.temperature

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def predict(self, tensor: torch.Tensor, *, unknown_threshold: float) -> Dict[str, object]:
        with torch.no_grad():
            logits = self.logits(tensor, require_grad=False)
            probs = logits.softmax(dim=-1).squeeze(0)

        top_prob, top_idx = probs.max(dim=0)
        raw_label = self.classes[top_idx.item()]
        probability = float(top_prob.item())
        label = raw_label if probability >= float(unknown_threshold) else "unknown"

        per_class = [
            {"label": cls, "probability": float(probs[i].item())}
            for i, cls in enumerate(self.classes)
        ]
        per_class.sort(key=lambda item: item["probability"], reverse=True)

        return {
            "family": self.name,
            "prediction": label,
            "raw_label": raw_label,
            "probability": probability,
            "classes": per_class,
        }

    predict_batch = predict

    # Alias for clarity when used outside
    predict_batch = predict



class FamilyAdaptedClipDetector:
    """Fine-tuned CLIP detector that owns both backbone and classifier."""

    def __init__(
        self,
        name: str,
        classes: Sequence[str],
        *,
        clip_model: nn.Module,
        preprocess,
        device: torch.device,
        feature_dropout: float = 0.0,
        mode: str = "linear_probe"
    ) -> None:
        self.name = name
        self.classes = list(classes)
        self.clip_model = clip_model
        self.device = device
        self.preprocess = preprocess
        self.temperature = None  
        self.mode = mode 
        
        embed_dim = getattr(getattr(clip_model, "visual", None), "output_dim", None)
        if embed_dim is None:
            embed_dim = clip_model.text_projection.shape[1]

        feature_dropout = max(0.0, float(feature_dropout))
        head_layers: List[nn.Module] = []
        if feature_dropout > 0:
            head_layers.append(nn.Dropout(feature_dropout))
        head_layers.append(nn.Linear(embed_dim, len(self.classes)))
        self.classifier = nn.Sequential(*head_layers).to(device)
        if self.mode == "linear_probe":
            for p in self.clip_model.parameters():
                p.requires_grad_(False)

    def parameters(self) -> Iterable[nn.Parameter]:
        if self.mode == "linear_probe":
            return self.classifier.parameters()
        return chain(self.clip_model.parameters(), self.classifier.parameters())

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        feats = self.clip_model.encode_image(images)
        return F.normalize(feats, dim=-1)

    def logits(self, images: torch.Tensor, *, require_grad: bool = False) -> torch.Tensor:
        images = images.to(self.device)
        if require_grad:
            self.clip_model.train()
            self.classifier.train()
            return self._forward(images)

        self.clip_model.eval()
        self.classifier.eval()
        with torch.no_grad():
            return self._forward(images)

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encode_images(images)
        return self.classifier(feats)

    def predict(self, tensor: torch.Tensor, *, unknown_threshold: float) -> Dict[str, object]:
        with torch.no_grad():
            logits = self.logits(tensor, require_grad=False)
            probs = logits.softmax(dim=-1).squeeze(0)

        top_prob, top_idx = probs.max(dim=0)
        raw_label = self.classes[top_idx.item()]
        probability = float(top_prob.item())
        label = raw_label if probability >= float(unknown_threshold) else "unknown"

        per_class = [
            {"label": cls, "probability": float(probs[i].item())}
            for i, cls in enumerate(self.classes)
        ]
        per_class.sort(key=lambda item: item["probability"], reverse=True)

        return {
            "family": self.name,
            "prediction": label,
            "raw_label": raw_label,
            "probability": probability,
            "classes": per_class,
        }

    predict_batch = predict
