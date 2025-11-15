from __future__ import annotations

from itertools import chain
from typing import Callable, Tuple, List, Optional, Dict, Sequence, Iterable, Literal

import open_clip
import timm
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
    extra = {}
    if model_name.startswith("MobileCLIP-S"):
        extra.update(dict(image_mean=(0,0,0), image_std=(1,1,1)))
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_tag,
        **extra
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
        # rileva dove stanno i componenti testuali
        if hasattr(clip_model, "text"):  
            text_mod = clip_model.text
        else:                          
            text_mod = clip_model

        # salva riferimenti low-level
        self.token_embedding = text_mod.token_embedding          
        self.positional_embedding = text_mod.positional_embedding 
        self.transformer = text_mod.transformer
        self.ln_final = text_mod.ln_final
        self.text_projection = text_mod.text_projection        

        self.context_length = getattr(text_mod, "context_length", None)
        self.dtype = next(clip_model.parameters()).dtype
        self.device = next(clip_model.parameters()).device


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



class DiseaseClipDetector:
    """Shared CLIP inference helper for an arbitrary set of disease classes."""

    def __init__(
        self,
        classes: Sequence[str],
        temperature: float,
        *,
        clip_model: nn.Module,
        text_encoder: nn.Module,
        device: torch.device,
        text_features: Optional[torch.Tensor] = None,
        prompt_learner: Optional[nn.Module] = None,
        detector_id: Optional[str] = None,
    ) -> None:
        if text_features is None and prompt_learner is None:
            raise ValueError("Provide either text_features or a prompt_learner.")

        self.classes = list(classes)
        self.temperature = float(temperature)
        self.detector_id = detector_id

        self.clip_model = clip_model
        self.text_encoder = text_encoder
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
            raise RuntimeError("DiseaseClipDetector has no prompt learner or static text features.")
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

        result = {
            "prediction": label,
            "raw_label": raw_label,
            "probability": probability,
            "classes": per_class,
        }
        if self.detector_id is not None:
            result["detector_id"] = self.detector_id
        return result

    predict_batch = predict



class ClipClassifierDetector:
    """Fine-tuned detector that mirrors the DiseaseClipDetector API."""

    def __init__(
        self,
        name: str,
        classes: Sequence[str],
        *,
        clip_model: nn.Module,
        preprocess,
        device: torch.device,
        feature_dropout: float = 0.0,
        train_backbone: bool = False,
        temperature: float | None = None,
        text_encoder: Optional[nn.Module] = None,
    ) -> None:
        self.name = name
        self.classes = list(classes)
        self.clip_model = clip_model
        self.device = device
        self.preprocess = preprocess
        self.temperature = temperature if temperature is not None else 0.07
        self.train_backbone = bool(train_backbone)
        self.text_encoder = text_encoder

        embed_dim = self._infer_embed_dim()

        feature_dropout = max(0.0, float(feature_dropout))
        head_layers: List[nn.Module] = []
        if feature_dropout > 0:
            head_layers.append(nn.Dropout(feature_dropout))
        head_layers.append(nn.Linear(embed_dim, len(self.classes)))
        self.classifier = nn.Sequential(*head_layers).to(device)

        if not self.train_backbone:
            self._freeze_backbone()

    def parameters(self) -> Iterable[nn.Parameter]:
        if not self.train_backbone:
            return self.classifier.parameters()
        return chain(self._backbone_parameters(), self.classifier.parameters())

    def _infer_embed_dim(self) -> int:
        visual = getattr(self.clip_model, "visual", None)
        if visual is not None and hasattr(visual, "output_dim"):
            return visual.output_dim
        text = getattr(self.clip_model, "text", None)
        if text is not None and hasattr(text, "output_dim"):
            return text.output_dim
        if hasattr(self.clip_model, "text_projection"):
            return self.clip_model.text_projection.shape[1]
        raise ValueError("Unable to infer embedding dimension for classifier head.")

    def _backbone_module(self) -> nn.Module:
        visual = getattr(self.clip_model, "visual", None)
        if isinstance(visual, nn.Module):
            return visual
        return self.clip_model

    def _backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self._backbone_module().parameters()

    def _freeze_backbone(self) -> None:
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        feats = self.clip_model.encode_image(images)
        return F.normalize(feats, dim=-1)

    def logits(
        self,
        images: torch.Tensor,
        *,
        require_grad: bool = False,
        prompts_embeds: Optional[torch.Tensor] = None,
        tokenized_prompts: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        images = images.to(self.device)
        use_prompt_logits = (
            text_features is not None
            or (prompts_embeds is not None and tokenized_prompts is not None)
        )

        backbone_mode = require_grad and self.train_backbone
        self._set_module_mode(self._backbone_module(), training=backbone_mode)
        self._set_module_mode(self.classifier, training=require_grad and not use_prompt_logits)

        if use_prompt_logits:
            feats = self.encode_images(images)
            txt = self._resolve_text_features(
                prompts_embeds=prompts_embeds,
                tokenized_prompts=tokenized_prompts,
                text_features=text_features,
                require_grad=require_grad,
            )
            return (feats @ txt.t()) / float(self.temperature)

        if require_grad:
            return self._forward(images)
        with torch.no_grad():
            return self._forward(images)

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encode_images(images)
        return self.classifier(feats)

    @staticmethod
    def _set_module_mode(module: nn.Module, *, training: bool) -> None:
        if training:
            module.train()
        else:
            module.eval()

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
            "detector": self.name,
            "family": self.name,  # backward compatibility
            "prediction": label,
            "raw_label": raw_label,
            "probability": probability,
            "classes": per_class,
        }

    predict_batch = predict

    def _resolve_text_features(
        self,
        *,
        prompts_embeds: Optional[torch.Tensor],
        tokenized_prompts: Optional[torch.Tensor],
        text_features: Optional[torch.Tensor],
        require_grad: bool,
    ) -> torch.Tensor:
        if text_features is not None:
            feats = torch.as_tensor(text_features).to(self.device)
            feats = F.normalize(feats, dim=-1)
            if feats.dim() == 3:
                feats = feats[0]
            return feats

        if prompts_embeds is None or tokenized_prompts is None:
            raise ValueError("Provide either text_features or (prompts_embeds, tokenized_prompts).")
        if self.text_encoder is None:
            raise RuntimeError(
                "ClipClassifierDetector was not initialized with a text_encoder, "
                "cannot encode prompt embeddings."
            )

        if require_grad:
            txt = self.text_encoder(prompts_embeds, tokenized_prompts)
        else:
            with torch.no_grad():
                txt = self.text_encoder(prompts_embeds, tokenized_prompts)
        txt = F.normalize(txt.to(self.device), dim=-1)
        if txt.dim() == 3:
            txt = txt[0]
        return txt


class ConvNextDetector:
    """Detector wrapper for ConvNeXt-based classifiers trained within EcoGrow."""

    def __init__(
        self,
        classes: Sequence[str],
        *,
        model_name: str = "convnext_small",
        pretrained: bool = True,
        device: torch.device | str = "cpu",
        preprocess=None,
        train_backbone: bool = False,
        drop_rate: float = 0.0,
        **model_kwargs,
    ) -> None:
        self.classes = list(classes)
        self.device = torch.device(device)
        self.preprocess = preprocess
        self.train_backbone = bool(train_backbone)

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=len(self.classes),
            drop_rate=drop_rate,
            **model_kwargs,
        )
        self.model.to(self.device)

        if not self.train_backbone:
            self._freeze_backbone()

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.model.parameters()
    
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        self.model.load_state_dict(state_dict, strict=strict)

    def logits(self, images: torch.Tensor, *, require_grad: bool = False) -> torch.Tensor:
        images = images.to(self.device)
        if require_grad:
            self.model.train(self.train_backbone)
            return self.model(images)
        self.model.eval()
        with torch.no_grad():
            return self.model(images)

    def predict(self, tensor: torch.Tensor, *, unknown_threshold: float = 0.0) -> Dict[str, object]:
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
            "detector": self.name,
            "family": self.name,
            "prediction": label,
            "raw_label": raw_label,
            "probability": probability,
            "classes": per_class,
        }

    predict_batch = predict

    def _freeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        classifier = self._get_classifier()
        for p in classifier.parameters():
            p.requires_grad_(True)

    def _get_classifier(self) -> nn.Module:
        classifier = self.model.get_classifier()
        if isinstance(classifier, nn.Module):
            return classifier
        # Some timm models return tensors; fall back to searching common attributes
        if hasattr(self.model, "head") and isinstance(self.model.head, nn.Module):
            return self.model.head
        raise RuntimeError("Unable to locate ConvNeXt classifier head.")
