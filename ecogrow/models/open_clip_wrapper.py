from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class OpenClipWrapper:
    def __init__(   
        self,
        model_name: str = "ViT-B-32",
        pretrained_tag: str = "laion2b_s34b_b79k",
        device: str = "cpu"
    ) -> None:
        
        self.model_name = model_name
        self.pretrained_tag = pretrained_tag
        self.device = device
  
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                                self.model_name,
                                pretrained=self.pretrained_tag)
            
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.float()
        self.model = self.model.to(self.device).eval()
        self.text_encoder = TextEncoderOpenCLIP(self.model).float().to(device).eval()

    def freeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)

    def encode_texts(self, prompts: Sequence[str], normalize: bool = True) -> torch.Tensor:
        if not isinstance(prompts, (list, tuple)):
            raise TypeError("prompts must be a sequence of strings.")
        if not prompts:
            raise ValueError("prompts must contain at least one element.")
        assert all(isinstance(p, str) for p in prompts), "prompts must be strings."

        with torch.no_grad():
            tokens = self.tokenizer(list(prompts)).to(self.device)
            text_features = self.model.encode_text(tokens)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def predict_image(
        self,
        img: Image.Image,
        class_names: Sequence[str],
        text_embeddings: torch.Tensor,
        *,
        temperature: float = 0.01,
        unknown_threshold: float = 0.5,
        segment_fn=None,
    ) -> Tuple[str, float, torch.Tensor]:
        if segment_fn is not None:
            img = segment_fn(img)
        else:
            img = img.convert("RGB")

        image_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1)

        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.mean(dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        logits = image_features @ text_embeddings.t()
        probs = (logits / temperature).softmax(dim=-1).squeeze(0)

        max_prob, idx = probs.max(dim=0)
        pred_class = class_names[idx.item()]
        if max_prob.item() < unknown_threshold:
            pred_class = "unknown"

        return pred_class, float(max_prob.item()), probs.detach().cpu()



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
        self.text_projection = clip_model.text_projection           # [D_out, D]
        self.dtype = next(clip_model.parameters()).dtype

    def _encode(self, x, tokenized_prompts):
        # x: [C, L, D]
        C, L, D = x.shape
        pos = self.positional_embedding[:L, :].to(x.dtype)
        x = x + pos.unsqueeze(0)
        x = self.transformer(x)           # [L,C,D] se il tuo transformer vuole [L,C,D], altrimenti regola i permute
        x = x.permute(1, 0, 2) if x.shape[0] == L else x  # mantieni [C,L,D]
        x = self.ln_final(x)

        # --- EOT SHIFT FIX ---
        # L0 = lunghezza dei token originali (senza ctx)
        L0 = tokenized_prompts.shape[-1]
        shift = L - L0                     # = n_ctx, in tutti i posizionamenti ('end','middle','front')
        eot_idx = tokenized_prompts.argmax(dim=-1) + shift  # [C]

        feats = x[torch.arange(C), eot_idx]   # [C, D]
        feats = feats @ self.text_projection.T
        return feats

    def forward(self, prompts_embeds, tokenized_prompts):
        if prompts_embeds.dim() == 3:                       # [C, L, D]
            return self._encode(prompts_embeds, tokenized_prompts)
        elif prompts_embeds.dim() == 4:                     # [B, C, L, D]
            B, C, L, D = prompts_embeds.shape
            outs = []
            for b in range(B):
                o = self._encode(prompts_embeds[b], tokenized_prompts)  # [C, D_out]
                outs.append(o)
            return torch.stack(outs, dim=0)                 # [B, C, D_out]
        else:
            raise ValueError("prompts_embeds must be [C,L,D] or [B,C,L,D]")
