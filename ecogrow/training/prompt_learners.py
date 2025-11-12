# prompt_learning_min.py
from typing import Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipPromptLearner(nn.Module):
    """Minimal prompt learner for CLIP (end placement only).

    - Requires precomputed context vectors `ctx_vectors` of shape [n_ctx, D].
    - Always places the learnable context right after SOS and before class tokens.
    - Builds tokenized prompts using a simple placeholder of `n_ctx` tokens to
      align shapes; the actual embeddings for those tokens come from `ctx_vectors`.
    """

    def __init__(
        self,
        classnames,
        text_encoder,
        *,
        ctx_vectors: torch.Tensor,
        model_name: str = "ViT-B-32",
    ):
        super().__init__()
        self.classnames = [c.replace("_", " ") for c in classnames]
        self.n_cls = len(self.classnames)

        dtype = text_encoder.dtype
        ctx_dim = text_encoder.text_projection.shape[1]
        self.text_encoder = text_encoder
        self._tokenizer = open_clip.get_tokenizer(model_name)

        if not isinstance(ctx_vectors, torch.Tensor) or ctx_vectors.dim() != 2:
            raise ValueError("ctx_vectors must be a 2D torch.Tensor [n_ctx, D]")
        if ctx_vectors.size(1) != ctx_dim:
            raise ValueError(f"ctx_vectors second dimension must be {ctx_dim}")

        self.n_ctx = ctx_vectors.size(0)
        embedding_device = text_encoder.token_embedding.weight.device
        self.ctx = nn.Parameter(ctx_vectors.to(embedding_device, dtype=dtype).clone())
        self.ctx.requires_grad_(True)

        placeholder = " ".join(["X"] * self.n_ctx)
        prompts_txt = [f"{placeholder} {name}." for name in self.classnames]
        tokenized = torch.cat([self._tokenizer([p]) for p in prompts_txt], dim=0).long()
        self.register_buffer("tokenized_prompts", tokenized, persistent=False)

        with torch.no_grad():
            emb_full = self.text_encoder.token_embedding(self.tokenized_prompts.to(embedding_device)).type(dtype)

        # Keep SOS as prefix and everything after the n_ctx placeholders as suffix
        self.register_buffer("token_prefix", emb_full[:, :1, :])            # [C,1,D]
        self.register_buffer("token_suffix", emb_full[:, 1 + self.n_ctx :, :])  # [C,*,D]

    def forward(self, img_features=None):  
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).repeat(self.n_cls, 1, 1)  # [C, n_ctx, D]

        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)  # [C,L,D]
        tokenized = self.tokenized_prompts.to(prompts.device)
        return prompts, tokenized
    
        
