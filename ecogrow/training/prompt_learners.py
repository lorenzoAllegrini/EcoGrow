# prompt_learning_min.py
from typing import Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptLearnerOpenCLIP(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        n_ctx=16,
        ctx_init="a photo of a",
        class_token_position="end",
        model_name="ViT-B-32",
        ctx_vectors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.classnames = [c.replace("_", " ") for c in classnames]
        self.class_token_position = class_token_position
        self.n_cls = len(self.classnames)
        self.n_ctx = n_ctx

        # dimensioni dal text branch
        dtype = next(clip_model.parameters()).dtype
        ctx_dim = clip_model.text_projection.shape[1]  # transformer width
        self.dtype = dtype

        # tokenizer open_clip per contare i token del nome
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self.clip_model = clip_model
        # init context tokens
        embedding_device = clip_model.token_embedding.weight.device
        if ctx_vectors is not None:
            if not isinstance(ctx_vectors, torch.Tensor):
                raise TypeError("ctx_vectors must be a torch.Tensor")
            if ctx_vectors.dim() != 2:
                raise ValueError("ctx_vectors must have shape [n_ctx, ctx_dim]")
            if ctx_vectors.size(1) != ctx_dim:
                raise ValueError(f"ctx_vectors second dimension must be {ctx_dim}")
            self.n_ctx = ctx_vectors.size(0)
            ctx_prepared = ctx_vectors.to(device=embedding_device, dtype=dtype)
            self.ctx = nn.Parameter(ctx_prepared.clone())
            prompt_prefix = ctx_init or " ".join(["X"] * self.n_ctx)
        elif ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx_from_text = len(ctx_init.split(" "))
            if n_ctx_from_text != n_ctx:
                # allinea automaticamente ai token contati dal tokenizer
                tokens = self._tokenizer([ctx_init])
                n_ctx_from_tok = (tokens != 0).sum().item() - 2  # -2 per SOT/EOT
                self.n_ctx = n_ctx_from_tok
            tokens = self._tokenizer([ctx_init])
            with torch.no_grad():
                emb_full = self.clip_model.token_embedding(tokens).type(dtype)  # [1,L,D]
            # prendi i token dopo SOT per n_ctx posizioni
            self.ctx = nn.Parameter(emb_full[0, 1:1+self.n_ctx, :])  # [n_ctx, D]
            prompt_prefix = ctx_init
        else:
            self.ctx = nn.Parameter(torch.empty(self.n_ctx, ctx_dim, dtype=dtype))
            nn.init.normal_(self.ctx, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)

        # tokenizzazione dei nomi e prompt di riferimento (solo per prefix/suffix)
        prompts_txt = [f"{prompt_prefix} {name}." for name in self.classnames]
        tokenized = torch.cat([self._tokenizer([p]) for p in prompts_txt], dim=0).long()  # [C,L]
        self.register_buffer("tokenized_prompts", tokenized, persistent=False)
        with torch.no_grad():
            emb_full = self.clip_model.token_embedding(
                self.tokenized_prompts.to(embedding_device)
            ).type(dtype)  # [C,L,D]

        # salva SOS come prefix, e tutto dopo i n_ctx come suffix (inclusi class tokens + EOS)
        self.register_buffer("token_prefix", emb_full[:, :1, :])             # [C,1,D]
        self.register_buffer("token_suffix", emb_full[:, 1+self.n_ctx:, :])  # [C,*,D]

        # lunghezze del nome (per “middle” e “front”)
        self.name_lens = [len(self._tokenizer.encode(n)) for n in self.classnames]
        self.ctx.requires_grad_(True)

    def forward(self, img_features=None):
        # (per CoCoOp useresti img_features per condizionare self.ctx; qui è CoOp “semplice”)
        ctx = self.ctx
        if ctx.dim() == 2:
            # repeat evita stride a zero di expand che in PyTorch 2.9 può annullare i gradienti
            ctx = ctx.unsqueeze(0).repeat(self.n_cls, 1, 1)     # [C, n_ctx, D]

        prefix = self.token_prefix                               # [C,1,D]
        suffix = self.token_suffix                               # [C,*,D]

        if self.class_token_position == "end":
            # [C, 1+n_ctx+*, D]
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half = self.n_ctx // 2
            chunks = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1]               # [1,1,D]
                class_i  = suffix[i:i+1, :name_len]    # [1,L,D]
                suffix_i = suffix[i:i+1, name_len:]    # [1,*,D]
                ctx1 = ctx[i:i+1, :half]               # [1, n_ctx//2, D]
                ctx2 = ctx[i:i+1, half:]               # [1, n_ctx - half, D]
                chunks.append(torch.cat([prefix_i, ctx1, class_i, ctx2, suffix_i], dim=1))
            prompts = torch.cat(chunks, dim=0)         # [C, L, D]

        elif self.class_token_position == "front":
            chunks = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1]
                class_i  = suffix[i:i+1, :name_len]
                suffix_i = suffix[i:i+1, name_len:]
                ctx_i    = ctx[i:i+1]
                chunks.append(torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1))
            prompts = torch.cat(chunks, dim=0)

        else:
            raise ValueError("class_token_position must be one of {'end','middle','front'}")

        tokenized = self.tokenized_prompts.to(prompts.device)
        return prompts, tokenized
    
        
