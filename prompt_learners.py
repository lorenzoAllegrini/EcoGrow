# prompt_learning_min.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

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
        pos = self.positional_embedding[:L, :].to(x.dtype)  # [L, D]
        x = x + pos.unsqueeze(0)                            # [C, L, D]
        x = x.permute(1, 0, 2)                              # [L, C, D]
        # open_clip transformer accetta attn_mask registrata internamente
        x = self.transformer(x)                             # [L, C, D]
        x = x.permute(1, 0, 2)                              # [C, L, D]
        x = self.ln_final(x)                                # [C, L, D]

        # posizione EOT per ogni classe (heuristic standard: argmax dell’ID > 0)
        # tokenized_prompts: [C, L]
        eot_idx = tokenized_prompts.argmax(dim=-1)          # [C]
        feats = x[torch.arange(C), eot_idx]                 # [C, D]
        feats = feats @ self.text_projection.T              # [C, D_out]
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


class PromptLearnerOpenCLIP(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="a photo of a", 
                 class_token_position="end", model_name="ViT-B-32"):
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

        # init context tokens
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx_from_text = len(ctx_init.split(" "))
            if n_ctx_from_text != n_ctx:
                # allinea automaticamente ai token contati dal tokenizer
                tokens = self._tokenizer([ctx_init])
                n_ctx_from_tok = (tokens != 0).sum().item() - 2  # -2 per SOT/EOT
                self.n_ctx = n_ctx_from_tok
            tokens = self._tokenizer([ctx_init])
            with torch.no_grad():
                emb_full = clip_model.token_embedding(tokens).type(dtype)  # [1,L,D]
            # prendi i token dopo SOT per n_ctx posizioni
            self.ctx = nn.Parameter(emb_full[0, 1:1+self.n_ctx, :])  # [n_ctx, D]
            prompt_prefix = ctx_init
        else:
            self.ctx = nn.Parameter(torch.empty(self.n_ctx, ctx_dim, dtype=dtype))
            nn.init.normal_(self.ctx, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)

        # tokenizzazione dei nomi e prompt di riferimento (solo per prefix/suffix)
        prompts_txt = [f"{prompt_prefix} {name}." for name in self.classnames]
        self.tokenized_prompts = torch.cat([self._tokenizer([p]) for p in prompts_txt], dim=0)  # [C,L]
        with torch.no_grad():
            emb_full = clip_model.token_embedding(self.tokenized_prompts).type(dtype)           # [C,L,D]

        # salva SOS come prefix, e tutto dopo i n_ctx come suffix (inclusi class tokens + EOS)
        self.register_buffer("token_prefix", emb_full[:, :1, :])             # [C,1,D]
        self.register_buffer("token_suffix", emb_full[:, 1+self.n_ctx:, :])  # [C,*,D]

        # lunghezze del nome (per “middle” e “front”)
        self.name_lens = [len(self._tokenizer.encode(n)) for n in self.classnames]

    def forward(self, img_features=None):
        # (per CoCoOp useresti img_features per condizionare self.ctx; qui è CoOp “semplice”)
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)   # [C, n_ctx, D]

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

        return prompts, self.tokenized_prompts
