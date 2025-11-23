"""Utilities for experiment scripts."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Dict, Callable
from pathlib import Path
import torch
import torch.nn.functional as F
from disease_detection.models.model_wrappers import TextEncoderOpenCLIP
from disease_detection.training.prompt_learners import ClipPromptLearner

@torch.no_grad()
def compute_init_ctx(
    n_ctx: int,
    text_encoder: torch.nn.Module,
    tokenizer: Callable[[Sequence[str]], torch.Tensor],
    class_prompts: Sequence[str],
    alpha_seed: float = 0.0,
    seed_text: str = "a photo of a",
) -> torch.Tensor:
    """
    Crea il contesto iniziale [n_ctx, D] per il Prompt Learner di CLIP.

    Converte un prompt per classe in token embedding, prende i primi n_ctx token
    dopo SOT e ne fa la media tra tutte le classi. Facoltativamente fonde il risultato
    con il prompt seed ("a photo of a") pesato da alpha_seed.

    Args:
        n_ctx: numero di token di contesto da usare.
        clip_model: modello CLIP già caricato.
        tokenizer: tokenizer compatibile con CLIP.
        class_prompts: lista di prompt, uno per classe.
        alpha_seed: peso di blending con il seed_text (0 = disattivo).
        seed_text: prompt di riferimento per il blending.

    Returns:
        torch.Tensor: tensore [n_ctx, D] con il contesto iniziale.
    """
    device = text_encoder.device
    dtype = text_encoder.dtype

    prompts = [(p.strip() if p.strip().endswith(".") else p.strip() + ".") for p in class_prompts]
    if not prompts:
        raise ValueError("Provide at least one prompt (one per class).")

    tokens = torch.cat([tokenizer([p]) for p in prompts], dim=0).to(device)         # [C,L]
    emb    = text_encoder.token_embedding(tokens).type(dtype)                          # [C,L,D]
    ctx    = emb[:, 1:1+n_ctx, :].mean(dim=0)                                        # [n_ctx,D]

    if alpha_seed > 0.0:
        seed_tok = tokenizer([seed_text]).to(device)
        seed_emb = text_encoder.token_embedding(seed_tok).type(dtype)[0, 1:1+n_ctx, :] # [n_ctx,D]
        ctx = alpha_seed * seed_emb + (1.0 - alpha_seed) * ctx

    return ctx  


@torch.no_grad()
def compute_class_ctx(
    n_ctx: int,
    clip_model: torch.nn.Module,
    tokenizer: Callable[[Sequence[str]], torch.Tensor],
    prompts_per_class: Sequence[Sequence[str]],
    alpha_seed: float = 0.0,
    seed_text: str = "a photo of a",
) -> torch.Tensor:
    """
    Costruisce un contesto per-classe di shape [C, n_ctx, D] mediando i token
    di contesto (dopo SOT) dei prompt forniti per ciascuna classe.

    prompts_per_class: lista di liste; ogni elemento è l'elenco di prompt testuali
                       associati alla classe corrispondente.
    """
    device = next(clip_model.parameters()).device
    dtype = next(clip_model.parameters()).dtype

    class_ctx = []
    for cls_prompts in prompts_per_class:
        prompts = [p.strip() + "." if not p.strip().endswith(".") else p.strip() for p in cls_prompts]
        if not prompts:
            raise ValueError("Ogni classe deve avere almeno un prompt.")
        tokens = torch.cat([tokenizer([p]) for p in prompts], dim=0).to(device)     # [N,L]
        emb = clip_model.token_embedding(tokens).type(dtype)                         # [N,L,D]
        ctx_i = emb[:, 1:1+n_ctx, :].mean(dim=0)                                     # [n_ctx,D]

        if alpha_seed > 0.0:
            seed_tok = tokenizer([seed_text]).to(device)
            seed_emb = clip_model.token_embedding(seed_tok).type(dtype)[0, 1:1+n_ctx, :]
            ctx_i = alpha_seed * seed_emb + (1.0 - alpha_seed) * ctx_i

        class_ctx.append(ctx_i)

    ctx_stack = torch.stack(class_ctx, dim=0)                                        # [C,n_ctx,D]
    return ctx_stack


def export_detector_embeddings(
    out_path: Path,
    detector_name: str,
    class_order: Iterable[str],
    prompt_learner: ClipPromptLearner,
    text_encoder: TextEncoderOpenCLIP,
    temperature: float,
) -> Dict:
    with torch.no_grad():
        prompts_embeds, tokenized_prompts = prompt_learner()
        text_features = text_encoder(prompts_embeds, tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)

    payload = {
        "detector": detector_name,
        "classes": list(class_order),
        "text_features": text_features.cpu(),
        "temperature": float(temperature),
    }
    torch.save(payload, out_path)
    return payload

def list_leaf_modules(model):
    leaves = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            leaves.append((name, module.__class__.__name__))
    
    return leaves


def save_lora_adapter(
    model: torch.nn.Module,
    lora_config,
    out_dir: Path,
    adapter_name: str = "default",
) -> Path:
    """Persist only LoRA adapter weights and config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lora_" in k
    }

    try:
        config_dict = lora_config.to_dict()  # type: ignore[attr-defined]
    except AttributeError:
        config_dict = dict(lora_config.__dict__)

    payload = {
        "peft_type": "LORA",
        "adapter_name": adapter_name,
        "config": config_dict,
        "state_dict": state,
    }

    path = out_dir / f"{adapter_name}.pt"
    torch.save(payload, path)
    return path



__all__ = [
    "export_detector_embeddings",
    "compute_init_ctx",
    "compute_class_ctx",
    "list_leaf_modules",
    "save_lora_adapter",
]
