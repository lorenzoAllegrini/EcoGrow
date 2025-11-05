"""Utilities for experiment scripts."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

from ecogrow.models.open_clip_wrapper import OpenClipWrapper
from ecogrow.training.prompt_learners import PromptLearnerOpenCLIP


def _compute_ctx_from_prompts(
    n_ctx: int,
    clip_model,
    tokenizer,
    prompts_per_class: Sequence[Iterable[str]],
    alpha_seed: float,
    seed_text: str,
) -> torch.Tensor:
    device = next(clip_model.parameters()).device
    dtype = next(clip_model.parameters()).dtype

    flat: List[str] = []
    slices = []
    start = 0
    for prompt_list in prompts_per_class:
        cleaned = [
            prompt if prompt.strip().endswith(".") else f"{prompt.strip()}."
            for prompt in prompt_list
        ]
        flat.extend(cleaned)
        end = start + len(cleaned)
        slices.append((start, end))
        start = end

    if not flat:
        raise ValueError("prompts_per_class must contain at least one prompt.")

    with torch.no_grad():
        tokens = torch.cat([tokenizer([prompt]) for prompt in flat], dim=0).to(device)
        embeddings = clip_model.token_embedding(tokens).type(dtype)
        ctx_chunks = embeddings[:, 1 : 1 + n_ctx, :]

    per_class_ctx = [ctx_chunks[s:e].mean(dim=0) for s, e in slices]
    ctx_init = torch.stack(per_class_ctx, dim=0).mean(dim=0)

    if alpha_seed > 0.0:
        seed_tokens = tokenizer([seed_text]).to(device)
        with torch.no_grad():
            seed_embeddings = clip_model.token_embedding(seed_tokens).type(dtype)[
                0, 1 : 1 + n_ctx, :
            ]
        ctx_init = alpha_seed * seed_embeddings + (1 - alpha_seed) * ctx_init

    return ctx_init


def build_prompt_learner(
    classnames: Sequence[str],
    clip_wrapper: OpenClipWrapper,
    device: torch.device,
    prompts_per_class: Sequence[Iterable[str]],
    *,
    model_name: str = "ViT-B-32",
    n_ctx: int = 16,
    ctx_init: str | None = None,
    class_token_position: str = "end",
    alpha_seed: float = 0.0,
    seed_text: str = "a photo of a",
) -> PromptLearnerOpenCLIP:
    """Create a PromptLearnerOpenCLIP initialised from textual prompts."""

    if not classnames:
        raise ValueError("classnames must contain at least one label.")

    clip_model = clip_wrapper.model
    tokenizer = clip_wrapper.tokenizer

    ctx_tensor = _compute_ctx_from_prompts(
        n_ctx,
        clip_model,
        tokenizer,
        prompts_per_class,
        alpha_seed=alpha_seed,
        seed_text=seed_text,
    )

    learner = PromptLearnerOpenCLIP(
        classnames=classnames,
        clip_model=clip_model,
        n_ctx=n_ctx,
        ctx_init=ctx_init or seed_text,
        class_token_position=class_token_position,
        model_name=model_name,
        ctx_vectors=ctx_tensor,
    ).to(device)

    for _, param in learner.named_parameters():
        param.requires_grad = True

    return learner


__all__ = ["build_prompt_learner"]
