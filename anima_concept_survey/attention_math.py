from __future__ import annotations

from collections.abc import Sequence

import torch


def compute_attention_logits(q: torch.Tensor, k: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)


def compute_attention_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def attention_output_from_probs(probs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.matmul(probs.to(v.dtype), v)


def apply_token_attention_scale(
    probs: torch.Tensor,
    token_indices: Sequence[int],
    scale: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    if scale < 0:
        raise ValueError("token attention scale must be non-negative")
    modified = probs.clone()
    if not token_indices:
        return modified
    modified[..., list(token_indices)] *= float(scale)
    normalizer = modified.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return modified / normalizer


def apply_attention_logit_bias(
    logits: torch.Tensor,
    token_indices: Sequence[int],
    bias: float,
) -> torch.Tensor:
    modified = logits.clone()
    if not token_indices:
        return modified
    modified[..., list(token_indices)] += float(bias)
    return modified
