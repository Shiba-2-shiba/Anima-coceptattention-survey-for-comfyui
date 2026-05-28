from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import math

import torch

from .concepts import ConceptTokenMatch


def normalized_entropy(values: torch.Tensor, dim: int, eps: float = 1e-12) -> torch.Tensor:
    probs = values.clamp_min(eps)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=dim)
    denom = math.log(max(int(values.shape[dim]), 2))
    return entropy / denom


def token_scores_from_attention(attention_probs: torch.Tensor, max_tokens: int) -> list[dict[str, Any]]:
    # attention_probs: batch, heads, image_query, text_key
    token_mean = attention_probs.mean(dim=(0, 1, 2))
    token_max = attention_probs.amax(dim=(0, 1, 2))
    spatial_mean = attention_probs.mean(dim=(0, 1)).transpose(0, 1)
    token_entropy = normalized_entropy(spatial_mean, dim=1)
    k = min(max_tokens, int(token_mean.numel()))
    top_values, top_indices = torch.topk(token_mean, k=k)
    records: list[dict[str, Any]] = []
    for rank, (value, token_index) in enumerate(zip(top_values, top_indices, strict=False), start=1):
        idx = int(token_index.detach().cpu().item())
        records.append({
            "rank": rank,
            "token_index": idx,
            "score_mean": float(value.detach().cpu().item()),
            "score_max": float(token_max[idx].detach().cpu().item()),
            "score_entropy": float(token_entropy[idx].detach().cpu().item()),
        })
    return records


def concept_scores_from_attention(
    attention_probs: torch.Tensor,
    spatial: tuple[int, int],
    matches: Sequence[ConceptTokenMatch],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text_len = int(attention_probs.shape[-1])
    for match in matches:
        indices = list(match.token_indices)
        if not indices or any(index < 0 or index >= text_len for index in indices):
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=attention_probs.device)
        concept_attention = attention_probs.index_select(-1, index_tensor).sum(dim=-1)
        heatmap = concept_attention.mean(dim=(0, 1)).reshape(spatial)
        entropy = normalized_entropy(heatmap.flatten(), dim=0)
        rows.append({
            "term": match.term,
            "token_indices": indices,
            "source_token_indices": list(match.source_token_indices),
            "token_texts": list(match.token_texts),
            "token_sources": [match.token_source for _ in match.token_indices],
            "token_ids": list(match.token_ids),
            "ignored_token_indices": list(match.ignored_token_indices),
            "occurrence_index": match.occurrence_index,
            "match_warnings": list(match.match_warnings),
            "score_mean": float(concept_attention.mean().detach().cpu().item()),
            "score_max": float(concept_attention.amax().detach().cpu().item()),
            "score_entropy": float(entropy.detach().cpu().item()),
            "_heatmap": heatmap.detach().cpu().to(torch.float32),
        })
    return rows
