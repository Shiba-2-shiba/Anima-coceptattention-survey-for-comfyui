from __future__ import annotations

import torch


def selected_branch_indices(batch: int, cond_or_uncond: list[int] | None, branch_mode: str, device: torch.device) -> torch.Tensor:
    if branch_mode == "both":
        return torch.arange(batch, device=device)
    if not cond_or_uncond:
        return torch.arange(0, 0, dtype=torch.long, device=device)
    chunks = len(cond_or_uncond)
    if chunks <= 0 or batch % chunks != 0:
        return torch.arange(0, 0, dtype=torch.long, device=device)
    per_chunk = batch // chunks
    selected: list[int] = []
    for chunk_index, branch in enumerate(cond_or_uncond):
        if branch_mode == "positive_only" and branch != 0:
            continue
        if branch_mode == "negative_only" and branch != 1:
            continue
        selected.extend(range(chunk_index * per_chunk, (chunk_index + 1) * per_chunk))
    return torch.tensor(selected, dtype=torch.long, device=device)


def branch_index_groups(batch: int, cond_or_uncond: list[int] | None, selected: torch.Tensor) -> list[tuple[str, list[int]]]:
    selected_set = set(int(index) for index in selected.detach().cpu().tolist())
    if not cond_or_uncond:
        return [("all", sorted(selected_set))]
    chunks = len(cond_or_uncond)
    if chunks <= 0 or batch % chunks != 0:
        return [("all", sorted(selected_set))]
    per_chunk = batch // chunks
    groups: list[tuple[str, list[int]]] = []
    for chunk_index, branch_value in enumerate(cond_or_uncond):
        label = "positive" if branch_value == 0 else "negative" if branch_value == 1 else f"branch_{branch_value}"
        indices = [
            index
            for index in range(chunk_index * per_chunk, (chunk_index + 1) * per_chunk)
            if index in selected_set
        ]
        if indices:
            groups.append((label, indices))
    return groups or [("all", sorted(selected_set))]
