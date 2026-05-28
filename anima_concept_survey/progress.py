from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ProgressInfo:
    index: int
    num_steps: int
    last_index: int
    progress: float
    sigma: float


def progress_from_sigmas(transformer_options: dict[str, Any]) -> ProgressInfo | None:
    sigmas = transformer_options.get("sigmas")
    sample_sigmas = transformer_options.get("sample_sigmas")
    if sigmas is None or sample_sigmas is None:
        return None
    if not torch.is_tensor(sigmas) or not torch.is_tensor(sample_sigmas):
        return None
    if sigmas.numel() == 0 or sample_sigmas.numel() < 2:
        return None

    sigma = sigmas.detach().float().flatten()[0].to(device=sample_sigmas.device)
    candidates = sample_sigmas.detach().float().flatten()
    matches = torch.where(torch.isclose(candidates, sigma, rtol=1e-4, atol=1e-5))[0]
    if matches.numel() > 0:
        index = int(matches[0].item())
    else:
        index = int(torch.argmin((candidates - sigma).abs()).item())
        if torch.abs(candidates[index] - sigma) > max(1e-5, float(torch.abs(sigma).item()) * 1e-4):
            return None

    num_steps = max(int(candidates.numel()) - 1, 1)
    last_index = max(num_steps - 1, 1)
    index = min(index, num_steps - 1)
    progress = min(max(index / last_index, 0.0), 1.0)
    return ProgressInfo(
        index=index,
        num_steps=num_steps,
        last_index=last_index,
        progress=progress,
        sigma=float(sigma.item()),
    )
