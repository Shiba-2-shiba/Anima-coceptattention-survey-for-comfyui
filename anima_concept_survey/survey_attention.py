from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import torch

from .token_text import build_token_text_map


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[AnimaConceptSurvey]"

MODES = ["observe", "off"]
CAPTURE_LEVELS = ["summary", "tokens", "heatmap"]
BRANCH_MODES = ["both", "positive_only", "negative_only"]
FAIL_MODES = ["fallback", "raise"]


@dataclass(frozen=True)
class SurveyConfig:
    mode: str = "observe"
    capture_level: str = "tokens"
    target_call_indices: str = "all"
    diagnostic_call_indices: str = "all"
    branch_mode: str = "both"
    max_tokens: int = 16
    max_steps: int = 0
    jsonl_path: str | None = None
    save_heatmaps: bool = False
    heatmap_dir: str | None = None
    max_logits_mib: float = 1024.0
    fail_mode: str = "fallback"
    prompt_text: str = ""
    token_text_map: dict[int, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unsupported survey mode: {self.mode!r}")
        if self.capture_level not in CAPTURE_LEVELS:
            raise ValueError(f"Unsupported capture_level: {self.capture_level!r}")
        if self.branch_mode not in BRANCH_MODES:
            raise ValueError(f"Unsupported branch_mode: {self.branch_mode!r}")
        if self.fail_mode not in FAIL_MODES:
            raise ValueError(f"Unsupported fail_mode: {self.fail_mode!r}")
        parse_call_index_scope(self.target_call_indices)
        parse_call_index_scope(self.diagnostic_call_indices)
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_steps < 0:
            raise ValueError("max_steps must be zero or positive")
        if self.max_logits_mib <= 0:
            raise ValueError("max_logits_mib must be positive")
        if self.save_heatmaps and not self.heatmap_dir:
            raise ValueError("heatmap_dir is required when save_heatmaps is true")


@dataclass(frozen=True)
class ProgressInfo:
    index: int
    num_steps: int
    last_index: int
    progress: float
    sigma: float


@dataclass
class StepStats:
    index: int
    num_steps: int
    total_calls: int = 0
    eligible_calls: int = 0
    observed_calls: int = 0
    skipped_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    shape_counts: Counter[str] = field(default_factory=Counter)
    eligible_call_indices: Counter[int] = field(default_factory=Counter)


@dataclass
class SurveyStats:
    observed_calls: int = 0
    skipped_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    steps: dict[int, StepStats] = field(default_factory=dict)

    def step_for(self, progress: ProgressInfo) -> StepStats:
        step = self.steps.get(progress.index)
        if step is None:
            step = StepStats(index=progress.index, num_steps=progress.num_steps)
            self.steps[progress.index] = step
        return step


class SurveyFallback(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def parse_call_index_scope(spec: str) -> set[int] | None:
    normalized = str(spec).strip().lower()
    if normalized == "all":
        return None
    if not normalized:
        raise ValueError("call index scope must be 'all' or a comma-separated list of non-negative integers/ranges")

    indices: set[int] = set()
    for raw_part in normalized.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"Invalid call index scope entry in {spec!r}")
        if "-" in part:
            start_text, _, end_text = part.partition("-")
            if not start_text or not end_text:
                raise ValueError(f"Invalid call index range: {part!r}")
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(f"Invalid call index range: {part!r}") from exc
            if start < 0 or end < 0:
                raise ValueError("call index scope entries must be non-negative")
            if end < start:
                raise ValueError(f"Invalid descending call index range: {part!r}")
            indices.update(range(start, end + 1))
        else:
            try:
                index = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid call index scope entry: {part!r}") from exc
            if index < 0:
                raise ValueError("call index scope entries must be non-negative")
            indices.add(index)
    return indices


def infer_square_spatial_shape(query_len: int) -> tuple[int, int] | None:
    side = math.isqrt(int(query_len))
    if side * side != query_len:
        return None
    return side, side


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


def estimate_logits_mib(batch: int, heads: int, query_len: int, text_len: int, bytes_per: int = 4) -> float:
    return batch * heads * query_len * text_len * bytes_per / (1024**2)


def shape_key(q: torch.Tensor, k: torch.Tensor) -> str:
    return f"q{int(q.shape[-2])},k{int(k.shape[-2])},h{int(q.shape[1])},d{int(q.shape[-1])}"


def _safe_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if torch.is_tensor(value):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, tuple | list):
        if all(item is None or isinstance(item, str | int | float | bool) for item in value):
            return list(value)
        return {"type": type(value).__name__, "len": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "count": len(value), "keys": [str(key) for key in list(value.keys())[:16]]}
    return str(value)


def discover_transformer_metadata(transformer_options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    keys = ("block", "block_index", "transformer_index", "module_path", "patches_replace")
    metadata = {key: _safe_metadata_value(transformer_options[key]) for key in keys if key in transformer_options}
    block = metadata.get("block")
    if isinstance(block, list) and block:
        block_id = ":".join(str(part) for part in block)
    elif "block_index" in metadata:
        block_id = str(metadata["block_index"])
    elif "transformer_index" in metadata:
        block_id = str(metadata["transformer_index"])
    elif "module_path" in metadata:
        block_id = str(metadata["module_path"])
    else:
        block_id = "unknown"
    return block_id, metadata


def is_anima_like_model(model: Any) -> bool:
    inner = getattr(model, "model", model)
    diffusion_model = getattr(inner, "diffusion_model", inner)
    if diffusion_model.__class__.__name__ == "Anima":
        return True
    return hasattr(diffusion_model, "llm_adapter") and hasattr(diffusion_model, "blocks")


def _normalized_entropy(values: torch.Tensor, dim: int, eps: float = 1e-12) -> torch.Tensor:
    probs = values.clamp_min(eps)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=dim)
    denom = math.log(max(int(values.shape[dim]), 2))
    return entropy / denom


def _token_scores_from_attention(attention_probs: torch.Tensor, max_tokens: int) -> list[dict[str, Any]]:
    # attention_probs: batch, heads, image_query, text_key
    token_mean = attention_probs.mean(dim=(0, 1, 2))
    token_max = attention_probs.amax(dim=(0, 1, 2))
    spatial_mean = attention_probs.mean(dim=(0, 1)).transpose(0, 1)
    token_entropy = _normalized_entropy(spatial_mean, dim=1)
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


def _heatmap_for_token(attention_probs: torch.Tensor, token_index: int, spatial: tuple[int, int]) -> torch.Tensor:
    # Mean over batch/head, then reshape image queries to spatial grid.
    heatmap = attention_probs[:, :, :, token_index].mean(dim=(0, 1))
    return heatmap.reshape(spatial)


class AnimaConceptSurveyAttentionOverride:
    def __init__(self, config: SurveyConfig, clip: Any | None = None):
        config.validate()
        self.config = config
        self.stats = SurveyStats()
        self.run_id = f"survey-{id(self):x}"
        self._target_call_indices = parse_call_index_scope(config.target_call_indices)
        self._diagnostic_call_indices = parse_call_index_scope(config.diagnostic_call_indices)
        self.token_text_map = dict(config.token_text_map)
        if not self.token_text_map and config.prompt_text:
            self.token_text_map = build_token_text_map(clip, config.prompt_text)

    def finalize(self) -> None:
        self._emit_jsonl(self._run_summary_record())

    def __call__(self, original_func: Callable, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.config.mode == "off":
            return original_func(*args, **kwargs)
        progress = self._progress_from_kwargs(kwargs)
        try:
            self._observe(*args, progress=progress, **kwargs)
        except Exception as exc:
            if self.config.fail_mode == "raise":
                LOGGER.exception("%s observer failed", LOG_PREFIX)
                raise
            reason = exc.reason if isinstance(exc, SurveyFallback) else "observer_runtime_error"
            self._record_fallback(reason, progress)
            LOGGER.debug("%s fallback reason=%s detail=%s", LOG_PREFIX, reason, exc)
        return original_func(*args, **kwargs)

    def _progress_from_kwargs(self, kwargs: dict[str, Any]) -> ProgressInfo | None:
        transformer_options = kwargs.get("transformer_options") or {}
        return progress_from_sigmas(transformer_options)

    def _observe(self, *args: Any, progress: ProgressInfo | None, **kwargs: Any) -> None:
        q, k, v, heads = self._extract_attention_args(args)
        transformer_options = kwargs.get("transformer_options") or {}
        progress = progress or progress_from_sigmas(transformer_options)
        if progress is None:
            raise SurveyFallback("missing_sigmas")
        if self.config.max_steps and progress.index >= self.config.max_steps:
            self._record_skipped("max_steps_reached", progress)
            return

        step = self.stats.step_for(progress)
        step.total_calls += 1
        if len(args) > 4 and args[4] is not None:
            raise SurveyFallback("mask_shape_unsupported")
        if kwargs.get("mask") is not None:
            raise SurveyFallback("mask_shape_unsupported")
        if not kwargs.get("skip_reshape", False):
            raise SurveyFallback("not_skip_reshape")

        query_len = int(q.shape[-2])
        text_len = int(k.shape[-2])
        if query_len == text_len:
            raise SurveyFallback("not_cross_attention")
        spatial = infer_square_spatial_shape(query_len)
        if spatial is None:
            raise SurveyFallback("cannot_infer_spatial_shape")

        batch = int(q.shape[0])
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        selected = selected_branch_indices(batch, cond_or_uncond, self.config.branch_mode, q.device)
        if selected.numel() == 0:
            reason = "branch_layout_unknown" if self.config.branch_mode != "both" and not cond_or_uncond else "branch_not_selected"
            raise SurveyFallback(reason)

        eligible_call_index = step.eligible_calls
        step.eligible_calls += 1
        step.eligible_call_indices[eligible_call_index] += 1
        step.shape_counts[shape_key(q, k)] += 1
        if not self._scope_contains(self._target_call_indices, eligible_call_index):
            self._record_skipped("target_call_skipped", progress, step)
            return
        if not self._scope_contains(self._diagnostic_call_indices, eligible_call_index):
            self._record_skipped("diagnostic_call_skipped", progress, step)
            return

        estimated_logits_mib = estimate_logits_mib(int(selected.numel()), int(q.shape[1]), query_len, text_len)
        if estimated_logits_mib > self.config.max_logits_mib:
            raise SurveyFallback("vram_guard_exceeded")

        block_id, metadata = discover_transformer_metadata(transformer_options)
        self._emit_attention_records(
            q=q,
            k=k,
            v=v,
            progress=progress,
            selected=selected,
            cond_or_uncond=cond_or_uncond,
            heads=heads,
            spatial=spatial,
            estimated_logits_mib=estimated_logits_mib,
            eligible_call_index=eligible_call_index,
            block_id=block_id,
            metadata=metadata,
            kwargs=kwargs,
        )
        step.observed_calls += 1
        self.stats.observed_calls += 1

    def _extract_attention_args(self, args: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if len(args) < 4:
            raise SurveyFallback("bad_signature")
        q, k, v, heads = args[:4]
        if not all(torch.is_tensor(tensor) and tensor.ndim == 4 for tensor in (q, k, v)):
            raise SurveyFallback("bad_rank")
        return q, k, v, int(heads)

    def _emit_attention_records(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        progress: ProgressInfo,
        selected: torch.Tensor,
        cond_or_uncond: list[int] | None,
        heads: int,
        spatial: tuple[int, int],
        estimated_logits_mib: float,
        eligible_call_index: int,
        block_id: str,
        metadata: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        selected_q = q.index_select(0, selected)
        selected_k = k.index_select(0, selected)
        scale = float(kwargs.get("scale", q.shape[-1] ** -0.5))
        logits = torch.matmul(selected_q.float(), selected_k.float().transpose(-2, -1)) * scale
        attention_probs = torch.softmax(logits, dim=-1)

        selected_batch_indices = [int(index) for index in selected.detach().cpu().tolist()]
        position_by_batch = {batch_index: pos for pos, batch_index in enumerate(selected_batch_indices)}
        for branch, batch_indices in branch_index_groups(int(q.shape[0]), cond_or_uncond, selected):
            positions = [position_by_batch[index] for index in batch_indices if index in position_by_batch]
            if not positions:
                continue
            branch_index = torch.tensor(positions, dtype=torch.long, device=attention_probs.device)
            branch_probs = attention_probs.index_select(0, branch_index)
            token_scores = []
            if self.config.capture_level in ("tokens", "heatmap"):
                token_scores = _token_scores_from_attention(branch_probs, self.config.max_tokens)
                for token in token_scores:
                    token_meta = self.token_text_map.get(int(token["token_index"]))
                    if token_meta:
                        token.update({
                            "token_id": token_meta.get("token_id"),
                            "token_text": token_meta.get("token_text"),
                            "token_source": token_meta.get("token_source"),
                            "token_weight": token_meta.get("weight"),
                        })
            record = {
                "schema_version": 1,
                "event": "attention_observation",
                "run_id": self.run_id,
                "step_index": progress.index,
                "num_steps": progress.num_steps,
                "last_index": progress.last_index,
                "u": progress.progress,
                "sigma": progress.sigma,
                "eligible_call_index": eligible_call_index,
                "branch": branch,
                "batch_indices": batch_indices,
                "selected_indices": selected_batch_indices,
                "branch_mode": self.config.branch_mode,
                "block": block_id,
                "metadata": metadata,
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "heads": int(heads),
                "spatial": list(spatial),
                "text_len": int(k.shape[-2]),
                "image_len": int(q.shape[-2]),
                "estimated_logits_mib": estimated_logits_mib,
                "prompt_text": self.config.prompt_text,
                "token_scores": token_scores,
            }
            self._emit_jsonl(record)
            if self.config.capture_level == "heatmap" and self.config.save_heatmaps:
                self._save_heatmaps(branch_probs, token_scores, spatial, progress, eligible_call_index, branch)

    def _save_heatmaps(
        self,
        attention_probs: torch.Tensor,
        token_scores: list[dict[str, Any]],
        spatial: tuple[int, int],
        progress: ProgressInfo,
        eligible_call_index: int,
        branch: str,
    ) -> None:
        import numpy as np

        out_dir = Path(self.config.heatmap_dir or "")
        out_dir.mkdir(parents=True, exist_ok=True)
        for token in token_scores:
            token_index = int(token["token_index"])
            heatmap = _heatmap_for_token(attention_probs, token_index, spatial).detach().cpu().to(torch.float32).numpy()
            stem = f"step{progress.index:03d}_call{eligible_call_index:03d}_{branch}_token{token_index:03d}"
            np.save(out_dir / f"{stem}.npy", heatmap)
            self._save_heatmap_png(out_dir / f"{stem}.png", heatmap)

    def _save_heatmap_png(self, path: Path, heatmap: Any) -> None:
        try:
            from PIL import Image
            import numpy as np
        except Exception:
            LOGGER.warning("%s could not import PIL/numpy for PNG heatmap export", LOG_PREFIX)
            return
        arr = np.asarray(heatmap, dtype=np.float32)
        min_value = float(arr.min())
        max_value = float(arr.max())
        if max_value > min_value:
            arr = (arr - min_value) / (max_value - min_value)
        else:
            arr = np.zeros_like(arr)
        Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L").save(path)

    def _scope_contains(self, scope: set[int] | None, eligible_call_index: int) -> bool:
        return scope is None or eligible_call_index in scope

    def _record_skipped(self, reason: str, progress: ProgressInfo, step: StepStats | None = None) -> None:
        step = step or self.stats.step_for(progress)
        step.skipped_calls += 1
        self.stats.skipped_calls += 1
        self._emit_jsonl({
            "schema_version": 1,
            "event": "attention_skipped",
            "run_id": self.run_id,
            "step_index": progress.index,
            "num_steps": progress.num_steps,
            "reason": reason,
        })

    def _record_fallback(self, reason: str, progress: ProgressInfo | None) -> None:
        self.stats.fallback_calls += 1
        self.stats.fallback_reasons[reason] += 1
        if progress is not None:
            step = self.stats.step_for(progress)
            step.fallback_calls += 1
            step.fallback_reasons[reason] += 1
        self._emit_jsonl({
            "schema_version": 1,
            "event": "attention_fallback",
            "run_id": self.run_id,
            "step_index": None if progress is None else progress.index,
            "num_steps": None if progress is None else progress.num_steps,
            "reason": reason,
        })

    def _run_summary_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event": "run_summary",
            "run_id": self.run_id,
            "prompt_text": self.config.prompt_text,
            "token_text_count": len(self.token_text_map),
            "token_text_map": [
                self.token_text_map[index]
                for index in sorted(self.token_text_map)
            ],
            "observed_calls": self.stats.observed_calls,
            "skipped_calls": self.stats.skipped_calls,
            "fallback_calls": self.stats.fallback_calls,
            "fallback_reasons": dict(self.stats.fallback_reasons),
            "steps": {
                str(index): {
                    "total_calls": step.total_calls,
                    "eligible_calls": step.eligible_calls,
                    "observed_calls": step.observed_calls,
                    "skipped_calls": step.skipped_calls,
                    "fallback_calls": step.fallback_calls,
                    "fallback_reasons": dict(step.fallback_reasons),
                    "shape_counts": dict(step.shape_counts),
                    "eligible_call_indices": {str(key): value for key, value in step.eligible_call_indices.items()},
                }
                for index, step in sorted(self.stats.steps.items())
            },
        }

    def _emit_jsonl(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        LOGGER.info("%s %s", LOG_PREFIX, line)
        if self.config.jsonl_path:
            path = Path(self.config.jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
