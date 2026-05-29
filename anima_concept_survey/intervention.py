from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
from typing import Any, Callable

import torch

from .attention_math import (
    apply_attention_logit_bias,
    apply_token_attention_scale,
    attention_output_from_probs,
    compute_attention_logits,
    compute_attention_probs,
)
from .branches import branch_index_groups, selected_branch_indices
from .concepts import ConceptMatchReport, ConceptTokenMatch, build_concept_token_matches
from .intervention_config import InterventionConfig
from .intervention_records import (
    intervention_fallback_record,
    intervention_observation_record,
    intervention_plan_record,
    intervention_skipped_record,
    intervention_summary_record,
)
from .metadata import discover_transformer_metadata
from .progress import ProgressInfo, progress_from_sigmas
from .selectors import estimate_logits_mib, infer_square_spatial_shape, shape_key
from .token_text import build_token_text_map
from .writer import JsonlWriter


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[AnimaConceptIntervention]"


@dataclass
class InterventionStepStats:
    index: int
    num_steps: int
    total_calls: int = 0
    eligible_calls: int = 0
    observed_calls: int = 0
    modified_calls: int = 0
    skipped_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    shape_counts: Counter[str] = field(default_factory=Counter)
    eligible_call_indices: Counter[int] = field(default_factory=Counter)


@dataclass
class InterventionStats:
    observed_calls: int = 0
    modified_calls: int = 0
    skipped_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    steps: dict[int, InterventionStepStats] = field(default_factory=dict)

    def step_for(self, progress: ProgressInfo) -> InterventionStepStats:
        step = self.steps.get(progress.index)
        if step is None:
            step = InterventionStepStats(index=progress.index, num_steps=progress.num_steps)
            self.steps[progress.index] = step
        return step


class InterventionFallback(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class AnimaConceptInterventionAttentionOverride:
    def __init__(self, config: InterventionConfig, clip: Any | None = None):
        config.validate()
        self.config = config
        self.stats = InterventionStats()
        self.run_id = f"intervention-{id(self):x}"
        self._target_call_indices = config.target_call_scope()
        self._target_step_indices = config.target_step_scope()
        self.token_text_map = dict(config.token_text_map)
        if not self.token_text_map and config.prompt_text:
            self.token_text_map = build_token_text_map(clip, config.prompt_text)
        self.concept_match_report: ConceptMatchReport = build_concept_token_matches(
            config.intervention_terms,
            self.token_text_map,
        )
        self.concept_token_groups = list(self.concept_match_report.matches)
        self._writer = JsonlWriter(config.jsonl_path, logger=LOGGER, log_prefix=LOG_PREFIX)
        self._emit_jsonl(intervention_plan_record(self.run_id, config, self.concept_token_groups))

    def finalize(self) -> None:
        self._emit_jsonl(intervention_summary_record(
            run_id=self.run_id,
            observed_calls=self.stats.observed_calls,
            modified_calls=self.stats.modified_calls,
            skipped_calls=self.stats.skipped_calls,
            fallback_calls=self.stats.fallback_calls,
            fallback_reasons=self.stats.fallback_reasons,
        ))

    def __call__(self, original_func: Callable, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.config.mode == "off":
            return original_func(*args, **kwargs)
        progress = self._progress_from_kwargs(kwargs)
        if self.config.mode == "shadow":
            try:
                with torch.no_grad():
                    self._observe_shadow(*args, progress=progress, **kwargs)
            except Exception as exc:
                if self.config.fail_mode == "raise":
                    LOGGER.exception("%s shadow intervention failed", LOG_PREFIX)
                    raise
                reason = exc.reason if isinstance(exc, InterventionFallback) else "runtime_error"
                self._record_fallback(reason, progress)
                LOGGER.debug("%s fallback reason=%s detail=%s", LOG_PREFIX, reason, exc)
            return original_func(*args, **kwargs)
        if self.config.mode == "intervene":
            try:
                with torch.no_grad():
                    modified = self._intervene(*args, progress=progress, **kwargs)
                if modified is not None:
                    return modified
            except Exception as exc:
                if self.config.fail_mode == "raise":
                    LOGGER.exception("%s intervention failed", LOG_PREFIX)
                    raise
                reason = exc.reason if isinstance(exc, InterventionFallback) else "runtime_error"
                self._record_fallback(reason, progress)
                LOGGER.debug("%s fallback reason=%s detail=%s", LOG_PREFIX, reason, exc)
            return original_func(*args, **kwargs)
        raise InterventionFallback("unsupported_mode")

    def _progress_from_kwargs(self, kwargs: dict[str, Any]) -> ProgressInfo | None:
        transformer_options = kwargs.get("transformer_options") or {}
        return progress_from_sigmas(transformer_options)

    def _observe_shadow(self, *args: Any, progress: ProgressInfo | None, **kwargs: Any) -> None:
        q, k, v, heads = self._extract_attention_args(args)
        transformer_options = kwargs.get("transformer_options") or {}
        progress = progress or progress_from_sigmas(transformer_options)
        if progress is None:
            raise InterventionFallback("missing_sigmas")
        if self.config.max_steps and progress.index >= self.config.max_steps:
            self._record_skipped("max_steps_reached", progress)
            return

        step = self.stats.step_for(progress)
        step.total_calls += 1
        if len(args) > 4 and args[4] is not None:
            raise InterventionFallback("mask_shape_unsupported")
        if kwargs.get("mask") is not None:
            raise InterventionFallback("mask_shape_unsupported")
        if not kwargs.get("skip_reshape", False):
            raise InterventionFallback("not_skip_reshape")

        query_len = int(q.shape[-2])
        text_len = int(k.shape[-2])
        if query_len == text_len:
            raise InterventionFallback("not_cross_attention")
        spatial = infer_square_spatial_shape(query_len)
        if spatial is None:
            raise InterventionFallback("cannot_infer_spatial_shape")

        batch = int(q.shape[0])
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        selected = selected_branch_indices(batch, cond_or_uncond, self.config.branch_mode, q.device)
        if selected.numel() == 0:
            reason = "branch_layout_unknown" if self.config.branch_mode != "both" and not cond_or_uncond else "branch_not_selected"
            raise InterventionFallback(reason)

        eligible_call_index = step.eligible_calls
        step.eligible_calls += 1
        step.eligible_call_indices[eligible_call_index] += 1
        step.shape_counts[shape_key(q, k)] += 1
        if not self._scope_contains(self._target_step_indices, progress.index):
            self._record_skipped("target_step_skipped", progress, step, eligible_call_index=eligible_call_index)
            return
        if not self._scope_contains(self._target_call_indices, eligible_call_index):
            self._record_skipped("target_call_skipped", progress, step, eligible_call_index=eligible_call_index)
            return

        estimated_logits_mib = estimate_logits_mib(int(selected.numel()), int(q.shape[1]), query_len, text_len)
        if estimated_logits_mib > self.config.max_logits_mib:
            raise InterventionFallback("vram_guard_exceeded")

        if not self.concept_token_groups:
            self._record_skipped("concept_not_matched", progress, step, eligible_call_index=eligible_call_index)
            return

        block_id, metadata = discover_transformer_metadata(transformer_options)
        self._emit_shadow_records(
            q=q,
            k=k,
            v=v,
            selected=selected,
            cond_or_uncond=cond_or_uncond,
            progress=progress,
            eligible_call_index=eligible_call_index,
            block_id=block_id,
            metadata=metadata,
            estimated_logits_mib=estimated_logits_mib,
            kwargs=kwargs,
        )
        step.observed_calls += 1
        self.stats.observed_calls += 1

    def _extract_attention_args(self, args: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if len(args) < 4:
            raise InterventionFallback("bad_signature")
        q, k, v, heads = args[:4]
        if not all(torch.is_tensor(tensor) and tensor.ndim == 4 for tensor in (q, k, v)):
            raise InterventionFallback("bad_rank")
        return q, k, v, int(heads)

    def _intervene(self, *args: Any, progress: ProgressInfo | None, **kwargs: Any) -> torch.Tensor | None:
        q, k, v, heads = self._extract_attention_args(args)
        transformer_options = kwargs.get("transformer_options") or {}
        progress = progress or progress_from_sigmas(transformer_options)
        if progress is None:
            raise InterventionFallback("missing_sigmas")
        if self.config.max_steps and progress.index >= self.config.max_steps:
            self._record_skipped("max_steps_reached", progress)
            return None

        step = self.stats.step_for(progress)
        step.total_calls += 1
        if len(args) > 4 and args[4] is not None:
            raise InterventionFallback("mask_shape_unsupported")
        if kwargs.get("mask") is not None:
            raise InterventionFallback("mask_shape_unsupported")
        if not kwargs.get("skip_reshape", False):
            raise InterventionFallback("not_skip_reshape")

        query_len = int(q.shape[-2])
        text_len = int(k.shape[-2])
        if query_len == text_len:
            raise InterventionFallback("not_cross_attention")
        spatial = infer_square_spatial_shape(query_len)
        if spatial is None:
            raise InterventionFallback("cannot_infer_spatial_shape")

        batch = int(q.shape[0])
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        selected = selected_branch_indices(batch, cond_or_uncond, self.config.branch_mode, q.device)
        if selected.numel() == 0:
            reason = "branch_layout_unknown" if self.config.branch_mode != "both" and not cond_or_uncond else "branch_not_selected"
            raise InterventionFallback(reason)

        eligible_call_index = step.eligible_calls
        step.eligible_calls += 1
        step.eligible_call_indices[eligible_call_index] += 1
        step.shape_counts[shape_key(q, k)] += 1
        if not self._scope_contains(self._target_step_indices, progress.index):
            self._record_skipped("target_step_skipped", progress, step, eligible_call_index=eligible_call_index)
            return None
        if not self._scope_contains(self._target_call_indices, eligible_call_index):
            self._record_skipped("target_call_skipped", progress, step, eligible_call_index=eligible_call_index)
            return None

        estimated_logits_mib = estimate_logits_mib(int(selected.numel()), int(q.shape[1]), query_len, text_len)
        if estimated_logits_mib > self.config.max_logits_mib:
            raise InterventionFallback("vram_guard_exceeded")
        if not self.concept_token_groups:
            self._record_skipped("concept_not_matched", progress, step, eligible_call_index=eligible_call_index)
            return None

        scale = float(kwargs.get("scale", q.shape[-1] ** -0.5))
        logits = compute_attention_logits(q, k, scale)
        original_probs = compute_attention_probs(logits)
        modified_probs = self._modified_probs_for_intervention(
            logits=logits,
            original_probs=original_probs,
            selected=selected,
            cond_or_uncond=cond_or_uncond,
            v=v,
            progress=progress,
            eligible_call_index=eligible_call_index,
            transformer_options=transformer_options,
            estimated_logits_mib=estimated_logits_mib,
        )
        step.observed_calls += 1
        self.stats.observed_calls += 1
        modified_out = attention_output_from_probs(modified_probs, v)
        return self._format_attention_output(modified_out, q, heads, kwargs)

    def _modified_probs_for_intervention(
        self,
        *,
        logits: torch.Tensor,
        original_probs: torch.Tensor,
        selected: torch.Tensor,
        cond_or_uncond: list[int] | None,
        v: torch.Tensor,
        progress: ProgressInfo,
        eligible_call_index: int,
        transformer_options: dict[str, Any],
        estimated_logits_mib: float,
    ) -> torch.Tensor:
        block_id, metadata = discover_transformer_metadata(transformer_options)
        if self.config.intervention_kind == "attention_logit_bias":
            modified_logits = logits.clone()
            selected_set = set(int(index) for index in selected.detach().cpu().tolist())
            for branch, batch_indices in branch_index_groups(int(logits.shape[0]), cond_or_uncond, selected):
                scoped_indices = [index for index in batch_indices if index in selected_set]
                if not scoped_indices:
                    continue
                batch_index = torch.tensor(scoped_indices, dtype=torch.long, device=logits.device)
                for match in self.concept_token_groups:
                    token_indices = self._valid_token_indices_or_skip(
                        match,
                        int(logits.shape[-1]),
                        progress,
                        eligible_call_index,
                        branch,
                    )
                    if token_indices is None:
                        continue
                    before_branch_probs = compute_attention_probs(modified_logits.index_select(0, batch_index))
                    branch_logits = modified_logits.index_select(0, batch_index)
                    after_branch_logits = apply_attention_logit_bias(
                        branch_logits,
                        token_indices,
                        self.config.logit_bias,
                    )
                    after_branch_probs = compute_attention_probs(after_branch_logits)
                    modified_logits.index_copy_(0, batch_index, after_branch_logits)
                    self._emit_probs_observation(
                        match=match,
                        branch=branch,
                        branch_probs=before_branch_probs,
                        modified_probs=after_branch_probs,
                        branch_v=v.index_select(0, batch_index),
                        progress=progress,
                        eligible_call_index=eligible_call_index,
                        block_id=block_id,
                        metadata=metadata,
                        estimated_logits_mib=estimated_logits_mib,
                    )
            return compute_attention_probs(modified_logits)

        if self.config.intervention_kind == "token_attention_scale":
            modified_probs = original_probs.clone()
            selected_set = set(int(index) for index in selected.detach().cpu().tolist())
            for branch, batch_indices in branch_index_groups(int(logits.shape[0]), cond_or_uncond, selected):
                scoped_indices = [index for index in batch_indices if index in selected_set]
                if not scoped_indices:
                    continue
                batch_index = torch.tensor(scoped_indices, dtype=torch.long, device=logits.device)
                for match in self.concept_token_groups:
                    token_indices = self._valid_token_indices_or_skip(
                        match,
                        int(modified_probs.shape[-1]),
                        progress,
                        eligible_call_index,
                        branch,
                    )
                    if token_indices is None:
                        continue
                    before_branch_probs = modified_probs.index_select(0, batch_index)
                    after_branch_probs = apply_token_attention_scale(
                        before_branch_probs,
                        token_indices,
                        self.config.strength,
                    )
                    modified_probs.index_copy_(0, batch_index, after_branch_probs)
                    self._emit_probs_observation(
                        match=match,
                        branch=branch,
                        branch_probs=before_branch_probs,
                        modified_probs=after_branch_probs,
                        branch_v=v.index_select(0, batch_index),
                        progress=progress,
                        eligible_call_index=eligible_call_index,
                        block_id=block_id,
                        metadata=metadata,
                        estimated_logits_mib=estimated_logits_mib,
                    )
            return modified_probs

        raise InterventionFallback("unsupported_intervention_kind")

    def _format_attention_output(self, attention_out: torch.Tensor, q: torch.Tensor, heads: int, kwargs: dict[str, Any]) -> torch.Tensor:
        if kwargs.get("skip_output_reshape", False):
            return attention_out
        return attention_out.transpose(1, 2).reshape(
            int(q.shape[0]),
            int(q.shape[-2]),
            int(heads) * int(attention_out.shape[-1]),
        )

    def _emit_shadow_records(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected: torch.Tensor,
        cond_or_uncond: list[int] | None,
        progress: ProgressInfo,
        eligible_call_index: int,
        block_id: str,
        metadata: dict[str, Any],
        estimated_logits_mib: float,
        kwargs: dict[str, Any],
    ) -> None:
        selected_q = q.index_select(0, selected)
        selected_k = k.index_select(0, selected)
        selected_v = v.index_select(0, selected)
        scale = float(kwargs.get("scale", q.shape[-1] ** -0.5))
        logits = compute_attention_logits(selected_q, selected_k, scale)
        attention_probs = compute_attention_probs(logits)

        selected_batch_indices = [int(index) for index in selected.detach().cpu().tolist()]
        position_by_batch = {batch_index: pos for pos, batch_index in enumerate(selected_batch_indices)}
        for branch, batch_indices in branch_index_groups(int(q.shape[0]), cond_or_uncond, selected):
            positions = [position_by_batch[index] for index in batch_indices if index in position_by_batch]
            if not positions:
                continue
            branch_index = torch.tensor(positions, dtype=torch.long, device=attention_probs.device)
            branch_logits = logits.index_select(0, branch_index)
            branch_probs = attention_probs.index_select(0, branch_index)
            branch_v = selected_v.index_select(0, branch_index)
            for match in self.concept_token_groups:
                self._emit_match_shadow_record(
                    match=match,
                    branch=branch,
                    branch_logits=branch_logits,
                    branch_probs=branch_probs,
                    branch_v=branch_v,
                    progress=progress,
                    eligible_call_index=eligible_call_index,
                    block_id=block_id,
                    metadata=metadata,
                    estimated_logits_mib=estimated_logits_mib,
                )

    def _emit_match_shadow_record(
        self,
        *,
        match: ConceptTokenMatch,
        branch: str,
        branch_logits: torch.Tensor,
        branch_probs: torch.Tensor,
        branch_v: torch.Tensor,
        progress: ProgressInfo,
        eligible_call_index: int,
        block_id: str,
        metadata: dict[str, Any],
        estimated_logits_mib: float,
    ) -> None:
        text_len = int(branch_probs.shape[-1])
        token_indices = list(match.token_indices)
        if any(index < 0 or index >= text_len for index in token_indices):
            self._emit_jsonl(intervention_skipped_record(
                run_id=self.run_id,
                progress=progress,
                reason="concept_token_index_out_of_range",
                eligible_call_index=eligible_call_index,
                branch=branch,
            ))
            return

        if self.config.intervention_kind == "attention_logit_bias":
            modified_probs = compute_attention_probs(apply_attention_logit_bias(
                branch_logits,
                token_indices,
                self.config.logit_bias,
            ))
        elif self.config.intervention_kind == "token_attention_scale":
            modified_probs = apply_token_attention_scale(
                branch_probs,
                token_indices,
                self.config.strength,
            )
        else:
            raise InterventionFallback("unsupported_intervention_kind")

        original_mass = branch_probs[..., token_indices].sum(dim=-1)
        modified_mass = modified_probs[..., token_indices].sum(dim=-1)
        prob_delta = (modified_probs - branch_probs).abs()
        original_out = attention_output_from_probs(branch_probs, branch_v)
        modified_out = attention_output_from_probs(modified_probs, branch_v)
        output_delta_norm = torch.linalg.vector_norm((modified_out - original_out).float())
        mean_abs_prob_delta = float(prob_delta.mean().item())
        max_abs_prob_delta = float(prob_delta.max().item())

        if mean_abs_prob_delta > 0 or float(output_delta_norm.item()) > 0:
            self.stats.modified_calls += 1
            step = self.stats.step_for(progress)
            step.modified_calls += 1

        self._emit_jsonl(intervention_observation_record(
            run_id=self.run_id,
            progress=progress,
            eligible_call_index=eligible_call_index,
            branch=branch,
            block=block_id,
            metadata=metadata,
            intervention_kind=self.config.intervention_kind,
            match=match,
            original_score_mean=float(original_mass.mean().item()),
            modified_score_mean=float(modified_mass.mean().item()),
            mean_abs_prob_delta=mean_abs_prob_delta,
            max_abs_prob_delta=max_abs_prob_delta,
            output_delta_norm=float(output_delta_norm.item()),
            estimated_logits_mib=estimated_logits_mib,
            strength=self.config.strength,
            logit_bias=self.config.logit_bias,
        ))

    def _valid_token_indices_or_skip(
        self,
        match: ConceptTokenMatch,
        text_len: int,
        progress: ProgressInfo,
        eligible_call_index: int,
        branch: str,
    ) -> list[int] | None:
        token_indices = list(match.token_indices)
        if any(index < 0 or index >= text_len for index in token_indices):
            self._emit_jsonl(intervention_skipped_record(
                run_id=self.run_id,
                progress=progress,
                reason="concept_token_index_out_of_range",
                eligible_call_index=eligible_call_index,
                branch=branch,
            ))
            return None
        return token_indices

    def _emit_probs_observation(
        self,
        *,
        match: ConceptTokenMatch,
        branch: str,
        branch_probs: torch.Tensor,
        modified_probs: torch.Tensor,
        branch_v: torch.Tensor,
        progress: ProgressInfo,
        eligible_call_index: int,
        block_id: str,
        metadata: dict[str, Any],
        estimated_logits_mib: float,
    ) -> None:
        token_indices = list(match.token_indices)
        original_mass = branch_probs[..., token_indices].sum(dim=-1)
        modified_mass = modified_probs[..., token_indices].sum(dim=-1)
        prob_delta = (modified_probs - branch_probs).abs()
        original_out = attention_output_from_probs(branch_probs, branch_v)
        modified_out = attention_output_from_probs(modified_probs, branch_v)
        output_delta_norm = torch.linalg.vector_norm((modified_out - original_out).float())
        mean_abs_prob_delta = float(prob_delta.mean().item())
        max_abs_prob_delta = float(prob_delta.max().item())

        if mean_abs_prob_delta > 0 or float(output_delta_norm.item()) > 0:
            self.stats.modified_calls += 1
            step = self.stats.step_for(progress)
            step.modified_calls += 1

        self._emit_jsonl(intervention_observation_record(
            run_id=self.run_id,
            progress=progress,
            eligible_call_index=eligible_call_index,
            branch=branch,
            block=block_id,
            metadata=metadata,
            intervention_kind=self.config.intervention_kind,
            match=match,
            original_score_mean=float(original_mass.mean().item()),
            modified_score_mean=float(modified_mass.mean().item()),
            mean_abs_prob_delta=mean_abs_prob_delta,
            max_abs_prob_delta=max_abs_prob_delta,
            output_delta_norm=float(output_delta_norm.item()),
            estimated_logits_mib=estimated_logits_mib,
            strength=self.config.strength,
            logit_bias=self.config.logit_bias,
        ))

    def _scope_contains(self, scope: set[int] | None, index: int) -> bool:
        return scope is None or index in scope

    def _record_skipped(
        self,
        reason: str,
        progress: ProgressInfo,
        step: InterventionStepStats | None = None,
        *,
        eligible_call_index: int | None = None,
        branch: str | None = None,
    ) -> None:
        step = step or self.stats.step_for(progress)
        step.skipped_calls += 1
        self.stats.skipped_calls += 1
        self._emit_jsonl(intervention_skipped_record(
            run_id=self.run_id,
            progress=progress,
            reason=reason,
            eligible_call_index=eligible_call_index,
            branch=branch,
        ))

    def _record_fallback(self, reason: str, progress: ProgressInfo | None) -> None:
        self.stats.fallback_calls += 1
        self.stats.fallback_reasons[reason] += 1
        if progress is not None:
            step = self.stats.step_for(progress)
            step.fallback_calls += 1
            step.fallback_reasons[reason] += 1
        self._emit_jsonl(intervention_fallback_record(
            run_id=self.run_id,
            progress=progress,
            reason=reason,
        ))

    def _emit_jsonl(self, record: dict[str, Any]) -> None:
        self._writer.emit(record)
