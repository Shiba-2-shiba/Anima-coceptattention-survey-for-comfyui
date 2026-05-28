from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
from typing import Any, Callable

import torch

from .branches import branch_index_groups, selected_branch_indices
from .config import SurveyConfig
from .concepts import (
    ConceptMatchReport,
    ConceptTokenMatch,
    build_concept_token_matches,
)
from .heatmaps import HeatmapStore
from .metadata import discover_transformer_metadata
from .progress import ProgressInfo, progress_from_sigmas
from .records import (
    attention_fallback_record,
    attention_skipped_record,
    concept_alignment_warning_record,
    concept_ambiguity_warning_records,
    concept_match_summary_record,
    concept_unmatched_records,
    public_concept_record,
    run_summary_record,
)
from .scoring import concept_scores_from_attention, token_scores_from_attention
from .selectors import estimate_logits_mib, infer_square_spatial_shape, parse_call_index_scope, shape_key
from .token_text import build_token_text_map
from .writer import JsonlWriter


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[AnimaConceptSurvey]"


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


@dataclass(frozen=True)
class ConceptTokenGroup:
    term: str
    token_indices: tuple[int, ...]
    token_texts: tuple[str, ...]
    token_sources: tuple[str, ...]


class SurveyFallback(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def build_concept_token_groups(concept_terms: str, token_text_map: dict[int, dict[str, Any]]) -> list[ConceptTokenGroup]:
    report = build_concept_token_matches(concept_terms, token_text_map, allow_ambiguous=True)
    return [
        ConceptTokenGroup(
            term=match.term,
            token_indices=match.token_indices,
            token_texts=match.token_texts,
            token_sources=tuple(match.token_source for _ in match.token_indices),
        )
        for match in report.matches
    ]


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
        self._heatmaps = HeatmapStore(config.heatmap_dir)
        self._heatmap_accumulators = self._heatmaps.token_accumulators
        self.concept_match_report: ConceptMatchReport = build_concept_token_matches(config.concept_terms, self.token_text_map)
        self.concept_token_groups = list(self.concept_match_report.matches)
        self._concept_heatmap_accumulators = self._heatmaps.concept_accumulators
        self._writer = JsonlWriter(config.jsonl_path, logger=LOGGER, log_prefix=LOG_PREFIX)
        self._emit_concept_match_diagnostics()

    def finalize(self) -> None:
        if self.config.capture_level == "heatmap" and self.config.save_heatmaps:
            if self._wants_token_heatmaps():
                self._save_aggregate_heatmaps()
            if self._wants_concept_heatmaps():
                self._save_aggregate_concept_heatmaps()
        self._emit_jsonl(self._run_summary_record())

    def __call__(self, original_func: Callable, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.config.mode == "off":
            return original_func(*args, **kwargs)
        progress = self._progress_from_kwargs(kwargs)
        try:
            with torch.no_grad():
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
            if self.config.capture_level == "tokens" or (
                self.config.capture_level == "heatmap" and self._wants_token_heatmaps()
            ):
                token_scores = token_scores_from_attention(branch_probs, self.config.max_tokens)
                for token in token_scores:
                    token_meta = self.token_text_map.get(int(token["token_index"]))
                    if token_meta:
                        token.update({
                            "token_id": token_meta.get("token_id"),
                            "token_text": token_meta.get("token_text"),
                            "token_source": token_meta.get("token_source"),
                            "token_weight": token_meta.get("weight"),
                        })
            concept_scores = []
            if self.config.capture_level == "heatmap" and self._wants_concept_heatmaps() and self.concept_token_groups:
                concept_scores = self._concept_scores_from_attention(
                    branch_probs,
                    spatial,
                    progress=progress,
                    eligible_call_index=eligible_call_index,
                    branch=branch,
                )
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
                "concept_scores": [public_concept_record(concept) for concept in concept_scores],
            }
            self._emit_jsonl(record)
            if self.config.capture_level == "heatmap" and self.config.save_heatmaps:
                if self._wants_token_heatmaps():
                    self._save_heatmaps(branch_probs, token_scores, spatial, progress, eligible_call_index, branch, record)
                if self._wants_concept_heatmaps():
                    self._save_concept_heatmaps(concept_scores, spatial, progress, eligible_call_index, branch)

    def _wants_token_heatmaps(self) -> bool:
        return self.config.heatmap_output in ("tokens_only", "tokens_and_concepts")

    def _wants_concept_heatmaps(self) -> bool:
        return self.config.heatmap_output in ("concepts_only", "tokens_and_concepts")

    def _save_heatmaps(
        self,
        attention_probs: torch.Tensor,
        token_scores: list[dict[str, Any]],
        spatial: tuple[int, int],
        progress: ProgressInfo,
        eligible_call_index: int,
        branch: str,
        record: dict[str, Any],
    ) -> None:
        self._heatmaps.save_token_heatmaps(
            attention_probs,
            token_scores,
            spatial,
            progress,
            eligible_call_index,
            branch,
            record.get("block"),
        )

    def _concept_scores_from_attention(
        self,
        attention_probs: torch.Tensor,
        spatial: tuple[int, int],
        *,
        progress: ProgressInfo | None = None,
        eligible_call_index: int | None = None,
        branch: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        text_len = int(attention_probs.shape[-1])
        valid_matches: list[ConceptTokenMatch] = []
        for group in self.concept_token_groups:
            indices = list(group.token_indices)
            if any(index < 0 or index >= text_len for index in indices):
                self._emit_concept_alignment_warning(
                    group,
                    reason="concept_token_index_out_of_range",
                    text_len=text_len,
                    progress=progress,
                    eligible_call_index=eligible_call_index,
                    branch=branch,
                )
                continue
            valid_matches.append(group)
        rows.extend(concept_scores_from_attention(attention_probs, spatial, valid_matches))
        return rows

    def _save_concept_heatmaps(
        self,
        concept_scores: list[dict[str, Any]],
        spatial: tuple[int, int],
        progress: ProgressInfo,
        eligible_call_index: int,
        branch: str,
    ) -> None:
        self._heatmaps.save_concept_heatmaps(concept_scores, spatial, progress, eligible_call_index, branch)

    def _update_heatmap_accumulator(self, branch: str, token: dict[str, Any], heatmap: torch.Tensor) -> None:
        self._heatmaps.update_token_accumulator(branch, token, heatmap)

    def _update_concept_heatmap_accumulator(self, branch: str, concept: dict[str, Any], heatmap: torch.Tensor) -> None:
        self._heatmaps.update_concept_accumulator(branch, concept, heatmap)

    def _save_aggregate_heatmaps(self) -> None:
        self._heatmaps.save_aggregate_heatmaps()

    def _save_aggregate_concept_heatmaps(self) -> None:
        self._heatmaps.save_aggregate_concept_heatmaps()

    def _scope_contains(self, scope: set[int] | None, eligible_call_index: int) -> bool:
        return scope is None or eligible_call_index in scope

    def _record_skipped(self, reason: str, progress: ProgressInfo, step: StepStats | None = None) -> None:
        step = step or self.stats.step_for(progress)
        step.skipped_calls += 1
        self.stats.skipped_calls += 1
        self._emit_jsonl(attention_skipped_record(self.run_id, progress, reason))

    def _record_fallback(self, reason: str, progress: ProgressInfo | None) -> None:
        self.stats.fallback_calls += 1
        self.stats.fallback_reasons[reason] += 1
        if progress is not None:
            step = self.stats.step_for(progress)
            step.fallback_calls += 1
            step.fallback_reasons[reason] += 1
        self._emit_jsonl(attention_fallback_record(self.run_id, progress, reason))

    def _emit_concept_match_diagnostics(self) -> None:
        if not self.config.concept_terms:
            return
        available_sources = sorted({
            str(meta.get("token_source") or "")
            for meta in self.token_text_map.values()
        })
        self._emit_jsonl(concept_match_summary_record(
            run_id=self.run_id,
            prompt_text=self.config.prompt_text,
            concept_terms=self.config.concept_terms,
            report=self.concept_match_report,
        ))
        for record in concept_unmatched_records(
            run_id=self.run_id,
            report=self.concept_match_report,
            available_sources=available_sources,
        ):
            self._emit_jsonl(record)
        for record in concept_ambiguity_warning_records(
            run_id=self.run_id,
            report=self.concept_match_report,
            available_sources=available_sources,
        ):
            self._emit_jsonl(record)

    def _emit_concept_alignment_warning(
        self,
        match: ConceptTokenMatch,
        *,
        reason: str,
        text_len: int | None = None,
        progress: ProgressInfo | None = None,
        eligible_call_index: int | None = None,
        branch: str | None = None,
    ) -> None:
        self._emit_jsonl(concept_alignment_warning_record(
            run_id=self.run_id,
            match=match,
            reason=reason,
            text_len=text_len,
            progress=progress,
            eligible_call_index=eligible_call_index,
            branch=branch,
        ))

    def _run_summary_record(self) -> dict[str, Any]:
        return run_summary_record(
            run_id=self.run_id,
            prompt_text=self.config.prompt_text,
            token_text_map=self.token_text_map,
            concept_match_report=self.concept_match_report,
            stats=self.stats,
        )

    def _emit_jsonl(self, record: dict[str, Any]) -> None:
        self._writer.emit(record)
