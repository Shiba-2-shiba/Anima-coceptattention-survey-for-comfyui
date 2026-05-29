from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .concepts import ConceptTokenMatch
from .intervention_config import InterventionConfig
from .progress import ProgressInfo


def intervention_plan_record(
    run_id: str,
    config: InterventionConfig,
    matches: Iterable[ConceptTokenMatch],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "intervention_plan",
        "run_id": run_id,
        "mode": config.mode,
        "intervention_kind": config.intervention_kind,
        "prompt_text": config.prompt_text,
        "intervention_terms": config.intervention_terms,
        "matches": [_match_fields(match) for match in matches],
        "target_call_indices": config.target_call_indices,
        "target_step_indices": config.target_step_indices,
        "branch_mode": config.branch_mode,
        "strength": float(config.strength),
        "logit_bias": float(config.logit_bias),
        "max_steps": int(config.max_steps),
        "max_logits_mib": float(config.max_logits_mib),
        "fail_mode": config.fail_mode,
    }


def intervention_observation_record(
    *,
    run_id: str,
    progress: ProgressInfo,
    eligible_call_index: int,
    branch: str,
    block: str,
    metadata: dict[str, Any],
    intervention_kind: str,
    match: ConceptTokenMatch,
    original_score_mean: float,
    modified_score_mean: float,
    mean_abs_prob_delta: float,
    max_abs_prob_delta: float,
    output_delta_norm: float,
    estimated_logits_mib: float,
    strength: float,
    logit_bias: float,
) -> dict[str, Any]:
    score_delta = float(modified_score_mean) - float(original_score_mean)
    return {
        "schema_version": 1,
        "event": "intervention_observation",
        "run_id": run_id,
        "step_index": progress.index,
        "num_steps": progress.num_steps,
        "last_index": progress.last_index,
        "u": progress.progress,
        "sigma": progress.sigma,
        "eligible_call_index": int(eligible_call_index),
        "branch": branch,
        "block": block,
        "metadata": metadata,
        "intervention_kind": intervention_kind,
        **_match_fields(match),
        "original_score_mean": float(original_score_mean),
        "modified_score_mean": float(modified_score_mean),
        "score_delta": score_delta,
        "mean_abs_prob_delta": float(mean_abs_prob_delta),
        "max_abs_prob_delta": float(max_abs_prob_delta),
        "output_delta_norm": float(output_delta_norm),
        "estimated_logits_mib": float(estimated_logits_mib),
        "strength": float(strength),
        "logit_bias": float(logit_bias),
    }


def intervention_skipped_record(
    *,
    run_id: str,
    progress: ProgressInfo,
    reason: str,
    eligible_call_index: int | None = None,
    branch: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "intervention_skipped",
        "run_id": run_id,
        "step_index": progress.index,
        "num_steps": progress.num_steps,
        "eligible_call_index": eligible_call_index,
        "branch": branch,
        "reason": reason,
    }


def intervention_fallback_record(
    *,
    run_id: str,
    progress: ProgressInfo | None,
    reason: str,
    eligible_call_index: int | None = None,
    branch: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "intervention_fallback",
        "run_id": run_id,
        "step_index": None if progress is None else progress.index,
        "num_steps": None if progress is None else progress.num_steps,
        "eligible_call_index": eligible_call_index,
        "branch": branch,
        "reason": reason,
    }


def intervention_summary_record(
    *,
    run_id: str,
    observed_calls: int,
    modified_calls: int,
    skipped_calls: int,
    fallback_calls: int,
    fallback_reasons: Counter[str] | dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "intervention_summary",
        "run_id": run_id,
        "observed_calls": int(observed_calls),
        "modified_calls": int(modified_calls),
        "skipped_calls": int(skipped_calls),
        "fallback_calls": int(fallback_calls),
        "fallback_reasons": dict(fallback_reasons),
    }


def _match_fields(match: ConceptTokenMatch) -> dict[str, Any]:
    return {
        "concept_uid": match.concept_uid,
        "term": match.term,
        "normalized_term": match.normalized_term,
        "token_source": match.token_source,
        "token_indices": list(match.token_indices),
        "source_token_indices": list(match.source_token_indices),
        "token_texts": list(match.token_texts),
        "token_ids": list(match.token_ids),
        "occurrence_index": match.occurrence_index,
        "ignored_token_indices": list(match.ignored_token_indices),
        "match_warnings": list(match.match_warnings),
    }
