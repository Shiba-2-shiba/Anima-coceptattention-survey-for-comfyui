from __future__ import annotations

from typing import Any

from .concepts import ConceptMatchReport, ConceptTokenMatch, parse_concept_terms
from .progress import ProgressInfo


def public_concept_record(concept: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in concept.items()
        if not key.startswith("_")
    }


def attention_skipped_record(run_id: str, progress: ProgressInfo, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "attention_skipped",
        "run_id": run_id,
        "step_index": progress.index,
        "num_steps": progress.num_steps,
        "reason": reason,
    }


def attention_fallback_record(run_id: str, progress: ProgressInfo | None, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "attention_fallback",
        "run_id": run_id,
        "step_index": None if progress is None else progress.index,
        "num_steps": None if progress is None else progress.num_steps,
        "reason": reason,
    }


def concept_match_summary_record(
    *,
    run_id: str,
    prompt_text: str,
    concept_terms: str,
    report: ConceptMatchReport,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "concept_match_summary",
        "run_id": run_id,
        "prompt_text": prompt_text,
        "terms": [term.term for term in parse_concept_terms(concept_terms)],
        "matches": [
            {
                "term": match.term,
                "normalized_term": match.normalized_term,
                "token_source": match.token_source,
                "token_indices": list(match.token_indices),
                "source_token_indices": list(match.source_token_indices),
                "token_texts": list(match.token_texts),
                "occurrence_index": match.occurrence_index,
                "ignored_token_indices": list(match.ignored_token_indices),
                "match_warnings": list(match.match_warnings),
            }
            for match in report.matches
        ],
        "unmatched_terms": [term.term for term in report.unmatched_terms],
        "ambiguous_terms": [term.term for term in report.ambiguous_terms],
        "warnings": list(report.warnings),
    }


def concept_unmatched_records(
    *,
    run_id: str,
    report: ConceptMatchReport,
    available_sources: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": 1,
            "event": "concept_unmatched",
            "run_id": run_id,
            "term": term.term,
            "normalized_term": term.normalized,
            "source_filter": term.source_filter,
            "available_sources": available_sources,
            "reason": "no_contiguous_token_match",
        }
        for term in report.unmatched_terms
    ]


def concept_ambiguity_warning_records(
    *,
    run_id: str,
    report: ConceptMatchReport,
    available_sources: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": 1,
            "event": "concept_alignment_warning",
            "run_id": run_id,
            "term": term.term,
            "normalized_term": term.normalized,
            "source_filter": term.source_filter,
            "available_sources": available_sources,
            "reason": "ambiguous_across_sources",
        }
        for term in report.ambiguous_terms
    ]


def concept_alignment_warning_record(
    *,
    run_id: str,
    match: ConceptTokenMatch,
    reason: str,
    text_len: int | None = None,
    progress: ProgressInfo | None = None,
    eligible_call_index: int | None = None,
    branch: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "concept_alignment_warning",
        "run_id": run_id,
        "term": match.term,
        "normalized_term": match.normalized_term,
        "token_source": match.token_source,
        "token_indices": list(match.token_indices),
        "source_token_indices": list(match.source_token_indices),
        "text_len": text_len,
        "step_index": None if progress is None else progress.index,
        "num_steps": None if progress is None else progress.num_steps,
        "eligible_call_index": eligible_call_index,
        "branch": branch,
        "reason": reason,
    }


def run_summary_record(
    *,
    run_id: str,
    prompt_text: str,
    token_text_map: dict[int, dict[str, Any]],
    concept_match_report: ConceptMatchReport,
    stats: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "run_summary",
        "run_id": run_id,
        "prompt_text": prompt_text,
        "token_text_count": len(token_text_map),
        "token_text_map": [
            token_text_map[index]
            for index in sorted(token_text_map)
        ],
        "token_sources": sorted({
            str(meta.get("token_source") or "")
            for meta in token_text_map.values()
        }),
        "concept_match_count": len(concept_match_report.matches),
        "concept_unmatched_terms": [term.term for term in concept_match_report.unmatched_terms],
        "concept_ambiguous_terms": [term.term for term in concept_match_report.ambiguous_terms],
        "observed_calls": stats.observed_calls,
        "skipped_calls": stats.skipped_calls,
        "fallback_calls": stats.fallback_calls,
        "fallback_reasons": dict(stats.fallback_reasons),
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
            for index, step in sorted(stats.steps.items())
        },
    }
