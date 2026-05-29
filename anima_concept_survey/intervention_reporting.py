from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def intervention_observation_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("event") == "intervention_observation"]


def summarize_intervention_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    record_list = list(records)
    observations = intervention_observation_records(record_list)
    plans = [record for record in record_list if record.get("event") == "intervention_plan"]
    skipped = [record for record in record_list if record.get("event") == "intervention_skipped"]
    fallbacks = [record for record in record_list if record.get("event") == "intervention_fallback"]
    summaries = [record for record in record_list if record.get("event") == "intervention_summary"]

    by_call: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_concept: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in observations:
        call = _optional_int(record.get("eligible_call_index"), default=-1)
        branch = str(record.get("branch") or "unknown")
        concept_uid = str(record.get("concept_uid") or _concept_fallback_key(record))
        term = str(record.get("term") or "")
        by_call[(call, branch)].append(record)
        by_concept[(concept_uid, term, branch, call, str(record.get("token_source") or ""))].append(record)

    result = {
        "schema_version": 1,
        "record_count": len(record_list),
        "run_ids": sorted({str(record.get("run_id")) for record in record_list if record.get("run_id") not in (None, "")}),
        "plan_count": len(plans),
        "observation_count": len(observations),
        "skipped_count": len(skipped),
        "fallback_count": len(fallbacks),
        "summary_count": len(summaries),
        "plans": [_plan_summary(record) for record in plans],
        "fallback_reasons": dict(sorted(Counter(str(record.get("reason") or "unknown") for record in fallbacks).items())),
        "skipped_reasons": dict(sorted(Counter(str(record.get("reason") or "unknown") for record in skipped).items())),
        "intervention_by_call": _rank_rows([
            _summarize_observation_group({"eligible_call_index": call, "branch": branch}, rows)
            for (call, branch), rows in sorted(by_call.items())
        ]),
        "intervention_by_concept": _rank_rows([
            _summarize_observation_group({
                "concept_uid": concept_uid,
                "term": term,
                "branch": branch,
                "eligible_call_index": call,
                "token_source": token_source,
            }, rows)
            for (concept_uid, term, branch, call, token_source), rows in sorted(by_concept.items())
        ]),
    }
    return result


def write_intervention_summary_outputs(result: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": out / "intervention_summary.json",
        "by_call_csv": out / "intervention_by_call.csv",
        "by_concept_csv": out / "intervention_by_concept.csv",
        "report_md": out / "intervention_report.md",
    }
    paths["summary_json"].write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(paths["by_call_csv"], result["intervention_by_call"])
    _write_csv(paths["by_concept_csv"], result["intervention_by_concept"])
    paths["report_md"].write_text(intervention_report_markdown(result), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def intervention_report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Anima Concept Intervention Report",
        "",
        "Intervention output is a causal-probe heuristic. It is not a final-image segmentation mask, and image-level claims require fixed-seed image differences.",
        "",
        f"- run_ids: {', '.join(result.get('run_ids') or []) or 'none'}",
        f"- plans: {result.get('plan_count', 0)}",
        f"- observations: {result.get('observation_count', 0)}",
        f"- skipped: {result.get('skipped_count', 0)}",
        f"- fallbacks: {result.get('fallback_count', 0)}",
        "",
    ]
    if result.get("plans"):
        plan = result["plans"][0]
        lines.extend([
            "## Plan",
            "",
            f"- mode: {plan.get('mode')}",
            f"- intervention_kind: {plan.get('intervention_kind')}",
            f"- terms: {plan.get('intervention_terms')}",
            f"- branch_mode: {plan.get('branch_mode')}",
            f"- target_call_indices: {plan.get('target_call_indices')}",
            f"- target_step_indices: {plan.get('target_step_indices')}",
            "",
        ])
    if not result.get("observation_count"):
        lines.extend([
            "## Top Intervention Calls",
            "",
            "No intervention observations found.",
            "",
        ])
    else:
        lines.extend([
            "## Top Intervention Calls",
            "",
            "| rank | call | branch | obs | mean_delta | abs_mean_delta | output_delta_max | max_prob_delta |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in result["intervention_by_call"][:20]:
            lines.append(
                f"| {row['rank']} | {row.get('eligible_call_index')} | {row.get('branch')} | "
                f"{row.get('observation_count')} | {row.get('score_delta_mean')} | "
                f"{row.get('abs_score_delta_mean')} | {row.get('output_delta_norm_max')} | "
                f"{row.get('max_abs_prob_delta_max')} |"
            )
        lines.extend([
            "",
            "## Top Intervention Concepts",
            "",
            "| rank | term | concept_uid | call | branch | obs | mean_delta | output_delta_max |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in result["intervention_by_concept"][:20]:
            lines.append(
                f"| {row['rank']} | {row.get('term')} | {row.get('concept_uid')} | "
                f"{row.get('eligible_call_index')} | {row.get('branch')} | "
                f"{row.get('observation_count')} | {row.get('score_delta_mean')} | "
                f"{row.get('output_delta_norm_max')} |"
            )
    if result.get("fallback_reasons"):
        lines.extend(["", "## Fallback Reasons", "", "| reason | count |", "| --- | --- |"])
        for reason, count in result["fallback_reasons"].items():
            lines.append(f"| {reason} | {count} |")
    if result.get("skipped_reasons"):
        lines.extend(["", "## Skipped Reasons", "", "| reason | count |", "| --- | --- |"])
        for reason, count in result["skipped_reasons"].items():
            lines.append(f"| {reason} | {count} |")
    lines.extend([
        "",
        "## Interpretation Notes",
        "",
        "- Use fixed seed, same sampler, same steps, same CFG, same negative prompt, and same latent size for image comparisons.",
        "- A manual ROI is only a measurement box supplied by the user, not an automatic semantic segmentation mask.",
        "- Unsupported masks or layouts should be treated as conservative fallback behavior, not as negative evidence for the concept.",
        "",
    ])
    return "\n".join(lines)


def _summarize_observation_group(prefix: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_deltas = _numbers(row.get("score_delta") for row in rows)
    mean_prob_deltas = _numbers(row.get("mean_abs_prob_delta") for row in rows)
    max_prob_deltas = _numbers(row.get("max_abs_prob_delta") for row in rows)
    output_deltas = _numbers(row.get("output_delta_norm") for row in rows)
    original_scores = _numbers(row.get("original_score_mean") for row in rows)
    modified_scores = _numbers(row.get("modified_score_mean") for row in rows)
    return {
        **prefix,
        "observation_count": len(rows),
        "original_score_mean": _mean(original_scores),
        "modified_score_mean": _mean(modified_scores),
        "score_delta_mean": _mean(score_deltas),
        "abs_score_delta_mean": _mean([abs(value) for value in score_deltas]),
        "score_delta_min": min(score_deltas) if score_deltas else None,
        "score_delta_max": max(score_deltas) if score_deltas else None,
        "mean_abs_prob_delta_mean": _mean(mean_prob_deltas),
        "max_abs_prob_delta_max": max(max_prob_deltas) if max_prob_deltas else None,
        "output_delta_norm_mean": _mean(output_deltas),
        "output_delta_norm_max": max(output_deltas) if output_deltas else None,
        "step_min": _min_int(row.get("step_index") for row in rows),
        "step_max": _max_int(row.get("step_index") for row in rows),
        "block": _mode_text(row.get("block") for row in rows),
        "intervention_kind": _mode_text(row.get("intervention_kind") for row in rows),
        "strength": _mode_number(row.get("strength") for row in rows),
        "logit_bias": _mode_number(row.get("logit_bias") for row in rows),
        "estimated_logits_mib_max": _max_float(row.get("estimated_logits_mib") for row in rows),
        "token_indices": _mode_json(row.get("token_indices") for row in rows),
        "token_texts": _mode_json(row.get("token_texts") for row in rows),
    }


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda row: (
            _optional_float(row.get("abs_score_delta_mean")) or 0.0,
            _optional_float(row.get("output_delta_norm_max")) or 0.0,
        ),
        reverse=True,
    )
    return [{**row, "rank": rank} for rank, row in enumerate(rows, start=1)]


def _plan_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "run_id",
            "mode",
            "intervention_kind",
            "prompt_text",
            "intervention_terms",
            "target_call_indices",
            "target_step_indices",
            "branch_mode",
            "strength",
            "logit_bias",
            "max_steps",
            "max_logits_mib",
            "fail_mode",
        )
    }


def _concept_fallback_key(record: dict[str, Any]) -> str:
    return "|".join(str(record.get(key) or "") for key in ("term", "token_source", "occurrence_index", "token_indices"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _numbers(values: Iterable[Any]) -> list[float]:
    parsed = []
    for value in values:
        parsed_value = _optional_float(value)
        if parsed_value is not None:
            parsed.append(parsed_value)
    return parsed


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any, *, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _min_int(values: Iterable[Any]) -> int | None:
    parsed = [_optional_int(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return min(parsed) if parsed else None


def _max_int(values: Iterable[Any]) -> int | None:
    parsed = [_optional_int(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return max(parsed) if parsed else None


def _max_float(values: Iterable[Any]) -> float | None:
    parsed = _numbers(values)
    return max(parsed) if parsed else None


def _mode_text(values: Iterable[Any]) -> Any:
    counts: dict[Any, int] = {}
    for value in values:
        if value in (None, ""):
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def _mode_number(values: Iterable[Any]) -> float | None:
    parsed = [_optional_float(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return None
    counts = Counter(parsed)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _mode_json(values: Iterable[Any]) -> str | None:
    encoded = [
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        for value in values
        if value not in (None, "")
    ]
    return _mode_text(encoded)
