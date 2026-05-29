from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    first_record_index: int
    last_record_index: int
    observation_count: int
    events: dict[str, int]
    prompt_text: str | None
    branches: tuple[str, ...]
    concept_terms: tuple[str, ...]


def iter_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def list_runs(records: Iterable[dict[str, Any]]) -> list[RunInfo]:
    builders: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        run_id = record.get("run_id")
        if run_id in (None, ""):
            continue
        run_id = str(run_id)
        builder = builders.get(run_id)
        if builder is None:
            builder = {
                "run_id": run_id,
                "first_record_index": index,
                "last_record_index": index,
                "observation_count": 0,
                "events": Counter(),
                "prompt_text": None,
                "branches": [],
                "concept_terms": [],
            }
            builders[run_id] = builder
        builder["last_record_index"] = index
        event = str(record.get("event") or "unknown")
        builder["events"][event] += 1
        if event == "attention_observation":
            builder["observation_count"] += 1
        prompt_text = record.get("prompt_text")
        if builder["prompt_text"] is None and prompt_text not in (None, ""):
            builder["prompt_text"] = str(prompt_text)
        branch = record.get("branch")
        if branch not in (None, ""):
            _append_unique(builder["branches"], str(branch))
        for term in _record_concept_terms(record):
            _append_unique(builder["concept_terms"], term)

    return [
        RunInfo(
            run_id=str(builder["run_id"]),
            first_record_index=int(builder["first_record_index"]),
            last_record_index=int(builder["last_record_index"]),
            observation_count=int(builder["observation_count"]),
            events=dict(sorted(builder["events"].items())),
            prompt_text=builder["prompt_text"],
            branches=tuple(builder["branches"]),
            concept_terms=tuple(builder["concept_terms"]),
        )
        for builder in sorted(builders.values(), key=lambda item: item["first_record_index"])
    ]


def latest_run_id(records: Iterable[dict[str, Any]]) -> str | None:
    runs = list_runs(records)
    if not runs:
        return None
    return max(runs, key=lambda run: run.last_record_index).run_id


def filter_records_by_run_id(records: Iterable[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("run_id") or "") == str(run_id)]


def attention_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("event") == "attention_observation"]


def summarize_records(
    records: Iterable[dict[str, Any]],
    top_k: int = 16,
    late_start_step: int | None = None,
    *,
    selected_run_id: str | None = None,
    available_runs: list[RunInfo] | None = None,
    run_count_in_input: int | None = None,
    mixed_runs: bool | None = None,
) -> dict[str, Any]:
    record_list = list(records)
    runs_for_summary = list_runs(record_list)
    available = available_runs if available_runs is not None else runs_for_summary
    if selected_run_id is None and len(runs_for_summary) == 1:
        selected_run_id = runs_for_summary[0].run_id
    if run_count_in_input is None:
        run_count_in_input = len(available)
    if mixed_runs is None:
        mixed_runs = len(runs_for_summary) > 1
    observations = attention_records(record_list)
    by_call: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    token_rows: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    concept_rows: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in observations:
        call = int(record.get("eligible_call_index", -1))
        branch = str(record.get("branch", "unknown"))
        step = int(record.get("step_index", -1))
        by_call[(call, branch)].append(record)
        by_step[step].append(record)
        by_branch[branch].append(record)
        for token in record.get("token_scores") or []:
            token_rows[(call, branch, int(token["token_index"]))].append({
                **token,
                "step_index": step,
                "block": record.get("block", "unknown"),
                "image_len": record.get("image_len"),
                "text_len": record.get("text_len"),
            })
        for concept in record.get("concept_scores") or []:
            term = str(concept.get("term") or "")
            concept_uid = str(concept.get("concept_uid") or term)
            concept_rows[(call, branch, concept_uid)].append({
                **concept,
                "step_index": step,
                "block": record.get("block", "unknown"),
                "image_len": record.get("image_len"),
                "text_len": record.get("text_len"),
            })

    summary_by_call = [_summarize_group({"eligible_call_index": call, "branch": branch}, rows, late_start_step) for (call, branch), rows in sorted(by_call.items())]
    summary_by_step = [_summarize_group({"step_index": step}, rows, late_start_step) for step, rows in sorted(by_step.items())]
    summary_by_branch = [_summarize_group({"branch": branch}, rows, late_start_step) for branch, rows in sorted(by_branch.items())]
    summary_by_token = _summarize_tokens(token_rows, top_k)
    summary_by_concept = _summarize_concepts(concept_rows)
    recommended_concept_targets = _recommend_concept_targets(summary_by_concept)
    preview_warnings = _preview_warnings(summary_by_concept)
    survey_branch_concept_delta = _branch_concept_delta(summary_by_concept)

    recommended_targets = [
        {
            "rank": rank,
            "eligible_call_index": row["eligible_call_index"],
            "branch": row["branch"],
            "mean_top_token_score": row["mean_top_token_score"],
            "late_mean_top_token_score": row["late_mean_top_token_score"],
            "observation_count": row["observation_count"],
            "target_hint": f"call_{row['eligible_call_index']}",
        }
        for rank, row in enumerate(
            sorted(summary_by_call, key=lambda item: (item["late_mean_top_token_score"] or item["mean_top_token_score"] or 0.0), reverse=True),
            start=1,
        )
    ]

    return {
        "schema_version": 1,
        "observation_count": len(observations),
        "top_k": top_k,
        "late_start_step": late_start_step,
        "selected_run_id": selected_run_id,
        "run_count_in_input": run_count_in_input,
        "mixed_runs": mixed_runs,
        "available_runs": [_run_info_dict(run) for run in available],
        "survey_runs": [_run_info_csv_row(run) for run in available],
        "summary_by_call": summary_by_call,
        "summary_by_step": summary_by_step,
        "summary_by_branch": summary_by_branch,
        "summary_by_token": summary_by_token,
        "summary_by_concept": summary_by_concept,
        "recommended_lora_targets": recommended_targets,
        "recommended_concept_targets": recommended_concept_targets,
        "preview_warnings": preview_warnings,
        "survey_branch_concept_delta": survey_branch_concept_delta,
    }


def write_summary_outputs(result: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": out / "survey_summary.json",
        "by_call_csv": out / "survey_by_call.csv",
        "by_step_csv": out / "survey_by_step.csv",
        "by_branch_csv": out / "survey_by_branch.csv",
        "by_token_csv": out / "survey_by_token.csv",
        "by_concept_csv": out / "survey_by_concept.csv",
        "recommended_lora_targets_csv": out / "recommended_lora_targets.csv",
        "recommended_concept_targets_csv": out / "recommended_concept_targets.csv",
        "branch_concept_delta_csv": out / "survey_branch_concept_delta.csv",
        "survey_runs_csv": out / "survey_runs.csv",
        "report_md": out / "survey_report.md",
    }
    paths["summary_json"].write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(paths["by_call_csv"], result["summary_by_call"])
    _write_csv(paths["by_step_csv"], result["summary_by_step"])
    _write_csv(paths["by_branch_csv"], result["summary_by_branch"])
    _write_csv(paths["by_token_csv"], result["summary_by_token"])
    _write_csv(paths["by_concept_csv"], result["summary_by_concept"])
    _write_csv(paths["recommended_lora_targets_csv"], result["recommended_lora_targets"])
    _write_csv(paths["recommended_concept_targets_csv"], result.get("recommended_concept_targets", []))
    _write_csv(paths["branch_concept_delta_csv"], result.get("survey_branch_concept_delta", []))
    _write_csv(paths["survey_runs_csv"], result.get("survey_runs", []))
    paths["report_md"].write_text(_report_markdown(result), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _summarize_group(prefix: dict[str, Any], rows: list[dict[str, Any]], late_start_step: int | None) -> dict[str, Any]:
    token_means = [
        float(token["score_mean"])
        for row in rows
        for token in (row.get("token_scores") or [])[:1]
        if token.get("score_mean") is not None
    ]
    late_rows = rows if late_start_step is None else [row for row in rows if int(row.get("step_index", -1)) >= late_start_step]
    late_token_means = [
        float(token["score_mean"])
        for row in late_rows
        for token in (row.get("token_scores") or [])[:1]
        if token.get("score_mean") is not None
    ]
    return {
        **prefix,
        "observation_count": len(rows),
        "mean_top_token_score": _mean(token_means),
        "late_mean_top_token_score": _mean(late_token_means),
        "max_estimated_logits_mib": _max(row.get("estimated_logits_mib") for row in rows),
    }


def _summarize_tokens(token_rows: dict[tuple[int, str, int], list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    rows = []
    for (call, branch, token_index), items in token_rows.items():
        score_means = [float(item["score_mean"]) for item in items if item.get("score_mean") is not None]
        score_maxes = [float(item["score_max"]) for item in items if item.get("score_max") is not None]
        entropies = [float(item["score_entropy"]) for item in items if item.get("score_entropy") is not None]
        rows.append({
            "eligible_call_index": call,
            "branch": branch,
            "token_index": token_index,
            "token_text": _mode_text(item.get("token_text") for item in items),
            "token_source": _mode_text(item.get("token_source") for item in items),
            "token_id": _mode_text(item.get("token_id") for item in items),
            "observation_count": len(items),
            "score_mean": _mean(score_means),
            "score_max": _max(score_maxes),
            "score_entropy": _mean(entropies),
        })
    rows.sort(key=lambda item: item["score_mean"] or 0.0, reverse=True)
    return [{**row, "rank": rank} for rank, row in enumerate(rows[:top_k], start=1)]


def _summarize_concepts(concept_rows: dict[tuple[int, str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for (call, branch, concept_uid), items in concept_rows.items():
        score_means = [float(item["score_mean"]) for item in items if item.get("score_mean") is not None]
        score_maxes = [float(item["score_max"]) for item in items if item.get("score_max") is not None]
        entropies = [float(item["score_entropy"]) for item in items if item.get("score_entropy") is not None]
        heatmap_means = _numbers(item.get("heatmap_mean") for item in items)
        heatmap_maxes = _numbers(item.get("heatmap_max") for item in items)
        heatmap_stds = _numbers(item.get("heatmap_std") for item in items)
        heatmap_focus = _numbers(item.get("heatmap_max_over_mean") for item in items)
        uniform_baselines = _numbers(item.get("uniform_baseline") for item in items)
        mean_over_uniform = _numbers(item.get("score_mean_over_uniform") for item in items)
        rows.append({
            "term": _mode_text(item.get("term") for item in items),
            "concept_uid": concept_uid if concept_uid else None,
            "normalized_term": _mode_text(item.get("normalized_term") for item in items),
            "token_source": _mode_text(item.get("token_source") for item in items),
            "occurrence_index": _mode_text(item.get("occurrence_index") for item in items),
            "branch": branch,
            "eligible_call_index": call,
            "observation_count": len(items),
            "score_mean": _mean(score_means),
            "score_max": _max(score_maxes),
            "score_entropy": _mean(entropies),
            "heatmap_mean": _mean(heatmap_means),
            "heatmap_max": _max(heatmap_maxes),
            "heatmap_std": _mean(heatmap_stds),
            "heatmap_max_over_mean": _mean(heatmap_focus),
            "uniform_baseline": _mean(uniform_baselines),
            "score_mean_over_uniform": _mean(mean_over_uniform),
            "near_uniform": _any_true(item.get("near_uniform") for item in items),
            "token_indices": _mode_json(item.get("token_indices") for item in items),
            "source_token_indices": _mode_json(item.get("source_token_indices") for item in items),
            "token_texts": _mode_json(item.get("token_texts") for item in items),
            "token_sources": _mode_json(item.get("token_sources") for item in items),
        })
    rows.sort(key=lambda item: item["score_mean"] or 0.0, reverse=True)
    return rows


def _recommend_concept_targets(summary_by_concept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in summary_by_concept:
        score_mean_over_uniform = _optional_float(row.get("score_mean_over_uniform"))
        heatmap_max_over_mean = _optional_float(row.get("heatmap_max_over_mean"))
        if score_mean_over_uniform is None or heatmap_max_over_mean is None:
            continue
        focus_factor = max(0.0, heatmap_max_over_mean - 1.0)
        focus_factor = min(focus_factor, 4.0)
        near_uniform_penalty = 0.25 if heatmap_max_over_mean < 1.05 else 1.0
        branch_factor = 1.0 if row.get("branch") == "positive" else 0.5
        rank_score = score_mean_over_uniform * (1.0 + focus_factor) * near_uniform_penalty * branch_factor
        target_hint = f"call_{row['eligible_call_index']}_{row['branch']}_{row.get('concept_uid') or row.get('term')}"
        ranked.append({
            "rank_score": rank_score,
            "term": row.get("term"),
            "concept_uid": row.get("concept_uid"),
            "token_source": row.get("token_source"),
            "occurrence_index": row.get("occurrence_index"),
            "branch": row.get("branch"),
            "eligible_call_index": row.get("eligible_call_index"),
            "observation_count": row.get("observation_count"),
            "score_mean": row.get("score_mean"),
            "score_max": row.get("score_max"),
            "score_entropy": row.get("score_entropy"),
            "heatmap_mean": row.get("heatmap_mean"),
            "heatmap_max": row.get("heatmap_max"),
            "heatmap_std": row.get("heatmap_std"),
            "heatmap_max_over_mean": row.get("heatmap_max_over_mean"),
            "uniform_baseline": row.get("uniform_baseline"),
            "score_mean_over_uniform": row.get("score_mean_over_uniform"),
            "near_uniform": bool(row.get("near_uniform")),
            "target_hint": target_hint,
        })
    ranked.sort(key=lambda item: item["rank_score"], reverse=True)
    return [{**row, "rank": rank} for rank, row in enumerate(ranked, start=1)]


def _preview_warnings(summary_by_concept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for row in summary_by_concept:
        focus = _optional_float(row.get("heatmap_max_over_mean"))
        if focus is None:
            continue
        severity = None
        phrase = None
        if focus < 1.05:
            severity = "near_uniform"
            phrase = "is near-uniform"
        elif focus < 1.15:
            severity = "weak_focus"
            phrase = "has weak focus"
        if severity is None or phrase is None:
            continue
        message = (
            f"Warning: {row.get('branch')} / {row.get('term')} / call {row.get('eligible_call_index')} {phrase}. "
            "Preview PNG is min-max normalized and may overstate localization. "
            "Use raw .npy and heatmap_max_over_mean for interpretation."
        )
        warnings.append({
            "severity": severity,
            "term": row.get("term"),
            "concept_uid": row.get("concept_uid"),
            "branch": row.get("branch"),
            "eligible_call_index": row.get("eligible_call_index"),
            "heatmap_max_over_mean": focus,
            "message": message,
        })
    return warnings


def _branch_concept_delta(summary_by_concept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_by_concept:
        branch = row.get("branch")
        if branch not in ("positive", "negative"):
            continue
        call = row.get("eligible_call_index")
        concept_key = str(row.get("concept_uid") or _concept_fallback_key(row))
        groups[(int(call), concept_key)][str(branch)] = row

    deltas: list[dict[str, Any]] = []
    for (call, concept_key), branches in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        positive = branches.get("positive")
        negative = branches.get("negative")
        if positive is None or negative is None:
            continue
        positive_mean = _optional_float(positive.get("score_mean"))
        negative_mean = _optional_float(negative.get("score_mean"))
        positive_focus = _optional_float(positive.get("heatmap_max_over_mean"))
        negative_focus = _optional_float(negative.get("heatmap_max_over_mean"))
        pos_minus_neg_mean = _subtract_optional(positive_mean, negative_mean)
        pos_focus_minus_neg_focus = _subtract_optional(positive_focus, negative_focus)
        deltas.append({
            "term": positive.get("term") or negative.get("term"),
            "concept_uid": positive.get("concept_uid") or negative.get("concept_uid") or concept_key,
            "eligible_call_index": call,
            "positive_score_mean": positive_mean,
            "negative_score_mean": negative_mean,
            "positive_heatmap_max_over_mean": positive_focus,
            "negative_heatmap_max_over_mean": negative_focus,
            "pos_minus_neg_mean": pos_minus_neg_mean,
            "pos_focus_minus_neg_focus": pos_focus_minus_neg_focus,
            "interpretation": _branch_delta_interpretation(
                positive_mean=positive_mean,
                negative_mean=negative_mean,
                positive_focus=positive_focus,
                negative_focus=negative_focus,
            ),
        })
    return deltas


def _branch_delta_interpretation(
    *,
    positive_mean: float | None,
    negative_mean: float | None,
    positive_focus: float | None,
    negative_focus: float | None,
) -> str:
    if (
        positive_focus is not None
        and positive_mean is not None
        and negative_mean is not None
        and positive_focus >= 1.15
        and positive_mean > negative_mean
    ):
        return "positive-localized"
    if negative_focus is not None and negative_focus < 1.05:
        return "negative-uniform"
    if (
        positive_focus is not None
        and negative_focus is not None
        and positive_focus < 1.15
        and negative_focus < 1.15
    ):
        return "both-diffuse"
    return "branch-ambiguous"


def _concept_fallback_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(key) or "")
        for key in ("term", "token_source", "occurrence_index", "token_indices")
    )


def _subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Anima Concept Survey Report",
        "",
        f"- run_id: {result.get('selected_run_id') or 'mixed'}",
        f"- run_count_in_input: {result.get('run_count_in_input')}",
        f"- mixed_runs: {result.get('mixed_runs')}",
        f"- observations: {result['observation_count']}",
        f"- late_start_step: {result['late_start_step']}",
        "",
    ]
    selected_run = _selected_run_info(result)
    if selected_run:
        lines.extend([
            "## Run",
            "",
            f"- Run ID: {selected_run['run_id']}",
            f"- Prompt summary: {_short_text(selected_run.get('prompt_text'))}",
            f"- Branches: {selected_run.get('branches') or ''}",
            f"- Concept terms: {selected_run.get('concept_terms') or ''}",
            "",
        ])
    if result.get("available_runs"):
        lines.extend([
            "## Available Runs",
            "",
            "| run_id | observations | branches | concept_terms |",
            "| --- | --- | --- | --- |",
        ])
        for run in result["available_runs"]:
            lines.append(
                f"| {run['run_id']} | {run['observation_count']} | "
                f"{run.get('branches') or ''} | {run.get('concept_terms') or ''} |"
            )
        lines.append("")
    lines.extend([
        "## Recommended LoRA Targets",
        "",
        "| rank | call | branch | score | late_score |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in result["recommended_lora_targets"][:20]:
        lines.append(
            f"| {row['rank']} | {row['eligible_call_index']} | {row['branch']} | "
            f"{row['mean_top_token_score']} | {row['late_mean_top_token_score']} |"
        )
    if result.get("recommended_concept_targets"):
        lines.extend([
            "",
            "## Recommended Concept Targets",
            "",
            "| rank | term | call | branch | rank_score | mean | mean/uniform | focus | warning |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in result["recommended_concept_targets"][:20]:
            warning = "near-uniform" if row.get("near_uniform") else ""
            lines.append(
                f"| {row['rank']} | {row.get('term')} | {row.get('eligible_call_index')} | {row.get('branch')} | "
                f"{row.get('rank_score')} | {row.get('score_mean')} | {row.get('score_mean_over_uniform')} | "
                f"{row.get('heatmap_max_over_mean')} | {warning} |"
            )
        positive_targets = [row for row in result["recommended_concept_targets"] if row.get("branch") == "positive"]
        if positive_targets:
            lines.extend([
                "",
                "## Positive Concept Targets",
                "",
                "| rank | term | call | rank_score | mean | mean/uniform | focus |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ])
            for row in positive_targets[:20]:
                lines.append(
                    f"| {row.get('rank')} | {row.get('term')} | {row.get('eligible_call_index')} | "
                    f"{row.get('rank_score')} | {row.get('score_mean')} | {row.get('score_mean_over_uniform')} | "
                    f"{row.get('heatmap_max_over_mean')} |"
                )
        negative_targets = [row for row in result["recommended_concept_targets"] if row.get("branch") == "negative"]
        if negative_targets:
            lines.extend([
                "",
                "## Negative Concept Diagnostics",
                "",
                "| rank | term | call | rank_score | mean | mean/uniform | focus | warning |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ])
            for row in negative_targets[:20]:
                warning = "near-uniform" if row.get("near_uniform") else ""
                lines.append(
                    f"| {row.get('rank')} | {row.get('term')} | {row.get('eligible_call_index')} | "
                    f"{row.get('rank_score')} | {row.get('score_mean')} | {row.get('score_mean_over_uniform')} | "
                    f"{row.get('heatmap_max_over_mean')} | {warning} |"
                )
    if result.get("survey_branch_concept_delta"):
        lines.extend([
            "",
            "## Branch Delta",
            "",
            "| term | call | pos_mean | neg_mean | pos_focus | neg_focus | mean_delta | focus_delta | interpretation |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in result["survey_branch_concept_delta"][:20]:
            lines.append(
                f"| {row.get('term')} | {row.get('eligible_call_index')} | "
                f"{row.get('positive_score_mean')} | {row.get('negative_score_mean')} | "
                f"{row.get('positive_heatmap_max_over_mean')} | {row.get('negative_heatmap_max_over_mean')} | "
                f"{row.get('pos_minus_neg_mean')} | {row.get('pos_focus_minus_neg_focus')} | "
                f"{row.get('interpretation')} |"
            )
    if result.get("preview_warnings"):
        lines.extend(["", "## Preview Warnings", ""])
        for warning in result["preview_warnings"]:
            lines.append(f"- {warning['message']}")
    lines.extend(["", "## Top Tokens", "", "| rank | call | branch | token | text | mean | max | entropy |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in result["summary_by_token"][:20]:
        lines.append(
            f"| {row['rank']} | {row['eligible_call_index']} | {row['branch']} | {row['token_index']} | {row.get('token_text')} | "
            f"{row['score_mean']} | {row['score_max']} | {row['score_entropy']} |"
        )
    if result.get("summary_by_concept"):
        lines.extend(["", "## Concepts", "", "| call | branch | term | mean | max | entropy | tokens |", "| --- | --- | --- | --- | --- | --- | --- |"])
        for row in result["summary_by_concept"][:20]:
            lines.append(
                f"| {row['eligible_call_index']} | {row['branch']} | {row['term']} | "
                f"{row['score_mean']} | {row['score_max']} | {row['score_entropy']} | {row['token_texts']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _numbers(values: Iterable[Any]) -> list[float]:
    parsed = []
    for value in values:
        parsed_value = _optional_float(value)
        if parsed_value is not None:
            parsed.append(parsed_value)
    return parsed


def _max(values: Iterable[Any]) -> float | None:
    parsed = [float(value) for value in values if value is not None]
    return None if not parsed else max(parsed)


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _any_true(values: Iterable[Any]) -> bool:
    return any(_bool_value(value) for value in values)


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _mode_text(values: Iterable[Any]) -> Any:
    counts: dict[Any, int] = {}
    for value in values:
        if value in (None, ""):
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def _mode_json(values: Iterable[Any]) -> str | None:
    encoded = [
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        for value in values
        if value not in (None, "")
    ]
    return _mode_text(encoded)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _record_concept_terms(record: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for term in record.get("terms") or record.get("concept_terms") or []:
        if term not in (None, ""):
            _append_unique(terms, str(term))
    for concept in record.get("concept_scores") or []:
        term = concept.get("term")
        if term not in (None, ""):
            _append_unique(terms, str(term))
    return terms


def _run_info_dict(run: RunInfo) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "first_record_index": run.first_record_index,
        "last_record_index": run.last_record_index,
        "observation_count": run.observation_count,
        "events": dict(run.events),
        "prompt_text": run.prompt_text,
        "branches": list(run.branches),
        "concept_terms": list(run.concept_terms),
    }


def _run_info_csv_row(run: RunInfo) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "first_record_index": run.first_record_index,
        "last_record_index": run.last_record_index,
        "observation_count": run.observation_count,
        "event_counts": json.dumps(run.events, ensure_ascii=False, sort_keys=True),
        "prompt_text": run.prompt_text,
        "prompt_summary": _short_text(run.prompt_text),
        "branches": ", ".join(run.branches),
        "concept_terms": ", ".join(run.concept_terms),
    }


def _selected_run_info(result: dict[str, Any]) -> dict[str, Any] | None:
    selected = result.get("selected_run_id")
    if not selected:
        return None
    for run in result.get("available_runs") or []:
        if run.get("run_id") == selected:
            return run
    return None


def _short_text(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
