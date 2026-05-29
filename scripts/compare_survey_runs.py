from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anima_concept_survey.reporting import filter_records_by_run_id, iter_json_records, list_runs, summarize_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two Anima Concept Survey run IDs from one JSONL file.")
    parser.add_argument("jsonl_path")
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    records = list(iter_json_records(args.jsonl_path))
    runs = list_runs(records)
    run_ids = {run.run_id for run in runs}
    missing = [run_id for run_id in (args.run_a, args.run_b) if run_id not in run_ids]
    if missing:
        print(f"Run ID not found: {', '.join(missing)}", file=sys.stderr)
        print("Available runs:", file=sys.stderr)
        for run in runs:
            print(f"- {run.run_id}", file=sys.stderr)
        return 2

    run_a_records = filter_records_by_run_id(records, args.run_a)
    run_b_records = filter_records_by_run_id(records, args.run_b)
    run_a_summary = summarize_records(run_a_records, selected_run_id=args.run_a)
    run_b_summary = summarize_records(run_b_records, selected_run_id=args.run_b)

    comparison = build_comparison(
        run_a=args.run_a,
        run_b=args.run_b,
        run_a_records=run_a_records,
        run_b_records=run_b_records,
        run_a_summary=run_a_summary,
        run_b_summary=run_b_summary,
    )
    paths = write_comparison_outputs(comparison, args.out_dir)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


def build_comparison(
    *,
    run_a: str,
    run_b: str,
    run_a_records: list[dict[str, Any]],
    run_b_records: list[dict[str, Any]],
    run_a_summary: dict[str, Any],
    run_b_summary: dict[str, Any],
) -> dict[str, Any]:
    concept_score_delta = _concept_score_delta(
        run_a,
        run_b,
        run_a_summary.get("summary_by_concept", []),
        run_b_summary.get("summary_by_concept", []),
    )
    concept_call_delta = _concept_call_delta(
        run_a,
        run_b,
        run_a_summary.get("summary_by_concept", []),
        run_b_summary.get("summary_by_concept", []),
    )
    prompt_token_span_delta = _prompt_token_span_delta(run_a_records, run_b_records)
    branch_delta = _combined_branch_delta(
        run_a,
        run_b,
        run_a_summary.get("survey_branch_concept_delta", []),
        run_b_summary.get("survey_branch_concept_delta", []),
    )
    summary = {
        "schema_version": 1,
        "run_a": run_a,
        "run_b": run_b,
        "run_a_observation_count": run_a_summary.get("observation_count", 0),
        "run_b_observation_count": run_b_summary.get("observation_count", 0),
        "run_a_branch_mode": _mode_value(record.get("branch_mode") for record in run_a_records),
        "run_b_branch_mode": _mode_value(record.get("branch_mode") for record in run_b_records),
        "run_a_prompt": _first_prompt(run_a_records),
        "run_b_prompt": _first_prompt(run_b_records),
        "concept_score_delta_count": len(concept_score_delta),
        "concept_call_delta_count": len(concept_call_delta),
        "branch_delta_count": len(branch_delta),
        "prompt_token_span_delta_count": len(prompt_token_span_delta),
    }
    return {
        "summary": summary,
        "concept_score_delta": concept_score_delta,
        "concept_call_delta": concept_call_delta,
        "branch_delta": branch_delta,
        "prompt_token_span_delta": prompt_token_span_delta,
    }


def write_comparison_outputs(comparison: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "compare_summary_json": out / "compare_summary.json",
        "concept_score_delta_csv": out / "concept_score_delta.csv",
        "concept_call_delta_csv": out / "concept_call_delta.csv",
        "branch_delta_csv": out / "branch_delta.csv",
        "prompt_token_span_delta_csv": out / "prompt_token_span_delta.csv",
        "compare_report_md": out / "compare_report.md",
    }
    paths["compare_summary_json"].write_text(
        json.dumps(comparison["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(paths["concept_score_delta_csv"], comparison["concept_score_delta"])
    _write_csv(paths["concept_call_delta_csv"], comparison["concept_call_delta"])
    _write_csv(paths["branch_delta_csv"], comparison["branch_delta"])
    _write_csv(paths["prompt_token_span_delta_csv"], comparison["prompt_token_span_delta"])
    paths["compare_report_md"].write_text(_compare_markdown(comparison), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _concept_score_delta(
    run_a: str,
    run_b: str,
    run_a_rows: list[dict[str, Any]],
    run_b_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    a_groups = _aggregate_concepts(run_a_rows, include_call=False)
    b_groups = _aggregate_concepts(run_b_rows, include_call=False)
    rows = []
    for key in sorted(set(a_groups) | set(b_groups)):
        a = a_groups.get(key, {})
        b = b_groups.get(key, {})
        a_mean = _optional_float(a.get("score_mean"))
        b_mean = _optional_float(b.get("score_mean"))
        a_focus = _optional_float(a.get("heatmap_max_over_mean"))
        b_focus = _optional_float(b.get("heatmap_max_over_mean"))
        rows.append({
            "term": a.get("term") or b.get("term"),
            "token_source": a.get("token_source") or b.get("token_source"),
            "occurrence_index": a.get("occurrence_index") if a.get("occurrence_index") is not None else b.get("occurrence_index"),
            "branch": a.get("branch") or b.get("branch"),
            "run_a": run_a,
            "run_b": run_b,
            "run_a_score_mean": a_mean,
            "run_b_score_mean": b_mean,
            "score_mean_delta": _subtract_optional(a_mean, b_mean),
            "run_a_heatmap_max_over_mean": a_focus,
            "run_b_heatmap_max_over_mean": b_focus,
            "focus_delta": _subtract_optional(a_focus, b_focus),
            "stronger_run": _stronger_run(run_a, run_b, a_mean, b_mean),
        })
    rows.sort(key=lambda row: abs(row.get("score_mean_delta") or 0.0), reverse=True)
    return rows


def _concept_call_delta(
    run_a: str,
    run_b: str,
    run_a_rows: list[dict[str, Any]],
    run_b_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    a_groups = _aggregate_concepts(run_a_rows, include_call=True)
    b_groups = _aggregate_concepts(run_b_rows, include_call=True)
    rows = []
    for key in sorted(set(a_groups) | set(b_groups)):
        a = a_groups.get(key, {})
        b = b_groups.get(key, {})
        a_mean = _optional_float(a.get("score_mean"))
        b_mean = _optional_float(b.get("score_mean"))
        rows.append({
            "term": a.get("term") or b.get("term"),
            "token_source": a.get("token_source") or b.get("token_source"),
            "occurrence_index": a.get("occurrence_index") if a.get("occurrence_index") is not None else b.get("occurrence_index"),
            "branch": a.get("branch") or b.get("branch"),
            "eligible_call_index": a.get("eligible_call_index") if a.get("eligible_call_index") is not None else b.get("eligible_call_index"),
            "run_a": run_a,
            "run_b": run_b,
            "run_a_score_mean": a_mean,
            "run_b_score_mean": b_mean,
            "score_mean_delta": _subtract_optional(a_mean, b_mean),
            "run_a_heatmap_max_over_mean": _optional_float(a.get("heatmap_max_over_mean")),
            "run_b_heatmap_max_over_mean": _optional_float(b.get("heatmap_max_over_mean")),
            "stronger_run": _stronger_run(run_a, run_b, a_mean, b_mean),
        })
    rows.sort(key=lambda row: (row.get("eligible_call_index") is None, row.get("eligible_call_index") or -1, row.get("term") or ""))
    return rows


def _aggregate_concepts(rows: list[dict[str, Any]], *, include_call: bool) -> dict[tuple[Any, ...], dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = _concept_compare_key(row, include_call=include_call)
        groups.setdefault(key, []).append(row)
    return {key: _aggregate_group(items) for key, items in groups.items()}


def _aggregate_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    first = items[0]
    return {
        "term": first.get("term"),
        "token_source": first.get("token_source"),
        "occurrence_index": first.get("occurrence_index"),
        "branch": first.get("branch"),
        "eligible_call_index": first.get("eligible_call_index"),
        "score_mean": _mean(_numbers(item.get("score_mean") for item in items)),
        "heatmap_max_over_mean": _mean(_numbers(item.get("heatmap_max_over_mean") for item in items)),
    }


def _concept_compare_key(row: dict[str, Any], *, include_call: bool) -> tuple[Any, ...]:
    key = (
        row.get("term"),
        row.get("token_source"),
        row.get("occurrence_index"),
        row.get("branch"),
    )
    if include_call:
        return (*key, row.get("eligible_call_index"))
    return key


def _prompt_token_span_delta(run_a_records: list[dict[str, Any]], run_b_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_matches = _concept_matches_by_key(run_a_records)
    b_matches = _concept_matches_by_key(run_b_records)
    prompt_a = _first_prompt(run_a_records)
    prompt_b = _first_prompt(run_b_records)
    rows = []
    for key in sorted(set(a_matches) | set(b_matches)):
        a = a_matches.get(key, {})
        b = b_matches.get(key, {})
        a_tokens = a.get("token_indices")
        b_tokens = b.get("token_indices")
        rows.append({
            "term": a.get("term") or b.get("term"),
            "token_source": a.get("token_source") or b.get("token_source"),
            "occurrence_index": a.get("occurrence_index") if a.get("occurrence_index") is not None else b.get("occurrence_index"),
            "run_a_token_indices": json.dumps(a_tokens, ensure_ascii=False) if a_tokens is not None else None,
            "run_b_token_indices": json.dumps(b_tokens, ensure_ascii=False) if b_tokens is not None else None,
            "run_a_source_token_indices": json.dumps(a.get("source_token_indices"), ensure_ascii=False) if a.get("source_token_indices") is not None else None,
            "run_b_source_token_indices": json.dumps(b.get("source_token_indices"), ensure_ascii=False) if b.get("source_token_indices") is not None else None,
            "token_span_changed": a_tokens != b_tokens,
            "prompt_changed": prompt_a != prompt_b,
            "run_a_prompt": prompt_a,
            "run_b_prompt": prompt_b,
        })
    return rows


def _concept_matches_by_key(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    matches: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("event") != "concept_match_summary":
            continue
        for match in record.get("matches") or []:
            key = (
                match.get("term"),
                match.get("token_source"),
                match.get("occurrence_index"),
            )
            matches[key] = match
    return matches


def _combined_branch_delta(
    run_a: str,
    run_b: str,
    run_a_rows: list[dict[str, Any]],
    run_b_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for run_id, source_rows in ((run_a, run_a_rows), (run_b, run_b_rows)):
        for row in source_rows:
            rows.append({"run_id": run_id, **row})
    return rows


def _compare_markdown(comparison: dict[str, Any]) -> str:
    summary = comparison["summary"]
    lines = [
        "# Anima Concept Survey Run Comparison",
        "",
        f"- Run A: {summary['run_a']}",
        f"- Run B: {summary['run_b']}",
        f"- Run A branch mode: {summary.get('run_a_branch_mode')}",
        f"- Run B branch mode: {summary.get('run_b_branch_mode')}",
        "",
        "## Concept Score Delta",
        "",
        "| term | branch | run_a_mean | run_b_mean | delta | stronger_run |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    a_stronger = []
    for row in comparison["concept_score_delta"][:20]:
        if row.get("stronger_run") == summary["run_a"]:
            a_stronger.append(row)
        lines.append(
            f"| {row.get('term')} | {row.get('branch')} | {row.get('run_a_score_mean')} | "
            f"{row.get('run_b_score_mean')} | {row.get('score_mean_delta')} | {row.get('stronger_run')} |"
        )
    if a_stronger:
        lines.extend(["", "## Run A stronger", ""])
        for row in a_stronger[:10]:
            lines.append(f"- {row.get('term')} / {row.get('branch')}: delta={row.get('score_mean_delta')}")
    lines.extend([
        "",
        "## Prompt / Token Span Differences",
        "",
        "| term | run_a_tokens | run_b_tokens | token_span_changed | prompt_changed |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in comparison["prompt_token_span_delta"][:20]:
        lines.append(
            f"| {row.get('term')} | {row.get('run_a_token_indices')} | {row.get('run_b_token_indices')} | "
            f"{row.get('token_span_changed')} | {row.get('prompt_changed')} |"
        )
    if comparison["branch_delta"]:
        lines.extend([
            "",
            "## Branch Delta",
            "",
            "| run_id | term | call | interpretation |",
            "| --- | --- | --- | --- |",
        ])
        for row in comparison["branch_delta"][:20]:
            lines.append(
                f"| {row.get('run_id')} | {row.get('term')} | {row.get('eligible_call_index')} | "
                f"{row.get('interpretation')} |"
            )
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _first_prompt(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        prompt = record.get("prompt_text")
        if prompt not in (None, ""):
            return str(prompt)
    return None


def _mode_value(values: Iterable[Any]) -> Any:
    counts: dict[Any, int] = {}
    for value in values:
        if value in (None, ""):
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


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


def _subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _stronger_run(run_a: str, run_b: str, a_mean: float | None, b_mean: float | None) -> str | None:
    if a_mean is None and b_mean is None:
        return None
    if b_mean is None:
        return run_a
    if a_mean is None:
        return run_b
    if a_mean == b_mean:
        return "tie"
    return run_a if a_mean > b_mean else run_b


if __name__ == "__main__":
    raise SystemExit(main())
