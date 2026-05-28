from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def iter_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def attention_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("event") == "attention_observation"]


def summarize_records(records: Iterable[dict[str, Any]], top_k: int = 16, late_start_step: int | None = None) -> dict[str, Any]:
    observations = attention_records(records)
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
            concept_rows[(call, branch, term)].append({
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
        "summary_by_call": summary_by_call,
        "summary_by_step": summary_by_step,
        "summary_by_branch": summary_by_branch,
        "summary_by_token": summary_by_token,
        "summary_by_concept": summary_by_concept,
        "recommended_lora_targets": recommended_targets,
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
        "report_md": out / "survey_report.md",
    }
    paths["summary_json"].write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(paths["by_call_csv"], result["summary_by_call"])
    _write_csv(paths["by_step_csv"], result["summary_by_step"])
    _write_csv(paths["by_branch_csv"], result["summary_by_branch"])
    _write_csv(paths["by_token_csv"], result["summary_by_token"])
    _write_csv(paths["by_concept_csv"], result["summary_by_concept"])
    _write_csv(paths["recommended_lora_targets_csv"], result["recommended_lora_targets"])
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
    for (call, branch, term), items in concept_rows.items():
        score_means = [float(item["score_mean"]) for item in items if item.get("score_mean") is not None]
        score_maxes = [float(item["score_max"]) for item in items if item.get("score_max") is not None]
        entropies = [float(item["score_entropy"]) for item in items if item.get("score_entropy") is not None]
        rows.append({
            "term": term,
            "branch": branch,
            "eligible_call_index": call,
            "observation_count": len(items),
            "score_mean": _mean(score_means),
            "score_max": _max(score_maxes),
            "score_entropy": _mean(entropies),
            "token_indices": _mode_json(item.get("token_indices") for item in items),
            "token_texts": _mode_json(item.get("token_texts") for item in items),
            "token_sources": _mode_json(item.get("token_sources") for item in items),
        })
    rows.sort(key=lambda item: item["score_mean"] or 0.0, reverse=True)
    return rows


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
        f"- observations: {result['observation_count']}",
        f"- late_start_step: {result['late_start_step']}",
        "",
        "## Recommended LoRA Targets",
        "",
        "| rank | call | branch | score | late_score |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result["recommended_lora_targets"][:20]:
        lines.append(
            f"| {row['rank']} | {row['eligible_call_index']} | {row['branch']} | "
            f"{row['mean_top_token_score']} | {row['late_mean_top_token_score']} |"
        )
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


def _max(values: Iterable[Any]) -> float | None:
    parsed = [float(value) for value in values if value is not None]
    return None if not parsed else max(parsed)


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
