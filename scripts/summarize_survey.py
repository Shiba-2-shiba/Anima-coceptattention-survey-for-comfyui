from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anima_concept_survey.reporting import (
    filter_records_by_run_id,
    iter_json_records,
    latest_run_id,
    list_runs,
    summarize_records,
    write_summary_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Anima Concept Survey JSONL into JSON, CSV, and Markdown reports.")
    parser.add_argument("jsonl_path")
    parser.add_argument("--out-dir")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--late-start-step", type=int, default=None)
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--latest-run", action="store_true")
    parser.add_argument("--allow-mixed-runs", action="store_true")
    args = parser.parse_args(argv)

    records = list(iter_json_records(args.jsonl_path))
    runs = list_runs(records)
    if args.list_runs:
        print(json.dumps([_run_json(run) for run in runs], indent=2, sort_keys=True))
        return 0

    if not args.out_dir:
        parser.error("--out-dir is required unless --list-runs is used")
    if args.run_id and args.latest_run:
        parser.error("--run-id and --latest-run are mutually exclusive")
    if args.allow_mixed_runs and (args.run_id or args.latest_run):
        parser.error("--allow-mixed-runs cannot be combined with --run-id or --latest-run")

    selected_run_id = args.run_id
    selected_records = records
    mixed_runs = False
    if args.latest_run:
        selected_run_id = latest_run_id(records)
        if selected_run_id is None:
            print("No run_id values found in JSONL.", file=sys.stderr)
            return 2

    if selected_run_id:
        selected_records = filter_records_by_run_id(records, selected_run_id)
        if not selected_records:
            print(f"Run ID not found: {selected_run_id}", file=sys.stderr)
            _print_available_runs(runs)
            return 2
    elif len(runs) > 1 and not args.allow_mixed_runs:
        print("Multiple run_id values found. Specify --run-id, --latest-run, or --allow-mixed-runs.", file=sys.stderr)
        _print_available_runs(runs)
        return 2
    elif len(runs) > 1:
        mixed_runs = True

    result = summarize_records(
        selected_records,
        top_k=args.top_k,
        late_start_step=args.late_start_step,
        selected_run_id=selected_run_id,
        available_runs=runs,
        run_count_in_input=len(runs),
        mixed_runs=mixed_runs,
    )
    paths = write_summary_outputs(result, args.out_dir)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


def _run_json(run) -> dict:
    return {
        "run_id": run.run_id,
        "first_record_index": run.first_record_index,
        "last_record_index": run.last_record_index,
        "observation_count": run.observation_count,
        "events": run.events,
        "prompt_text": run.prompt_text,
        "branches": list(run.branches),
        "concept_terms": list(run.concept_terms),
    }


def _print_available_runs(runs) -> None:
    if not runs:
        print("Available runs: none", file=sys.stderr)
        return
    print("Available runs:", file=sys.stderr)
    for run in runs:
        branches = ", ".join(run.branches) or "-"
        print(f"- {run.run_id} observations={run.observation_count} branches={branches}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
