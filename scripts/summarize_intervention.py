from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anima_concept_survey.intervention_reporting import (
    summarize_intervention_records,
    write_intervention_summary_outputs,
)
from anima_concept_survey.reporting import iter_json_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Anima Concept Intervention JSONL into JSON, CSV, and Markdown reports.")
    parser.add_argument("jsonl_path")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    records = list(iter_json_records(args.jsonl_path))
    result = summarize_intervention_records(records)
    paths = write_intervention_summary_outputs(result, args.out_dir)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
