from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anima_concept_survey.reporting import iter_json_records, summarize_records, write_summary_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Anima Concept Survey JSONL into JSON, CSV, and Markdown reports.")
    parser.add_argument("jsonl_path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--late-start-step", type=int, default=None)
    args = parser.parse_args(argv)

    result = summarize_records(iter_json_records(args.jsonl_path), top_k=args.top_k, late_start_step=args.late_start_step)
    paths = write_summary_outputs(result, args.out_dir)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
