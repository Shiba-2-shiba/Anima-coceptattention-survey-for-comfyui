from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anima_concept_survey.reporting import iter_json_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print Anima Concept Survey JSONL records as pretty JSON.")
    parser.add_argument("jsonl_path")
    parser.add_argument("--event", default="", help="Optional event filter.")
    args = parser.parse_args(argv)

    for record in iter_json_records(args.jsonl_path):
        if args.event and record.get("event") != args.event:
            continue
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
