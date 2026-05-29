from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anima_concept_survey.image_diff import compare_images, parse_roi_bbox


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare fixed-seed intervention images and write diff artifacts.")
    parser.add_argument("baseline_png")
    parser.add_argument("variant_png")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--roi-bbox", help="Optional manual ROI as x,y,w,h.")
    args = parser.parse_args(argv)

    try:
        result = compare_images(
            args.baseline_png,
            args.variant_png,
            args.out_dir,
            roi_bbox=parse_roi_bbox(args.roi_bbox),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "diff_abs_png": result["diff_abs_png"],
        "diff_summary_json": result["diff_summary_json"],
        "diff_report_md": result["diff_report_md"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
