from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_images(
    baseline_path: str | Path,
    variant_path: str | Path,
    out_dir: str | Path,
    *,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError("PIL and numpy are required for image difference reporting") from exc

    baseline_file = Path(baseline_path)
    variant_file = Path(variant_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline = Image.open(baseline_file).convert("RGB")
    variant = Image.open(variant_file).convert("RGB")
    if baseline.size != variant.size:
        raise ValueError(f"size mismatch: baseline={baseline.size} variant={variant.size}")

    baseline_arr = np.asarray(baseline, dtype=np.float32)
    variant_arr = np.asarray(variant, dtype=np.float32)
    diff = np.abs(variant_arr - baseline_arr)
    pixel_changed = np.any(diff > 0.0, axis=-1)
    width, height = baseline.size
    whole_delta_sum = float(diff.sum())

    summary: dict[str, Any] = {
        "schema_version": 1,
        "baseline_path": str(baseline_file),
        "variant_path": str(variant_file),
        "width": int(width),
        "height": int(height),
        "channels": 3,
        "mean_abs_delta": float(diff.mean()),
        "max_abs_delta": float(diff.max()),
        "p95_abs_delta": float(np.percentile(diff, 95)),
        "changed_pixel_ratio": float(pixel_changed.mean()),
        "sum_abs_delta": whole_delta_sum,
        "roi_bbox": None,
        "roi_mean_abs_delta": None,
        "roi_sum_abs_delta": None,
        "roi_delta_over_whole_delta": None,
    }

    if roi_bbox is not None:
        x, y, w, h = _validate_roi_bbox(roi_bbox, width, height)
        roi = diff[y:y + h, x:x + w, :]
        roi_delta_sum = float(roi.sum())
        summary.update({
            "roi_bbox": [x, y, w, h],
            "roi_mean_abs_delta": float(roi.mean()),
            "roi_sum_abs_delta": roi_delta_sum,
            "roi_delta_over_whole_delta": roi_delta_sum / whole_delta_sum if whole_delta_sum else None,
        })

    diff_image = Image.fromarray(diff.clip(0, 255).astype(np.uint8), mode="RGB")
    diff_path = out / "diff_abs.png"
    summary_path = out / "diff_summary.json"
    report_path = out / "diff_report.md"
    diff_image.save(diff_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(image_diff_report_markdown(summary), encoding="utf-8")
    return {
        **summary,
        "diff_abs_png": str(diff_path),
        "diff_summary_json": str(summary_path),
        "diff_report_md": str(report_path),
    }


def parse_roi_bbox(value: str | None) -> tuple[int, int, int, int] | None:
    if value in (None, ""):
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("--roi-bbox must use x,y,w,h")
    try:
        x, y, w, h = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--roi-bbox must contain integers: x,y,w,h") from exc
    return x, y, w, h


def image_diff_report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Intervention Image Difference Report",
        "",
        "This report measures fixed-seed image differences. A manual ROI is a user-supplied measurement box, not an automatic semantic segmentation mask.",
        "",
        f"- baseline: {summary.get('baseline_path')}",
        f"- variant: {summary.get('variant_path')}",
        f"- size: {summary.get('width')}x{summary.get('height')}",
        f"- mean_abs_delta: {summary.get('mean_abs_delta')}",
        f"- max_abs_delta: {summary.get('max_abs_delta')}",
        f"- p95_abs_delta: {summary.get('p95_abs_delta')}",
        f"- changed_pixel_ratio: {summary.get('changed_pixel_ratio')}",
        f"- sum_abs_delta: {summary.get('sum_abs_delta')}",
        "",
    ]
    if summary.get("roi_bbox") is not None:
        lines.extend([
            "## Manual ROI",
            "",
            f"- roi_bbox: {summary.get('roi_bbox')}",
            f"- roi_mean_abs_delta: {summary.get('roi_mean_abs_delta')}",
            f"- roi_sum_abs_delta: {summary.get('roi_sum_abs_delta')}",
            f"- roi_delta_over_whole_delta: {summary.get('roi_delta_over_whole_delta')}",
            "",
        ])
    lines.extend([
        "## Interpretation Notes",
        "",
        "- Only compare images generated with the same seed, sampler, steps, CFG, negative prompt, and latent size.",
        "- Pixel differences do not identify which prompt token caused a region by themselves.",
        "- Combine this report with intervention JSONL records and prompt-pair controls.",
        "",
    ])
    return "\n".join(lines)


def _validate_roi_bbox(roi_bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = roi_bbox
    if w <= 0 or h <= 0:
        raise ValueError("--roi-bbox width and height must be positive")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"--roi-bbox is outside image bounds: bbox={roi_bbox} image={width}x{height}")
    return x, y, w, h
