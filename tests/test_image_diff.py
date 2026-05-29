import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


try:
    from PIL import Image
except Exception:  # pragma: no cover - environment-dependent skip path
    Image = None


@unittest.skipIf(Image is None, "PIL is required for image diff tests")
class ImageDiffTests(unittest.TestCase):
    def test_compare_intervention_images_cli_writes_diff_summary_and_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.png"
            variant = tmp_path / "variant.png"
            out_dir = tmp_path / "diff"
            _write_rgb_image(baseline, [
                [(0, 0, 0), (0, 0, 0)],
                [(0, 0, 0), (0, 0, 0)],
            ])
            _write_rgb_image(variant, [
                [(10, 20, 30), (0, 0, 0)],
                [(0, 0, 0), (0, 0, 0)],
            ])

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/compare_intervention_images.py",
                    str(baseline),
                    str(variant),
                    "--out-dir",
                    str(out_dir),
                    "--roi-bbox",
                    "0,0,1,1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out_dir / "diff_abs.png").exists())
            self.assertTrue((out_dir / "diff_summary.json").exists())
            self.assertTrue((out_dir / "diff_report.md").exists())

            summary = json.loads((out_dir / "diff_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["width"], 2)
            self.assertEqual(summary["height"], 2)
            self.assertEqual(summary["max_abs_delta"], 30.0)
            self.assertEqual(summary["changed_pixel_ratio"], 0.25)
            self.assertEqual(summary["roi_bbox"], [0, 0, 1, 1])
            self.assertEqual(summary["roi_delta_over_whole_delta"], 1.0)

            report = (out_dir / "diff_report.md").read_text(encoding="utf-8")
            self.assertIn("manual ROI", report)
            self.assertIn("not an automatic semantic segmentation mask", report)

    def test_compare_intervention_images_cli_fails_on_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.png"
            variant = tmp_path / "variant.png"
            _write_rgb_image(baseline, [[(0, 0, 0)]])
            _write_rgb_image(variant, [[(0, 0, 0), (0, 0, 0)]])

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/compare_intervention_images.py",
                    str(baseline),
                    str(variant),
                    "--out-dir",
                    str(tmp_path / "diff"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("size mismatch", result.stderr.lower())


def _write_rgb_image(path, rows):
    image = Image.new("RGB", (len(rows[0]), len(rows)))
    for y, row in enumerate(rows):
        for x, pixel in enumerate(row):
            image.putpixel((x, y), pixel)
    image.save(path)


if __name__ == "__main__":
    unittest.main()
