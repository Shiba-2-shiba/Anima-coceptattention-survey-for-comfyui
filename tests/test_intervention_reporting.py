import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class InterventionReportingTests(unittest.TestCase):
    def test_summarize_intervention_cli_writes_json_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "intervention.jsonl"
            out_dir = Path(tmp) / "report"
            _write_jsonl(jsonl_path, _intervention_fixture_records())

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_intervention.py",
                    str(jsonl_path),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for name in {
                "intervention_summary.json",
                "intervention_by_call.csv",
                "intervention_by_concept.csv",
                "intervention_report.md",
            }:
                self.assertTrue((out_dir / name).exists(), name)

            summary = json.loads((out_dir / "intervention_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["observation_count"], 3)
            self.assertEqual(summary["fallback_count"], 1)
            self.assertEqual(summary["plan_count"], 1)

            by_call = list(csv.DictReader((out_dir / "intervention_by_call.csv").open(encoding="utf-8")))
            self.assertEqual(by_call[0]["eligible_call_index"], "5")
            self.assertEqual(by_call[0]["branch"], "positive")
            self.assertIn("abs_score_delta_mean", by_call[0])

            by_concept = list(csv.DictReader((out_dir / "intervention_by_concept.csv").open(encoding="utf-8")))
            self.assertEqual(by_concept[0]["concept_uid"], "bigbreasts__qwen__occ0__tok078-079")
            self.assertIn("output_delta_norm_max", by_concept[0])

            report = (out_dir / "intervention_report.md").read_text(encoding="utf-8")
            self.assertIn("Top Intervention Calls", report)
            self.assertIn("causal-probe heuristic", report)
            self.assertIn("mask", report)

    def test_summarize_intervention_cli_handles_empty_observation_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "intervention.jsonl"
            out_dir = Path(tmp) / "report"
            _write_jsonl(jsonl_path, [_plan_record()])

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_intervention.py",
                    str(jsonl_path),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out_dir / "intervention_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["observation_count"], 0)
            report = (out_dir / "intervention_report.md").read_text(encoding="utf-8")
            self.assertIn("No intervention observations found", report)


def _intervention_fixture_records():
    return [
        _plan_record(),
        _observation_record(call=5, branch="positive", delta=-0.2, output_delta=4.0),
        _observation_record(call=5, branch="positive", delta=-0.1, output_delta=2.0),
        _observation_record(
            call=7,
            branch="positive",
            concept_uid="hair__qwen__occ0__tok010",
            term="hair",
            delta=0.0,
            output_delta=0.0,
        ),
        {
            "schema_version": 1,
            "event": "intervention_fallback",
            "run_id": "intervention-a",
            "step_index": 0,
            "num_steps": 20,
            "eligible_call_index": None,
            "branch": None,
            "reason": "mask_shape_unsupported",
        },
    ]


def _plan_record():
    return {
        "schema_version": 1,
        "event": "intervention_plan",
        "run_id": "intervention-a",
        "mode": "intervene",
        "intervention_kind": "attention_logit_bias",
        "prompt_text": "1girl, big breasts",
        "intervention_terms": "big breasts",
        "target_call_indices": "5",
        "target_step_indices": "all",
        "branch_mode": "positive_only",
        "strength": 1.0,
        "logit_bias": -5.0,
        "matches": [
            {
                "concept_uid": "bigbreasts__qwen__occ0__tok078-079",
                "term": "big breasts",
                "normalized_term": "bigbreasts",
                "token_source": "qwen",
                "token_indices": [78, 79],
                "token_texts": ["big", " breasts"],
                "occurrence_index": 0,
            },
        ],
    }


def _observation_record(*, call, branch, delta, output_delta, concept_uid="bigbreasts__qwen__occ0__tok078-079", term="big breasts"):
    return {
        "schema_version": 1,
        "event": "intervention_observation",
        "run_id": "intervention-a",
        "step_index": 0,
        "num_steps": 20,
        "eligible_call_index": call,
        "branch": branch,
        "block": "unknown",
        "metadata": {},
        "intervention_kind": "attention_logit_bias",
        "concept_uid": concept_uid,
        "term": term,
        "normalized_term": term.replace(" ", ""),
        "token_source": "qwen",
        "token_indices": [78, 79],
        "token_texts": ["big", " breasts"],
        "occurrence_index": 0,
        "original_score_mean": 0.3,
        "modified_score_mean": 0.3 + delta,
        "score_delta": delta,
        "mean_abs_prob_delta": abs(delta) / 2,
        "max_abs_prob_delta": abs(delta),
        "output_delta_norm": output_delta,
        "estimated_logits_mib": 12.0,
        "strength": 1.0,
        "logit_bias": -5.0,
    }


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
