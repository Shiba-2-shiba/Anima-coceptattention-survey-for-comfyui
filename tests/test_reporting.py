import tempfile
import unittest
import json
import subprocess
import sys
from pathlib import Path

from anima_concept_survey.reporting import (
    filter_records_by_run_id,
    latest_run_id,
    list_runs,
    summarize_records,
    write_summary_outputs,
)


class ReportingTests(unittest.TestCase):
    def test_list_runs_detects_run_ids_and_latest_by_file_order(self):
        records = [
            {"event": "concept_match_summary", "run_id": "survey-a", "prompt_text": "first", "terms": ["big breasts"]},
            {"event": "attention_observation", "run_id": "survey-a", "branch": "positive"},
            {"event": "attention_observation", "run_id": "survey-b", "branch": "negative", "prompt_text": "second"},
            {"event": "run_summary", "run_id": "survey-b", "prompt_text": "second"},
            {"event": "attention_fallback", "run_id": "survey-a"},
        ]

        runs = list_runs(records)

        self.assertEqual([run.run_id for run in runs], ["survey-a", "survey-b"])
        self.assertEqual(runs[0].first_record_index, 0)
        self.assertEqual(runs[0].last_record_index, 4)
        self.assertEqual(runs[0].observation_count, 1)
        self.assertEqual(runs[0].events["attention_observation"], 1)
        self.assertEqual(runs[0].events["attention_fallback"], 1)
        self.assertEqual(runs[0].prompt_text, "first")
        self.assertEqual(runs[0].branches, ("positive",))
        self.assertEqual(runs[0].concept_terms, ("big breasts",))
        self.assertEqual(latest_run_id(records), "survey-a")

    def test_filter_records_by_run_id_selects_only_requested_run(self):
        records = [
            {"event": "attention_observation", "run_id": "survey-a"},
            {"event": "attention_observation", "run_id": "survey-b"},
            {"event": "run_summary", "run_id": "survey-a"},
        ]

        filtered = filter_records_by_run_id(records, "survey-a")

        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(record["run_id"] == "survey-a" for record in filtered))

    def test_summarize_records_ranks_calls_and_tokens(self):
        records = [
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 7,
                "branch": "positive",
                "estimated_logits_mib": 1.0,
                "token_scores": [
                    {"token_index": 10, "token_text": "dragon", "token_source": "l", "score_mean": 0.2, "score_max": 0.5, "score_entropy": 0.8},
                    {"token_index": 11, "token_text": "rock", "token_source": "l", "score_mean": 0.1, "score_max": 0.4, "score_entropy": 0.9},
                ],
            },
            {
                "event": "attention_observation",
                "step_index": 1,
                "eligible_call_index": 0,
                "branch": "positive",
                "estimated_logits_mib": 1.0,
                "token_scores": [
                    {"token_index": 5, "score_mean": 0.05, "score_max": 0.2, "score_entropy": 0.7},
                ],
            },
        ]
        result = summarize_records(records, top_k=2, late_start_step=1)
        self.assertEqual(result["observation_count"], 2)
        self.assertEqual(result["summary_by_token"][0]["token_index"], 10)
        self.assertEqual(result["summary_by_token"][0]["token_text"], "dragon")
        self.assertEqual(result["recommended_lora_targets"][0]["eligible_call_index"], 7)

    def test_write_summary_outputs(self):
        result = summarize_records([], top_k=2)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_summary_outputs(result, tmp)
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            self.assertTrue((Path(tmp) / "survey_by_concept.csv").exists())
            self.assertTrue((Path(tmp) / "survey_runs.csv").exists())
            self.assertTrue((Path(tmp) / "recommended_concept_targets.csv").exists())

    def test_summarize_records_aggregates_concept_scores_without_token_scores(self):
        records = [
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 7,
                "branch": "positive",
                "estimated_logits_mib": 1.0,
                "token_scores": [],
                "concept_scores": [
                    {
                        "term": "big breasts",
                        "score_mean": 0.2,
                        "score_max": 0.5,
                        "score_entropy": 0.8,
                        "token_indices": [0, 1],
                        "token_texts": ["big", " breasts"],
                        "token_sources": ["qwen", "qwen"],
                    },
                ],
            },
            {
                "event": "attention_observation",
                "step_index": 1,
                "eligible_call_index": 7,
                "branch": "positive",
                "estimated_logits_mib": 1.0,
                "token_scores": [],
                "concept_scores": [
                    {
                        "term": "big breasts",
                        "score_mean": 0.4,
                        "score_max": 0.7,
                        "score_entropy": 0.6,
                        "token_indices": [0, 1],
                        "token_texts": ["big", " breasts"],
                        "token_sources": ["qwen", "qwen"],
                    },
                ],
            },
        ]

        result = summarize_records(records, top_k=2)

        self.assertEqual(result["observation_count"], 2)
        self.assertEqual(result["summary_by_token"], [])
        self.assertEqual(result["recommended_lora_targets"][0]["mean_top_token_score"], None)
        self.assertEqual(len(result["summary_by_concept"]), 1)
        concept = result["summary_by_concept"][0]
        self.assertEqual(concept["term"], "big breasts")
        self.assertEqual(concept["observation_count"], 2)
        self.assertAlmostEqual(concept["score_mean"], 0.3)
        self.assertEqual(concept["score_max"], 0.7)
        self.assertEqual(concept["token_indices"], "[0, 1]")

    def test_summarize_records_keeps_distinct_concept_uids_separate(self):
        records = [
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 7,
                "branch": "positive",
                "estimated_logits_mib": 1.0,
                "token_scores": [],
                "concept_scores": [
                    {
                        "concept_uid": "bigbreasts__qwen__occ0__tok000-001",
                        "term": "big breasts",
                        "normalized_term": "bigbreasts",
                        "token_source": "qwen",
                        "occurrence_index": 0,
                        "score_mean": 0.2,
                        "score_max": 0.5,
                        "score_entropy": 0.8,
                        "token_indices": [0, 1],
                        "source_token_indices": [0, 1],
                        "token_texts": ["big", " breasts"],
                        "token_sources": ["qwen", "qwen"],
                    },
                    {
                        "concept_uid": "bigbreasts__qwen__occ1__tok003-004",
                        "term": "big breasts",
                        "normalized_term": "bigbreasts",
                        "token_source": "qwen",
                        "occurrence_index": 1,
                        "score_mean": 0.4,
                        "score_max": 0.7,
                        "score_entropy": 0.6,
                        "token_indices": [3, 4],
                        "source_token_indices": [3, 4],
                        "token_texts": ["big", " breasts"],
                        "token_sources": ["qwen", "qwen"],
                    },
                ],
            },
        ]

        result = summarize_records(records, top_k=2)

        self.assertEqual(len(result["summary_by_concept"]), 2)
        self.assertEqual(
            {row["concept_uid"] for row in result["summary_by_concept"]},
            {"bigbreasts__qwen__occ0__tok000-001", "bigbreasts__qwen__occ1__tok003-004"},
        )

    def test_summarize_records_recommends_concept_targets_from_concept_only_records(self):
        records = [
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 9,
                "branch": "positive",
                "estimated_logits_mib": 1.0,
                "token_scores": [],
                "concept_scores": [
                    _concept_score(
                        concept_uid="bigbreasts__qwen__occ0__tok027-028",
                        branch="positive",
                        score_mean=0.004,
                        heatmap_max_over_mean=2.0,
                        near_uniform=False,
                    ),
                ],
            },
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 13,
                "branch": "negative",
                "estimated_logits_mib": 1.0,
                "token_scores": [],
                "concept_scores": [
                    _concept_score(
                        concept_uid="bigbreasts__qwen__occ0__tok027-028",
                        branch="negative",
                        score_mean=0.004,
                        heatmap_max_over_mean=1.0,
                        near_uniform=True,
                    ),
                ],
            },
        ]

        result = summarize_records(records, top_k=2)

        self.assertEqual(len(result["recommended_concept_targets"]), 2)
        self.assertEqual(result["recommended_concept_targets"][0]["branch"], "positive")
        self.assertEqual(result["recommended_concept_targets"][0]["eligible_call_index"], 9)
        self.assertGreater(
            result["recommended_concept_targets"][0]["rank_score"],
            result["recommended_concept_targets"][1]["rank_score"],
        )
        concept = result["summary_by_concept"][0]
        self.assertIn("heatmap_max_over_mean", concept)
        self.assertIn("uniform_baseline", concept)
        self.assertIn("score_mean_over_uniform", concept)
        self.assertIn("near_uniform", concept)

        with tempfile.TemporaryDirectory() as tmp:
            write_summary_outputs(result, tmp)
            concept_targets = Path(tmp) / "recommended_concept_targets.csv"
            by_concept = Path(tmp) / "survey_by_concept.csv"
            report = Path(tmp) / "survey_report.md"
            self.assertIn("rank_score", concept_targets.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("heatmap_max_over_mean", by_concept.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("Recommended Concept Targets", report.read_text(encoding="utf-8"))

    def test_markdown_warns_for_near_uniform_and_weak_focus_previews(self):
        records = [
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 9,
                "branch": "positive",
                "token_scores": [],
                "concept_scores": [
                    _concept_score(
                        concept_uid="bigbreasts__qwen__occ0__tok027-028",
                        branch="positive",
                        score_mean=0.004,
                        heatmap_max_over_mean=1.5,
                        near_uniform=False,
                    ),
                ],
            },
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 13,
                "branch": "negative",
                "token_scores": [],
                "concept_scores": [
                    _concept_score(
                        concept_uid="bigbreasts__qwen__occ0__tok027-028",
                        branch="negative",
                        score_mean=0.004,
                        heatmap_max_over_mean=1.0,
                        near_uniform=True,
                    ),
                ],
            },
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 14,
                "branch": "negative",
                "token_scores": [],
                "concept_scores": [
                    _concept_score(
                        concept_uid="bigbreasts__qwen__occ1__tok030-031",
                        branch="negative",
                        score_mean=0.004,
                        heatmap_max_over_mean=1.1,
                        near_uniform=False,
                    ),
                ],
            },
        ]

        result = summarize_records(records, top_k=2)

        self.assertEqual([warning["severity"] for warning in result["preview_warnings"]], ["near_uniform", "weak_focus"])
        markdown = result["report_markdown"] if "report_markdown" in result else None
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_summary_outputs(result, tmp)
            markdown = Path(paths["report_md"]).read_text(encoding="utf-8")
        self.assertIn("negative / big breasts / call 13 is near-uniform", markdown)
        self.assertIn("negative / big breasts / call 14 has weak focus", markdown)
        self.assertIn("Preview PNG is min-max normalized and may overstate localization.", markdown)
        self.assertNotIn("positive / big breasts / call 9 is near-uniform", markdown)

    def test_branch_concept_delta_csv_and_markdown_sections(self):
        records = [
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 9,
                "branch": "positive",
                "token_scores": [],
                "concept_scores": [
                    _concept_score(
                        concept_uid="bigbreasts__qwen__occ0__tok027-028",
                        branch="positive",
                        score_mean=0.006,
                        heatmap_max_over_mean=1.5,
                        near_uniform=False,
                    ),
                ],
            },
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 9,
                "branch": "negative",
                "token_scores": [],
                "concept_scores": [
                    _concept_score(
                        concept_uid="bigbreasts__qwen__occ0__tok027-028",
                        branch="negative",
                        score_mean=0.004,
                        heatmap_max_over_mean=1.0,
                        near_uniform=True,
                    ),
                ],
            },
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 10,
                "branch": "positive",
                "token_scores": [],
                "concept_scores": [
                    _concept_score(
                        concept_uid="bigbreasts__qwen__occ1__tok030-031",
                        branch="positive",
                        score_mean=0.004,
                        heatmap_max_over_mean=1.05,
                        near_uniform=False,
                    ),
                ],
            },
            {
                "event": "attention_observation",
                "step_index": 0,
                "eligible_call_index": 10,
                "branch": "negative",
                "token_scores": [],
                "concept_scores": [
                    _concept_score(
                        concept_uid="bigbreasts__qwen__occ1__tok030-031",
                        branch="negative",
                        score_mean=0.004,
                        heatmap_max_over_mean=1.04,
                        near_uniform=True,
                    ),
                ],
            },
        ]

        result = summarize_records(records, top_k=2)

        self.assertEqual(len(result["survey_branch_concept_delta"]), 2)
        first = result["survey_branch_concept_delta"][0]
        self.assertEqual(first["eligible_call_index"], 9)
        self.assertEqual(first["interpretation"], "positive-localized")
        self.assertAlmostEqual(first["pos_minus_neg_mean"], 0.002)
        self.assertAlmostEqual(first["pos_focus_minus_neg_focus"], 0.5)
        self.assertEqual(result["survey_branch_concept_delta"][1]["interpretation"], "negative-uniform")

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_summary_outputs(result, tmp)
            delta_csv = Path(tmp) / "survey_branch_concept_delta.csv"
            report = Path(paths["report_md"]).read_text(encoding="utf-8")

            self.assertTrue(delta_csv.exists())
            self.assertIn("pos_minus_neg_mean", delta_csv.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("## Positive Concept Targets", report)
            self.assertIn("## Negative Concept Diagnostics", report)
            self.assertIn("## Branch Delta", report)
            self.assertIn("positive-localized", report)
            self.assertIn("negative-uniform", report)

    def test_summarize_survey_cli_refuses_mixed_runs_without_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "survey.jsonl"
            out_dir = Path(tmp) / "report"
            _write_jsonl(jsonl_path, _run_fixture_records())

            result = subprocess.run(
                [sys.executable, "scripts/summarize_survey.py", str(jsonl_path), "--out-dir", str(out_dir)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("survey-a", result.stderr)
            self.assertIn("survey-b", result.stderr)
            self.assertFalse((out_dir / "survey_summary.json").exists())

    def test_summarize_survey_cli_list_runs_and_filters_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "survey.jsonl"
            out_dir = Path(tmp) / "report"
            _write_jsonl(jsonl_path, _run_fixture_records())

            list_result = subprocess.run(
                [sys.executable, "scripts/summarize_survey.py", str(jsonl_path), "--list-runs"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(list_result.returncode, 0)
            self.assertIn("survey-a", list_result.stdout)
            self.assertIn("survey-b", list_result.stdout)

            run_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_survey.py",
                    str(jsonl_path),
                    "--out-dir",
                    str(out_dir),
                    "--run-id",
                    "survey-a",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run_result.returncode, 0)
            summary = json.loads((out_dir / "survey_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_run_id"], "survey-a")
            self.assertFalse(summary["mixed_runs"])
            self.assertEqual(summary["observation_count"], 1)
            self.assertEqual(summary["run_count_in_input"], 2)
            self.assertEqual(len(summary["available_runs"]), 2)

    def test_summarize_survey_cli_latest_and_allow_mixed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "survey.jsonl"
            latest_dir = Path(tmp) / "latest"
            mixed_dir = Path(tmp) / "mixed"
            _write_jsonl(jsonl_path, _run_fixture_records())

            latest_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_survey.py",
                    str(jsonl_path),
                    "--out-dir",
                    str(latest_dir),
                    "--latest-run",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(latest_result.returncode, 0)
            latest_summary = json.loads((latest_dir / "survey_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(latest_summary["selected_run_id"], "survey-b")
            self.assertEqual(latest_summary["observation_count"], 1)

            mixed_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_survey.py",
                    str(jsonl_path),
                    "--out-dir",
                    str(mixed_dir),
                    "--allow-mixed-runs",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(mixed_result.returncode, 0)
            mixed_summary = json.loads((mixed_dir / "survey_summary.json").read_text(encoding="utf-8"))
            self.assertIsNone(mixed_summary["selected_run_id"])
            self.assertTrue(mixed_summary["mixed_runs"])
            self.assertEqual(mixed_summary["observation_count"], 2)

    def test_compare_survey_runs_cli_writes_delta_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "survey.jsonl"
            out_dir = Path(tmp) / "compare"
            _write_jsonl(jsonl_path, _compare_fixture_records())

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/compare_survey_runs.py",
                    str(jsonl_path),
                    "--run-a",
                    "survey-a",
                    "--run-b",
                    "survey-b",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {
                "compare_summary.json",
                "concept_score_delta.csv",
                "concept_call_delta.csv",
                "branch_delta.csv",
                "prompt_token_span_delta.csv",
                "compare_report.md",
            }
            self.assertTrue(all((out_dir / name).exists() for name in expected))

            summary = json.loads((out_dir / "compare_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["run_a"], "survey-a")
            self.assertEqual(summary["run_b"], "survey-b")
            self.assertEqual(summary["run_a_branch_mode"], "positive_only")
            self.assertEqual(summary["run_b_branch_mode"], "both")

            score_delta = (out_dir / "concept_score_delta.csv").read_text(encoding="utf-8")
            self.assertIn("stronger_run", score_delta.splitlines()[0])
            self.assertIn("survey-a", score_delta)
            self.assertIn("big breasts", score_delta)

            span_delta = (out_dir / "prompt_token_span_delta.csv").read_text(encoding="utf-8")
            self.assertIn("[27, 28]", span_delta)
            self.assertIn("[24, 25]", span_delta)

            report = (out_dir / "compare_report.md").read_text(encoding="utf-8")
            self.assertIn("Run A stronger", report)
            self.assertIn("Prompt / Token Span Differences", report)


def _run_fixture_records():
    return [
        {"event": "concept_match_summary", "run_id": "survey-a", "prompt_text": "first prompt", "terms": ["big breasts"]},
        {
            "event": "attention_observation",
            "run_id": "survey-a",
            "step_index": 0,
            "eligible_call_index": 1,
            "branch": "positive",
            "prompt_text": "first prompt",
            "token_scores": [],
            "concept_scores": [],
        },
        {"event": "concept_match_summary", "run_id": "survey-b", "prompt_text": "second prompt", "terms": ["big breasts"]},
        {
            "event": "attention_observation",
            "run_id": "survey-b",
            "step_index": 0,
            "eligible_call_index": 2,
            "branch": "negative",
            "prompt_text": "second prompt",
            "token_scores": [],
            "concept_scores": [],
        },
    ]


def _compare_fixture_records():
    return [
        {
            "event": "concept_match_summary",
            "run_id": "survey-a",
            "prompt_text": "first prompt big breasts",
            "terms": ["big breasts"],
            "matches": [
                {
                    "concept_uid": "bigbreasts__qwen__occ0__tok027-028",
                    "term": "big breasts",
                    "normalized_term": "bigbreasts",
                    "token_source": "qwen",
                    "occurrence_index": 0,
                    "token_indices": [27, 28],
                    "source_token_indices": [27, 28],
                    "token_texts": [" big", " breasts"],
                },
            ],
        },
        {
            "event": "attention_observation",
            "run_id": "survey-a",
            "step_index": 0,
            "eligible_call_index": 9,
            "branch": "positive",
            "branch_mode": "positive_only",
            "prompt_text": "first prompt big breasts",
            "token_scores": [],
            "concept_scores": [
                _concept_score(
                    concept_uid="bigbreasts__qwen__occ0__tok027-028",
                    branch="positive",
                    score_mean=0.006,
                    heatmap_max_over_mean=1.8,
                    near_uniform=False,
                ),
            ],
        },
        {
            "event": "concept_match_summary",
            "run_id": "survey-b",
            "prompt_text": "second prompt big breasts",
            "terms": ["big breasts"],
            "matches": [
                {
                    "concept_uid": "bigbreasts__qwen__occ0__tok024-025",
                    "term": "big breasts",
                    "normalized_term": "bigbreasts",
                    "token_source": "qwen",
                    "occurrence_index": 0,
                    "token_indices": [24, 25],
                    "source_token_indices": [24, 25],
                    "token_texts": [" big", " breasts"],
                },
            ],
        },
        {
            "event": "attention_observation",
            "run_id": "survey-b",
            "step_index": 0,
            "eligible_call_index": 9,
            "branch": "positive",
            "branch_mode": "both",
            "prompt_text": "second prompt big breasts",
            "token_scores": [],
            "concept_scores": [
                _concept_score(
                    concept_uid="bigbreasts__qwen__occ0__tok024-025",
                    branch="positive",
                    score_mean=0.003,
                    heatmap_max_over_mean=1.2,
                    near_uniform=False,
                ),
            ],
        },
        {
            "event": "attention_observation",
            "run_id": "survey-b",
            "step_index": 0,
            "eligible_call_index": 9,
            "branch": "negative",
            "branch_mode": "both",
            "prompt_text": "second prompt big breasts",
            "token_scores": [],
            "concept_scores": [
                _concept_score(
                    concept_uid="bigbreasts__qwen__occ0__tok024-025",
                    branch="negative",
                    score_mean=0.002,
                    heatmap_max_over_mean=1.0,
                    near_uniform=True,
                ),
            ],
        },
    ]


def _concept_score(*, concept_uid, branch, score_mean, heatmap_max_over_mean, near_uniform):
    del branch
    return {
        "concept_uid": concept_uid,
        "term": "big breasts",
        "normalized_term": "bigbreasts",
        "token_source": "qwen",
        "occurrence_index": 0,
        "score_mean": score_mean,
        "score_max": score_mean * 4,
        "score_entropy": 0.5,
        "token_indices": [27, 28],
        "source_token_indices": [27, 28],
        "token_texts": [" big", " breasts"],
        "token_sources": ["qwen", "qwen"],
        "heatmap_mean": score_mean,
        "heatmap_max": score_mean * heatmap_max_over_mean,
        "heatmap_std": 0.001,
        "heatmap_max_over_mean": heatmap_max_over_mean,
        "uniform_baseline": 0.002,
        "score_mean_over_uniform": score_mean / 0.002,
        "near_uniform": near_uniform,
    }


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
