import tempfile
import unittest
from pathlib import Path

from anima_concept_survey.reporting import summarize_records, write_summary_outputs


class ReportingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
