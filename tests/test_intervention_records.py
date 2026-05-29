import json
import unittest
from collections import Counter

from anima_concept_survey.concepts import ConceptTokenMatch
from anima_concept_survey.intervention_config import InterventionConfig
from anima_concept_survey.intervention_records import (
    intervention_fallback_record,
    intervention_observation_record,
    intervention_plan_record,
    intervention_skipped_record,
    intervention_summary_record,
)
from anima_concept_survey.progress import ProgressInfo


def sample_match() -> ConceptTokenMatch:
    return ConceptTokenMatch(
        term="big breasts",
        normalized_term="bigbreasts",
        token_source="qwen3_06b",
        token_indices=(78, 79),
        source_token_indices=(78, 79),
        token_texts=(" big", " breasts"),
        token_ids=(2409, 36349),
        occurrence_index=0,
        concept_uid="bigbreasts__qwen3_06b__occ0__tok078-079",
    )


def sample_progress() -> ProgressInfo:
    return ProgressInfo(index=4, num_steps=30, last_index=29, progress=0.25, sigma=0.7)


class InterventionRecordTests(unittest.TestCase):
    def test_plan_record_is_json_serializable_and_includes_config_and_matches(self):
        config = InterventionConfig(
            mode="shadow",
            intervention_kind="attention_logit_bias",
            prompt_text="1girl, big breasts",
            intervention_terms="big breasts",
            target_call_indices="10,14",
            target_step_indices="4-6",
            branch_mode="positive_only",
            logit_bias=-5.0,
        )

        record = intervention_plan_record("intervention-1", config, [sample_match()])

        json.dumps(record)
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["event"], "intervention_plan")
        self.assertEqual(record["run_id"], "intervention-1")
        self.assertEqual(record["intervention_kind"], "attention_logit_bias")
        self.assertEqual(record["matches"][0]["concept_uid"], "bigbreasts__qwen3_06b__occ0__tok078-079")
        self.assertEqual(record["matches"][0]["token_ids"], [2409, 36349])

    def test_observation_record_contains_progress_match_and_delta_stats(self):
        record = intervention_observation_record(
            run_id="intervention-1",
            progress=sample_progress(),
            eligible_call_index=10,
            branch="positive",
            block="double_blocks.10.attn",
            metadata={"module_path": "double_blocks.10.attn"},
            intervention_kind="attention_logit_bias",
            match=sample_match(),
            original_score_mean=0.2,
            modified_score_mean=0.05,
            mean_abs_prob_delta=0.01,
            max_abs_prob_delta=0.2,
            output_delta_norm=1.5,
            estimated_logits_mib=128.0,
            strength=1.0,
            logit_bias=-5.0,
        )

        json.dumps(record)
        self.assertEqual(record["event"], "intervention_observation")
        self.assertEqual(record["step_index"], 4)
        self.assertEqual(record["eligible_call_index"], 10)
        self.assertAlmostEqual(record["score_delta"], -0.15)
        self.assertEqual(record["token_texts"], [" big", " breasts"])
        self.assertEqual(record["metadata"], {"module_path": "double_blocks.10.attn"})

    def test_skipped_fallback_and_summary_records_are_json_serializable(self):
        skipped = intervention_skipped_record(
            run_id="intervention-1",
            progress=sample_progress(),
            reason="target_call_skipped",
            eligible_call_index=3,
            branch="positive",
        )
        fallback = intervention_fallback_record(
            run_id="intervention-1",
            progress=None,
            reason="not_cross_attention",
        )
        summary = intervention_summary_record(
            run_id="intervention-1",
            observed_calls=5,
            modified_calls=2,
            skipped_calls=3,
            fallback_calls=1,
            fallback_reasons=Counter({"not_cross_attention": 1}),
        )

        json.dumps(skipped)
        json.dumps(fallback)
        json.dumps(summary)
        self.assertEqual(skipped["event"], "intervention_skipped")
        self.assertEqual(fallback["step_index"], None)
        self.assertEqual(summary["fallback_reasons"], {"not_cross_attention": 1})


if __name__ == "__main__":
    unittest.main()
