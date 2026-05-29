import unittest

from anima_concept_survey.intervention_config import (
    INTERVENTION_KINDS,
    INTERVENTION_MODES,
    InterventionConfig,
)
from anima_concept_survey.selectors import parse_step_index_scope


class InterventionConfigTests(unittest.TestCase):
    def test_defaults_are_noop_safe(self):
        config = InterventionConfig()

        self.assertEqual(config.mode, "shadow")
        self.assertEqual(config.intervention_kind, "attention_logit_bias")
        self.assertEqual(config.strength, 1.0)
        self.assertEqual(config.logit_bias, 0.0)
        config.validate()

    def test_validate_accepts_supported_modes_and_kinds(self):
        for mode in INTERVENTION_MODES:
            for kind in INTERVENTION_KINDS:
                with self.subTest(mode=mode, kind=kind):
                    InterventionConfig(mode=mode, intervention_kind=kind).validate()

    def test_validate_rejects_invalid_enums_and_numeric_ranges(self):
        invalid_configs = [
            InterventionConfig(mode="observe"),
            InterventionConfig(intervention_kind="token_logit_bias"),
            InterventionConfig(branch_mode="positive"),
            InterventionConfig(fail_mode="ignore"),
            InterventionConfig(strength=-0.1),
            InterventionConfig(max_steps=-1),
            InterventionConfig(max_logits_mib=0),
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()

    def test_validate_parses_call_and_step_scopes(self):
        config = InterventionConfig(target_call_indices="0,2-3", target_step_indices="4-6")

        self.assertEqual(config.target_call_scope(), {0, 2, 3})
        self.assertEqual(config.target_step_scope(), {4, 5, 6})
        config.validate()

    def test_parse_step_index_scope_matches_call_scope_rules(self):
        self.assertIsNone(parse_step_index_scope("all"))
        self.assertEqual(parse_step_index_scope("0"), {0})
        self.assertEqual(parse_step_index_scope("0,3,5"), {0, 3, 5})
        self.assertEqual(parse_step_index_scope("4-6"), {4, 5, 6})
        for invalid in ("", "-1", "6-4", "x", "1,,2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_step_index_scope(invalid)


if __name__ == "__main__":
    unittest.main()
