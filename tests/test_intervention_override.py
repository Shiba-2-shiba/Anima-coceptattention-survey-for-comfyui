import json
import tempfile
import unittest
from pathlib import Path

import torch

from anima_concept_survey.intervention import AnimaConceptInterventionAttentionOverride
from anima_concept_survey.intervention_config import InterventionConfig


def reference_attention(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    del mask, kwargs
    assert skip_reshape
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    out = torch.matmul(torch.softmax(logits, dim=-1).to(v.dtype), v)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], heads * q.shape[-1])


class SentinelAttention:
    def __init__(self):
        self.calls = 0
        self.reference_output = None

    def __call__(self, *args, **kwargs):
        del kwargs
        self.calls += 1
        q, _k, _v, heads = args[:4]
        self.reference_output = torch.full(
            (q.shape[0], q.shape[2], heads * q.shape[-1]),
            123.0,
            dtype=q.dtype,
            device=q.device,
        )
        return self.reference_output


def token_text_map():
    return {
        0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
        1: {"token_index": 1, "source_token_index": 1, "token_text": " breasts", "token_source": "qwen"},
        2: {"token_index": 2, "source_token_index": 2, "token_text": ",", "token_source": "qwen"},
    }


def tensors():
    torch.manual_seed(501)
    return (
        torch.randn(1, 2, 4, 3),
        torch.randn(1, 2, 3, 3),
        torch.randn(1, 2, 3, 3),
    )


def branch_tensors():
    torch.manual_seed(601)
    return (
        torch.randn(2, 2, 4, 3),
        torch.randn(2, 2, 3, 3),
        torch.randn(2, 2, 3, 3),
    )


def progress_kwargs(**overrides):
    kwargs = {
        "skip_reshape": True,
        "transformer_options": {
            "sigmas": torch.tensor([0.5]),
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            "block_index": 3,
        },
    }
    kwargs.update(overrides)
    return kwargs


def progress_kwargs_for_sigma(sigma):
    return progress_kwargs(transformer_options={
        "sigmas": torch.tensor([sigma]),
        "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
        "block_index": 3,
    })


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class InterventionOverrideTests(unittest.TestCase):
    def test_shadow_returns_original_output_and_emits_plan_and_observation(self):
        q, k, v = tensors()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intervention.jsonl"
            override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
                jsonl_path=str(path),
                branch_mode="both",
                intervention_terms="big breasts",
                token_text_map=token_text_map(),
                logit_bias=-5.0,
            ))
            original = SentinelAttention()

            out = override(original, q, k, v, 2, **progress_kwargs())

            self.assertIs(out, original.reference_output)
            self.assertEqual(original.calls, 1)

            records = read_jsonl(path)
            self.assertEqual(records[0]["event"], "intervention_plan")
            observation = next(record for record in records if record["event"] == "intervention_observation")
            self.assertEqual(observation["term"], "big breasts")
            self.assertEqual(observation["eligible_call_index"], 0)
            self.assertEqual(observation["branch"], "all")
            self.assertLess(observation["modified_score_mean"], observation["original_score_mean"])
            self.assertGreater(observation["mean_abs_prob_delta"], 0.0)
            self.assertGreater(observation["output_delta_norm"], 0.0)

    def test_shadow_noop_bias_records_zero_deltas(self):
        q, k, v = tensors()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intervention.jsonl"
            override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
                jsonl_path=str(path),
                branch_mode="both",
                intervention_terms="big breasts",
                token_text_map=token_text_map(),
                logit_bias=0.0,
            ))

            override(reference_attention, q, k, v, 2, **progress_kwargs())

            observation = next(record for record in read_jsonl(path) if record["event"] == "intervention_observation")
            self.assertAlmostEqual(observation["score_delta"], 0.0)
            self.assertAlmostEqual(observation["mean_abs_prob_delta"], 0.0)
            self.assertAlmostEqual(observation["max_abs_prob_delta"], 0.0)
            self.assertAlmostEqual(observation["output_delta_norm"], 0.0)

    def test_shadow_scale_suppression_records_nonzero_deltas(self):
        q, k, v = tensors()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intervention.jsonl"
            override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
                jsonl_path=str(path),
                branch_mode="both",
                intervention_kind="token_attention_scale",
                intervention_terms="big breasts",
                token_text_map=token_text_map(),
                strength=0.0,
            ))

            override(reference_attention, q, k, v, 2, **progress_kwargs())

            observation = next(record for record in read_jsonl(path) if record["event"] == "intervention_observation")
            self.assertLess(observation["modified_score_mean"], observation["original_score_mean"])
            self.assertGreater(observation["mean_abs_prob_delta"], 0.0)

    def test_shadow_emits_skipped_for_out_of_scope_step(self):
        q, k, v = tensors()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intervention.jsonl"
            override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
                jsonl_path=str(path),
                branch_mode="both",
                intervention_terms="big breasts",
                target_step_indices="0",
                token_text_map=token_text_map(),
            ))

            override(reference_attention, q, k, v, 2, **progress_kwargs())

            skipped = next(record for record in read_jsonl(path) if record["event"] == "intervention_skipped")
            self.assertEqual(skipped["reason"], "target_step_skipped")
            self.assertEqual(skipped["eligible_call_index"], 0)

    def test_shadow_fallbacks_return_original_output(self):
        fallback_cases = [
            ("mask_shape_unsupported", {"mask": torch.ones(1)}),
            ("not_skip_reshape", {"skip_reshape": False}),
            ("vram_guard_exceeded", {}, InterventionConfig(
                branch_mode="both",
                intervention_terms="big breasts",
                token_text_map=token_text_map(),
                max_logits_mib=0.00001,
            )),
        ]
        for reason, kwargs_overrides, *config_override in fallback_cases:
            with self.subTest(reason=reason):
                q, k, v = tensors()
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "intervention.jsonl"
                    config = config_override[0] if config_override else InterventionConfig(
                        branch_mode="both",
                        intervention_terms="big breasts",
                        token_text_map=token_text_map(),
                    )
                    config = InterventionConfig(**{**config.__dict__, "jsonl_path": str(path)})
                    override = AnimaConceptInterventionAttentionOverride(config)
                    original = SentinelAttention()

                    out = override(original, q, k, v, 2, **progress_kwargs(**kwargs_overrides))

                    self.assertIs(out, original.reference_output)
                    fallback = next(record for record in read_jsonl(path) if record["event"] == "intervention_fallback")
                    self.assertEqual(fallback["reason"], reason)

    def test_shadow_fallbacks_self_attention(self):
        q = torch.randn(1, 2, 4, 3)
        k = torch.randn(1, 2, 4, 3)
        v = torch.randn(1, 2, 4, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intervention.jsonl"
            override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
                jsonl_path=str(path),
                branch_mode="both",
                intervention_terms="big breasts",
                token_text_map=token_text_map(),
            ))

            override(reference_attention, q, k, v, 2, **progress_kwargs())

            fallback = next(record for record in read_jsonl(path) if record["event"] == "intervention_fallback")
            self.assertEqual(fallback["reason"], "not_cross_attention")

    def test_intervene_attention_logit_bias_noop_matches_reference(self):
        q, k, v = tensors()
        override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
            mode="intervene",
            branch_mode="both",
            intervention_terms="big breasts",
            token_text_map=token_text_map(),
            logit_bias=0.0,
        ))

        out = override(reference_attention, q, k, v, 2, **progress_kwargs())
        expected = reference_attention(q, k, v, 2, **progress_kwargs())

        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_intervene_token_attention_scale_noop_matches_reference(self):
        q, k, v = tensors()
        override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
            mode="intervene",
            intervention_kind="token_attention_scale",
            branch_mode="both",
            intervention_terms="big breasts",
            token_text_map=token_text_map(),
            strength=1.0,
        ))

        out = override(reference_attention, q, k, v, 2, **progress_kwargs())
        expected = reference_attention(q, k, v, 2, **progress_kwargs())

        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_intervene_modifies_only_selected_positive_branch(self):
        q, k, v = branch_tensors()
        kwargs = progress_kwargs(transformer_options={
            "sigmas": torch.tensor([0.5]),
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            "cond_or_uncond": [1, 0],
            "block_index": 3,
        })
        override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
            mode="intervene",
            branch_mode="positive_only",
            intervention_terms="big breasts",
            token_text_map=token_text_map(),
            logit_bias=-5.0,
        ))

        out = override(reference_attention, q, k, v, 2, **kwargs)
        expected = reference_attention(q, k, v, 2, **kwargs)

        self.assertTrue(torch.allclose(out[0], expected[0], atol=1e-6))
        self.assertFalse(torch.allclose(out[1], expected[1], atol=1e-6))

    def test_intervene_respects_target_call_filter(self):
        q, k, v = tensors()
        override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
            mode="intervene",
            branch_mode="both",
            intervention_terms="big breasts",
            target_call_indices="1",
            token_text_map=token_text_map(),
            logit_bias=-5.0,
        ))
        kwargs = progress_kwargs()

        first = override(reference_attention, q, k, v, 2, **kwargs)
        second = override(reference_attention, q, k, v, 2, **kwargs)
        expected = reference_attention(q, k, v, 2, **kwargs)

        self.assertTrue(torch.allclose(first, expected, atol=1e-6))
        self.assertFalse(torch.allclose(second, expected, atol=1e-6))

    def test_intervene_respects_target_step_filter(self):
        q, k, v = tensors()
        override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
            mode="intervene",
            branch_mode="both",
            intervention_terms="big breasts",
            target_step_indices="1",
            token_text_map=token_text_map(),
            logit_bias=-5.0,
        ))

        first = override(reference_attention, q, k, v, 2, **progress_kwargs_for_sigma(1.0))
        second = override(reference_attention, q, k, v, 2, **progress_kwargs_for_sigma(0.5))
        expected_first = reference_attention(q, k, v, 2, **progress_kwargs_for_sigma(1.0))
        expected_second = reference_attention(q, k, v, 2, **progress_kwargs_for_sigma(0.5))

        self.assertTrue(torch.allclose(first, expected_first, atol=1e-6))
        self.assertFalse(torch.allclose(second, expected_second, atol=1e-6))

    def test_intervene_unsupported_layout_falls_back_to_original(self):
        q, k, v = tensors()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intervention.jsonl"
            override = AnimaConceptInterventionAttentionOverride(InterventionConfig(
                mode="intervene",
                jsonl_path=str(path),
                branch_mode="both",
                intervention_terms="big breasts",
                token_text_map=token_text_map(),
                logit_bias=-5.0,
            ))
            original = SentinelAttention()

            out = override(original, q, k, v, 2, **progress_kwargs(skip_reshape=False))

            self.assertIs(out, original.reference_output)
            fallback = next(record for record in read_jsonl(path) if record["event"] == "intervention_fallback")
            self.assertEqual(fallback["reason"], "not_skip_reshape")


if __name__ == "__main__":
    unittest.main()
