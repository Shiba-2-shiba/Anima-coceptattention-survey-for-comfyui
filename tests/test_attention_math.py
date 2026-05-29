import unittest

import torch

from anima_concept_survey.attention_math import (
    apply_attention_logit_bias,
    apply_token_attention_scale,
    attention_output_from_probs,
    compute_attention_logits,
    compute_attention_probs,
)


class AttentionMathTests(unittest.TestCase):
    def test_compute_logits_probs_and_output_shapes(self):
        torch.manual_seed(101)
        q = torch.randn(2, 3, 4, 5)
        k = torch.randn(2, 3, 7, 5)
        v = torch.randn(2, 3, 7, 6)

        logits = compute_attention_logits(q, k, scale=0.25)
        probs = compute_attention_probs(logits)
        out = attention_output_from_probs(probs, v)

        self.assertEqual(tuple(logits.shape), (2, 3, 4, 7))
        self.assertEqual(tuple(probs.shape), (2, 3, 4, 7))
        self.assertEqual(tuple(out.shape), (2, 3, 4, 6))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(2, 3, 4), atol=1e-6))

    def test_compute_attention_logits_matches_reference(self):
        q = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        k = torch.tensor([[[[0.5, 1.0], [1.5, -1.0], [2.0, 0.25]]]])

        logits = compute_attention_logits(q, k, scale=0.5)
        expected = torch.matmul(q.float(), k.float().transpose(-2, -1)) * 0.5

        self.assertTrue(torch.allclose(logits, expected))

    def test_token_attention_scale_noop_suppression_boost_and_normalization(self):
        probs = torch.tensor([[[[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]]], dtype=torch.float32)

        noop = apply_token_attention_scale(probs, [1], 1.0)
        suppressed = apply_token_attention_scale(probs, [1], 0.0)
        boosted = apply_token_attention_scale(probs, [1], 2.0)

        self.assertTrue(torch.allclose(noop, probs))
        self.assertTrue(torch.allclose(probs, torch.tensor([[[[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]]])))
        self.assertTrue(torch.allclose(suppressed.sum(dim=-1), torch.ones(1, 1, 2), atol=1e-6))
        self.assertTrue(torch.allclose(boosted.sum(dim=-1), torch.ones(1, 1, 2), atol=1e-6))
        self.assertLess(float(suppressed[..., 1].sum()), float(probs[..., 1].sum()))
        self.assertGreater(float(boosted[..., 1].sum()), float(probs[..., 1].sum()))

    def test_token_attention_scale_empty_indices_returns_unchanged_copy(self):
        probs = torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float32)

        out = apply_token_attention_scale(probs, [], 0.0)

        self.assertTrue(torch.allclose(out, probs))
        self.assertIsNot(out, probs)

    def test_attention_logit_bias_noop_negative_and_positive_bias(self):
        logits = torch.tensor([[[[0.0, 1.0, 2.0], [1.0, 0.5, -1.0]]]], dtype=torch.float32)
        base = torch.softmax(logits, dim=-1)

        noop = torch.softmax(apply_attention_logit_bias(logits, [1], 0.0), dim=-1)
        suppressed = torch.softmax(apply_attention_logit_bias(logits, [1], -5.0), dim=-1)
        boosted = torch.softmax(apply_attention_logit_bias(logits, [1], 2.0), dim=-1)

        self.assertTrue(torch.allclose(noop, base))
        self.assertLess(float(suppressed[..., 1].sum()), float(base[..., 1].sum()))
        self.assertGreater(float(boosted[..., 1].sum()), float(base[..., 1].sum()))
        self.assertTrue(torch.allclose(apply_attention_logit_bias(logits, [1], 0.0), logits))

    def test_attention_logit_bias_empty_indices_returns_unchanged_copy(self):
        logits = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)

        out = apply_attention_logit_bias(logits, [], -5.0)

        self.assertTrue(torch.allclose(out, logits))
        self.assertIsNot(out, logits)


if __name__ == "__main__":
    unittest.main()
