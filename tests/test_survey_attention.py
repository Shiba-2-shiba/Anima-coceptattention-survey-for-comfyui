import json
import tempfile
import unittest
from pathlib import Path

import torch

from anima_concept_survey.survey_attention import (
    AnimaConceptSurveyAttentionOverride,
    SurveyConfig,
    build_concept_token_groups,
    infer_square_spatial_shape,
    parse_call_index_scope,
    progress_from_sigmas,
    resolve_comfy_jsonl_path,
    resolve_comfy_output_path,
    selected_branch_indices,
)


class MockTokenizer:
    def convert_ids_to_tokens(self, token_id):
        if isinstance(token_id, list):
            return [f"tok_{item}" for item in token_id]
        return f"tok_{token_id}"


class MockClip:
    tokenizer = MockTokenizer()

    def tokenize(self, text):
        del text
        return {"l": [[10, 11, 12, 13, 14]]}


def reference_attention(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    del mask, kwargs
    assert skip_reshape
    logits = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    out = torch.matmul(torch.softmax(logits, dim=-1), v)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], heads * q.shape[-1])


class SentinelAttention:
    def __init__(self):
        self.calls = 0
        self.reference_output = None

    def __call__(self, *args, **kwargs):
        self.calls += 1
        out = reference_attention(*args, **kwargs)
        offsets = torch.arange(out.shape[0], device=out.device, dtype=out.dtype).reshape(out.shape[0], *([1] * (out.ndim - 1)))
        self.reference_output = out + offsets
        return self.reference_output


class SurveyAttentionTests(unittest.TestCase):
    def test_parse_call_index_scope_supports_all_lists_and_ranges(self):
        self.assertIsNone(parse_call_index_scope("all"))
        self.assertEqual(parse_call_index_scope("0"), {0})
        self.assertEqual(parse_call_index_scope("0,7,14"), {0, 7, 14})
        self.assertEqual(parse_call_index_scope("7-13"), {7, 8, 9, 10, 11, 12, 13})
        for invalid in ("", "-1", "7-3", "a", "1,,2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_call_index_scope(invalid)

    def test_infer_square_spatial_shape(self):
        self.assertEqual(infer_square_spatial_shape(64), (8, 8))
        self.assertIsNone(infer_square_spatial_shape(65))

    def test_resolve_comfy_output_path_keeps_absolute_and_bases_relative(self):
        base = Path(tempfile.gettempdir()) / "survey-base"
        absolute = base / "abs" / "survey.jsonl"
        self.assertEqual(resolve_comfy_output_path(str(absolute), base_dir=base), str(absolute))
        self.assertEqual(
            resolve_comfy_output_path("'reports/survey.jsonl'", base_dir=base),
            str(base / "reports" / "survey.jsonl"),
        )
        self.assertEqual(
            resolve_comfy_output_path("", default_relative="anima/heatmaps", base_dir=base),
            str(base / "anima" / "heatmaps"),
        )
        self.assertIsNone(resolve_comfy_output_path("", base_dir=base))
        self.assertEqual(
            resolve_comfy_jsonl_path("logs", base_dir=base),
            str(base / "logs" / "survey.jsonl"),
        )
        self.assertEqual(
            resolve_comfy_jsonl_path("logs/custom.jsonl", base_dir=base),
            str(base / "logs" / "custom.jsonl"),
        )
        self.assertIsNone(resolve_comfy_jsonl_path("", base_dir=base))

    def test_build_concept_token_groups_matches_phrase_tokens(self):
        groups = build_concept_token_groups("big breasts", {
            0: {"token_text": "big", "token_source": "qwen"},
            1: {"token_text": " breasts", "token_source": "qwen"},
            2: {"token_text": ",", "token_source": "qwen"},
        })
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].token_indices, (0, 1))

    def test_progress_last_model_step_reaches_one(self):
        info = progress_from_sigmas({
            "sigmas": torch.tensor([0.5]),
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
        })
        self.assertIsNotNone(info)
        self.assertEqual(info.index, 1)
        self.assertEqual(info.num_steps, 2)
        self.assertEqual(info.last_index, 1)
        self.assertAlmostEqual(info.progress, 1.0)

    def test_branch_selection(self):
        idx = selected_branch_indices(4, [1, 0], "positive_only", torch.device("cpu"))
        self.assertEqual(idx.tolist(), [2, 3])
        idx = selected_branch_indices(4, [1, 0], "negative_only", torch.device("cpu"))
        self.assertEqual(idx.tolist(), [0, 1])

    def test_observe_mode_returns_original_and_records_jsonl(self):
        torch.manual_seed(22)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "survey.jsonl"
            original = SentinelAttention()
            override = AnimaConceptSurveyAttentionOverride(
                SurveyConfig(jsonl_path=str(path), max_tokens=3, prompt_text="hello"),
                clip=MockClip(),
            )
            kwargs = {
                "skip_reshape": True,
                "transformer_options": {
                    "sigmas": torch.tensor([0.5]),
                    "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                    "cond_or_uncond": [1, 0],
                    "block_index": 7,
                },
            }
            out = override(original, q, k, v, 2, **kwargs)
            override.finalize()

            self.assertTrue(torch.equal(out, original.reference_output))
            self.assertEqual(original.calls, 1)
            self.assertEqual(override.stats.observed_calls, 1)

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            observations = [record for record in records if record.get("event") == "attention_observation"]
            self.assertEqual(len(observations), 2)
            self.assertEqual({record["branch"] for record in observations}, {"negative", "positive"})
            self.assertEqual(observations[0]["spatial"], [4, 4])
            self.assertEqual(observations[0]["block"], "7")
            self.assertEqual(len(observations[0]["token_scores"]), 3)
            self.assertIn("token_text", observations[0]["token_scores"][0])

    def test_unsupported_shape_falls_back_to_original(self):
        q = torch.randn(2, 2, 15, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        original = SentinelAttention()
        override = AnimaConceptSurveyAttentionOverride(SurveyConfig())
        out = override(
            original,
            q,
            k,
            v,
            2,
            skip_reshape=True,
            transformer_options={
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            },
        )
        self.assertTrue(torch.equal(out, original.reference_output))
        self.assertEqual(override.stats.fallback_reasons["cannot_infer_spatial_shape"], 1)

    def test_heatmap_export_writes_npy_and_png(self):
        torch.manual_seed(31)
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "survey.jsonl"
            heatmap_dir = Path(tmp) / "heatmaps"
            override = AnimaConceptSurveyAttentionOverride(SurveyConfig(
                jsonl_path=str(jsonl_path),
                capture_level="heatmap",
                save_heatmaps=True,
                heatmap_dir=str(heatmap_dir),
                max_tokens=2,
            ))
            override(
                reference_attention,
                q,
                k,
                v,
                2,
                skip_reshape=True,
                transformer_options={
                    "sigmas": torch.tensor([0.5]),
                    "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                },
            )
            self.assertGreaterEqual(len(list(heatmap_dir.glob("*.npy"))), 2)
            self.assertGreaterEqual(len(list(heatmap_dir.glob("*.png"))), 2)
            self.assertTrue((heatmap_dir / "manifest.json").exists())
            self.assertGreaterEqual(len(list((heatmap_dir / "aggregate").glob("*_preview.png"))), 2)
            self.assertTrue((heatmap_dir / "aggregate" / "manifest.json").exists())

    def test_concept_terms_export_combined_phrase_heatmaps(self):
        torch.manual_seed(41)
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "survey.jsonl"
            heatmap_dir = Path(tmp) / "heatmaps"
            override = AnimaConceptSurveyAttentionOverride(SurveyConfig(
                jsonl_path=str(jsonl_path),
                capture_level="heatmap",
                save_heatmaps=True,
                heatmap_dir=str(heatmap_dir),
                max_tokens=1,
                concept_terms="big breasts",
                token_text_map={
                    0: {"token_index": 0, "token_text": "big", "token_source": "qwen"},
                    1: {"token_index": 1, "token_text": " breasts", "token_source": "qwen"},
                    2: {"token_index": 2, "token_text": ",", "token_source": "qwen"},
                },
            ))
            override(
                reference_attention,
                q,
                k,
                v,
                2,
                skip_reshape=True,
                transformer_options={
                    "sigmas": torch.tensor([0.5]),
                    "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                },
            )
            override.finalize()
            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            observation = next(record for record in records if record.get("event") == "attention_observation")
            self.assertEqual(observation["concept_scores"][0]["term"], "big breasts")
            self.assertEqual(observation["concept_scores"][0]["token_indices"], [0, 1])
            self.assertTrue((heatmap_dir / "concepts" / "manifest.json").exists())
            self.assertTrue((heatmap_dir / "concepts" / "aggregate" / "manifest.json").exists())
            self.assertGreaterEqual(len(list((heatmap_dir / "concepts" / "aggregate").glob("*big_breasts_preview.png"))), 1)


if __name__ == "__main__":
    unittest.main()
