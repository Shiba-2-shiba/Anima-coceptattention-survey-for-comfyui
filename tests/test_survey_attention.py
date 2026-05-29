import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
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
            summary = next(record for record in records if record.get("event") == "run_summary")
            self.assertEqual(len(observations), 2)
            self.assertEqual({record["branch"] for record in observations}, {"negative", "positive"})
            self.assertEqual(observations[0]["spatial"], [4, 4])
            self.assertEqual(observations[0]["block"], "7")
            self.assertEqual(len(observations[0]["token_scores"]), 3)
            self.assertIn("token_text", observations[0]["token_scores"][0])
            self.assertEqual(summary["token_sources"], ["l"])
            self.assertEqual(summary["token_text_map"][0]["source_token_index"], 0)

    def test_observe_mode_returns_backend_sentinel_exactly(self):
        torch.manual_seed(23)
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        sentinel = torch.randn(1, 16, 8)

        def original(*args, **kwargs):
            del args, kwargs
            return sentinel

        override = AnimaConceptSurveyAttentionOverride(SurveyConfig(prompt_text="hello"))
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

        self.assertIs(out, sentinel)

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
                heatmap_output="tokens_only",
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
            manifest = json.loads((heatmap_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("heatmap_mean", manifest[0])
            self.assertIn("heatmap_max", manifest[0])
            self.assertIn("heatmap_std", manifest[0])
            self.assertIn("heatmap_max_over_mean", manifest[0])
            self.assertEqual(manifest[0]["preview_normalization"], "per_file_minmax")
            self.assertGreaterEqual(len(list((heatmap_dir / "aggregate").glob("*_preview.png"))), 2)
            self.assertTrue((heatmap_dir / "aggregate" / "manifest.json").exists())
            aggregate_manifest = json.loads((heatmap_dir / "aggregate" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(aggregate_manifest[0]["preview_normalization"], "per_file_minmax")

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
            summary = next(record for record in records if record.get("event") == "run_summary")
            self.assertEqual(observation["concept_scores"][0]["term"], "big breasts")
            self.assertEqual(observation["concept_scores"][0]["token_indices"], [0, 1])
            self.assertEqual(observation["concept_scores"][0]["source_token_indices"], [0, 1])
            self.assertEqual(summary["concept_match_count"], 1)
            self.assertEqual(summary["concept_unmatched_terms"], [])
            self.assertEqual(summary["concept_ambiguous_terms"], [])
            self.assertTrue((heatmap_dir / "concepts" / "manifest.json").exists())
            self.assertTrue((heatmap_dir / "concepts" / "aggregate" / "manifest.json").exists())
            self.assertEqual(list(heatmap_dir.glob("step*_token*.npy")), [])
            self.assertEqual(list(heatmap_dir.glob("step*_token*.png")), [])
            manifest = json.loads((heatmap_dir / "concepts" / "aggregate" / "manifest.json").read_text(encoding="utf-8"))
            row = manifest[0]
            self.assertEqual(row["concept_uid"], observation["concept_scores"][0]["concept_uid"])
            self.assertEqual(row["preview_normalization"], "per_file_minmax")
            self.assertTrue((heatmap_dir / "concepts" / "aggregate" / row["preview_png"]).exists())
            arr = np.load(heatmap_dir / "concepts" / "aggregate" / row["npy"])
            self.assertTrue(math.isclose(row["heatmap_mean"], float(arr.mean()), rel_tol=1e-6))
            self.assertTrue(math.isclose(row["heatmap_max"], float(arr.max()), rel_tol=1e-6))
            self.assertTrue(math.isclose(row["heatmap_std"], float(arr.std()), rel_tol=1e-6))
            self.assertTrue(math.isclose(row["heatmap_max_over_mean"], float(arr.max() / arr.mean()), rel_tol=1e-6))

    def test_repeated_concept_heatmaps_use_distinct_identity_and_files(self):
        torch.manual_seed(44)
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
                    0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
                    1: {"token_index": 1, "source_token_index": 1, "token_text": " breasts", "token_source": "qwen"},
                    2: {"token_index": 2, "source_token_index": 2, "token_text": ",", "token_source": "qwen"},
                    3: {"token_index": 3, "source_token_index": 3, "token_text": "big", "token_source": "qwen"},
                    4: {"token_index": 4, "source_token_index": 4, "token_text": " breasts", "token_source": "qwen"},
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
            match_summary = next(record for record in records if record.get("event") == "concept_match_summary")
            observation = next(record for record in records if record.get("event") == "attention_observation")
            concept_uids = [score["concept_uid"] for score in observation["concept_scores"]]

            self.assertEqual(len(match_summary["matches"]), 2)
            self.assertEqual(len(concept_uids), 2)
            self.assertEqual(len(set(concept_uids)), 2)
            self.assertTrue(all(match["concept_uid"] for match in match_summary["matches"]))

            manifest = json.loads((heatmap_dir / "concepts" / "manifest.json").read_text(encoding="utf-8"))
            aggregate_manifest = json.loads((heatmap_dir / "concepts" / "aggregate" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest), 2)
            self.assertEqual(len(aggregate_manifest), 2)
            self.assertEqual(len({row["npy"] for row in manifest}), 2)
            self.assertEqual({row["concept_uid"] for row in aggregate_manifest}, set(concept_uids))

    def test_unmatched_concept_terms_emit_jsonl_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "survey.jsonl"
            AnimaConceptSurveyAttentionOverride(SurveyConfig(
                jsonl_path=str(jsonl_path),
                concept_terms="missing term",
                token_text_map={
                    0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
                },
            ))

            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["event"], "concept_match_summary")
            unmatched = next(record for record in records if record.get("event") == "concept_unmatched")
            self.assertEqual(unmatched["term"], "missing term")
            self.assertEqual(unmatched["available_sources"], ["qwen"])

    def test_concept_heatmaps_preserve_positive_negative_branch_separation(self):
        torch.manual_seed(42)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "survey.jsonl"
            heatmap_dir = Path(tmp) / "heatmaps"
            override = AnimaConceptSurveyAttentionOverride(SurveyConfig(
                jsonl_path=str(jsonl_path),
                capture_level="heatmap",
                save_heatmaps=True,
                heatmap_dir=str(heatmap_dir),
                concept_terms="big breasts",
                token_text_map={
                    0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
                    1: {"token_index": 1, "source_token_index": 1, "token_text": " breasts", "token_source": "qwen"},
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
                    "cond_or_uncond": [1, 0],
                },
            )
            override.finalize()

            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            observations = [record for record in records if record.get("event") == "attention_observation"]
            self.assertEqual({record["branch"] for record in observations}, {"negative", "positive"})
            self.assertTrue(all(record["concept_scores"] for record in observations))

            manifest = json.loads((heatmap_dir / "concepts" / "aggregate" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual({row["branch"] for row in manifest}, {"negative", "positive"})

    def test_concept_heatmaps_write_call_level_aggregate_manifest(self):
        torch.manual_seed(45)
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
                concept_terms="big breasts",
                token_text_map={
                    0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
                    1: {"token_index": 1, "source_token_index": 1, "token_text": " breasts", "token_source": "qwen"},
                },
            ))
            base_options = {
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            }
            override(reference_attention, q, k, v, 2, skip_reshape=True, transformer_options=base_options)
            override(reference_attention, q, k, v, 2, skip_reshape=True, transformer_options=base_options)
            override.finalize()

            by_call_dir = heatmap_dir / "concepts" / "aggregate_by_call"
            self.assertTrue((by_call_dir / "manifest.json").exists())
            manifest = json.loads((by_call_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual({row["eligible_call_index"] for row in manifest}, {0, 1})
            self.assertTrue(all(row["preview_normalization"] == "per_file_minmax" for row in manifest))
            self.assertTrue(all((by_call_dir / row["preview_png"]).exists() for row in manifest))
            self.assertTrue(all(row["concept_uid"] for row in manifest))

    def test_out_of_range_concept_match_emits_warning_without_score(self):
        torch.manual_seed(43)
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "survey.jsonl"
            override = AnimaConceptSurveyAttentionOverride(SurveyConfig(
                jsonl_path=str(jsonl_path),
                capture_level="heatmap",
                concept_terms="distant",
                token_text_map={
                    7: {"token_index": 7, "source_token_index": 7, "token_text": "distant", "token_source": "qwen"},
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

            records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            observation = next(record for record in records if record.get("event") == "attention_observation")
            warning = next(
                record for record in records
                if record.get("event") == "concept_alignment_warning"
                and record.get("reason") == "concept_token_index_out_of_range"
            )
            self.assertEqual(observation["concept_scores"], [])
            self.assertEqual(warning["token_indices"], [7])
            self.assertEqual(warning["text_len"], 5)

    def test_concept_heatmap_math_sums_matched_token_probability_mass(self):
        attention_probs = torch.tensor([[[[
            0.10, 0.20, 0.70,
        ], [
            0.30, 0.40, 0.30,
        ], [
            0.25, 0.50, 0.25,
        ], [
            0.60, 0.10, 0.30,
        ]]]], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            heatmap_dir = Path(tmp) / "heatmaps"
            override = AnimaConceptSurveyAttentionOverride(SurveyConfig(
                capture_level="heatmap",
                save_heatmaps=True,
                heatmap_dir=str(heatmap_dir),
                concept_terms="big breasts",
                token_text_map={
                    0: {"token_index": 0, "token_text": "big", "token_source": "qwen"},
                    2: {"token_index": 2, "token_text": " breasts", "token_source": "qwen"},
                },
            ))

            scores = override._concept_scores_from_attention(attention_probs, (2, 2))
            expected_mass = attention_probs[..., [0, 2]].sum(dim=-1)
            expected_heatmap = expected_mass.mean(dim=(0, 1)).reshape(2, 2)

            self.assertEqual(len(scores), 1)
            self.assertTrue(torch.allclose(scores[0]["_heatmap"], expected_heatmap))
            self.assertTrue(math.isclose(scores[0]["score_mean"], float(expected_mass.mean()), rel_tol=1e-6))
            self.assertTrue(math.isclose(scores[0]["score_max"], float(expected_mass.max()), rel_tol=1e-6))
            self.assertTrue(math.isclose(scores[0]["heatmap_mean"], float(expected_heatmap.mean()), rel_tol=1e-6))
            self.assertTrue(math.isclose(scores[0]["heatmap_max"], float(expected_heatmap.max()), rel_tol=1e-6))
            self.assertTrue(math.isclose(scores[0]["heatmap_std"], float(expected_heatmap.numpy().std()), rel_tol=1e-6))
            self.assertTrue(math.isclose(
                scores[0]["heatmap_max_over_mean"],
                float(expected_heatmap.max() / expected_heatmap.mean()),
                rel_tol=1e-6,
            ))
            self.assertEqual(scores[0]["attention_key_indices"], [0, 2])
            self.assertEqual(scores[0]["uniform_baseline"], 2 / 3)
            self.assertTrue(math.isclose(
                scores[0]["score_mean_over_uniform"],
                float(expected_mass.mean()) / (2 / 3),
                rel_tol=1e-6,
            ))
            self.assertFalse(scores[0]["near_uniform"])

            progress = progress_from_sigmas({
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            })
            self.assertIsNotNone(progress)
            override._save_concept_heatmaps(scores, (2, 2), progress, 0, "positive")

            manifest = json.loads((heatmap_dir / "concepts" / "manifest.json").read_text(encoding="utf-8"))
            saved = np.load(heatmap_dir / "concepts" / manifest[0]["npy"])
            self.assertTrue(np.allclose(saved, expected_heatmap.numpy()))
            self.assertTrue(math.isclose(manifest[0]["heatmap_mean"], float(saved.mean()), rel_tol=1e-6))
            self.assertTrue(math.isclose(manifest[0]["heatmap_max"], float(saved.max()), rel_tol=1e-6))
            self.assertEqual(manifest[0]["preview_normalization"], "per_file_minmax")


if __name__ == "__main__":
    unittest.main()
