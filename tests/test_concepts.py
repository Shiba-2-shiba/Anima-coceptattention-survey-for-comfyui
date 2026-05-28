import unittest

from anima_concept_survey.concepts import (
    build_concept_token_matches,
    normalize_concept_text,
    parse_concept_terms,
)


class ConceptTests(unittest.TestCase):
    def test_parse_concept_terms_splits_dedupes_and_preserves_source_prefix(self):
        terms = parse_concept_terms(" big breasts\nred hair; blue eyes, qwen:big breasts, BIG   BREASTS")

        self.assertEqual([term.term for term in terms], ["big breasts", "red hair", "blue eyes", "big breasts"])
        self.assertEqual([term.source_filter for term in terms], [None, None, None, "qwen"])
        self.assertEqual([term.normalized for term in terms], ["bigbreasts", "redhair", "blueeyes", "bigbreasts"])

    def test_normalize_concept_text_handles_tokenizer_markers_and_fallback_tokens(self):
        self.assertEqual(normalize_concept_text("▁big Ġbreasts"), "bigbreasts")
        self.assertEqual(normalize_concept_text(","), "")
        self.assertEqual(normalize_concept_text("<token:123>"), "")

    def test_source_scoped_matching_keeps_token_sources_separate(self):
        report = build_concept_token_matches("big breasts", {
            0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
            1: {"token_index": 1, "source_token_index": 0, "token_text": " breasts", "token_source": "clip_l"},
        })

        self.assertEqual(report.matches, ())
        self.assertEqual([term.term for term in report.unmatched_terms], ["big breasts"])

    def test_matching_supports_tokenizer_markers(self):
        for first, second in (("▁big", "▁breasts"), ("Ġbig", "Ġbreasts")):
            with self.subTest(first=first, second=second):
                report = build_concept_token_matches("big breasts", {
                    0: {"token_index": 0, "source_token_index": 0, "token_text": first, "token_source": "qwen"},
                    1: {"token_index": 1, "source_token_index": 1, "token_text": second, "token_source": "qwen"},
                })

                self.assertEqual(len(report.matches), 1)
                self.assertEqual(report.matches[0].token_indices, (0, 1))

    def test_source_prefix_selects_requested_stream(self):
        report = build_concept_token_matches("clip_l:big breasts", {
            0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
            1: {"token_index": 1, "source_token_index": 1, "token_text": " breasts", "token_source": "qwen"},
            2: {"token_index": 2, "source_token_index": 0, "token_text": "big", "token_source": "clip_l"},
            3: {"token_index": 3, "source_token_index": 1, "token_text": " breasts", "token_source": "clip_l"},
        })

        self.assertEqual(len(report.matches), 1)
        self.assertEqual(report.matches[0].token_source, "clip_l")
        self.assertEqual(report.matches[0].token_indices, (2, 3))

    def test_unsourced_multi_source_match_is_ambiguous_by_default(self):
        report = build_concept_token_matches("big breasts", {
            0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
            1: {"token_index": 1, "source_token_index": 1, "token_text": " breasts", "token_source": "qwen"},
            2: {"token_index": 2, "source_token_index": 0, "token_text": "big", "token_source": "clip_l"},
            3: {"token_index": 3, "source_token_index": 1, "token_text": " breasts", "token_source": "clip_l"},
        })

        self.assertEqual(report.matches, ())
        self.assertEqual([term.term for term in report.ambiguous_terms], ["big breasts"])
        self.assertIn("ambiguous_across_sources:big breasts", report.warnings)

    def test_duplicate_occurrences_produce_occurrence_indexed_matches(self):
        report = build_concept_token_matches("breasts", {
            0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
            1: {"token_index": 1, "source_token_index": 1, "token_text": " breasts", "token_source": "qwen"},
            2: {"token_index": 2, "source_token_index": 2, "token_text": "small", "token_source": "qwen"},
            3: {"token_index": 3, "source_token_index": 3, "token_text": " breasts", "token_source": "qwen"},
        })

        self.assertEqual([match.token_indices for match in report.matches], [(1,), (3,)])
        self.assertEqual([match.occurrence_index for match in report.matches], [0, 1])

    def test_punctuation_inside_match_is_recorded_as_ignored(self):
        report = build_concept_token_matches("big breasts", {
            0: {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
            1: {"token_index": 1, "source_token_index": 1, "token_text": ",", "token_source": "qwen"},
            2: {"token_index": 2, "source_token_index": 2, "token_text": " breasts", "token_source": "qwen"},
        })

        self.assertEqual(len(report.matches), 1)
        self.assertEqual(report.matches[0].token_indices, (0, 2))
        self.assertEqual(report.matches[0].ignored_token_indices, (1,))
        self.assertEqual(report.matches[0].match_warnings, ("punctuation_tokens_ignored_inside_match",))

    def test_fallback_token_text_does_not_match_human_terms(self):
        report = build_concept_token_matches("token 123", {
            0: {"token_index": 0, "source_token_index": 0, "token_text": "<token:123>", "token_source": "qwen"},
        })

        self.assertEqual(report.matches, ())


if __name__ == "__main__":
    unittest.main()
