import unittest

from anima_concept_survey.token_text import build_token_text_map, flatten_tokenized


class MockTokenizer:
    def convert_ids_to_tokens(self, token_id):
        if isinstance(token_id, list):
            return [f"tok_{item}" for item in token_id]
        return f"tok_{token_id}"


class MockClip:
    def __init__(self):
        self.tokenizer = MockTokenizer()

    def tokenize(self, text):
        del text
        return {
            "l": [[(101, 1.0), (102, 0.5)]],
            "t5xxl": [[201, 202]],
        }


class TokenTextTests(unittest.TestCase):
    def test_flatten_tokenized_preserves_stream_order(self):
        flat = flatten_tokenized({"a": [[(1, 1.0), (2, 0.8)]], "b": [[3]]})
        self.assertEqual([item.token_id for item in flat], [1, 2, 3])
        self.assertEqual([item.token_source for item in flat], ["a", "a", "b"])
        self.assertEqual([item.token_index for item in flat], [0, 1, 2])
        self.assertEqual([item.source_token_index for item in flat], [0, 1, 0])
        self.assertEqual(flat[1].weight, 0.8)

    def test_build_token_text_map_uses_clip_decoder(self):
        mapping = build_token_text_map(MockClip(), "hello")
        self.assertEqual(mapping[0]["token_text"], "tok_101")
        self.assertEqual(mapping[0]["token_source"], "l")
        self.assertEqual(mapping[0]["source_token_index"], 0)
        self.assertEqual(mapping[2]["source_token_index"], 0)
        self.assertEqual(mapping[2]["token_text"], "tok_201")
        self.assertEqual(mapping[1]["weight"], 0.5)


if __name__ == "__main__":
    unittest.main()
