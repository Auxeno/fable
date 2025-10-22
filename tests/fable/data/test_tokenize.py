import pytest

from fable.data.tokenize import detokenize, tokenize


class TestTokenize:
    def test_tokenize_basic_characters(self):
        tokenizer_config = {"char_to_id": {"a": 1, "b": 2, " ": 0}}

        assert tokenize("ab ba", tokenizer_config) == [1, 2, 0, 2, 1]

    def test_tokenize_unknown_character_raises(self):
        tokenizer_config = {"char_to_id": {"a": 1}}

        with pytest.raises(ValueError, match=r"Character 'b' missing"):
            tokenize("ab", tokenizer_config)

    def test_tokenize_prefers_special_tokens(self):
        tokenizer_config = {
            "char_to_id": {
                "f": 1,
                "o": 2,
                "<": 3,
                "|": 4,
                "e": 5,
                "n": 6,
                "d": 7,
                "t": 8,
                "x": 9,
            },
            "special_tokens": {"<|endoftext|>": 99},
            "eot_token": "<|endoftext|>",
        }

        assert tokenize("foo<|endoftext|>", tokenizer_config) == [1, 2, 2, 99]


class TestDetokenize:
    def test_detokenize_basic_ids(self):
        tokenizer_config = {"char_to_id": {"x": 5, "y": 6}}

        assert detokenize([5, 6, 5], tokenizer_config) == "xyx"

    def test_detokenize_unknown_id_raises(self):
        tokenizer_config = {"char_to_id": {"x": 5}}

        with pytest.raises(ValueError, match="Token ID 2 missing"):
            detokenize([5, 2], tokenizer_config)

    def test_detokenize_special_tokens(self):
        tokenizer_config = {
            "char_to_id": {"h": 1, "!": 2},
            "special_tokens": {"<|endoftext|>": 3},
            "eot_token": "<|endoftext|>",
        }

        assert detokenize([1, 2, 3], tokenizer_config) == "h!<|endoftext|>"
