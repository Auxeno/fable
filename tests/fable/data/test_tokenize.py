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


class TestDetokenize:
    def test_detokenize_basic_ids(self):
        tokenizer_config = {"char_to_id": {"x": 5, "y": 6}}

        assert detokenize([5, 6, 5], tokenizer_config) == "xyx"

    def test_detokenize_unknown_id_raises(self):
        tokenizer_config = {"char_to_id": {"x": 5}}

        with pytest.raises(ValueError, match="Token ID 2 missing"):
            detokenize([5, 2], tokenizer_config)
