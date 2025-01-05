# **************************************************************************
# Description: Test file for tokenizer and tokenizer utils
# **************************************************************************

import pytest
from playground_gpt.models.vocab import Vocab
from playground_gpt.tokenizer import Tokenizer
from playground_gpt.tokenizer_utils import TokenizerUtils


@pytest.fixture
def sample_tokens():
    return ["hello", "world", "!", "123", "test", "@", "python"]


@pytest.fixture
def sample_vocabulary(sample_tokens):
    tokenizer_utils = TokenizerUtils(verbose=False)
    return tokenizer_utils.create_vocabulary(sample_tokens)


@pytest.fixture
def tokenizer(sample_vocabulary):
    return Tokenizer(vocab=sample_vocabulary, verbose=False)


class TestTokenizerUtils:
    def test_create_vocabulary(self, sample_tokens):
        tokenizer_utils = TokenizerUtils(verbose=False)
        vocab = tokenizer_utils.create_vocabulary(sample_tokens)

        # Check if vocabulary is created correctly
        assert len(vocab) == len(set(sample_tokens))
        assert all(isinstance(v, Vocab) for v in vocab)
        assert all(hasattr(v, "token") and hasattr(v, "token_id") for v in vocab)

        # Check if all tokens are present
        vocab_tokens = {v.token for v in vocab}
        assert vocab_tokens == set(sample_tokens)

        # Check if IDs are unique and sequential
        vocab_ids = {v.token_id for v in vocab}
        assert vocab_ids == set(range(len(vocab)))


class TestTokenizer:
    def test_tokenize_basic(self, tokenizer):
        text = "hello world"
        tokens = tokenizer.tokenize(text)
        assert tokens == ["hello", "world"]

    def test_tokenize_with_special_chars(self, tokenizer):
        text = "hello! world@python"
        tokens = tokenizer.tokenize(text)
        assert tokens == ["hello", "!", "world", "@", "python"]

    def test_tokenize_with_numbers(self, tokenizer):
        text = "test123 world"
        tokens = tokenizer.tokenize(text)
        assert tokens == ["test", "1", "2", "3", "world"]

    def test_separate_special_chars_and_digits(self, tokenizer):
        words = ["hello123!", "world@"]
        result = tokenizer.separate_special_chars_and_digits(words)
        assert result == ["hello", "1", "2", "3", "!", "world", "@"]

    def test_encode_decode(self, tokenizer):
        text = "hello world !"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        # Check if encoding produces integers
        assert all(isinstance(id_, int) for id_ in encoded)

        # Check if decoding produces original tokens
        assert decoded == ["hello", "world", "!"]

    def test_vocab_maps(self, tokenizer, sample_vocabulary):
        # Test encode map
        assert len(tokenizer.encode_map) == len(sample_vocabulary)
        assert all(isinstance(v, int) for v in tokenizer.encode_map.values())

        # Test decode map
        assert len(tokenizer.decode_map) == len(sample_vocabulary)
        assert all(isinstance(k, int) for k in tokenizer.decode_map.keys())

        # Test bidirectional mapping
        for vocab_item in sample_vocabulary:
            encoded = tokenizer.encode_map[vocab_item.token]
            decoded = tokenizer.decode_map[encoded]
            assert decoded == vocab_item.token

    def test_empty_text(self, tokenizer):
        assert tokenizer.tokenize("") == []
        assert tokenizer.encode("") == []

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("hello123!", ["hello", "1", "2", "3", "!"]),
            ("test@world", ["test", "@", "world"]),
            ("python3.9", ["python", "3", ".", "9"]),
            ("a,b,c", ["a", ",", "b", ",", "c"]),
        ],
    )
    def test_tokenize_various_inputs(self, tokenizer, text, expected):
        assert tokenizer.tokenize(text) == expected

    def test_invalid_direction(self, sample_vocabulary):
        with pytest.raises(ValueError, match="Invalid direction:"):
            Tokenizer._Tokenizer__vocab_to_map(sample_vocabulary, "invalid")
