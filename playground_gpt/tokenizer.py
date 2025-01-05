# **************************************************************************
# Description: This file contains the code for the tokenizer
# **************************************************************************

from typing import List

from .models.vocab import Vocab
from .tokenizer_utils import TokenizerUtils


class Tokenizer:
    def __init__(self, vocab: List[Vocab], verbose: bool = True):
        self.vocab = vocab
        self.verbose = verbose
        self.utils = TokenizerUtils(verbose=verbose)
        self.encode_map = self.__vocab_to_map(vocab, direction="encode")
        self.decode_map = self.__vocab_to_map(vocab, direction="decode")

    @staticmethod
    def __vocab_to_map(vocab: List[Vocab], direction: str) -> dict:
        if direction == "encode":
            return {elem.token: elem.token_id for elem in vocab}
        elif direction == "decode":
            return {elem.token_id: elem.token for elem in vocab}
        raise ValueError(f"Invalid direction: {direction}.")

    def encode(self, text: str) -> List[int]:
        tokens = self.utils.tokenize(text)
        return [self.encode_map.get(tk) for tk in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.decode_map.get(_id) for _id in ids]
