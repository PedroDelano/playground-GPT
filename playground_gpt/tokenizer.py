# **************************************************************************
# Description: This file contains the code for the tokenizer
# **************************************************************************

import re
from typing import List

from .log import logger
from .models.vocab import Vocab


class Tokenizer:
    def __init__(self, vocab: List[Vocab], verbose: bool = True):
        self.vocab = vocab
        self.verbose = verbose
        self.encode_map = self.__vocab_to_map(vocab, direction="encode")
        self.decode_map = self.__vocab_to_map(vocab, direction="decode")

    @staticmethod
    def __vocab_to_map(vocab: List[Vocab], direction: str) -> dict:
        if direction == "encode":
            return {elem.token: elem.token_id for elem in vocab}
        elif direction == "decode":
            return {elem.token_id: elem.token for elem in vocab}
        raise ValueError(f"Invalid direction: {direction}.")

    @staticmethod
    def separate_special_chars_and_digits(word_list):
        """
        Separates special characters and individual digits from words in a list.

        Args:
            word_list (list): List of strings potentially containing special characters and numbers

        Returns:
            list: New list with special characters and digits as separate elements
        """
        result = []

        for word in word_list:
            # First split into parts (words, special chars, numbers)
            parts = re.findall(r"[a-zA-Z]+|[0-9]|[^a-zA-Z0-9\s]", word)
            result.extend(parts)

        return result

    def tokenize(self, text: str) -> List[str]:
        """Tokenization is the process, through which we
        separate words of the text, including special characters.
        We can also add some special tokens, such as start or
        end string tokens.

        Args:
            text (str): Text to be tokenized

        Returns:
            List[str]: List of tokens
        """
        splits = text.split()
        tokens = [tk for tk in splits if tk != " " and tk != ""]
        tokens = self.separate_special_chars_and_digits(tokens)
        if self.verbose is True:
            logger.info(f"Tokenized {len(text)} characters into {len(tokens)} tokens.")
        return tokens

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.encode_map.get(tk) for tk in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.decode_map.get(_id) for _id in ids]
