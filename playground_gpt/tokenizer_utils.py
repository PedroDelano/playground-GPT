# **************************************************************************
# Description: This file contains utils for the tokenizer
# **************************************************************************

import re
from typing import List

from .log import logger
from .models.vocab import Vocab


class TokenizerUtils:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.special_tokens = {
            "unk": "<unk>",
            "start": "<startoftext>",
            "end": "<endoftext>",
        }

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
        tokens = [tk for tk in splits if tk != ""]
        tokens = self.separate_special_chars_and_digits(tokens)

        # inserting the start and end tokens
        tokens.append(self.special_tokens["end"])
        tokens.insert(0, self.special_tokens["start"])

        if self.verbose is True:
            logger.info(f"Tokenized {len(text)} characters into {len(tokens)} tokens.")
        return tokens

    def create_vocabulary(self, tokens: List[str]) -> List[Vocab]:
        """A vocabulary is a map from tokens to unique IDs. This is
        an intermediary step before embedding words into a vector space.

        Args:
            tokens (List[str]): The list of all available tokens in the token-space.

        Returns:
            List[Vocab]: The mapping of each token to the respective ID
        """
        unique_tokens = list(set(tokens))
        vocabulary = [
            Vocab(token=tk, token_id=tk_id) for tk_id, tk in enumerate(unique_tokens)
        ]

        # adding the unkown token
        vocabulary.append(
            Vocab(token=self.special_tokens["unk"], token_id=len(vocabulary))
        )

        # adding the start and end tokens
        vocabulary.append(
            Vocab(token=self.special_tokens["start"], token_id=len(vocabulary))
        )
        vocabulary.append(
            Vocab(token=self.special_tokens["end"], token_id=len(vocabulary))
        )

        if self.verbose is True:
            logger.info(f"Created a vocabulary of {len(vocabulary)} IDs.")
        return vocabulary
