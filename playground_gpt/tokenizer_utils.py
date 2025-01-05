# **************************************************************************
# Description: This file contains utils for the tokenizer
# **************************************************************************

from typing import List

from .log import logger
from .models.vocab import Vocab


class TokenizerUtils:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

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
        if self.verbose is True:
            logger.info(f"Created a vocabulary of {len(vocabulary)} IDs.")
        return vocabulary
