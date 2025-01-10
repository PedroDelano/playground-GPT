import torch

from playground_gpt.log import logger


class Embedding:
    """Embedding is just a look-up table of vector of a
    specific fixed size, defined by the vocabulary. Basically
    each token ID has a specific vector associated with it.

    input_size (int): The number of tokens in the input sequence.
    embed_size (int): The size of the embedding vector.
    vocab_size (int): The size of the vocabulary.
    seed (int): The seed for reproducing the results.
    """

    def __init__(
        self,
        input_size: int,
        embed_size: int,
        vocab_size: int,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.input_size = input_size
        self.embed_size = embed_size
        self.embedding_matrix = torch.rand(vocab_size, embed_size)
        self.positional_embedding_matrix = torch.rand(input_size, embed_size)

    @staticmethod
    def __convert_token_id(token_ids: list) -> torch.Tensor:
        logger.warning(
            "Inputting token_ids as a list. Ideally you would convert it to torch.Tensor before embedding."
        )
        return torch.tensor(token_ids)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        if isinstance(token_ids, list):
            token_ids = self.__convert_token_id(token_ids)
        return self.embedding_matrix[token_ids]

    def positional_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        if isinstance(token_ids, list):
            token_ids = self.__convert_token_id(token_ids)
        absolute_embeddings = self.embed(token_ids)
        positional_matrix = torch.rand(self.input_size, self.embed_size)
        return absolute_embeddings + positional_matrix
