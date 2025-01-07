# ***************************************************************
# This creates a dataset to train our embedding model
# ***************************************************************

import torch
from torch.utils.data import Dataset

from playground_gpt.tokenizer import Tokenizer

# For more info regarding datasets, read:
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files


class PlaygroundDataset(Dataset):
    """A GPT Style dataset. This will create a input / target tensor
    creating a fixed-size input and the next token the model has to
    predict is just the next token in the sentence. This will be repeated
    in a rolling-window, so that we can increase our datapoints.
    """

    def __init__(
        self, txt: str, tokenizer: Tokenizer, input_size: int, window_size: int
    ):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = self.tokenizer.encode(txt)

        for i in range(0, len(token_ids) - input_size, window_size):
            input_chunk = token_ids[i : i + input_size]
            target_chunk = token_ids[i + 1 : i + input_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]
