# Playground-GPT <img src="man/figures/logo.webp" width="120" align="right" />

Welcome to Playground-GPT, an educational project to build a Language Model from scratch! This repository provides a step-by-step guide to understanding and implementing core concepts of Large Language Models. Perfect for learning and experimenting with AI fundamentals.

## Build Stages

```■◧□□□□□□□□□□ 12.5%``` | Currently working on: Stage 1.1 

### Stage 1: Building a LLM 🧱

Laying the groundwork for our language model by implementing core components from scratch.

1.1) Data preparation and Sampling 📚

1.2) Attention Mechanism 🔍

1.3) LLM Architecture 🏗️

## Stage 2: Foundation model 🌟

Taking our model from basic implementation to a functioning foundation model.

2.1) Training Loop ⚡

2.2) Model evaluation 📊

2.3) Load pretrained weights 🔄

## Stage 3: Finetunning 🎯

Specializing our model for specific tasks and applications.

3.1) Classifier 🏷️

3.2) Personal Assistant 🤖


## Examples

### Tokenizer

First we need to download a large text to be used as reference to the vocabulary.

```python
import requests
shakespeare_complete = requests.get("https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt").text
```

Now we can construct the vocabulary and use the tokenizer.

```python
from playground_gpt.tokenizer import Tokenizer
from playground_gpt.tokenizer_utils import TokenizerUtils

utils = TokenizerUtils()
tokens = utils.tokenize(shakespeare_complete)
vocab = utils.create_vocabulary(tokens)

tokenizer = Tokenizer(vocab=vocab)

encoded_str = tokenizer.encode("Hey there! I'm a string with numbers such as: 1, 2 and also 3!")
print(encoded_str)

decoded_str = tokenizer.decode(encoded_str)
print(decoded_str)
```
