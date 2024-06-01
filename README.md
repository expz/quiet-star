# Quiet STAR

An implementation of the Quiet-STAR paper (https://arxiv.org/pdf/2403.09629.pdf) including a full test suite to guarantee correctness.

This implementation is optimized so that it performs a minimal amount of computation when generating thoughts. The tricks used to reduce the computation were alluded to in the paper. However, it still needs to be configured to use Flash Attention, so a batch of size 1 takes a little over 1 second on a 4090. (Recall that one batch requires, among other things, generating multiple thoughts of length N at all locations in the input sequence, so all things considered, it is still fairly fast.)

See TODO.md for planned improvements.

## Usage
To set up the environment:
```bash
poetry config keyring.enabled false
poetry install
source .venv/bin/activate
```
Then to fine tune a Qwen 0.5B model (default config requires an nvidia GPU like a 3090 or 4090 with 24GB of RAM):
```bash
python run_train_qwen.py
```
By default, the dataset used is just a small part of the dataset used in the paper.

There is code in this repository to train a model using MLX on Apple silicon, but it is not currently set up to use a pretrained HuggingFace model.

## Development
To begin:
```bash
pre-commit install
```
Before committing:
```
pytest
```

