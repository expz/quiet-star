# Quiet STAR

An implementation of the Quiet-STAR paper (https://arxiv.org/pdf/2403.09629.pdf) including a full test suite to guarantee correctness.

This implementation is optimized so that it performs a minimal amount of computation when generating thoughts. The tricks used to reduce the computation were alluded to in the paper.

See TODO.md for planned improvements.

## Usage
To set up the environment (if you do not have poetry installed, see appendix):
```bash
poetry config keyring.enabled false
poetry install
source .venv/bin/activate
```

### OpenELM
To fine tune an OpenELM 270M model (default config requires an nvidia GPU with 11GB of memory):
```bash
python run_train_openelm.py
```
To see options, run `python run_train_openelm.py --help`.

### Qwen
To fine tune a Qwen 0.5B model (default config requires an nvidia GPU with 24GB of RAM):
```bash
python run_train_qwen.py
```
By default, the dataset used is just a small part of the dataset used in the paper. Also, by default the sequence length is shorter than the 256 tokens used in the paper.

### MLX
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

## Appendix: Installing Poetry

```bash
# If you do not have virtualenv installed yet
pip install virtualenv

# Create a virtual environment with Python 3.10 called .venv and install poetry
virtualenv --python=/usr/bin/python3.10 .venv
source .venv/bin/activate
pip install poetry
```
