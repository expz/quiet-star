# Quiet STAR

An implementation of the Quiet-STAR paper (https://arxiv.org/pdf/2403.09629.pdf).

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
There is code (with tests!) in this repository to fine tune a model using MLX on Apple silicon, but it is not currently set up to use a pretrained HuggingFace model.

## Development
To begin:
```bash
pre-commit install
```
Before committing:
```
pytest
```

