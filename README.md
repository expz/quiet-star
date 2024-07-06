# Quiet STAR

An implementation of the Quiet-STAR paper (https://arxiv.org/pdf/2403.09629.pdf) including a full test suite to guarantee correctness.

This implementation is optimized so that it performs a minimal amount of computation when generating thoughts. The tricks used to reduce the computation were alluded to in the paper.

Full disclosure: I have not trained the model on enough data yet to see a performance improvement. With the default settings and 32K samples, the trained Qwen model using thought generation comes close (35%) to the performance of the original model without thoughts (45%).

See TODO.md for planned improvements.

## Usage
To set up the environment (if you do not have poetry installed, see appendix):
```bash
poetry config keyring.enabled false
poetry install
source .venv/bin/activate
```

### Qwen
To fine tune a Qwen2 0.5B model (default config requires an nvidia GPU with 24GB of RAM):
```bash
python run_train_qwen.py
```
By default, the dataset used is just a small part (2K samples) of the dataset used in the paper. A larger dataset is required to improve performance. Also, due to memory constraints the default sequence length is 80, shorter than the 256 tokens used in the paper. A simple way to get the model to train on a GPU with 11GB of RAM would be to shorten the training sequence length to 32.

To see options, run `python run_train_qwen.py --help`.

Afterward, you can evaluate the model on GSM8K:
```bash
python run_eval_qwen.py --version 0 --limit 20
```
To find the correct version to use, check the output of the `run_train_qwen.py` script and look for this version in an expression like `v_num = `. Alternatively, check out the `lightning_logs` folder to find the version number.

The generation is obviously quite slow because we have to generate thoughts before every token and pass the resulting logits through a mixing head, etc, so you probably want to limit the testing to a small number of samples.

### OpenELM
The Quiet-STAR algorithm aims to improve performance on, among other things, GSM8K type math problems. After implementing support for OpenELM I found that the model cannot answer GSM8K question, so I cannot recommend using it. But in any case, to fine tune an OpenELM 270M model (default config requires an nvidia GPU with 11GB of memory):
```bash
python run_train_openelm.py
```
To see options, run `python run_train_openelm.py --help`.

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
