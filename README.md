# Quiet STAR

An implementation of the Quiet-STAR paper (https://arxiv.org/pdf/2403.09629.pdf) including a full test suite to guarantee correctness.

This implementation is optimized so that it performs a minimal amount of computation when generating thoughts. The tricks used to reduce the computation were alluded to in the paper.

Current status: Training on 32K samples with the default configs will improve the performance of the Qwen2 model on a sample of 100 problems from GSM8K (27% => 32%) cot-pass@1 using greedy sampling. The paper checks for improvements on cot-maj@8 (chain-of-thought majority vote with 8 samples), not pass@1, but evaluating using majority vote is not yet supported by this repo. 

## Usage
This repo requires Python 3.10 or later. To set up the environment on linux (if you do not have poetry installed, see appendix):
```bash
# create a virtual environment named .venv
# you might need to run something like `virtualenv --python=/usr/bin/python3.10 .venv` if you have multiple pythons installed
virtualenv --python=python3 .venv

# activate the virtual environment
source .venv/bin/activate

# install packages in the virtual environment
poetry install
```

### Qwen
To fine tune a Qwen2 0.5B model (default config requires an nvidia GPU with 24GB of RAM, but shortening the context window will reduce the memory requirement):
```bash
python run_train_qwen.py --max_samples 32768 --test_pct 0.015625 --accumulate_batches 8 --epochs 1
```
By default, the dataset used is just a tiny part (2K samples) of the dataset used in the paper. A larger dataset is required to improve performance. Also, due to memory constraints the default sequence length is 92, shorter than the 256 tokens used in the paper. A simple way to get the model to train on a GPU with 11GB of RAM would be to shorten the training sequence length to 32.

To see options, run `python run_train_qwen.py --help`.

Afterward, you can evaluate the model on GSM8K:
```bash
python run_eval_qwen.py --version 0 --limit 20
```
To find the correct version to use, check the output of the `run_train_qwen.py` script and look for this version in an expression like `v_num = `. Alternatively, check out the `lightning_logs` folder to find the version number.

The generation is obviously quite slow because we have to generate thoughts before every token and pass the resulting logits through a mixing head, etc, so you probably want to limit the testing to a small number of samples.

To check the untrained model's performance, run:
```bash
python run_eval_qwen.py --version -1 --limit 20
```

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

After activating the `.venv` virtual environment, try running:
```bash
pip install poetry
```
