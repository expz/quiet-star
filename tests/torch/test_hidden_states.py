import random

import lightning
import torch

from quiet_star.config import Config, GPTConfig, GPTModelConfig, ModelConfig
from quiet_star.constants import END_THOUGHT_TOKEN, START_THOUGHT_TOKEN
from quiet_star.torch.gpt import GPTModel
from quiet_star.torch.openelm import OpenELMThoughtModel
from quiet_star.torch.qwen import QwenThoughtModel
from quiet_star.torch.qwen_explicit import QwenExplicitThoughtModel
from quiet_star.torch.utils import torch_dtype


def tokenize(model: GPTModel, config: Config, text: str) -> list[int]:
    return model.tokenizer(
        text,
        padding="do_not_pad",
        truncation=True,
        max_length=config.model.max_length - config.thought_length - 2,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"][0].tolist()


def prepare_test_inputs(
    model: GPTModel,
    config: Config,
    text: str,
    max_thought_length: int,
    lookahead_length: int,
) -> tuple[list[list[int]], list[list[int]]]:
    x = tokenize(model, config, text)
    start_thought_tokens = tokenize(model, config, START_THOUGHT_TOKEN)
    end_thought_tokens = tokenize(model, config, END_THOUGHT_TOKEN)
    start_thought_token = (
        start_thought_tokens[1]
        if start_thought_tokens[0] == model.tokenizer.bos_token_id
        else start_thought_tokens[0]
    )
    end_thought_token = (
        end_thought_tokens[1]
        if end_thought_tokens[0] == model.tokenizer.bos_token_id
        else end_thought_tokens[0]
    )
    pad_token = model.tokenizer.pad_token_id
    if pad_token is None:
        pad_token = model.tokenizer.bos_token_id
    x1 = [
        x[:i]
        + [start_thought_token]
        + [random.randrange(0, len(model.tokenizer)) for _ in range(max_thought_length)]
        + [end_thought_token]
        + x[i:]
        for i in range(1, len(x) + 1)
    ]
    m = 1 + max_thought_length + 2 + lookahead_length
    x2 = [
        x1[i][i : i + m] + [pad_token] * (i + m - (len(x) + max_thought_length + 2))
        for i in range(len(x))
    ]
    return x1, x2


def extract_correct_hidden_states(config: Config, h1: torch.Tensor) -> torch.Tensor:
    b, l, _, _ = h1.shape
    k = 1 + config.thought_length + 1
    h1 = [
        [h1[j, i, i + k : i + k + config.lookahead_tokens].tolist() for i in range(l)]
        for j in range(b)
    ]

    h1 = torch.tensor(
        h1, dtype=torch_dtype(config.model.dtype), device=config.model.device
    )

    return h1


def run_hidden_states_test(model: lightning.LightningModule, config: Config) -> None:
    text = "This is a test."
    l1, l2 = [], []
    for _ in range(config.batch_size):
        i1, i2 = prepare_test_inputs(
            model, config, text, config.thought_length, config.lookahead_tokens
        )
        l1.append(i1)
        l2.append(i2)

    x1 = torch.tensor(l1, dtype=torch.int64, device=config.model.device)
    x2 = torch.tensor(l2, dtype=torch.int64, device=config.model.device)

    l = x1.shape[1]
    expected_x1_shape = (
        config.batch_size,
        l,
        l + config.thought_length + 2,
    )
    expected_x2_shape = (
        config.batch_size,
        l,
        1 + config.thought_length + 2 + config.lookahead_tokens,
    )
    assert (
        x1.shape == expected_x1_shape
    ), f"x1 has unexpected shape, expected shape: {expected_x1_shape}, actual shape: {x1.shape}"
    assert (
        x2.shape == expected_x2_shape
    ), f"x2 has unexpected shape, expected shape: {expected_x2_shape}, actual shape: {x2.shape}"

    # only include states where we have enough lookahead tokens
    x1 = x1[:, : l - (config.lookahead_tokens - 1)]
    x2 = x2[:, : l - (config.lookahead_tokens - 1)]

    b, l, m = x1.shape

    # x1 must be 2D to work with Qwen2
    _, h1, _ = model.forward(x1.reshape(b * l, m), return_hidden_state=True)
    h1 = h1.reshape(b, l, m, -1)

    # extract the hidden states we care about
    h1 = extract_correct_hidden_states(config, h1)

    h2 = model.hidden_states(x2)

    assert (
        h1.shape == h2.shape
    ), f"correct hidden states and calculate hidden states had different shape: {h1.shape} and {h2.shape}"

    assert torch.allclose(
        h1, h2, atol=1e-2
    ), f"the hidden states were not correct, correct hidden states: {h1}, actual hidden states: {h2}"


def test_gpt_hidden_states() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(
        batch_size=2,
        lookahead_tokens=3,
        thought_length=3,
        model=GPTModelConfig(
            attn_type="torch",
            device=device,
            dropout_attn=0.0,
            dropout_embed=0.0,
            dtype="float32",
            embed_dim=3 * 8,
            max_length=32,
            num_heads=3,
            num_layers=3,
        ),
    )
    model = GPTModel(config).to(config.model.device)
    run_hidden_states_test(model, config)


def test_qwen_hidden_states() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        lookahead_tokens=3,
        thought_length=3,
        model=ModelConfig(
            device=device,
            dtype="float32",
            model_name="Qwen/Qwen2-0.5B-Instruct",
            max_length=32,
        ),
    )
    model = QwenThoughtModel(config).to(config.model.device)
    run_hidden_states_test(model, config)


def test_qwen_explicit_hidden_states() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        lookahead_tokens=3,
        thought_length=3,
        model=ModelConfig(
            device=device,
            dtype="float32",
            model_name="Qwen/Qwen2-0.5B-Instruct",
            max_length=32,
        ),
    )
    model = QwenExplicitThoughtModel(config).to(config.model.device)
    run_hidden_states_test(model, config)


def test_openelm_hidden_states() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        lookahead_tokens=3,
        thought_length=3,
        model=ModelConfig(
            dtype="float32",
            device=device,
            model_name="apple/OpenELM-270M-Instruct",
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            max_length=32,
        ),
    )
    model = OpenELMThoughtModel(config).to(config.model.device)
    run_hidden_states_test(model, config)
