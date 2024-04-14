import random

import mlx.core

from quiet_star.config import Config, ModelConfig
from quiet_star.gpt_mlx import _GPTModel


def tokenize(model: _GPTModel, config: Config, text: str) -> list[int]:
    return model.tokenizer(
        text,
        padding="do_not_pad",
        truncation=True,
        max_length=config.model.max_length - config.thought_length - 2,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"][0].tolist()


def prepare_test_inputs(
    model: _GPTModel,
    config: Config,
    text: str,
    max_thought_length: int,
    lookahead_length: int,
) -> tuple[list[list[int]], list[list[int]]]:
    x = tokenize(model, config, text)
    start_thought_token = tokenize(model, config, "<|startofthought|>")[0]
    end_thought_token = tokenize(model, config, "<|endofthought|>")[0]
    pad_token = model.tokenizer.pad_token_id
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


def extract_correct_hidden_states(config: Config, h1: mlx.core.array) -> mlx.core.array:
    b, l, _, _ = h1.shape
    k = 1 + config.thought_length + 1
    h1 = [
        [
            mlx.core.take(
                h1[j, i],
                mlx.core.array(list(range(i + k, i + k + config.lookahead_tokens))),
                axis=0,
            ).tolist()
            for i in range(l)
        ]
        for j in range(b)
    ]
    h1 = mlx.core.array(h1)

    # drop the states where we don't have enough lookahead tokens
    return h1[:, : l - (config.lookahead_tokens - 1)]


def test_hidden_states() -> None:
    config = Config(
        batch_size=2,
        lookahead_tokens=3,
        thought_length=4,
        model=ModelConfig(
            attn_type="mlx",
            dropout_attn=0.0,
            dropout_embed=0.0,
            embed_dim=3 * 8,
            max_length=32,
            num_heads=3,
            num_layers=1,
        ),
    )
    model = _GPTModel(config)

    text = "This is a test."
    l1, l2 = [], []
    for _ in range(config.batch_size):
        i1, i2 = prepare_test_inputs(
            model, config, text, config.thought_length, config.lookahead_tokens
        )
        l1.append(i1)
        l2.append(i2)

    x1 = mlx.core.array(l1, dtype=mlx.core.uint32)
    x2 = mlx.core.array(l2, dtype=mlx.core.uint32)

    expected_x1_shape = (
        config.batch_size,
        x1.shape[1],
        x1.shape[1] + config.thought_length + 2,
    )
    expected_x2_shape = (
        config.batch_size,
        x2.shape[1],
        1 + config.thought_length + 2 + config.lookahead_tokens,
    )
    assert (
        x1.shape == expected_x1_shape
    ), f"x1 has unexpected shape, expected shape: {expected_x1_shape}, actual shape: {x1.shape}"
    assert (
        x2.shape == expected_x2_shape
    ), f"x2 has unexpected shape, expected shape: {expected_x2_shape}, actual shape: {x2.shape}"

    b, l, m = x1.shape

    # x1 must be 2D to work with mlx.nn.MultiHeadAttention
    _, h1 = model(x1.reshape(b * l, m), return_hidden_state=True)
    h1 = h1.reshape(b, l, m, -1)

    # extract the hidden states we care about
    h1 = extract_correct_hidden_states(config, h1)

    h2 = model.hidden_states(x2)

    assert (
        h1.shape == h2.shape
    ), f"correct hidden states and calculate hidden states had different shape: {h1.shape} and {h2.shape}"

    assert mlx.core.allclose(
        h1, h2, atol=1e-6
    ).item(), f"the hidden states were not correct, correct hidden states: {h1}, actual hidden states: {h2}"
