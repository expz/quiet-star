import random

import mlx.core

from quiet_star.config import Config, ModelConfig
from quiet_star.gpt_mlx import _GPTModel


BATCH_SIZE = 2

def prepare_test_inputs(
    model: _GPTModel, config: Config, text: str, max_thought_length: int
) -> tuple[list[int], list[list[int]]]:
    x = model.tokenizer(
        text,
        padding="do_not_pad",
        truncation=True,
        max_length=config.model.max_length - config.thought_length - 2,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"][0].tolist()
    start_thought_token = model.tokenizer(
        "<|startofthought|>", return_tensors="np", return_attention_mask=False
    )["input_ids"][0, 0].tolist()
    thought_tokens = [[start_thought_token] for _ in range(len(x))]
    if max_thought_length > 0:
        next_tokens = [
            [
                random.randrange(0, len(model.tokenizer))
                for _ in range(max_thought_length)
            ]
            for _ in range(len(x))
        ]
        thought_tokens = [thought_tokens[i] + next_tokens[i] for i in range(len(x))]

    return x, thought_tokens


def calculate_correct_logits(
    model: _GPTModel, x: list[int], thought_tokens: list[list[int]], thought_length: int
):
    correct_logits = []
    for i in range(len(x)):
        xi = mlx.core.array([x[: i + 1] + thought_tokens[i][:thought_length]])
        correct_logits.append(model(xi)[0, -1])
    return mlx.core.array([correct_logits for _ in range(BATCH_SIZE)])  # add batch dimension


def prepare_next_thought_token_input(
    x: list[int],
    thought_tokens: list[list[int]],
    thought_length: int,
    last_thought_token_only: bool,
) -> mlx.core.array:
    if not last_thought_token_only:
        thoughts = [tokens[:thought_length] for tokens in thought_tokens]
        x = mlx.core.array(x, dtype=mlx.core.uint32)
        x = mlx.core.expand_dims(x, axis=-1)
        thoughts = mlx.core.array(thoughts, dtype=mlx.core.uint32)
        inputs = mlx.core.concatenate([x, thoughts], axis=-1)
    else:
        thoughts = [[tokens[thought_length - 1]] for tokens in thought_tokens]
        inputs = mlx.core.array(thoughts, dtype=mlx.core.uint32)

    return mlx.core.array([inputs for _ in range(BATCH_SIZE)])  # add batch dimension


def generate_and_verify_logits(
    model: _GPTModel,
    x: list[int],
    thought_tokens: list[int],
    thought_length: int,
    activation_cache: list[dict[str, mlx.core.array]],
) -> list[dict[str, mlx.core.array]]:
    correct_logits = calculate_correct_logits(model, x, thought_tokens, thought_length)

    inputs = prepare_next_thought_token_input(
        x, thought_tokens, thought_length, last_thought_token_only=(thought_length > 1)
    )
    logits, activation_cache = model.generate_next_thought_token(
        inputs, thought_length, activation_cache
    )
    logits = logits[:, :, -1].squeeze()  # only compare logits of last thought tokens

    expected_shape = (BATCH_SIZE, len(x), len(model.tokenizer))
    assert (
        correct_logits.shape == expected_shape
    ), f"for thought length {thought_length}, correct logits has shape {correct_logits.shape}, expected shape {expected_shape}"
    assert (
        logits.shape == expected_shape
    ), f"for thought length {thought_length}, logits has shape {logits.shape}, expected shape {expected_shape}"

    assert mlx.core.allclose(
        correct_logits, logits, atol=1e-6
    ).item(), f"for thought length {thought_length}, logits were not close: correct logits {correct_logits} actual logits {logits}"

    return activation_cache


def test_thought_generation():
    max_thought_length = 3

    config = Config(
        batch_size=4,
        thought_length=4,
        model=ModelConfig(
            attn_type="mlx",
            dropout_attn=0.0,
            dropout_embed=0.0,
            embed_dim=3 * 8,
            max_length=32,
            num_heads=3,
            num_layers=3,
        ),
    )
    model = _GPTModel(config)

    activation_cache = None
    x, thought_tokens = prepare_test_inputs(
        model, config, "This is a test.", max_thought_length
    )
    for t in range(1, max_thought_length + 1):
        activation_cache = generate_and_verify_logits(
            model, x, thought_tokens, t, activation_cache
        )
