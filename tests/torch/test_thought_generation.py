import random

import torch

from quiet_star.config import Config, ModelConfig
from quiet_star.constants import START_THOUGHT_TOKEN
from quiet_star.torch.gpt import GPTModel


def prepare_test_inputs(
    model: GPTModel, config: Config, text: str, max_thought_length: int
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
        START_THOUGHT_TOKEN, return_tensors="np", return_attention_mask=False
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
    model: GPTModel,
    x: list[int],
    thought_tokens: list[list[int]],
    thought_length: int,
    batch_size: int,
) -> torch.Tensor:
    correct_logits = []
    for i in range(len(x)):
        xi = torch.LongTensor([x[: i + 1] + thought_tokens[i][:thought_length]])
        correct_logits.append(model(xi)[0, -1].tolist())
    return torch.FloatTensor(
        [correct_logits for _ in range(batch_size)]
    )  # add batch dimension


def prepare_next_thought_token_input(
    x: list[int],
    thought_tokens: list[list[int]],
    thought_length: int,
    batch_size: int,
    last_thought_token_only: bool,
) -> torch.Tensor:
    if not last_thought_token_only:
        thoughts = [tokens[:thought_length] for tokens in thought_tokens]
        x = torch.LongTensor(x)
        x = torch.unsqueeze(x, dim=-1)
        thoughts = torch.LongTensor(thoughts)
        inputs = torch.concatenate([x, thoughts], dim=-1).tolist()
    else:
        thoughts = [[tokens[thought_length - 1]] for tokens in thought_tokens]
        inputs = thoughts

    return torch.LongTensor([inputs for _ in range(batch_size)])  # add batch dimension


def generate_and_verify_logits(
    model: GPTModel,
    x: list[int],
    thought_tokens: list[list[int]],
    thought_length: int,
    batch_size: int,
    activation_cache: list[dict[str, torch.Tensor]] | None,
) -> list[dict[str, torch.Tensor]]:
    correct_logits = calculate_correct_logits(
        model, x, thought_tokens, thought_length, batch_size
    )

    inputs = prepare_next_thought_token_input(
        x,
        thought_tokens,
        thought_length,
        batch_size,
        last_thought_token_only=(thought_length > 1),
    )
    logits, activation_cache = model.generate_next_thought_token(
        inputs, thought_length, activation_cache
    )
    logits = logits[:, :, -1].squeeze()  # only compare logits of last thought tokens

    expected_shape = (batch_size, len(x), len(model.tokenizer))
    assert (
        correct_logits.shape == expected_shape
    ), f"for thought length {thought_length}, correct logits has shape {correct_logits.shape}, expected shape {expected_shape}"
    assert (
        logits.shape == expected_shape
    ), f"for thought length {thought_length}, logits has shape {logits.shape}, expected shape {expected_shape}"

    assert torch.allclose(
        correct_logits, logits, atol=1e-6
    ), f"for thought length {thought_length}, logits were not close: correct logits {correct_logits} actual logits {logits}"

    assert activation_cache is not None

    return activation_cache


def test_thought_generation() -> None:
    config = Config(
        batch_size=2,
        thought_length=3,
        model=ModelConfig(
            attn_type="torch",
            dropout_attn=0.0,
            dropout_embed=0.0,
            embed_dim=3 * 8,
            max_length=32,
            num_heads=3,
            num_layers=3,
        ),
    )
    model = GPTModel(config)

    activation_cache = None
    x, thought_tokens = prepare_test_inputs(
        model, config, "This is a test.", config.thought_length
    )
    for t in range(1, config.thought_length + 1):
        activation_cache = generate_and_verify_logits(
            model, x, thought_tokens, t, config.batch_size, activation_cache
        )