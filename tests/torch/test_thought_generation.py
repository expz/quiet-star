import random

import lightning
import torch

from quiet_star.config import Config, ModelConfig
from quiet_star.constants import START_THOUGHT_TOKEN
from quiet_star.torch.gpt import GPTModel
from quiet_star.torch.openelm import OpenELMThoughtModel
from quiet_star.torch.qwen import QwenThoughtModel
from quiet_star.torch.qwen_explicit import QwenExplicitThoughtModel


def prepare_test_inputs(
    model: lightning.LightningModule, config: Config, text: str, max_thought_length: int
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
    model: lightning.LightningModule,
    x: list[int],
    thought_tokens: list[list[int]],
    thought_length: int,
    batch_size: int,
) -> torch.Tensor:
    correct_logits = []
    for i in range(len(x)):
        xi = torch.tensor(
            [x[: i + 1] + thought_tokens[i][:thought_length]],
            dtype=torch.int64,
            device=model.device,
        )
        correct_logits.append(model(xi)[0, -1].tolist())
    return torch.tensor(
        [correct_logits for _ in range(batch_size)],
        dtype=model._dtype,
        device=model.device,
    )  # add batch dimension


def prepare_next_thought_token_input(
    x: list[int],
    thought_tokens: list[list[int]],
    thought_length: int,
    batch_size: int,
    device: str | torch.device,
    last_thought_token_only: bool,
) -> torch.Tensor:
    if not last_thought_token_only:
        thoughts = [tokens[:thought_length] for tokens in thought_tokens]
        x = torch.tensor(x, dtype=torch.int64, device=device)
        x = torch.unsqueeze(x, dim=-1)
        thoughts = torch.tensor(thoughts, dtype=torch.int64, device=device)
        inputs = torch.concatenate([x, thoughts], dim=-1).tolist()
    else:
        thoughts = [[tokens[thought_length - 1]] for tokens in thought_tokens]
        inputs = thoughts

    return torch.tensor(
        [inputs for _ in range(batch_size)], dtype=torch.int64, device=device
    )  # add batch dimension


def generate_and_verify_logits(
    model: lightning.LightningModule,
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
        model.device,
        last_thought_token_only=(thought_length > 1),
    )
    logits, activation_cache = model.generate_next_thought_token(
        inputs, thought_length, activation_cache
    )
    logits = logits[:, :, -1].squeeze()  # only compare logits of last thought tokens

    expected_shape = (batch_size, len(x), model.vocab_size)
    assert (
        correct_logits.shape == expected_shape
    ), f"for thought length {thought_length}, correct logits has shape {correct_logits.shape}, expected shape {expected_shape}"
    assert (
        logits.shape == expected_shape
    ), f"for thought length {thought_length}, logits has shape {logits.shape}, expected shape {expected_shape}"

    assert torch.allclose(
        correct_logits, logits, atol=1e-2
    ), f"for thought length {thought_length}, logits were not close: correct logits {correct_logits} actual logits {logits}"

    assert activation_cache is not None

    return activation_cache


def run_thought_generation_test(
    model: lightning.LightningModule, config: Config
) -> None:
    activation_cache = None
    x, thought_tokens = prepare_test_inputs(
        model, config, "This is a test.", config.thought_length
    )
    for t in range(1, config.thought_length + 1):
        activation_cache = generate_and_verify_logits(
            model, x, thought_tokens, t, config.batch_size, activation_cache
        )


def test_gpt_thought_generation() -> None:
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
    model = GPTModel(config).to(config.model.device)
    run_thought_generation_test(model, config)


def test_qwen_thought_generation() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        thought_length=3,
        model=ModelConfig(
            attn_type="torch",
            device=device,
            dropout_attn=0.0,
            dropout_embed=0.0,
            dtype="float32",
            model_name="Qwen/Qwen1.5-0.5B-Chat",
            max_length=32,
        ),
    )
    model = QwenThoughtModel(config).to(config.model.device)
    run_thought_generation_test(model, config)


def test_qwen_explicit_thought_generation() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        thought_length=3,
        model=ModelConfig(
            attn_type="torch",
            device=device,
            dropout_attn=0.0,
            dropout_embed=0.0,
            dtype="float32",
            model_name="Qwen/Qwen1.5-0.5B-Chat",
            max_length=32,
        ),
    )
    model = QwenExplicitThoughtModel(config).to(config.model.device)
    run_thought_generation_test(model, config)


def test_openelm_thought_generation() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        thought_length=3,
        model=ModelConfig(
            attn_type="torch",
            device=device,
            dropout_attn=0.0,
            dropout_embed=0.0,
            dtype="float32",
            model_name="apple/OpenELM-270M-Instruct",
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            max_length=32,
        ),
    )
    model = OpenELMThoughtModel(config).to(config.model.device)
    run_thought_generation_test(model, config)
