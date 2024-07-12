import random

import lightning
import torch

from quiet_star.config import Config, GPTConfig, GPTModelConfig, ModelConfig
from quiet_star.constants import END_THOUGHT_TOKEN, START_THOUGHT_TOKEN
from quiet_star.torch.gpt import GPTModel
from quiet_star.torch.openelm import OpenELMThoughtModel
from quiet_star.torch.pretrained import PretrainedThoughtModel
from quiet_star.torch.qwen import QwenThoughtModel
from quiet_star.torch.qwen_explicit import QwenExplicitThoughtModel
from quiet_star.torch.utils import torch_dtype


def tokenize(model: PretrainedThoughtModel, config: Config, text: str) -> torch.Tensor:
    return model.tokenizer(
        text,
        padding="do_not_pad",
        truncation=True,
        max_length=config.model.train_max_length - config.thought_length - 2,
        return_tensors="pt",
        return_attention_mask=False,
    )["input_ids"][0].to(model.device)


def calculate_correct_thoughts(
    model: PretrainedThoughtModel, input_t: torch.Tensor
) -> torch.Tensor:
    """
    This function loops through every position in the input sequence and
    generates a thought while saving the logits. After generating the thought,
    it does one more model inference to get the hidden states of the lookahead
    tokens.
    """
    b, l = input_t.shape
    l -= model.lookahead_tokens
    m = 1 + model.thought_length + 2 + model.lookahead_tokens

    correct_logits_thought = []
    correct_h_lookahead = []
    correct_tokens = []
    start_thought_tokens = torch.tensor(
        [[model.start_thought_token_id] for _ in range(b)],
        dtype=torch.int64,
        device=input_t.device,
    )
    end_thought_tokens = torch.tensor(
        [[model.end_thought_token_id] for _ in range(b)],
        dtype=torch.int64,
        device=input_t.device,
    )
    # loop through each position in the input sequence and generate a thought
    for token in range(l):
        x = input_t[:, : token + 1]
        x = torch.concatenate([x, start_thought_tokens], dim=-1)
        logits_thought = None
        # generate a thought
        for _ in range(model.thought_length):
            logits = model.model(x).logits[:, -1:]
            next_tokens = logits.argmax(dim=-1)
            x = torch.concatenate([x, next_tokens], dim=-1)
            if logits_thought is None:
                logits_thought = logits
            else:
                logits_thought = torch.cat([logits_thought, logits], dim=-2)

        x_with_thoughts = torch.concatenate([x, end_thought_tokens], dim=-1)

        lookahead_tokens = input_t[:, token + 1 : token + 1 + model.lookahead_tokens]
        x_with_thoughts = torch.concatenate([x_with_thoughts, lookahead_tokens], dim=-1)

        h = model.model(x_with_thoughts, output_hidden_states=True).hidden_states[-1]
        # the first lookahead token's hidden state is the output of the end thought token
        # hence the extra 1
        h_lookahead = h[:, -(model.lookahead_tokens + 1) : -1]
        correct_h_lookahead.append(h_lookahead)

        correct_logits_thought.append(logits_thought)

        # only save the token immediately before the thought, the thought, and
        # the lookahead tokens because that is the shape of the tensor we will
        # compare to
        correct_tokens.append(x_with_thoughts[:, token : token + m])

    correct_logits_thought_t = torch.stack(correct_logits_thought, dim=1)
    correct_h_lookahead_t = torch.stack(correct_h_lookahead, dim=1)
    correct_tokens_t = torch.stack(correct_tokens, dim=1)

    return correct_logits_thought_t, correct_h_lookahead_t, correct_tokens_t


def run_generate_thoughts_test(
    model: lightning.LightningModule, config: Config
) -> None:
    text = "This is an even longer test sentence."
    x = tokenize(model, config, text)
    x = x.repeat(config.batch_size, 1)

    with torch.no_grad():
        correct_logits_thought, correct_h_lookahead, correct_x_with_thoughts = (
            calculate_correct_thoughts(model, x)
        )

        logits_thought, h_lookahead, x_with_thoughts = model.generate_thoughts(x)

    expected_x_with_thoughts_shape = (
        config.batch_size,
        x.size(1) - model.lookahead_tokens,
        1 + model.thought_length + 2 + model.lookahead_tokens,
    )

    assert (
        correct_x_with_thoughts.shape == expected_x_with_thoughts_shape
    ), f"correct_x_with_thoughts has unexpected shape, expected shape: {expected_x_with_thoughts_shape}, actual shape: {correct_x_with_thoughts.shape}"
    assert (
        x_with_thoughts.shape == expected_x_with_thoughts_shape
    ), f"x_with_thoughts has unexpected shape, expected shape: {expected_x_with_thoughts_shape}, actual shape: {x_with_thoughts.shape}"
    assert torch.all(
        correct_x_with_thoughts == x_with_thoughts
    ), f"x_with_thoughts was not correct, expected token ids:\n{correct_x_with_thoughts},\nactual token ids:\n{x_with_thoughts}"
    assert (
        correct_logits_thought.shape == logits_thought.shape
    ), f"correct logits and calculate logits had different shape: {correct_logits_thought.shape} and {logits_thought.shape}"
    assert (
        correct_h_lookahead.shape == h_lookahead.shape
    ), f"correct hidden states and calculate hidden states had different shape: {correct_h_lookahead.shape} and {h_lookahead.shape}"
    assert torch.allclose(
        correct_logits_thought, logits_thought, atol=1e-2
    ), f"the logits were not correct, correct logits:\n{correct_logits_thought},\nactual logits:\n{logits_thought}"
    assert torch.allclose(
        correct_h_lookahead, h_lookahead, atol=1e-2
    ), f"the hidden states were not correct, correct hidden states:\n{correct_h_lookahead},\nactual hidden states:\n{h_lookahead}"


def test_qwen_hidden_states() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        lookahead_tokens=3,
        thought_length=4,
        model=ModelConfig(
            device=device,
            dtype="float32",
            model_name="Qwen/Qwen2-0.5B-Instruct",
            train_max_length=32,
        ),
    )
    model = QwenThoughtModel(config).to(config.model.device)
    run_generate_thoughts_test(model, config)


def test_openelm_thought_generation() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        thought_length=3,
        model=ModelConfig(
            device=device,
            dtype="float32",
            model_name="apple/OpenELM-270M-Instruct",
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            train_max_length=32,
        ),
    )
    model = OpenELMThoughtModel(config).to(config.model.device)
    run_generate_thoughts_test(model, config)
