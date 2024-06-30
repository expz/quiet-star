import sys

import pytest

from quiet_star.config import GPTConfig, GPTModelConfig

try:
    import mlx.core

    from quiet_star.mlx.gpt import GPTModel
except ModuleNotFoundError:
    pass

if sys.platform != "darwin":
    pytest.skip("All tests in this module require macOS", allow_module_level=True)


def test_training_step() -> None:
    config = GPTConfig(
        batch_size=5,
        thought_length=3,
        lookahead_tokens=2,
        model=GPTModelConfig(
            attn_type="mlx",
            dropout_attn=0.0,
            dropout_embed=0.0,
            embed_dim=3 * 8,
            max_length=32,
            num_heads=3,
            num_layers=3,
        ),
    )
    model = GPTModel(config)

    text = "This is a longer test sentence."
    x = model.tokenizer(
        text,
        padding="do_not_pad",
        truncation=True,
        max_length=config.model.max_length - config.thought_length - 2,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"][0].tolist()
    x = mlx.core.array([x for _ in range(config.batch_size)])

    GPTModel.forward_pass(model.model, x[:, :-1], x[:, 1:])
