import mlx.core

from quiet_star.config import Config, ModelConfig
from quiet_star.gpt_mlx import _GPTModel


def test_thought_generation():
    config = Config(
        batch_size=4,
        max_thought_length=4,
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
    x = model.tokenizer(
        "This is a test.",
        padding="do_not_pad",
        truncation=True,
        max_length=config.model.max_length - config.max_thought_length - 2,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"][0].tolist()
    start_thought_token = model.tokenizer(
        "<|startofthought|>", return_tensors="np", return_attention_mask=False
    )["input_ids"][0].tolist()
    correct_logits = []
    for i in range(1, len(x) + 1):
        xi = mlx.core.array([x[:i] + start_thought_token])
        correct_logits.append(model(xi)[0, i])
    correct_logits = mlx.core.array([correct_logits])
    expected_shape = (1, len(x), len(model.tokenizer))
    assert (
        correct_logits.shape == expected_shape
    ), f"correct_logits has shape {correct_logits.shape}, expected shape {expected_shape}"

    xp = mlx.core.array([x], dtype=mlx.core.uint32)
    xp = mlx.core.expand_dims(xp, axis=2)
    start_token = mlx.core.full(
        xp.shape, model.start_thought_token_id, dtype=mlx.core.uint32
    )
    xp = mlx.core.concatenate([xp, start_token], axis=2)
    activation_cache = None
    logits, activation_cache = model.generate_next_thought_token(xp, 1, activation_cache)
    expected_shape = (1, len(x), 2, len(model.tokenizer))
    assert logits.shape == expected_shape, f"logits has shape {logits.shape}, expected shape {expected_shape}"

    logits = logits[:, :, -1]

    assert mlx.core.allclose(correct_logits, logits, atol=1e-6).item(), f"logits were not close: {correct_logits} {logits}"
