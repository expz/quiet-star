import mlx.core

from quiet_star.config import Config, ModelConfig
from quiet_star.gpt_mlx import GPTModel


def test_training_step() -> None:
    config = Config(
        batch_size=5,
        thought_length=3,
        lookahead_tokens=2,
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
