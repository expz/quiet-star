import torch

from quiet_star.config import Config, ModelConfig
from quiet_star.torch.train import train_qwen


def main() -> None:
    config = Config(
        batch_size=1,
        epochs=2,
        lookahead_tokens=4,
        num_thoughts=2,
        seed=1,
        thought_length=10,
        model=ModelConfig(
            attn_type="torch",
            device=torch.device("cuda"),
            dropout_attn=0.0,
            dropout_embed=0.0,
            dtype="bfloat16",
            embed_dim=64 * 6,
            max_length=64,
            model_name="Qwen/Qwen1.5-0.5B-Chat",
            num_heads=6,
            num_layers=8,
        ),
    )
    train_qwen(config)


if __name__ == "__main__":
    main()
