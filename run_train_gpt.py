import torch

from quiet_star.config import Config, ModelConfig
from quiet_star.torch.train import train_gpt


def main() -> None:
    config = Config(
        batch_size=2,
        epochs=2,
        seed=1,
        model=ModelConfig(
            attn_type="torch",
            device=torch.device("cuda"),
            dropout_attn=0.0,
            dropout_embed=0.0,
            dtype="bfloat16",
            embed_dim=3 * 8,
            num_heads=3,
            num_layers=3,
        ),
    )
    train_gpt(config)


if __name__ == "__main__":
    main()
