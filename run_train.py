import torch

from quiet_star.config import Config, ModelConfig
from quiet_star.train import train_gpt


def main():
    config = Config(
        batch_size=16,
        epochs=2,
        seed=1,
        model=ModelConfig(
            attn_type="triton",
            dropout_attn=0.0,
            dropout_embed=0.0,
            dtype=torch.bfloat16,
        ),
    )
    train_gpt(config)


if __name__ == "__main__":
    main()
