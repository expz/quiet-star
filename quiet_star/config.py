import dataclasses
from typing import Tuple

import torch


@dataclasses.dataclass
class ModelConfig:
    attn_type: str = "torch"
    device: str = "cpu"
    dropout_attn: float = 0.0
    dropout_embed: float = 0.0
    dtype: str = "float32"
    embed_dim: int = 64 * 6
    max_length: int = 256
    model_name: str = "Qwen/Qwen1.5-0.5B"
    num_heads: int = 6
    num_layers: int = 8


@dataclasses.dataclass
class Config:
    batch_size: int = 16
    betas: Tuple[float, float] = (0.9, 0.999)
    epochs: int = 2
    learning_rate: float = 1e-5
    max_samples: int = 2048
    max_thought_length: int = 12
    seed: int = 123
    test_pct: float = 0.125
    weight_decay: float = 0.01

    model: ModelConfig = ModelConfig()
