import dataclasses
from typing import Tuple


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
    batch_size: int = 1
    betas: Tuple[float, float] = (0.9, 0.999)
    embedding_grad_weight: float = 100.0
    embedding_init_token: str = "---"
    epochs: int = 2
    learning_rate: float = 1e-6
    lookahead_tokens: int = 4
    max_samples: int = 2048
    num_thoughts: int = 2
    optimizer_warmup: int = 20
    policy_weight: float = 1e6
    seed: int = 123
    thought_length: int = 12
    test_pct: float = 0.125
    weight_decay: float = 0.001

    model: ModelConfig = ModelConfig()
