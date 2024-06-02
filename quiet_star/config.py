import dataclasses
from typing import Tuple


@dataclasses.dataclass
class ModelConfig:
    # which implementation of attention to use
    attn_type: str = "torch"
    # device to load the model on
    device: str = "cpu"
    # dropout of the attention layers, not supported by all models
    dropout_attn: float = 0.0
    # dropout of the initial embedding layer, not supported by all models
    dropout_embed: float = 0.0
    # data type of the model weights
    dtype: str = "float32"
    # dimension of the model
    embed_dim: int = 64 * 6
    # maximum context length
    max_length: int = 256
    # name of HuggingFace model to fine tune
    model_name: str = "Qwen/Qwen1.5-0.5B"
    # name of HuggingFace tokenizer to use
    tokenizer_name: str = "Qwen/Qwen1.5-0.5B"
    # number of heads in the multi-head attention
    num_heads: int = 6
    # number of model layers
    num_layers: int = 8


@dataclasses.dataclass
class Config:
    # number of samples per batch
    batch_size: int = 1
    # beta1 and beta2 for the Adam optimizer
    betas: Tuple[float, float] = (0.9, 0.999)
    # learning rate multiplier for the embeddings of the start and end thought tokens
    embedding_grad_weight: float = 100.0
    # string to use to initialize embedding weights of start and end thought tokens
    embedding_init_token: str = "---"
    # number of training epochs
    epochs: int = 2
    # learning rate of adam optimizer
    learning_rate: float = 1e-6
    # number of tokens to generate after the thought for checking the quality of the thought
    lookahead_tokens: int = 4
    # maximum number of samples in the combined training and validation datasets
    max_samples: int = 2048
    # number of thoughts to generate at each location in sequence of tokens
    num_thoughts: int = 2
    # number of warmup steps for the optimizer, not currently supported
    optimizer_warmup: int = 20
    # weight of the policy loss
    policy_weight: float = 1e6
    # seed for random number generators
    seed: int = 123
    # length of thought to generate
    thought_length: int = 12
    # percent of samples to use for the validation set
    test_pct: float = 0.125
    # strength of the weight decay of the Adam optimizer
    weight_decay: float = 0.001

    model: ModelConfig = ModelConfig()
