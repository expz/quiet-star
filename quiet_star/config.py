import dataclasses
from typing import Any, Tuple

import torch

from quiet_star.constants import MetricLoggerBackend


def flatten(d: dict[str, Any], parent_key: str | None = None) -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key is not None else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


@dataclasses.dataclass
class ModelConfig:
    # device to load the model on
    device: str = "cpu"
    # data type of the model weights
    dtype: str = "float32"
    # maximum context length while training
    train_max_length: int = 256
    # name of HuggingFace model to fine tune
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    # name of HuggingFace tokenizer to use
    tokenizer_name: str = "Qwen/Qwen2-0.5B-Instruct"


@dataclasses.dataclass
class GPTModelConfig(ModelConfig):
    # which implementation of attention to use
    attn_type: str = "torch"
    # dropout of the attention layers
    dropout_attn: float = 0.0
    # dropout of the initial embedding layer
    dropout_embed: float = 0.0
    # dimension of the model
    embed_dim: int = 64 * 6
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
    # backend to use for logging metrics
    logger: MetricLoggerBackend = MetricLoggerBackend.NONE
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
    # number of hours between saving checkpoints
    save_interval_hours: float = 4
    # keep the last `save_top_k` checkpoints or -1 to keep all checkpoints
    save_top_k: int = -1
    # seed for random number generators
    seed: int = 123
    # temperature to use when sampling thoughts
    temperature: float = 0.7
    # percent of samples to use for the validation set
    test_pct: float = 0.125
    # length of thought to generate
    thought_length: int = 12
    # strength of the weight decay of the Adam optimizer
    weight_decay: float = 0.001

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    def as_dict(self) -> dict[str, Any]:
        return flatten(dataclasses.asdict(self))


@dataclasses.dataclass
class GPTConfig(Config):
    model: GPTModelConfig = dataclasses.field(default_factory=GPTModelConfig)


@dataclasses.dataclass
class QwenDefaultModelConfig(ModelConfig):
    """
    Configuration for the Qwen model architecture.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    train_max_length: int = 92
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer_name: str = "Qwen/Qwen2-0.5B-Instruct"


@dataclasses.dataclass
class QwenDefaultConfig(Config):
    """
    Configuration for the Qwen model training process and dataset.
    """

    batch_size: int = 1
    epochs: int = 2
    lookahead_tokens: int = 4
    max_samples: int = 2048
    num_thoughts: int = 2
    seed: int = 1
    thought_length: int = 8

    model: QwenDefaultModelConfig = dataclasses.field(
        default_factory=QwenDefaultModelConfig
    )


@dataclasses.dataclass
class OpenELMDefaultModelConfig(ModelConfig):
    """
    Configuration for the OpenELM model architecture.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    train_max_length: int = 64
    model_name: str = "apple/OpenELM-270M-Instruct"
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"


@dataclasses.dataclass
class OpenELMDefaultConfig(Config):
    """
    Configuration for the OpenELM model training process and dataset.
    """

    batch_size: int = 1
    epochs: int = 2
    lookahead_tokens: int = 4
    max_samples: int = 2048
    num_thoughts: int = 2
    seed: int = 1
    thought_length: int = 8

    model: OpenELMDefaultModelConfig = dataclasses.field(
        default_factory=OpenELMDefaultModelConfig
    )


@dataclasses.dataclass
class EvalConfig:
    """
    Configuration for evaluating a model.
    """

    # version of model to load (see the lightning_logs directory) or -1 for untrained model
    version: int = 0
    # epoch of checkpoint to load
    epoch: int | None = None
    # step of epoch of checkpoint to load
    step: int | None = None
    # limit number of samples to evalute on
    limit: int | None = None
    # HuggingFace model name if using untrained model
    model_name: str | None = None
    # HuggingFace tokenizer name if using untrained model
    tokenizer_name: str | None = None


@dataclasses.dataclass
class QwenDefaultEvalConfig(EvalConfig):
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer_name: str = "Qwen/Qwen2-0.5B-Instruct"


@dataclasses.dataclass
class OpenELMDefaultEvalConfig(EvalConfig):
    model_name: str = "apple/OpenELM-270M-Instruct"
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
