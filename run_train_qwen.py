import dataclasses
import warnings

import torch
from simple_parsing import ArgumentParser

from quiet_star.config import Config, ModelConfig
from quiet_star.torch.train import train_qwen

warnings.filterwarnings("ignore")


@dataclasses.dataclass
class QwenDefaultModelConfig(ModelConfig):
    """
    Configuration for the Qwen model architecture.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    max_length: int = 80
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

    model: QwenDefaultModelConfig = QwenDefaultModelConfig()


def parse_args() -> QwenDefaultConfig:
    parser = ArgumentParser()
    parser.add_arguments(QwenDefaultConfig, dest="config")
    args = parser.parse_args()
    return args.config


def main(config: QwenDefaultConfig) -> None:
    train_qwen(config)


if __name__ == "__main__":
    config = parse_args()
    main(config)
