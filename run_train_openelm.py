import dataclasses
import warnings

import torch
from simple_parsing import ArgumentParser

from quiet_star.config import Config, ModelConfig
from quiet_star.torch.train import train_openelm

warnings.filterwarnings("ignore")


@dataclasses.dataclass
class OpenELMDefaultModelConfig(ModelConfig):
    """
    Configuration for the OpenELM model architecture.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    max_length: int = 64
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

    model: OpenELMDefaultModelConfig = OpenELMDefaultModelConfig()


def parse_args() -> OpenELMDefaultConfig:
    parser = ArgumentParser()
    parser.add_arguments(OpenELMDefaultConfig, dest="config")
    args = parser.parse_args()
    return args.config


def main(config: OpenELMDefaultConfig) -> None:
    train_openelm(config)


if __name__ == "__main__":
    config = parse_args()
    main(config)
