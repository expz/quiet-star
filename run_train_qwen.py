import warnings

from simple_parsing import ArgumentParser

from quiet_star.config import QwenDefaultConfig
from quiet_star.torch.train import train_qwen

warnings.filterwarnings("ignore")


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
