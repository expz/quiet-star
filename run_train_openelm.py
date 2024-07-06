import warnings

from simple_parsing import ArgumentParser

from quiet_star.config import OpenELMDefaultConfig
from quiet_star.torch.train import train_openelm

warnings.filterwarnings("ignore")


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
