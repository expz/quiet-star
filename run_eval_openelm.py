import dataclasses
from typing import Optional

from simple_parsing import ArgumentParser

from quiet_star.torch.eval import eval_openelm


@dataclasses.dataclass
class EvalConfig:
    """
    Configuration for evaluating a model.
    """

    # version of model to load (see the lightning_logs directory)
    version: int = 0
    # epoch of checkpoint to load
    epoch: Optional[int] = None


def parse_args() -> EvalConfig:
    parser = ArgumentParser()
    parser.add_arguments(EvalConfig, dest="config")
    args = parser.parse_args()
    return args.config


def main(config: EvalConfig) -> None:
    eval_openelm(config.version, config.epoch)


if __name__ == "__main__":
    config = parse_args()
    main(config)
