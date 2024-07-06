import dataclasses
from typing import Optional

from simple_parsing import ArgumentParser

# Used by load_from_checkpoint() in eval_qwen()
from quiet_star.config import QwenDefaultConfig, QwenDefaultModelConfig
from quiet_star.torch.eval import eval_qwen


@dataclasses.dataclass
class EvalConfig:
    """
    Configuration for evaluating a model.
    """

    # version of model to load (see the lightning_logs directory)
    version: int = 0
    # epoch of checkpoint to load
    epoch: Optional[int] = None
    # limit number of samples to evalute on
    limit: Optional[int] = None


def parse_args() -> EvalConfig:
    parser = ArgumentParser()
    parser.add_arguments(EvalConfig, dest="config")
    args = parser.parse_args()
    return args.config


def main(config: EvalConfig) -> None:
    eval_qwen(config.version, config.epoch, config.limit)


if __name__ == "__main__":
    config = parse_args()
    main(config)
