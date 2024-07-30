import logging
import warnings

from simple_parsing import ArgumentParser

# Used by load_from_checkpoint() in eval_openelm()
from quiet_star.config import (
    EvalConfig,
    OpenELMDefaultConfig,
    OpenELMDefaultEvalConfig,
    OpenELMDefaultModelConfig,
)
from quiet_star.torch.eval import eval_openelm

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


def parse_args() -> EvalConfig:
    parser = ArgumentParser()
    parser.add_arguments(OpenELMDefaultEvalConfig, dest="config")
    args = parser.parse_args()
    return args.config


def main(config: EvalConfig) -> None:
    eval_openelm(
        config.version,
        config.epoch,
        config.step,
        config.limit,
        config.eval_max_length,
        config.model_name,
        config.tokenizer_name,
    )


if __name__ == "__main__":
    config = parse_args()
    main(config)
