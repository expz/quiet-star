import logging
import warnings

from simple_parsing import ArgumentParser

# Unused imports are actually used by load_from_checkpoint() in eval_qwen()
from quiet_star.config import (
    EvalConfig,
    QwenDefaultConfig,
    QwenDefaultEvalConfig,
    QwenDefaultModelConfig,
)
from quiet_star.torch.eval import eval_qwen

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


def parse_args() -> EvalConfig:
    parser = ArgumentParser()
    parser.add_arguments(QwenDefaultEvalConfig, dest="config")
    args = parser.parse_args()
    return args.config


def main(config: EvalConfig) -> None:
    eval_qwen(
        config.version,
        config.epoch,
        config.step,
        config.limit,
        config.do_sample,
        config.temperature,
        config.eval_max_length,
        config.model_name,
        config.tokenizer_name,
        config.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    main(config)
