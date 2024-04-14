from functools import partial

import mlx.core.random  # type: ignore
import mlx.nn
import numpy as np

from quiet_star.config import Config
from quiet_star.dataset import get_open_web_math_dataset
from quiet_star.gpt_mlx import GPTModel
from quiet_star.mlx import MLXDataLoader, MLXTrainer


def train_gpt(config: Config) -> GPTModel:
    mlx.core.random.seed(config.seed)

    model = GPTModel(config)
    model.compile()

    dataset = get_open_web_math_dataset(
        model.tokenizer,
        config.model.max_length,
        2,
        config.max_samples,
        config.test_pct,
        tensor_type="mlx",
        # use_local_cache=False,
    )

    train_dataloader = MLXDataLoader(dataset["train"], batch_size=config.batch_size)
    test_dataloader = MLXDataLoader(dataset["test"], batch_size=config.batch_size)

    trainer = MLXTrainer(max_epochs=config.epochs)

    trainer.fit(model, train_dataloader, test_dataloader)

    return model
