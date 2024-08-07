import mlx.core.random  # type: ignore
import mlx.nn

from quiet_star.config import GPTConfig
from quiet_star.dataset import _format_tokenizer_name, get_open_web_math_dataset
from quiet_star.mlx.framework import MLXDataLoader, MLXTrainer
from quiet_star.mlx.gpt import GPTModel


def train_gpt(config: GPTConfig) -> GPTModel:
    mlx.core.random.seed(config.seed)

    model = GPTModel(config)
    model.compile()

    dataset = get_open_web_math_dataset(
        model.tokenizer,
        _format_tokenizer_name(config.model.tokenizer_name),
        config.model.train_max_length,
        config.max_samples,
        config.test_pct,
        tensor_type="mlx",
    )

    train_dataloader = MLXDataLoader(dataset["train"], batch_size=config.batch_size)
    test_dataloader = MLXDataLoader(dataset["test"], batch_size=config.batch_size)

    trainer = MLXTrainer(max_epochs=config.epochs)

    trainer.fit(model, train_dataloader, test_dataloader)

    return model
