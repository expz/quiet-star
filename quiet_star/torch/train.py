import lightning.pytorch
import torch
import torch.utils.data
from lightning.pytorch.callbacks import RichProgressBar

from quiet_star.config import Config, GPTConfig
from quiet_star.dataset import _format_tokenizer_name, get_open_web_math_dataset
from quiet_star.torch.gpt import GPTModel
from quiet_star.torch.openelm import OpenELMThoughtModel
from quiet_star.torch.qwen import QwenThoughtModel
from quiet_star.torch.qwen_explicit import QwenExplicitThoughtModel

# Properly utilize tensor cores
torch.set_float32_matmul_precision("medium")


def _train(
    config: Config, model: lightning.LightningModule
) -> lightning.LightningModule:
    dataset = get_open_web_math_dataset(
        model.tokenizer,
        _format_tokenizer_name(config.model.tokenizer_name),
        config.model.train_max_length,
        config.max_samples,
        config.test_pct,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=config.batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=config.batch_size
    )

    trainer = lightning.pytorch.Trainer(
        deterministic=True,
        accelerator="cpu" if config.model.device == "cpu" else "gpu",
        max_epochs=config.epochs,
        callbacks=[RichProgressBar(leave=True)],
    )

    trainer.fit(model, train_dataloader, test_dataloader)

    return model


def train_gpt(config: GPTConfig) -> GPTModel:
    lightning.pytorch.seed_everything(config.seed, workers=True)

    model = GPTModel(config).to(config.model.device)

    return _train(config, model)


def train_qwen(config: Config) -> QwenThoughtModel:
    lightning.pytorch.seed_everything(config.seed, workers=True)

    model = QwenThoughtModel(config)

    return _train(config, model)


def train_qwen_explicit(config: Config) -> QwenExplicitThoughtModel:
    lightning.pytorch.seed_everything(config.seed, workers=True)

    model = QwenExplicitThoughtModel(config)

    return _train(config, model)


def train_openelm(config: Config) -> OpenELMThoughtModel:
    lightning.pytorch.seed_everything(config.seed, workers=True)

    model = OpenELMThoughtModel(config)

    return _train(config, model)
