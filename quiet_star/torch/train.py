import dataclasses
import datetime

import lightning.pytorch
import torch
import torch.utils.data
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar

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
    print("************* Settings *************")
    print(yaml.dump(dataclasses.asdict(config), indent=4, sort_keys=True))
    print("************************************")

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
    hourly_checkpoint_callback = ModelCheckpoint(
        train_time_interval=datetime.timedelta(hours=config.save_interval_hours),
        save_top_k=config.save_top_k,
    )
    epoch_checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=-1,
    )

    trainer = lightning.pytorch.Trainer(
        deterministic=False,
        accelerator="cpu" if config.model.device == "cpu" else "gpu",
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_batches,
        callbacks=[
            hourly_checkpoint_callback,
            epoch_checkpoint_callback,
            RichProgressBar(leave=True),
        ],
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
    model.metric_logger.init()

    return _train(config, model)


def train_qwen_explicit(config: Config) -> QwenExplicitThoughtModel:
    lightning.pytorch.seed_everything(config.seed, workers=True)

    model = QwenExplicitThoughtModel(config)
    model.metric_logger.init()

    return _train(config, model)


def train_openelm(config: Config) -> OpenELMThoughtModel:
    lightning.pytorch.seed_everything(config.seed, workers=True)

    model = OpenELMThoughtModel(config)
    model.metric_logger.init()

    return _train(config, model)
