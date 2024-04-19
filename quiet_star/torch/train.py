import lightning.pytorch
import torch
import torch.utils.data
from lightning.pytorch.callbacks import RichProgressBar

from quiet_star.config import Config
from quiet_star.dataset import get_open_web_math_dataset
from quiet_star.torch.gpt import GPTModel


def train_gpt(config: Config) -> GPTModel:
    lightning.pytorch.seed_everything(config.seed, workers=True)

    model = GPTModel(config)

    dataset = get_open_web_math_dataset(
        model.tokenizer,
        config.model.max_length,
        2,
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
        accelerator="gpu",
        max_epochs=config.epochs,
        callbacks=[RichProgressBar(leave=True)],
    )

    trainer.fit(model, train_dataloader, test_dataloader)

    return model