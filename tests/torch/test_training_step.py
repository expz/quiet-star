import lightning
import torch

from quiet_star.config import Config, ModelConfig
from quiet_star.torch.gpt import GPTModel
from quiet_star.torch.pretrained import PretrainedThoughtModel


def run_training_step_test(model: lightning.LightningModule, config: Config) -> None:
    text = "This is a longer test sentence."
    x = model.tokenizer(
        text,
        padding="do_not_pad",
        truncation=True,
        max_length=config.model.max_length - config.thought_length - 2,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"][0].tolist()
    x = torch.tensor(
        [x for _ in range(config.batch_size)],
        dtype=torch.int64,
        device=config.model.device,
    )

    model.forward_pass(x[:, :-1], x[:, 1:])


def test_gpt_training_step() -> None:
    config = Config(
        batch_size=5,
        thought_length=3,
        lookahead_tokens=2,
        model=ModelConfig(
            attn_type="torch",
            dropout_attn=0.0,
            dropout_embed=0.0,
            embed_dim=3 * 8,
            max_length=32,
            num_heads=3,
            num_layers=3,
        ),
    )
    model = GPTModel(config).to(config.model.device)
    run_training_step_test(model, config)


def test_pretrained_training_step() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        thought_length=3,
        model=ModelConfig(
            attn_type="torch",
            device=device,
            dropout_attn=0.0,
            dropout_embed=0.0,
            model_name="Qwen/Qwen1.5-0.5B-Chat",
            max_length=32,
        ),
    )
    model = PretrainedThoughtModel(config).to(config.model.device)
    run_training_step_test(model, config)
