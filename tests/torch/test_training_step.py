import lightning
import torch

from quiet_star.config import Config, GPTConfig, GPTModelConfig, ModelConfig
from quiet_star.torch.gpt import GPTModel
from quiet_star.torch.openelm import OpenELMThoughtModel
from quiet_star.torch.qwen import QwenThoughtModel


def run_training_step_test(model: lightning.LightningModule, config: Config) -> None:
    text = "This is a longer test sentence."
    x = model.tokenizer(
        text,
        padding="do_not_pad",
        truncation=True,
        max_length=config.model.max_length,
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
    config = GPTConfig(
        batch_size=5,
        thought_length=3,
        lookahead_tokens=2,
        model=GPTModelConfig(
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


def test_qwen_training_step() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=1,
        thought_length=3,
        lookahead_tokens=4,
        model=ModelConfig(
            device=device,
            model_name="Qwen/Qwen2-0.5B-Instruct",
            tokenizer_name="Qwen/Qwen2-0.5B-Instruct",
            max_length=32,
        ),
    )
    model = QwenThoughtModel(config).to(config.model.device)
    run_training_step_test(model, config)


def test_openelm_training_step() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=1,
        thought_length=3,
        lookahead_tokens=4,
        model=ModelConfig(
            device=device,
            model_name="apple/OpenELM-270M-Instruct",
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            max_length=32,
        ),
    )
    model = OpenELMThoughtModel(config).to(config.model.device)
    run_training_step_test(model, config)
