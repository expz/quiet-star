import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from quiet_star.config import Config, ModelConfig
from quiet_star.torch.pretrained import PretrainedThoughtModel
from quiet_star.torch.qwen_explicit import QwenExplicitThoughtModel
from quiet_star.torch.utils import torch_dtype


def run_test_of_forward(thinking_model: PretrainedThoughtModel, config: Config) -> None:
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=torch_dtype(config.model.dtype),
        trust_remote_code=True,
    ).to(config.model.device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name,
        trust_remote_code=True,
    )

    text = "This is a longer test sentence."
    x = tokenizer(
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

    # WARNING: The vocab size of the model is larger than that of the
    #          the tokenizer because it support dummy tokens.
    #          Also, recall that thinking_model expands the vocabulary by two tokens
    #          The output will be different for those tokens.
    original_vocab_length = len(tokenizer)
    output = model(x)
    y_correct = output.logits[:, :, :original_vocab_length]

    thinking_output = thinking_model.forward_for_testing(x)
    assert isinstance(thinking_output, torch.Tensor)  # appease mypy
    y_actual = thinking_output[:, :, :original_vocab_length]

    assert (
        y_correct.shape == y_actual.shape
    ), f"model output did not have the expected shape! expected shape: {y_correct.shape}, actual shape: {y_actual.shape}"

    assert torch.allclose(
        y_correct, y_actual, atol=1e-4
    ), f"logits did not match!\nexpected logits: {y_correct},\nactual logits: {y_actual}"


def test_qwen_explicit_forward() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        batch_size=2,
        lookahead_tokens=3,
        thought_length=3,
        model=ModelConfig(
            attn_type="torch",
            dtype="float32",
            device=device,
            dropout_attn=0.0,
            dropout_embed=0.0,
            model_name="Qwen/Qwen1.5-0.5B",
            tokenizer_name="Qwen/Qwen1.5-0.5B",
            max_length=32,
        ),
    )
    thinking_model = QwenExplicitThoughtModel(config).to(config.model.device)

    run_test_of_forward(thinking_model, config)
