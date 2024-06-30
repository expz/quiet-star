import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from quiet_star.config import Config, ModelConfig
from quiet_star.torch.openelm import OpenELMThoughtModel
from quiet_star.torch.pretrained import PretrainedThoughtModel
from quiet_star.torch.qwen import QwenThoughtModel
from quiet_star.torch.qwen_explicit import QwenExplicitThoughtModel
from quiet_star.torch.utils import torch_dtype


def standard_setup(
    config: Config,
) -> tuple[PreTrainedModel, AutoTokenizer, torch.Tensor]:
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
    return model, tokenizer, x


def run_test_of_forward(thinking_model: PretrainedThoughtModel, config: Config) -> None:
    model, tokenizer, x = standard_setup(config)

    # WARNING: The vocab size of the model can be larger than that of the
    #          the tokenizer if it supports dummy tokens.
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


def run_test_of_forward_with_cache(
    thinking_model: PretrainedThoughtModel, config: Config
) -> None:
    """
    Generate a token saving the key-value cache. Then run the forward pass
    again with the key-value cahce and compare to a forward pass without it.
    """
    model, tokenizer, x = standard_setup(config)

    original_vocab_length = len(tokenizer)
    output = model(x)
    y_correct = output.logits[:, :, :original_vocab_length]

    print("x:", x.shape)
    y_actual1, _, key_value_cache = thinking_model.forward(
        x[:, :-2], return_key_value_cache=True
    )  # type: ignore
    y_actual2, _, kev_value_cache = thinking_model.forward(
        x[:, -2:-1], key_value_cache=key_value_cache, return_key_value_cache=True
    )  # type: ignore
    y_actual3, _, _ = thinking_model.forward(x[:, -1:], key_value_cache=kev_value_cache)

    y_actual = torch.cat([y_actual1, y_actual2, y_actual3], dim=1)
    y_actual = y_actual[:, :, :original_vocab_length]

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
            model_name="Qwen/Qwen2-0.5B-Instruct",
            tokenizer_name="Qwen/Qwen2-0.5B-Instruct",
            max_length=32,
        ),
    )
    thinking_model = QwenExplicitThoughtModel(config).to(config.model.device)

    run_test_of_forward(thinking_model, config)


def test_openelm_forward() -> None:
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
            model_name="apple/OpenELM-270M-Instruct",
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            max_length=32,
        ),
    )
    thinking_model = OpenELMThoughtModel(config).to(config.model.device)

    run_test_of_forward(thinking_model, config)


def test_qwen_forward_with_key_value_cache() -> None:
    """
    Test QwenThoughtModel with key-value cache.
    """
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
            model_name="Qwen/Qwen2-0.5B-Instruct",
            tokenizer_name="Qwen/Qwen2-0.5B-Instruct",
            max_length=32,
        ),
    )
    thinking_model = QwenThoughtModel(config).to(config.model.device)

    run_test_of_forward_with_cache(thinking_model, config)


def test_openelm_forward_with_key_value_cache() -> None:
    """
    Test OpenELMThoughtModel with key-value cache.
    """
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
            model_name="apple/OpenELM-270M-Instruct",
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            max_length=32,
        ),
    )
    thinking_model = OpenELMThoughtModel(config).to(config.model.device)

    run_test_of_forward_with_cache(thinking_model, config)
