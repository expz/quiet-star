import readline  # used by input to enable arrow keys
import warnings

from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from quiet_star.config import CompletionConfig
from quiet_star.torch.utils import torch_dtype

warnings.filterwarnings("ignore")


def parse_args() -> CompletionConfig:
    parser = ArgumentParser()
    parser.add_arguments(CompletionConfig, dest="config")
    args = parser.parse_args()
    return args.config


def init_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
    ]


def main(config: CompletionConfig) -> None:
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype(config.dtype),
        trust_remote_code=True,
    ).to(config.device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        trust_remote_code=True,
    )

    # Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the  value of the house by 150%.  Is the following statement correct: "The profit is the value of the house after the repairs minus the cost of the house"? Why or why not?
    messages = init_messages()
    while True:
        user_input = input("> ")

        normalized_user_input = user_input.lower().strip()
        if normalized_user_input == ":quit":
            break
        if normalized_user_input == ":reset":
            messages = init_messages()
            continue

        messages.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], padding=True, return_tensors="pt").to(
            config.device
        )

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            attention_mask=model_inputs.attention_mask,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    config = parse_args()
    main(config)
