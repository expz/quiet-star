import pathlib
import random
from typing import Callable

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfFileSystem

import torch.utils.data


def random_split_on_whitespace(text: str, min_remaining_whitespace: int = 256) -> str:
    """
    Splits a string on a random whitespace character and returns the second part.

    Args:
        text: The string to split.
        min_remaining_whitespace: The minimum number of whitespace characters to leave in string.

    Returns:
        The second part of the string after a random whitespace split,
        or the original string if no whitespace is found.
        Leading whitespace is stripped.
    """
    whitespace_indexes = [i for i, char in enumerate(text) if char.isspace()][
        :-min_remaining_whitespace
    ]

    if not whitespace_indexes:
        return text  # Not enough whitespace found, return original text

    random_index = random.choice(whitespace_indexes)

    return text[random_index + 1 :].lstrip()


def process_batch(
    tokenizer: AutoTokenizer, max_length: int = 256
) -> Callable[[dict[str, list]], dict[str, list]]:
    def _process(examples: dict[str, list]) -> dict[str, list]:
        result = {
            "input_ids": [
                tokenizer(
                    random_split_on_whitespace(text, max_length + 1),
                    max_length=max_length + 1,
                    truncation=True,
                    padding="max_length",
                )["input_ids"]
                for text in examples["text"]
            ],
        }
        result["text"] = tokenizer.batch_decode(result["input_ids"])
        return result

    return _process


def get_open_web_math_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    file_count: int = 1,
    max_samples: int | None = None,
    test_pct: float = 0.125,
) -> torch.utils.data.Dataset:
    fs = HfFileSystem()

    # List all ".parquet" files in the repo
    paths = fs.glob("datasets/open-web-math/open-web-math/**/*.parquet")
    relative_paths = [
        str(pathlib.Path(path).relative_to(*pathlib.Path(path).parts[:3]))
        for path in paths
    ]

    large_dataset = load_dataset(
        "open-web-math/open-web-math",
        data_files={"train": relative_paths[:file_count]},
        num_proc=8,
        verification_mode=datasets.VerificationMode.NO_CHECKS,
    )

    small_dataset = (
        large_dataset["train"]
        .select(range(max_samples))
        .map(
            process_batch(tokenizer, max_length),
            batched=True,
            remove_columns=["url", "date", "metadata"],
        )
    )
    small_dataset = small_dataset.remove_columns("text")
    small_dataset.set_format("pt", columns=["input_ids"], output_all_columns=True)

    split_dataset = small_dataset.train_test_split(test_size=test_pct, shuffle=True)

    split_dataset.save_to_disk("data/open-web-math")

    return split_dataset
