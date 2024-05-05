import os.path
import pathlib
import random
from typing import Callable

import datasets
import numpy as np
import torch.utils.data
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

DATASET_LOCAL_PATH = "data/open-web-math"


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


def to_mlx(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"input_ids": samples["input_ids"].astype("uint32")}


def get_open_web_math_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    file_count: int = 1,
    max_samples: int = 2048,
    test_pct: float = 0.125,
    tensor_type: str = "torch",
    use_local_cache: bool = True,
) -> torch.utils.data.Dataset:
    cache_path = os.path.join(DATASET_LOCAL_PATH, str(max_length), tensor_type)
    if use_local_cache:
        try:
            split_dataset = datasets.load_from_disk(cache_path)
            return split_dataset
        except Exception as e:
            print(f"local dataset loading failed ({e}). downloading dataset.")

    fs = HfFileSystem()

    # List all ".parquet" files in the repo
    paths = fs.glob("datasets/open-web-math/open-web-math/**/*.parquet")
    relative_paths = [
        str(pathlib.Path(path).relative_to(*pathlib.Path(path).parts[:3]))
        for path in paths
    ]

    large_dataset = datasets.load_dataset(
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

    if tensor_type == "mlx":
        small_dataset.set_format(
            "numpy", columns=["input_ids"], output_all_columns=True
        )
        small_dataset = small_dataset.map(to_mlx, batched=True)
    else:
        small_dataset.set_format(
            tensor_type, columns=["input_ids"], output_all_columns=True
        )

    split_dataset = small_dataset.train_test_split(test_size=test_pct, shuffle=True)

    split_dataset.save_to_disk(cache_path)

    return split_dataset
