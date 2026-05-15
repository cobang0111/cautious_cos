#!/usr/bin/env python3
"""Build UltraFeedback P_4 survey-context data.

Outputs mirror the layout consumed by ``uf_p_4_preprocessing.py``:

- data/UltraFeedback_<other_subsets>_<dataset_name>/<subset>/{train,test}.jsonl
- data/<dataset_name>_survey_<survey_size>/<subset>/{train,test}.jsonl
- data/UltraFeedback_<other_subsets>_<dataset_name>/<subset>/survey_<survey_size>.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_SUBSETS = {
    "single": ["8", "4", "2", "1"],
    "84": ["8", "4"],
}
USER_TYPES = {
    "8": (1, 0, 0, 0),
    "4": (0, 1, 0, 0),
    "2": (0, 0, 1, 0),
    "1": (0, 0, 0, 1),
}
RATING_KEYS = ["helpfulness", "honesty", "instruction_following", "truthfulness"]


@dataclass
class ContextConfig:
    output_dir: str
    data_path: str
    data_subset: str
    data_split: str
    other_subsets: str
    survey_size: int
    context_length: int
    num_duplicates: int = 1
    fixed_context_length: bool = True
    controversial_only: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/build UltraFeedback P_4 survey-context data.")
    parser.add_argument("--other_subsets", type=str, default="single", choices=sorted(DEFAULT_SUBSETS))
    parser.add_argument("--dataset_name", type=str, default="P_4")
    parser.add_argument("--survey_size", type=int, default=16)
    parser.add_argument("--history_items", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_root", type=str, default="")
    parser.add_argument("--generated_data_dir", type=str, default="")
    parser.add_argument("--skip_augment", action="store_true", help="Do not rebuild raw UltraFeedback subset JSONL.")
    parser.add_argument("--skip_contexts", action="store_true", help="Do not rebuild survey-context JSONL.")
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def default_source_root(args: argparse.Namespace) -> Path:
    return Path(args.source_root or f"data/UltraFeedback_{args.other_subsets}_{args.dataset_name}")


def default_generated_data_dir(args: argparse.Namespace) -> Path:
    return Path(args.generated_data_dir or f"data/{args.dataset_name}_survey_{args.survey_size}")


def subsets_for(other_subsets: str) -> List[str]:
    return list(DEFAULT_SUBSETS[other_subsets])


def random_greater_than_zero(values: np.ndarray) -> np.ndarray:
    return (np.random.randn(values.shape[0]) * (values == 0) > 0.0) | (values > 0.0)


def get_user_type(chosen_ratings: Dict[str, int], rejected_ratings: Dict[str, int], augment_type: str) -> Tuple[List[str], Dict[str, bool], Dict[str, bool]]:
    chosen_values = np.asarray([chosen_ratings[key] for key in RATING_KEYS])
    rejected_values = np.asarray([rejected_ratings[key] for key in RATING_KEYS])
    is_equal_arr = chosen_values == rejected_values
    if augment_type not in {"single", "84"}:
        raise ValueError(f"Invalid augment_type: {augment_type}")
    data_subsets = ["8", "4", "2", "1"]
    reversed_arr = random_greater_than_zero(rejected_values - chosen_values)
    reversed_labels = {subset: bool(reversed_arr[idx]) for idx, subset in enumerate(data_subsets)}
    is_equal = {subset: bool(is_equal_arr[idx]) for idx, subset in enumerate(data_subsets)}
    return data_subsets, reversed_labels, is_equal


def inner_join(original: Any, binarized: Any, augment_type: str, users: Dict[str, Tuple[int, int, int, int]]) -> Any:
    from datasets import Dataset

    agreed_counter = 0
    controversial_counter = 0
    user_counter = {key: 0 for key in users}
    reversed_counter = {key: 0 for key in users}
    out_idx = 0
    orig_idx = 0
    dataset_dict: Dict[str, List[Any]] = {
        "Index": [],
        "original_idx": [],
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "data_subset": [],
        "controversial": [],
        "reversed": [],
        "satisfied_subset": [],
        "survey_options": [],
    }

    for bin_idx in range(len(binarized)):
        prompt = binarized[bin_idx]["prompt"]
        while orig_idx < len(original) and prompt != original[orig_idx]["instruction"]:
            orig_idx += 1
        if orig_idx >= len(original):
            continue

        chosen = binarized[bin_idx]["chosen"][1]["content"]
        rejected = binarized[bin_idx]["rejected"][1]["content"]
        if not chosen or not rejected:
            continue

        chosen_ratings: Dict[str, int] = {}
        rejected_ratings: Dict[str, int] = {}
        complete = True
        for completion in original[orig_idx]["completions"]:
            response = completion.get("response", "")
            if response not in {chosen, rejected}:
                continue
            target = chosen_ratings if response == chosen else rejected_ratings
            for key in RATING_KEYS:
                rating = completion["annotations"][key]["Rating"]
                if rating == "N/A":
                    complete = False
                    continue
                target[key] = int(rating)
        if not complete or len(chosen_ratings) != len(RATING_KEYS) or len(rejected_ratings) != len(RATING_KEYS):
            continue

        data_subsets, reversed_labels, _is_equal = get_user_type(chosen_ratings, rejected_ratings, augment_type)
        is_controversial = True in reversed_labels.values() and False in reversed_labels.values()
        if is_controversial:
            controversial_counter += 1
        else:
            agreed_counter += 1

        for data_subset in users:
            if data_subset not in data_subsets:
                continue
            user_counter[data_subset] += 1
            reversed_label = reversed_labels[data_subset]
            if reversed_label:
                reversed_counter[data_subset] += 1

            dataset_dict["Index"].append(out_idx)
            dataset_dict["original_idx"].append(orig_idx)
            dataset_dict["prompt"].append(prompt)
            if not reversed_label:
                dataset_dict["chosen"].append("Human: " + prompt + "\n\nAssistant: " + chosen)
                dataset_dict["rejected"].append("Human: " + prompt + "\n\nAssistant: " + rejected)
            else:
                dataset_dict["chosen"].append("Human: " + prompt + "\n\nAssistant: " + rejected)
                dataset_dict["rejected"].append("Human: " + prompt + "\n\nAssistant: " + chosen)
            dataset_dict["data_subset"].append(data_subset)
            dataset_dict["controversial"].append(is_controversial)
            dataset_dict["reversed"].append(reversed_label)
            satisfied_subset = [
                key for key in users if key not in data_subsets or reversed_labels[key] == reversed_label
            ]
            dataset_dict["satisfied_subset"].append(satisfied_subset)
            dataset_dict["survey_options"].append(is_controversial and len(data_subsets) == len(users))
            out_idx += 1

    print(out_idx, agreed_counter, controversial_counter)
    print("User counter:", user_counter)
    print("Reversed counter:", reversed_counter)
    return Dataset.from_dict(dataset_dict)


def load_input_dataset(args: ContextConfig) -> Any:
    from datasets import load_dataset

    data_file = Path(args.data_path) / args.data_subset / f"{args.data_split}.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Could not find dataset split: {data_file}")
    return load_dataset("json", data_files=str(data_file), split="train")


def generate_contexts(args: ContextConfig, input_dataset: Any, survey_dataset: Any) -> Any:
    from datasets import concatenate_datasets
    from tqdm import tqdm

    output_dir = Path(args.output_dir) / args.data_subset
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.controversial_only:
        input_dataset = input_dataset.filter(lambda row: as_bool(row.get("controversial")))

    dataset_size = len(input_dataset)
    num_copies = args.num_duplicates if args.data_split == "train" else 1
    dataset_list = []

    def random_choice(max_context_length: int, survey_size: int) -> Tuple[List[Dict[str, Any]], int]:
        if max_context_length > survey_size:
            raise ValueError("Context length is larger than survey size.")
        if len(survey_dataset) < survey_size:
            raise ValueError(f"Need {survey_size} survey rows, found {len(survey_dataset)}.")

        while True:
            if args.fixed_context_length:
                context_length = max_context_length
            elif args.other_subsets == "84":
                context_length = random.randint(1, max_context_length)
            else:
                context_length = random.randint(2, max_context_length)

            chosen_ids = set(np.random.choice(survey_size, context_length, replace=False).tolist())
            chosen_dataset = [row for idx, row in enumerate(survey_dataset) if idx in chosen_ids]
            if args.other_subsets != "single":
                return chosen_dataset, context_length

            satisfied_sets = [set(row["satisfied_subset"]) for row in chosen_dataset]
            shared = set.intersection(*satisfied_sets)
            if len(shared) == 1:
                return chosen_dataset, context_length
            if context_length == survey_size:
                raise ValueError("Please choose another random seed.")

    for _ in range(num_copies):
        output_dataset = deepcopy(input_dataset)
        context_lengths = []
        contexts = []
        for _row_id in tqdm(range(dataset_size)):
            row_contexts = []
            context_dataset, context_length = random_choice(args.context_length, args.survey_size)
            context_lengths.append(context_length)
            for context_row in context_dataset:
                row_contexts.append(
                    {
                        "original_id": context_row["Index"],
                        "prompt": context_row.get("prompt", ""),
                        "chosen": context_row.get("chosen", ""),
                        "rejected": context_row.get("rejected", ""),
                    }
                )
            contexts.append(row_contexts)
        output_dataset = output_dataset.add_column("context_length", context_lengths)
        output_dataset = output_dataset.add_column("contexts", contexts)
        dataset_list.append(output_dataset)

    output = concatenate_datasets(dataset_list)
    output.to_json(str(output_dir / f"{args.data_split}.jsonl"))
    return output


def build_augmented_dataset(args: argparse.Namespace, source_root: Path) -> None:
    from datasets import load_dataset

    user_types = {subset: USER_TYPES[subset] for subset in subsets_for(args.other_subsets)}
    ultra_feedback = load_dataset("openbmb/UltraFeedback")
    binarized_cleaned = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
    length = len(binarized_cleaned["train"])
    test_ids = set(np.random.choice(length, int(length * 0.1), replace=False).tolist())
    train_split = binarized_cleaned["train"].filter(lambda _example, idx: idx not in test_ids, with_indices=True)
    test_split = binarized_cleaned["train"].filter(lambda _example, idx: idx in test_ids, with_indices=True)

    print("start processing train split")
    joined_train = inner_join(ultra_feedback["train"], train_split, args.other_subsets, user_types)
    print("start processing test split")
    joined_test = inner_join(ultra_feedback["train"], test_split, args.other_subsets, user_types)

    for subset in user_types:
        train_subset = joined_train.filter(lambda row, subset=subset: row["data_subset"] == subset)
        test_subset = joined_test.filter(lambda row, subset=subset: row["data_subset"] == subset)
        train_subset = train_subset.filter(lambda row: row["controversial"] is True)
        test_subset = test_subset.filter(lambda row: row["controversial"] is True)
        subset_dir = source_root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        print(user_types[subset], len(train_subset), len(test_subset))
        train_subset.to_json(str(subset_dir / "train.jsonl"))
        test_subset.to_json(str(subset_dir / "test.jsonl"))


def build_survey_contexts(args: argparse.Namespace, source_root: Path, generated_data_dir: Path) -> None:
    from datasets import load_dataset

    for subset in subsets_for(args.other_subsets):
        for split in ("train", "test"):
            context_args = ContextConfig(
                output_dir=str(generated_data_dir),
                data_path=str(source_root),
                data_subset=subset,
                data_split=split,
                other_subsets=args.other_subsets,
                survey_size=int(args.survey_size),
                context_length=int(args.history_items),
            )
            print(context_args)
            dataset = load_input_dataset(context_args)
            survey_path = source_root / subset / f"survey_{args.survey_size}.jsonl"
            if split == "train":
                survey_options = dataset.filter(lambda row: as_bool(row.get("survey_options")))
                if len(survey_options) < int(args.survey_size):
                    raise ValueError(f"{subset} has only {len(survey_options)} survey options; need {args.survey_size}.")
                survey_ids = np.random.choice(range(len(survey_options)), int(args.survey_size), replace=False)
                print(survey_ids)
                survey_data = survey_options.filter(lambda _example, idx: idx in survey_ids, with_indices=True)
                survey_data.to_json(str(survey_path))
            else:
                survey_data = load_dataset("json", data_files=str(survey_path), split="train")
            generate_contexts(context_args, dataset, survey_data)


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    source_root = default_source_root(args)
    generated_data_dir = default_generated_data_dir(args)

    if not args.skip_augment:
        build_augmented_dataset(args, source_root)
    if not args.skip_contexts:
        build_survey_contexts(args, source_root, generated_data_dir)

    summary = {
        "source_root": str(source_root.resolve()),
        "generated_data_dir": str(generated_data_dir.resolve()),
        "other_subsets": args.other_subsets,
        "dataset_name": args.dataset_name,
        "survey_size": int(args.survey_size),
        "history_items": int(args.history_items),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
