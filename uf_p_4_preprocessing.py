#!/usr/bin/env python3
"""Build UltraFeedback P_4 data and Pengram-compatible support/query splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data_utils.add_survey_contexts import ScriptArguments as ContextArguments
from data_utils.add_survey_contexts import as_bool, generate_contexts, load_input_dataset
from data_utils.ultrafeedback_augment import inner_join


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UltraFeedback P_4 data for cautious context-steering eval.")
    parser.add_argument("--model_type", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--other_subsets", type=str, default="single", choices=sorted(DEFAULT_SUBSETS))
    parser.add_argument("--dataset_name", type=str, default="P_4")
    parser.add_argument("--survey_size", type=int, default=16)
    parser.add_argument("--history_items", type=int, default=4)
    parser.add_argument("--with_embeddings", type=str, default="False")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--source_root", type=str, default="")
    parser.add_argument("--generated_data_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")

    parser.add_argument("--valid_frac", type=float, default=0.1)
    parser.add_argument("--max_history_items", type=int, default=0)
    parser.add_argument("--drop_empty_history", action="store_true")
    parser.add_argument("--eval_support_ratio", type=float, default=1.0)
    parser.add_argument("--min_eval_support_rows", type=int, default=4)

    parser.add_argument("--skip_augment", action="store_true", help="Do not rebuild data/UltraFeedback_<subset>_<name>.")
    parser.add_argument("--skip_contexts", action="store_true", help="Do not rebuild data/<name>_survey_<n>/<model_type>.")
    parser.add_argument("--skip_splits", action="store_true", help="Do not build train/valid/test/calib_unseen/test_unseen JSONL.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def default_prompt_text(prompt: str) -> str:
    prompt = ensure_text(prompt).strip()
    return f"User request:\n{prompt}\n\nAssistant:\n" if prompt else ""


def extract_prompt_from_transcript(text: Any) -> str:
    raw = ensure_text(text).strip()
    if not raw or "Assistant:" not in raw:
        return ""
    prefix = raw.split("Assistant:", 1)[0]
    return prefix.replace("Human:", "", 1).strip()


def extract_assistant_response(text: Any, prompt: str = "") -> str:
    raw = ensure_text(text).strip()
    if not raw:
        return ""
    if "Assistant:" in raw:
        return raw.rsplit("Assistant:", 1)[-1].strip()
    prompt = ensure_text(prompt).strip()
    if prompt and raw.startswith(prompt):
        remainder = raw[len(prompt) :].strip()
        if remainder.startswith("Assistant:"):
            remainder = remainder[len("Assistant:") :].strip()
        if remainder:
            return remainder
    return raw


def stable_user_id(subset: str) -> str:
    return f"uf_subset_{subset}"


def default_source_root(args: argparse.Namespace) -> Path:
    return Path(args.source_root or f"data/UltraFeedback_{args.other_subsets}_{args.dataset_name}")


def default_generated_data_dir(args: argparse.Namespace) -> Path:
    return Path(args.generated_data_dir or f"data/{args.dataset_name}_survey_{args.survey_size}/{args.model_type}")


def default_out_dir(args: argparse.Namespace) -> Path:
    return Path(args.out_dir or f"data/ultrafeedback_{args.other_subsets}_{args.dataset_name}_history")


def subsets_for(other_subsets: str) -> List[str]:
    return list(DEFAULT_SUBSETS[other_subsets])


def infer_generated_subsets(generated_data_dir: Path, other_subsets: str) -> List[str]:
    if generated_data_dir.exists():
        child_dirs = sorted(path.name for path in generated_data_dir.iterdir() if path.is_dir())
        if child_dirs:
            return child_dirs
    return subsets_for(other_subsets)


def load_survey_lookup(source_root: Path, subset: str, survey_size: int) -> Dict[str, Dict[str, Any]]:
    survey_path = source_root / subset / f"survey_{survey_size}.jsonl"
    if not survey_path.exists():
        return {}
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(survey_path):
        key = str(row.get("Index", row.get("original_id", ""))).strip()
        if key:
            lookup[key] = row
    return lookup


def resolve_context_row(context: Dict[str, Any], survey_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if any(ensure_text(context.get(field, "")).strip() for field in ("prompt", "chosen", "rejected")):
        return context
    original_id = str(context.get("original_id", "")).strip()
    if original_id and original_id in survey_lookup:
        return survey_lookup[original_id]
    return {}


def build_history_pairs(
    contexts: Sequence[Dict[str, Any]],
    survey_lookup: Dict[str, Dict[str, Any]],
    max_history_items: int,
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    use_contexts = list(contexts)
    if max_history_items > 0:
        use_contexts = use_contexts[:max_history_items]
    for context in use_contexts:
        if not isinstance(context, dict):
            continue
        resolved = resolve_context_row(context, survey_lookup)
        if not resolved:
            continue
        prompt = ensure_text(resolved.get("prompt", "")).strip()
        if not prompt:
            prompt = extract_prompt_from_transcript(resolved.get("chosen", ""))
        chosen = extract_assistant_response(resolved.get("chosen", ""), prompt)
        rejected = extract_assistant_response(resolved.get("rejected", ""), prompt)
        if not (prompt or chosen):
            continue
        pair = {"prompt": prompt, "chosen": chosen}
        if rejected:
            pair["rejected"] = rejected
        pairs.append(pair)
    return pairs


def convert_row(
    row: Dict[str, Any],
    subset: str,
    split_name: str,
    source_index: int,
    survey_lookup: Dict[str, Dict[str, Any]],
    other_subsets: str,
    max_history_items: int,
) -> Dict[str, Any]:
    prompt = ensure_text(row.get("prompt", "")).strip()
    if not prompt:
        prompt = extract_prompt_from_transcript(row.get("chosen", ""))
    chosen = extract_assistant_response(row.get("chosen", ""), prompt)
    rejected = extract_assistant_response(row.get("rejected", ""), prompt)
    history_pairs = build_history_pairs(
        contexts=list(row.get("contexts", []) or []),
        survey_lookup=survey_lookup,
        max_history_items=max_history_items,
    )
    history_chars = sum(len(pair.get("prompt", "")) + len(pair.get("chosen", "")) for pair in history_pairs)
    return {
        "user_id": stable_user_id(subset),
        "prompt_text": default_prompt_text(prompt),
        "chosen": chosen,
        "rejected": rejected,
        "user_history_pairs": history_pairs,
        "meta": {
            "task": "ultrafeedback",
            "setting": other_subsets,
            "source_dataset": "UltraFeedback",
            "user_subset": subset,
            "user_id": stable_user_id(subset),
            "source_split": split_name,
            "output_split": split_name,
            "history_len": len(history_pairs),
            "history_chars": history_chars,
            "context_length": int(row.get("context_length", len(history_pairs)) or len(history_pairs)),
            "row_index": int(row.get("Index", row.get("original_idx", -1)) or -1),
            "source_index": int(source_index),
            "turn": int(source_index),
        },
    }


def filter_rows(rows: Sequence[Dict[str, Any]], drop_empty_history: bool) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if not ensure_text(row.get("prompt_text", "")).strip():
            continue
        if not ensure_text(row.get("chosen", "")).strip():
            continue
        if drop_empty_history and not list(row.get("user_history_pairs", []) or []):
            continue
        filtered.append(row)
    return filtered


def train_valid_split(rows: Sequence[Dict[str, Any]], valid_frac: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows_list = list(rows)
    if not rows_list or valid_frac <= 0.0 or len(rows_list) < 2:
        return rows_list, []
    rng = random.Random(seed)
    rng.shuffle(rows_list)
    n_valid = max(1, int(round(len(rows_list) * valid_frac)))
    n_valid = min(n_valid, len(rows_list) - 1)
    return rows_list[n_valid:], rows_list[:n_valid]


def summarize_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    history_lens = [int((row.get("meta", {}) or {}).get("history_len", 0)) for row in rows]
    return {
        "n_rows": len(rows),
        "mean_history_len": float(sum(history_lens) / len(history_lens)) if history_lens else 0.0,
    }


def relabel_output_split(row: Dict[str, Any], output_split: str, clear_history: bool = False) -> Dict[str, Any]:
    out = dict(row)
    meta = dict(out.get("meta", {}) or {})
    meta["output_split"] = output_split
    if clear_history:
        out["user_history_pairs"] = []
        out["user_history_text"] = ""
        meta["history_len"] = 0
        meta["history_chars"] = 0
    out["meta"] = meta
    return out


def build_support_query_splits(
    train_rows_by_subset: Dict[str, Sequence[Dict[str, Any]]],
    test_rows_by_subset: Dict[str, Sequence[Dict[str, Any]]],
    support_ratio: float,
    min_support_rows: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    support_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []
    per_subset: Dict[str, Any] = {}
    rng = random.Random(seed)

    for subset in sorted(test_rows_by_subset, key=lambda value: int(value) if value.isdigit() else value):
        train_rows = list(train_rows_by_subset.get(subset, []))
        test_rows = list(test_rows_by_subset.get(subset, []))
        if not train_rows or not test_rows:
            per_subset[subset] = {
                "support_rows": 0,
                "query_rows": len(test_rows),
                "available_train_rows": len(train_rows),
            }
            continue

        requested_support = max(int(min_support_rows), int(round(len(test_rows) * float(support_ratio))))
        requested_support = min(max(1, requested_support), len(train_rows))
        chosen_support = list(train_rows)
        rng.shuffle(chosen_support)
        chosen_support = sorted(
            chosen_support[:requested_support],
            key=lambda row: int((row.get("meta", {}) or {}).get("source_index", 0) or 0),
        )

        support_rows.extend(relabel_output_split(row, "calib_unseen", clear_history=True) for row in chosen_support)
        query_rows.extend(relabel_output_split(row, "test_unseen", clear_history=True) for row in test_rows)
        per_subset[subset] = {
            "support_rows": len(chosen_support),
            "query_rows": len(test_rows),
            "available_train_rows": len(train_rows),
            "support_to_query_ratio": float(len(chosen_support) / max(1, len(test_rows))),
        }

    return support_rows, query_rows, per_subset


def build_augmented_dataset(args: argparse.Namespace, source_root: Path) -> None:
    from datasets import load_dataset

    user_types = {subset: USER_TYPES[subset] for subset in subsets_for(args.other_subsets)}
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))

    ultra_feedback = load_dataset("openbmb/UltraFeedback")
    binarized_cleaned = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
    length = len(binarized_cleaned["train"])
    test_ids = list(np.random.choice(length, int(length * 0.1), replace=False))
    train_split = binarized_cleaned["train"].filter(lambda example, idx: idx not in test_ids, with_indices=True)
    test_split = binarized_cleaned["train"].filter(lambda example, idx: idx in test_ids, with_indices=True)

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


def build_survey_contexts(args: argparse.Namespace, source_root: Path) -> None:
    with_embeddings = as_bool(args.with_embeddings)
    for subset in subsets_for(args.other_subsets):
        for split in ("train", "test"):
            context_args = ContextArguments(
                output_dir=f"data/{args.dataset_name}_survey_{args.survey_size}/",
                data_path=str(source_root),
                data_subset=subset,
                data_split=split,
                model_type=args.model_type,
                with_embeddings=with_embeddings,
                other_subsets=args.other_subsets,
                survey_size=int(args.survey_size),
                context_length=int(args.history_items),
                num_duplicates=1,
                fixed_context_length=True,
            )
            print(context_args)
            dataset = load_input_dataset(context_args)
            survey_path = source_root / subset / f"survey_{args.survey_size}.jsonl"
            if split == "train":
                survey_options = dataset.filter(lambda row: as_bool(row.get("survey_options")))
                survey_ids = np.random.choice(range(len(survey_options)), int(args.survey_size), replace=False)
                print(survey_ids)
                survey_data = survey_options.filter(lambda example, idx: idx in survey_ids, with_indices=True)
                survey_data.to_json(str(survey_path))
            else:
                from datasets import load_dataset

                survey_data = load_dataset("json", data_files=str(survey_path), split="train")
            generate_contexts(context_args, dataset, survey_data)


def build_pengram_splits(args: argparse.Namespace, generated_data_dir: Path, source_root: Path, out_dir: Path) -> None:
    subsets = infer_generated_subsets(generated_data_dir, args.other_subsets)
    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    train_rows_by_subset: Dict[str, List[Dict[str, Any]]] = {}
    test_rows_by_subset: Dict[str, List[Dict[str, Any]]] = {}
    per_subset_summary: Dict[str, Any] = {}

    for subset_idx, subset in enumerate(subsets):
        subset_dir = generated_data_dir / subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"Could not find generated subset directory: {subset_dir}")

        survey_lookup = load_survey_lookup(source_root, subset, int(args.survey_size))
        converted_by_split: Dict[str, List[Dict[str, Any]]] = {}
        for split_name in ["train", "test"]:
            split_path = subset_dir / f"{split_name}.jsonl"
            if not split_path.exists():
                raise FileNotFoundError(f"Could not find generated split: {split_path}")
            converted = [
                convert_row(
                    row=row,
                    subset=subset,
                    split_name=split_name,
                    source_index=source_index,
                    survey_lookup=survey_lookup,
                    other_subsets=args.other_subsets,
                    max_history_items=int(args.max_history_items),
                )
                for source_index, row in enumerate(read_jsonl(split_path))
            ]
            converted_by_split[split_name] = filter_rows(converted, bool(args.drop_empty_history))

        subset_train, subset_valid = train_valid_split(
            converted_by_split["train"],
            valid_frac=float(args.valid_frac),
            seed=int(args.seed) + subset_idx,
        )
        train_rows.extend(subset_train)
        valid_rows.extend(subset_valid)
        test_rows.extend(converted_by_split["test"])
        train_rows_by_subset[subset] = converted_by_split["train"]
        test_rows_by_subset[subset] = converted_by_split["test"]
        per_subset_summary[subset] = {
            "train_source_rows": len(converted_by_split["train"]),
            "train_rows": len(subset_train),
            "valid_rows": len(subset_valid),
            "test_rows": len(converted_by_split["test"]),
            "mean_history_len_train": summarize_split(subset_train)["mean_history_len"],
            "mean_history_len_test": summarize_split(converted_by_split["test"])["mean_history_len"],
        }

    calib_unseen, test_unseen, eval_per_subset = build_support_query_splits(
        train_rows_by_subset=train_rows_by_subset,
        test_rows_by_subset=test_rows_by_subset,
        support_ratio=float(args.eval_support_ratio),
        min_support_rows=int(args.min_eval_support_rows),
        seed=int(args.seed),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)
    write_jsonl(out_dir / "calib_unseen.jsonl", calib_unseen)
    write_jsonl(out_dir / "test_unseen.jsonl", test_unseen)

    summary = {
        "generated_data_dir": str(generated_data_dir.resolve()),
        "source_root": str(source_root.resolve()),
        "other_subsets": args.other_subsets,
        "subsets": subsets,
        "survey_size": int(args.survey_size),
        "valid_frac": float(args.valid_frac),
        "max_history_items": int(args.max_history_items),
        "drop_empty_history": bool(args.drop_empty_history),
        "eval_support_ratio": float(args.eval_support_ratio),
        "min_eval_support_rows": int(args.min_eval_support_rows),
        "splits": {
            "train": summarize_split(train_rows),
            "valid": summarize_split(valid_rows),
            "test": summarize_split(test_rows),
            "calib_unseen": summarize_split(calib_unseen),
            "test_unseen": summarize_split(test_unseen),
        },
        "per_subset": per_subset_summary,
        "eval_per_subset": eval_per_subset,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    source_root = default_source_root(args)
    generated_data_dir = default_generated_data_dir(args)
    out_dir = default_out_dir(args)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))

    if not args.skip_augment:
        build_augmented_dataset(args, source_root)
    if not args.skip_contexts:
        build_survey_contexts(args, source_root)
    if not args.skip_splits:
        build_pengram_splits(args, generated_data_dir, source_root, out_dir)


if __name__ == "__main__":
    main()
