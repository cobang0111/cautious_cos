#!/usr/bin/env python3
"""
Build PSOUPS -> PRISM/context-steering-compatible JSONL splits.

The RiverDong/psoups dataset exposes rows as:
{
  "prompt": str,
  "chosen": str,
  "rejected": str,
  "uid": int,
  ...
}

This script preserves each uid as a user, uses source row order as the
within-user history order, and emits the same JSONL shape consumed by
train_prism_cautious_context_steering_distill.py and
eval_cautious_context_steering_distill.py.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def normalize_ws(value: Any) -> str:
    return " ".join(str(value or "").split())


def clip_text(value: Any, max_chars: int) -> str:
    text = normalize_ws(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 3)] + "..."


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hf_rows(dataset_name: str, config_name: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: datasets. Install the project requirements or run "
            "`pip install datasets` in the environment used for preprocessing."
        ) from exc

    ds = load_dataset(dataset_name, config_name, split=split)
    return [dict(row) for row in ds]


def group_by_uid(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        user_id = str(row.get("uid", row.get("user_id", "unknown")))
        enriched = dict(row)
        enriched["_source_index"] = idx
        grouped[user_id].append(enriched)
    return grouped


def history_pairs_from_rows(
    rows: Sequence[Dict[str, Any]],
    max_items: int,
    max_chars: int,
) -> List[Dict[str, str]]:
    if max_items <= 0:
        selected = list(rows)
    else:
        selected = list(rows)[-max_items:]
    pairs: List[Dict[str, str]] = []
    for row in selected:
        prompt = clip_text(row.get("prompt", ""), max_chars)
        chosen = clip_text(row.get("chosen", ""), max_chars)
        rejected = clip_text(row.get("rejected", ""), max_chars)
        if not chosen:
            continue
        pair = {"prompt": prompt, "chosen": chosen}
        if rejected:
            pair["rejected"] = rejected
        pairs.append(pair)
    return pairs


def history_text_from_pairs(pairs: Sequence[Dict[str, str]], include_prompt: bool) -> str:
    if not pairs:
        return "No prior interaction history available."
    lines: List[str] = []
    for idx, pair in enumerate(pairs, start=1):
        prompt = normalize_ws(pair.get("prompt", ""))
        chosen = normalize_ws(pair.get("chosen", ""))
        rejected = normalize_ws(pair.get("rejected", ""))
        if include_prompt and prompt:
            lines.append(f"Support example {idx}: User request: {prompt}")
        if chosen:
            lines.append(f"Preferred assistant response: {chosen}")
        if rejected:
            lines.append(f"Less preferred assistant response: {rejected}")
    return "\n".join(lines)


def to_preference_record(
    row: Dict[str, Any],
    *,
    config_name: str,
    source_split: str,
    output_split: str,
    prior_rows: Sequence[Dict[str, Any]],
    max_history_pairs: int,
    history_max_chars: int,
    history_include_prompt: bool,
) -> Dict[str, Any]:
    prompt = normalize_ws(row.get("prompt", ""))
    chosen = normalize_ws(row.get("chosen", ""))
    rejected = normalize_ws(row.get("rejected", ""))
    pairs = history_pairs_from_rows(prior_rows, max_history_pairs, history_max_chars)
    meta = {
        "dataset": "psoups",
        "config": config_name,
        "source_split": source_split,
        "output_split": output_split,
        "user_id": str(row.get("uid", row.get("user_id", "unknown"))),
        "uid": row.get("uid", row.get("user_id", "unknown")),
        "source_index": int(row.get("_source_index", -1)),
        "turn": int(row.get("_user_index", 0)),
        "conversation_id": f"psoups-{row.get('uid', row.get('user_id', 'unknown'))}-{row.get('_source_index', -1)}",
    }
    if "preference_key" in row:
        meta["preference_key"] = row.get("preference_key")
    return {
        "messages": [{"role": "user", "content": prompt}],
        "chosen": chosen,
        "rejected": rejected,
        "user_profile_text": "",
        "user_history_text": history_text_from_pairs(pairs, include_prompt=history_include_prompt),
        "user_history_pairs": pairs,
        "pair_weight": 1.0,
        "meta": meta,
    }


def clear_eval_history(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    out["user_history_text"] = ""
    out["user_history_pairs"] = []
    meta = dict(out.get("meta", {}) or {})
    meta["history_len"] = 0
    out["meta"] = meta
    return out


def build_train_valid_splits(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    train_out: List[Dict[str, Any]] = []
    valid_out: List[Dict[str, Any]] = []
    skipped_no_history = 0
    grouped = group_by_uid(rows)

    for user_id in sorted(grouped):
        user_rows = sorted(grouped[user_id], key=lambda x: int(x.get("_source_index", 0)))
        for i, row in enumerate(user_rows):
            row["_user_index"] = i
        n_valid = 0
        if len(user_rows) > args.min_history_pairs + 1 and args.seen_valid_frac > 0:
            n_valid = max(1, int(math.floor(len(user_rows) * args.seen_valid_frac)))
            n_valid = min(n_valid, len(user_rows) - args.min_history_pairs)
        train_rows = user_rows[:-n_valid] if n_valid else user_rows
        valid_rows = user_rows[-n_valid:] if n_valid else []

        prior: List[Dict[str, Any]] = []
        for row in train_rows:
            if len(prior) >= args.min_history_pairs:
                train_out.append(
                    to_preference_record(
                        row,
                        config_name=args.config_name,
                        source_split=args.train_split,
                        output_split="seen_train",
                        prior_rows=prior,
                        max_history_pairs=args.history_pairs,
                        history_max_chars=args.history_max_chars,
                        history_include_prompt=args.history_include_prompt,
                    )
                )
            else:
                skipped_no_history += 1
            prior.append(row)

        for row in valid_rows:
            if len(prior) >= args.min_history_pairs:
                valid_out.append(
                    to_preference_record(
                        row,
                        config_name=args.config_name,
                        source_split=args.train_split,
                        output_split="seen_valid",
                        prior_rows=prior,
                        max_history_pairs=args.history_pairs,
                        history_max_chars=args.history_max_chars,
                        history_include_prompt=args.history_include_prompt,
                    )
                )
            else:
                skipped_no_history += 1
            prior.append(row)

    return train_out, valid_out, {"train_users": len(grouped), "skipped_no_history": skipped_no_history}


def build_support_query_splits(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    support_out: List[Dict[str, Any]] = []
    query_out: List[Dict[str, Any]] = []
    skipped_short_users = 0
    grouped = group_by_uid(rows)

    for user_id in sorted(grouped):
        user_rows = sorted(grouped[user_id], key=lambda x: int(x.get("_source_index", 0)))
        for i, row in enumerate(user_rows):
            row["_user_index"] = i
        if len(user_rows) < 2:
            skipped_short_users += 1
            continue
        n_support = max(args.min_support_rows, int(math.floor(len(user_rows) * args.unseen_support_frac)))
        n_support = min(max(1, n_support), len(user_rows) - 1)
        support_rows = user_rows[:n_support]
        query_rows = user_rows[n_support:]

        prior: List[Dict[str, Any]] = []
        for row in support_rows:
            support_out.append(
                clear_eval_history(
                    to_preference_record(
                        row,
                        config_name=args.config_name,
                        source_split=args.test_split,
                        output_split="calib_unseen",
                        prior_rows=prior,
                        max_history_pairs=args.history_pairs,
                        history_max_chars=args.history_max_chars,
                        history_include_prompt=args.history_include_prompt,
                    )
                )
            )
            prior.append(row)

        for row in query_rows:
            query_out.append(
                clear_eval_history(
                    to_preference_record(
                        row,
                        config_name=args.config_name,
                        source_split=args.test_split,
                        output_split="test_unseen",
                        prior_rows=prior,
                        max_history_pairs=args.history_pairs,
                        history_max_chars=args.history_max_chars,
                        history_include_prompt=args.history_include_prompt,
                    )
                )
            )
            prior.append(row)

    return support_out, query_out, {"eval_users": len(grouped), "skipped_short_users": skipped_short_users}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build PSOUPS JSONL splits for cautious context-steering distillation")
    ap.add_argument("--dataset_name", type=str, default="RiverDong/psoups")
    ap.add_argument("--config_name", type=str, default="default", choices=["default", "ood", "new-user"])
    ap.add_argument("--out_dir", type=Path, default=Path("data/psoups_cautious_cos_splits"))
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--test_split", type=str, default="test")
    ap.add_argument("--seen_valid_frac", type=float, default=0.10)
    ap.add_argument("--unseen_support_frac", type=float, default=0.50)
    ap.add_argument("--min_history_pairs", type=int, default=1)
    ap.add_argument("--min_support_rows", type=int, default=4)
    ap.add_argument("--history_pairs", type=int, default=4)
    ap.add_argument("--history_max_chars", type=int, default=512)
    ap.add_argument("--history_include_prompt", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.config_name == "new-user":
        raise SystemExit("PSOUPS new-user has no train split; use --config_name default or ood for this pipeline.")

    train_rows = load_hf_rows(args.dataset_name, args.config_name, args.train_split)
    test_rows = load_hf_rows(args.dataset_name, args.config_name, args.test_split)

    seen_train, seen_valid, train_info = build_train_valid_splits(train_rows, args)
    calib_unseen, test_unseen, eval_info = build_support_query_splits(test_rows, args)

    write_jsonl(args.out_dir / "seen_train.jsonl", seen_train)
    write_jsonl(args.out_dir / "seen_valid.jsonl", seen_valid)
    write_jsonl(args.out_dir / "calib_unseen.jsonl", calib_unseen)
    write_jsonl(args.out_dir / "test_unseen.jsonl", test_unseen)

    summary = {
        "dataset_name": args.dataset_name,
        "config_name": args.config_name,
        "rows_seen_train": len(seen_train),
        "rows_seen_valid": len(seen_valid),
        "rows_calib_unseen": len(calib_unseen),
        "rows_test_unseen": len(test_unseen),
        **train_info,
        **eval_info,
        "config": vars(args),
    }
    (args.out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
