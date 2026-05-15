#!/usr/bin/env python3
"""
Build OpenAI TLDR comparisons -> PRISM/context-steering-compatible JSONL splits.

The openai/summarize_from_feedback comparisons config exposes rows as:
{
  "info": {"post": str, "title": str, "subreddit": str, ...},
  "summaries": [{"text": str, "policy": str, ...}, {"text": str, ...}],
  "choice": int,
  "worker": str,
  ...
}

This script treats the most prolific annotation workers as users, then uses each
worker's previous summary comparisons as preference history.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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


def load_hf_rows(dataset_name: str, config_name: str, split: str, revision: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: datasets. Install the project requirements or run "
            "`pip install datasets` in the environment used for preprocessing."
        ) from exc

    if revision:
        ds = load_dataset(dataset_name, split=split, revision=revision, data_dir=config_name)
    else:
        ds = load_dataset(dataset_name, config_name, split=split)
    return [dict(row) for row in ds]


def is_sft_comparison(row: Dict[str, Any]) -> bool:
    summaries = list(row.get("summaries", []) or [])
    if len(summaries) < 2:
        return False
    for summary in summaries[:2]:
        policy = str(summary.get("policy", ""))
        if "sup" not in policy or "ppo" in policy or "cnn" in policy:
            return False
    return True


def choice_pair(row: Dict[str, Any]) -> Tuple[str, str]:
    summaries = list(row.get("summaries", []) or [])
    if len(summaries) < 2:
        return "", ""
    try:
        choice = int(row.get("choice", -1))
    except Exception:
        return "", ""
    if choice not in (0, 1):
        return "", ""
    chosen = normalize_ws(summaries[choice].get("text", ""))
    rejected = normalize_ws(summaries[1 - choice].get("text", ""))
    return chosen, rejected


def build_prompt(row: Dict[str, Any]) -> str:
    info = row.get("info", {}) or {}
    title = normalize_ws(info.get("title", ""))
    subreddit = normalize_ws(info.get("subreddit", ""))
    post = normalize_ws(info.get("post", info.get("article", "")))
    parts = ["Summarize the following Reddit post."]
    if subreddit:
        parts.append(f"Subreddit: {subreddit}")
    if title:
        parts.append(f"Title: {title}")
    if post:
        parts.append(f"Post: {post}")
    parts.append("Write a concise TL;DR summary.")
    return "\n\n".join(parts)


def valid_row(row: Dict[str, Any], sft_only: bool) -> bool:
    if sft_only and not is_sft_comparison(row):
        return False
    worker = normalize_ws(row.get("worker", ""))
    prompt = build_prompt(row)
    chosen, rejected = choice_pair(row)
    return bool(worker and prompt and chosen and rejected)


def select_top_workers(rows: Sequence[Dict[str, Any]], n_workers: int) -> List[str]:
    counts = Counter(normalize_ws(row.get("worker", "")) for row in rows if normalize_ws(row.get("worker", "")))
    return [worker for worker, _ in counts.most_common(n_workers)]


def group_by_worker(rows: Sequence[Dict[str, Any]], workers: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    allowed = set(workers)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        worker = normalize_ws(row.get("worker", ""))
        if worker not in allowed:
            continue
        enriched = dict(row)
        enriched["_source_index"] = idx
        grouped[worker].append(enriched)
    return grouped


def history_pairs_from_rows(
    rows: Sequence[Dict[str, Any]],
    max_items: int,
    max_chars: int,
) -> List[Dict[str, str]]:
    selected = list(rows) if max_items <= 0 else list(rows)[-max_items:]
    pairs: List[Dict[str, str]] = []
    for row in selected:
        chosen, rejected = choice_pair(row)
        if not chosen:
            continue
        pair = {
            "prompt": clip_text(build_prompt(row), max_chars),
            "chosen": clip_text(chosen, max_chars),
        }
        if rejected:
            pair["rejected"] = clip_text(rejected, max_chars)
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
            lines.append(f"Preferred summary: {chosen}")
        if rejected:
            lines.append(f"Less preferred summary: {rejected}")
    return "\n".join(lines)


def to_preference_record(
    row: Dict[str, Any],
    *,
    source_split: str,
    output_split: str,
    prior_rows: Sequence[Dict[str, Any]],
    max_history_pairs: int,
    history_max_chars: int,
    history_include_prompt: bool,
) -> Dict[str, Any]:
    prompt = build_prompt(row)
    chosen, rejected = choice_pair(row)
    worker = normalize_ws(row.get("worker", "unknown"))
    pairs = history_pairs_from_rows(prior_rows, max_history_pairs, history_max_chars)
    info = row.get("info", {}) or {}
    meta = {
        "dataset": "tldr",
        "source_dataset": "openai/summarize_from_feedback",
        "source_split": source_split,
        "output_split": output_split,
        "user_id": worker,
        "worker": worker,
        "source_index": int(row.get("_source_index", -1)),
        "turn": int(row.get("_worker_index", 0)),
        "conversation_id": f"tldr-{worker}-{row.get('_source_index', -1)}",
        "post_id": info.get("id", ""),
        "subreddit": info.get("subreddit", ""),
        "batch": row.get("batch", ""),
        "confidence": (row.get("extra", {}) or {}).get("confidence"),
    }
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
    workers: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    train_out: List[Dict[str, Any]] = []
    valid_out: List[Dict[str, Any]] = []
    skipped_no_history = 0
    grouped = group_by_worker(rows, workers)

    for worker in workers:
        worker_rows = sorted(grouped.get(worker, []), key=lambda x: int(x.get("_source_index", 0)))
        for i, row in enumerate(worker_rows):
            row["_worker_index"] = i
        n_valid = 0
        if len(worker_rows) > args.min_history_pairs + 1 and args.seen_valid_frac > 0:
            n_valid = max(1, int(math.floor(len(worker_rows) * args.seen_valid_frac)))
            n_valid = min(n_valid, len(worker_rows) - args.min_history_pairs)
        train_rows = worker_rows[:-n_valid] if n_valid else worker_rows
        valid_rows = worker_rows[-n_valid:] if n_valid else []

        prior: List[Dict[str, Any]] = []
        for row in train_rows:
            if len(prior) >= args.min_history_pairs:
                train_out.append(
                    to_preference_record(
                        row,
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
    workers: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    support_out: List[Dict[str, Any]] = []
    query_out: List[Dict[str, Any]] = []
    skipped_short_users = 0
    grouped = group_by_worker(rows, workers)

    for worker in workers:
        worker_rows = sorted(grouped.get(worker, []), key=lambda x: int(x.get("_source_index", 0)))
        for i, row in enumerate(worker_rows):
            row["_worker_index"] = i
        if len(worker_rows) < 2:
            skipped_short_users += 1
            continue
        n_support = max(args.min_support_rows, int(math.floor(len(worker_rows) * args.unseen_support_frac)))
        n_support = min(max(1, n_support), len(worker_rows) - 1)
        support_rows = worker_rows[:n_support]
        query_rows = worker_rows[n_support:]

        prior: List[Dict[str, Any]] = []
        for row in support_rows:
            support_out.append(
                clear_eval_history(
                    to_preference_record(
                        row,
                        source_split=args.eval_split,
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
                        source_split=args.eval_split,
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
    ap = argparse.ArgumentParser(description="Build TLDR JSONL splits for cautious context-steering distillation")
    ap.add_argument("--dataset_name", type=str, default="openai/summarize_from_feedback")
    ap.add_argument("--config_name", type=str, default="comparisons")
    ap.add_argument(
        "--revision",
        type=str,
        default="refs/convert/parquet",
        help="Use the parquet conversion branch because newer datasets versions reject dataset scripts.",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("data/tldr_top40_cautious_cos_splits"))
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="validation")
    ap.add_argument("--top_workers", type=int, default=40)
    ap.add_argument("--sft_only", type=int, default=1, choices=[0, 1])
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
    train_rows = [
        row
        for row in load_hf_rows(args.dataset_name, args.config_name, args.train_split, args.revision)
        if valid_row(row, bool(args.sft_only))
    ]
    eval_rows = [
        row
        for row in load_hf_rows(args.dataset_name, args.config_name, args.eval_split, args.revision)
        if valid_row(row, bool(args.sft_only))
    ]
    workers = select_top_workers(train_rows, int(args.top_workers))

    seen_train, seen_valid, train_info = build_train_valid_splits(train_rows, workers, args)
    calib_unseen, test_unseen, eval_info = build_support_query_splits(eval_rows, workers, args)

    write_jsonl(args.out_dir / "seen_train.jsonl", seen_train)
    write_jsonl(args.out_dir / "seen_valid.jsonl", seen_valid)
    write_jsonl(args.out_dir / "calib_unseen.jsonl", calib_unseen)
    write_jsonl(args.out_dir / "test_unseen.jsonl", test_unseen)
    (args.out_dir / "top_workers.txt").write_text("\n".join(workers) + "\n", encoding="utf-8")

    summary = {
        "dataset_name": args.dataset_name,
        "config_name": args.config_name,
        "top_workers": int(args.top_workers),
        "sft_only": bool(args.sft_only),
        "selected_workers": workers,
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
