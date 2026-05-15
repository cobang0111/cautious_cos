#!/usr/bin/env python3
"""
Build PersonalLLM InteractionDatabase -> PRISM/Pengram-compatible JSONL splits.

Each row in namkoong-lab/PersonalLLM_InteractionDatabase is one simulated person
with 50 pairwise interactions:
{
  "person_id": int,
  "person_weight": [float, ...],
  "prompt_1": str,
  "response_1_a": str,
  "response_1_b": str,
  "chosen_1": str,
  ...
}

This script uses the first interactions for support history and the remaining
interactions as target queries. The JSONL rows are compatible with
test_pengram_history_generation.py and train_prism_cautious_context_steering_distill.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def load_hf_rows(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: datasets. Install the project requirements or run "
            "`pip install datasets` in the environment used for preprocessing."
        ) from exc

    ds = load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]


def resolve_pair(row: Dict[str, Any], idx: int) -> Optional[Dict[str, str]]:
    prompt = normalize_ws(row.get(f"prompt_{idx}", ""))
    response_a = normalize_ws(row.get(f"response_{idx}_a", ""))
    response_b = normalize_ws(row.get(f"response_{idx}_b", ""))
    chosen_raw = normalize_ws(row.get(f"chosen_{idx}", ""))
    if not prompt or not response_a or not response_b or not chosen_raw:
        return None

    choice = chosen_raw.strip().lower()
    if choice in {"a", "1", "response_a", "response_1", "response 1"}:
        chosen, rejected = response_a, response_b
    elif choice in {"b", "2", "response_b", "response_2", "response 2"}:
        chosen, rejected = response_b, response_a
    elif normalize_ws(chosen_raw) == response_a:
        chosen, rejected = response_a, response_b
    elif normalize_ws(chosen_raw) == response_b:
        chosen, rejected = response_b, response_a
    else:
        # Some exported variants store the chosen response text with minor
        # formatting differences. Preserve it as the reference and keep a
        # deterministic unchosen fallback for JSONL compatibility.
        chosen = chosen_raw
        rejected = response_b if chosen_raw != response_b else response_a

    if not chosen:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def history_pairs_from_interactions(
    interactions: Sequence[Dict[str, str]],
    max_items: int,
    max_chars: int,
) -> List[Dict[str, str]]:
    selected = list(interactions) if max_items <= 0 else list(interactions)[-max_items:]
    pairs: List[Dict[str, str]] = []
    for item in selected:
        prompt = clip_text(item.get("prompt", ""), max_chars)
        chosen = clip_text(item.get("chosen", ""), max_chars)
        rejected = clip_text(item.get("rejected", ""), max_chars)
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
        if include_prompt and prompt:
            lines.append(f"Support example {idx}: User request: {prompt}")
        if chosen:
            lines.append(f"Preferred assistant response: {chosen}")
    return "\n".join(lines)


def to_pengram_record(
    *,
    row: Dict[str, Any],
    interaction: Dict[str, str],
    interaction_idx: int,
    output_split: str,
    prior_interactions: Sequence[Dict[str, str]],
    max_history_pairs: int,
    history_max_chars: int,
    history_include_prompt: bool,
) -> Dict[str, Any]:
    person_id = str(row.get("person_id", "unknown"))
    pairs = history_pairs_from_interactions(prior_interactions, max_history_pairs, history_max_chars)
    meta = {
        "dataset": "personalllm_interaction_database",
        "source_dataset": "namkoong-lab/PersonalLLM_InteractionDatabase",
        "output_split": output_split,
        "user_id": person_id,
        "person_id": row.get("person_id", "unknown"),
        "interaction_idx": int(interaction_idx),
        "turn": int(interaction_idx),
        "conversation_id": f"personalllm-{person_id}-{interaction_idx}",
    }
    person_weight = row.get("person_weight")
    if person_weight is not None:
        meta["person_weight"] = person_weight

    return {
        "messages": [{"role": "user", "content": interaction["prompt"]}],
        "chosen": interaction["chosen"],
        "rejected": interaction.get("rejected", ""),
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


def interactions_for_person(row: Dict[str, Any], n_interactions: int) -> List[Tuple[int, Dict[str, str]]]:
    interactions: List[Tuple[int, Dict[str, str]]] = []
    for idx in range(1, n_interactions + 1):
        item = resolve_pair(row, idx)
        if item is not None:
            interactions.append((idx, item))
    return interactions


def build_support_query_splits(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    support_out: List[Dict[str, Any]] = []
    query_out: List[Dict[str, Any]] = []
    skipped_short_persons = 0
    used_persons = 0

    selected_rows = list(rows)
    if args.max_persons > 0:
        selected_rows = selected_rows[: int(args.max_persons)]

    for row in selected_rows:
        interactions = interactions_for_person(row, int(args.n_interactions))
        if len(interactions) <= int(args.support_count):
            skipped_short_persons += 1
            continue
        used_persons += 1

        support = interactions[: int(args.support_count)]
        query = interactions[int(args.support_count) :]
        if args.max_query_per_person > 0:
            query = query[: int(args.max_query_per_person)]

        prior: List[Dict[str, str]] = []
        for interaction_idx, interaction in support:
            support_out.append(
                clear_eval_history(
                    to_pengram_record(
                        row=row,
                        interaction=interaction,
                        interaction_idx=interaction_idx,
                        output_split="calib_unseen",
                        prior_interactions=prior,
                        max_history_pairs=args.history_pairs,
                        history_max_chars=args.history_max_chars,
                        history_include_prompt=args.history_include_prompt,
                    )
                )
            )
            prior.append(interaction)

        support_prior = [interaction for _, interaction in support]
        for interaction_idx, interaction in query:
            query_out.append(
                clear_eval_history(
                    to_pengram_record(
                        row=row,
                        interaction=interaction,
                        interaction_idx=interaction_idx,
                        output_split="test_unseen",
                        prior_interactions=support_prior,
                        max_history_pairs=args.history_pairs,
                        history_max_chars=args.history_max_chars,
                        history_include_prompt=args.history_include_prompt,
                    )
                )
            )

    info = {
        "persons_total": len(selected_rows),
        "persons_used": used_persons,
        "skipped_short_persons": skipped_short_persons,
    }
    return support_out, query_out, info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build PersonalLLM support/query JSONL splits")
    ap.add_argument("--dataset_name", type=str, default="namkoong-lab/PersonalLLM_InteractionDatabase")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out_dir", type=Path, default=Path("data/personalllm_pengram_splits"))
    ap.add_argument("--n_interactions", type=int, default=50)
    ap.add_argument("--support_count", type=int, default=4)
    ap.add_argument("--history_pairs", type=int, default=4)
    ap.add_argument("--history_max_chars", type=int, default=512)
    ap.add_argument("--history_include_prompt", action="store_true")
    ap.add_argument("--max_persons", type=int, default=1000)
    ap.add_argument(
        "--max_query_per_person",
        type=int,
        default=0,
        help="Limit target queries per person. 0 keeps all interactions after support_count.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_hf_rows(args.dataset_name, args.split)
    calib_unseen, test_unseen, info = build_support_query_splits(rows, args)

    write_jsonl(args.out_dir / "calib_unseen.jsonl", calib_unseen)
    write_jsonl(args.out_dir / "test_unseen.jsonl", test_unseen)

    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "rows_calib_unseen": len(calib_unseen),
        "rows_test_unseen": len(test_unseen),
        **info,
        "config": vars(args),
    }
    (args.out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
