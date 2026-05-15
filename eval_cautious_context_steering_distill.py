#!/usr/bin/env python3
"""Dataset-selectable evaluation entrypoint for cautious context steering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_pengram_history_generation as history_eval


DATASET_DEFAULTS: Dict[str, Tuple[str, str]] = {
    "prism": (
        "data/prism_pengram_splits/calib_unseen.jsonl",
        "data/prism_pengram_splits/test_unseen.jsonl",
    ),
    "ultrafeedback": (
        "data/ultrafeedback_single_P_4_history/calib_unseen.jsonl",
        "data/ultrafeedback_single_P_4_history/test_unseen.jsonl",
    ),
    "psoups": (
        "data/psoups_pengram_splits/calib_unseen.jsonl",
        "data/psoups_pengram_splits/test_unseen.jsonl",
    ),
    "tldr": (
        "data/tldr_top40_pengram_splits/calib_unseen.jsonl",
        "data/tldr_top40_pengram_splits/test_unseen.jsonl",
    ),
    "personalllm": (
        "data/personalllm_pengram_splits/calib_unseen.jsonl",
        "data/personalllm_pengram_splits/test_unseen.jsonl",
    ),
}

DATASET_ALIASES = {
    "uf": "ultrafeedback",
    "ultra": "ultrafeedback",
    "personal_llm": "personalllm",
    "personal-llm": "personalllm",
}


def _has_flag(flag: str) -> bool:
    return flag in sys.argv[1:]


def _inject_pair(flag: str, value: str) -> None:
    if _has_flag(flag):
        return
    sys.argv.extend([flag, value])


def _parse_dataset_arg() -> str:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cautious context steering on one dataset. All remaining "
            "arguments are forwarded to test_pengram_history_generation.py."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(set(DATASET_DEFAULTS) | set(DATASET_ALIASES)),
        help="Dataset whose default support/query split paths should be used.",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return DATASET_ALIASES.get(known.dataset, known.dataset)


def main() -> None:
    dataset = _parse_dataset_arg()
    support_rel, query_rel = DATASET_DEFAULTS[dataset]

    _inject_pair("--train_script", str(REPO_ROOT / "train_prism_cautious_context_steering_distill.py"))
    _inject_pair("--support_jsonl", str(REPO_ROOT / support_rel))
    _inject_pair("--query_jsonl", str(REPO_ROOT / query_rel))

    history_eval.main()


if __name__ == "__main__":
    main()
