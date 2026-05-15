#!/usr/bin/env python3
"""TLDR evaluation entrypoint for train_prism_cautious_context_steering_distill.py."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_pengram_history_generation as history_eval


def _has_flag(flag: str) -> bool:
    return flag in sys.argv[1:]


def _inject_pair(flag: str, value: str, require_exists: bool = False) -> None:
    if _has_flag(flag):
        return
    if require_exists and not Path(value).exists():
        return
    sys.argv.extend([flag, value])


def main() -> None:
    _inject_pair("--train_script", str(SCRIPT_DIR / "train_prism_cautious_context_steering_distill.py"))
    _inject_pair("--support_jsonl", str(REPO_ROOT / "data/tldr_top40_pengram_splits/calib_unseen.jsonl"), require_exists=True)
    _inject_pair("--query_jsonl", str(REPO_ROOT / "data/tldr_top40_pengram_splits/test_unseen.jsonl"), require_exists=True)
    history_eval.main()


if __name__ == "__main__":
    main()
