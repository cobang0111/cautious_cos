#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
import time
import unicodedata
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from history_icl_utils import (
    build_history_exemplar_prefix,
    build_history_pairs_from_support as shared_build_history_pairs_from_support,
    extract_last_user_prompt as shared_extract_last_user_prompt,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from rouge_score import rouge_scorer  # type: ignore
except Exception:  # pragma: no cover
    rouge_scorer = None

try:
    from bert_score import BERTScorer  # type: ignore
except Exception:  # pragma: no cover
    BERTScorer = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_SCRIPT = REPO_ROOT / "train_prism_cautious_context_steering_distill.py"
DATASET_DEFAULTS = {
    "prism": (
        "data/prism_cautious_cos_splits/calib_unseen.jsonl",
        "data/prism_cautious_cos_splits/test_unseen.jsonl",
    ),
    "ultrafeedback": (
        "data/ultrafeedback_single_P_4_history/calib_unseen.jsonl",
        "data/ultrafeedback_single_P_4_history/test_unseen.jsonl",
    ),
    "psoups": (
        "data/psoups_cautious_cos_splits/calib_unseen.jsonl",
        "data/psoups_cautious_cos_splits/test_unseen.jsonl",
    ),
    "tldr": (
        "data/tldr_top40_cautious_cos_splits/calib_unseen.jsonl",
        "data/tldr_top40_cautious_cos_splits/test_unseen.jsonl",
    ),
    "personalllm": (
        "data/personalllm_cautious_cos_splits/calib_unseen.jsonl",
        "data/personalllm_cautious_cos_splits/test_unseen.jsonl",
    ),
}
DATASET_ALIASES = {
    "uf_p_4": "ultrafeedback",
    "uf-p-4": "ultrafeedback",
    "uf": "ultrafeedback",
    "ultra": "ultrafeedback",
    "personal_llm": "personalllm",
    "personal-llm": "personalllm",
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    dataset_choices = sorted(set(DATASET_DEFAULTS) | set(DATASET_ALIASES))
    system_aliases = {
        "cautious-cos": "steer_distill",
        "cautious_cos": "steer_distill",
    }
    system_choices = ["base", "icl", "icl_rag", "cos", "cos_history", "steer_distill", "cautious-cos", "cautious_cos", "lora_sft"]
    ap = argparse.ArgumentParser(description="Unified generation and policy evaluation for cautious context steering distill.")
    ap.add_argument("--dataset", type=str, default="", choices=dataset_choices)
    ap.add_argument("--train_script", type=str, default="")
    ap.add_argument("--support_jsonl", type=str, default="")
    ap.add_argument("--query_jsonl", type=str, default="")
    ap.add_argument("--steering_checkpoint", type=str, default="", help="Checkpoint for context-steering distillation models.")
    ap.add_argument("--lora_sft_checkpoint", type=str, default="")

    ap.add_argument("--support_budgets", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--strict_support_budget", action="store_true")
    ap.add_argument(
        "--no_support_budget_loop",
        action="store_true",
        help="Run once with budget=1 instead of looping over --support_budgets.",
    )
    ap.add_argument("--support_selection", type=str, default="earliest", choices=["earliest", "recent"])

    # ICL / ICL-RAG
    ap.add_argument("--icl_mode", type=str, default="chosen_only", choices=["chosen_only", "pairwise"])
    ap.add_argument("--icl_include_prompt", action="store_true")
    ap.add_argument("--icl_include_user_profile", action="store_true")
    ap.add_argument("--icl_example_max_chars", type=int, default=512)
    ap.add_argument("--icl_intro", type=str, default="")
    ap.add_argument("--icl_rag", action="store_true")
    ap.add_argument("--icl_rag_topk", type=int, default=1)
    ap.add_argument("--icl_rag_retriever_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--icl_rag_scope", type=str, default="selected_k", choices=["selected_k", "all_support"])
    ap.add_argument("--icl_rag_batch_size", type=int, default=32)
    ap.add_argument("--icl_rag_match_field", type=str, default="prompt_only", choices=["prompt_only", "prompt_plus_chosen"])
    ap.add_argument(
        "--steering_use_icl_prompt",
        dest="steering_use_icl_prompt",
        action="store_true",
        help="Use the same exemplar ICL prompt format for steering rows.",
    )

    # History synthesis used by cautious_cos and CoS-history
    ap.add_argument("--cautious_cos_history_include_prompt", action="store_true")
    ap.add_argument("--cautious_cos_history_mode", type=str, default="chosen_only", choices=["chosen_only", "pairwise"])
    ap.add_argument("--cautious_cos_history_max_chars", type=int, default=512)
    ap.add_argument("--steering_skip_first_n", type=int, default=0)

    # CoS-history baseline
    ap.add_argument("--cos_lambda", type=float, default=1.0)
    ap.add_argument("--cos_history_mode", type=str, default="", choices=["", "chosen_only", "pairwise"])
    ap.add_argument("--cos_history_include_prompt", action="store_true")
    ap.add_argument("--cos_history_max_chars", type=int, default=0)
    ap.add_argument(
        "--cos_history_template",
        type=str,
        default="Interaction history:\n{history}\n\nCurrent interaction:\n{prompt}",
    )

    # generation
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--metric_device", type=str, default="")
    ap.add_argument("--max_eval_rows", type=int, default=0)
    ap.add_argument("--print_examples", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default="")
    ap.add_argument("--policy_eval_batch_size", type=int, default=4)
    ap.add_argument(
        "--sanity_check_rerank_repeat",
        action="store_true",
        help="Run repeat-call sanity check for reranking tensors on one batch.",
    )

    # metric models
    ap.add_argument("--prefsim_model", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--bertscore_lang", type=str, default="en")
    ap.add_argument("--bertscore_model_type", type=str, default="")
    ap.add_argument("--bertscore_batch_size", type=int, default=16)

    # fallback model options if no checkpoint is supplied
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--prompt_template", type=str, default="User request:\n{prompt}\n\nAssistant:\n")
    ap.add_argument("--append_eos", action="store_true")
    ap.add_argument("--length_normalize", action="store_true")
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_answer_len", type=int, default=512)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_user_len", type=int, default=512)
    ap.add_argument("--max_history_pairs", type=int, default=4)
    ap.add_argument("--max_history_prompt_len", type=int, default=256)
    ap.add_argument("--max_history_answer_len", type=int, default=256)

    # optional checkpoint overrides
    ap.add_argument("--override_model_name", type=str, default="")
    ap.add_argument("--override_dtype", type=str, default="", choices=["", "bfloat16", "float16", "float32"])
    ap.add_argument("--override_attn_implementation", type=str, default="", choices=["", "sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--override_trust_remote_code", action="store_true")
    ap.add_argument("--override_entropy_threshold", type=float, default=None)
    ap.add_argument("--override_entropy_temperature", type=float, default=None)
    ap.add_argument("--override_gap_threshold", type=float, default=None)
    ap.add_argument("--override_gap_temperature", type=float, default=None)
    ap.add_argument("--context_pool", type=str, default="attn", choices=["attn", "last", "mean"])
    ap.add_argument("--require_history", type=int, default=0, choices=[0, 1])
    ap.add_argument("--pref_hash_buckets", type=int, default=16)
    ap.add_argument("--pref_hash_bits", type=int, default=10)

    ap.add_argument(
        "--systems",
        nargs="+",
        default=["base", "icl", "cos", "steer_distill"],
        choices=system_choices,
    )
    ap.add_argument("--gen_batch_size", type=int, default=16)
    args = ap.parse_args()
    args.systems = [system_aliases.get(system, system) for system in args.systems]

    if args.dataset:
        dataset = DATASET_ALIASES.get(args.dataset, args.dataset)
        args.dataset = dataset
        support_rel, query_rel = DATASET_DEFAULTS[dataset]
        if not args.support_jsonl:
            args.support_jsonl = str(REPO_ROOT / support_rel)
        if not args.query_jsonl:
            args.query_jsonl = str(REPO_ROOT / query_rel)

    if not args.train_script:
        args.train_script = str(DEFAULT_TRAIN_SCRIPT)
    if not args.support_jsonl or not args.query_jsonl:
        ap.error("--support_jsonl and --query_jsonl are required unless --dataset supplies defaults")

    return args


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


class ListDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def normalize_ws(x: Any) -> str:
    return " ".join(str(x or "").split())


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def clip_text(x: Any, max_chars: int) -> str:
    s = normalize_ws(x)
    if len(s) <= max_chars:
        return s
    return s[: max(1, max_chars - 3)] + "..."


def maybe_mkdir(path: str) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dynamic_import_module(module_path: str) -> ModuleType:
    module_path = str(Path(module_path).resolve())
    spec = importlib.util.spec_from_file_location("context_steering_train_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_steering_checkpoint_paths(path_str: str) -> Tuple[Path, Path]:
    p = Path(path_str)
    if p.is_dir():
        ckpt_path = p / "steering.pt"
        args_path = p / "steering_args.json"
    else:
        ckpt_path = p
        args_path = p.parent / "steering_args.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Could not find steering checkpoint at {ckpt_path}")
    if not args_path.exists():
        raise FileNotFoundError(f"Could not find steering_args.json near {ckpt_path}")
    return ckpt_path, args_path


def resolve_lora_checkpoint_dir(path_str: str) -> Tuple[Path, Path]:
    p = Path(path_str)
    ckpt_dir = p if p.is_dir() else p.parent
    adapter_config = ckpt_dir / "adapter_config.json"
    args_path = ckpt_dir / "sft_args.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Could not find LoRA adapter_config.json at {adapter_config}")
    if not args_path.exists():
        raise FileNotFoundError(f"Could not find sft_args.json at {args_path}")
    return ckpt_dir, args_path


def build_namespace_from_cli(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=cli.model_name,
        dtype=cli.dtype,
        attn_implementation=cli.attn_implementation,
        trust_remote_code=cli.trust_remote_code,
        use_chat_template=cli.use_chat_template,
        prompt_template=cli.prompt_template,
        append_eos=cli.append_eos,
        length_normalize=cli.length_normalize,
        max_prompt_len=cli.max_prompt_len,
        max_answer_len=cli.max_answer_len,
        max_seq_len=cli.max_seq_len,
        max_user_len=getattr(cli, "max_user_len", 512),
        user_dim=512,
        prompt_dim=128,
        memory_dim=512,
        memory_slots=2048,
        memory_topk=8,
        candidate_dim=512,
        module_dropout=0.0,
        memory_temperature=0.5,
        support_min_k=4,
        support_max_k=48,
        support_top_p_min=0.70,
        support_top_p_max=0.95,
        entropy_threshold=0.3,
        entropy_temperature=0.08,
        score_scale=1.0,
        score_scale_warmup_ratio=0.10,
        zero_mean_scores=1,
        score_clip=3.0,
        pairwise_pref_weight=1.0,
        pairwise_beta=1.0,
        chosen_ce_weight=0.35,
        chosen_gain_hinge_weight=0.15,
        chosen_gain_margin=0.0,
        anchor_kl_weight=5e-4,
        hard_pair_margin=0.0,
        hard_pair_temperature=0.25,
        hard_pair_min_weight=0.25,
        train_force_label_in_support=1,
        train_force_rejected_label_in_support=0,
        user_context_mode="history_only",
        max_history_pairs=getattr(cli, "max_history_pairs", 4),
        max_history_prompt_len=getattr(cli, "max_history_prompt_len", 256),
        max_history_answer_len=getattr(cli, "max_history_answer_len", 256),
        history_attn_temperature=1.0,
        history_empty_fallback="zero",
        context_pool=getattr(cli, "context_pool", "attn"),
        require_history=getattr(cli, "require_history", 0),
        pref_hash_buckets=getattr(cli, "pref_hash_buckets", 16),
        pref_hash_bits=getattr(cli, "pref_hash_bits", 10),
        personalize_eos=0,
        ignore_eos_in_loss=1,
    )


def load_train_namespace(args_path: Path, cli: argparse.Namespace) -> SimpleNamespace:
    with open(args_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    if cli.override_model_name:
        saved["model_name"] = cli.override_model_name
    if cli.override_dtype:
        saved["dtype"] = cli.override_dtype
    if cli.override_attn_implementation:
        saved["attn_implementation"] = cli.override_attn_implementation
    if cli.override_trust_remote_code:
        saved["trust_remote_code"] = True
    if cli.override_entropy_threshold is not None:
        saved["entropy_threshold"] = cli.override_entropy_threshold
    if cli.override_entropy_temperature is not None:
        saved["entropy_temperature"] = cli.override_entropy_temperature
    if cli.override_gap_threshold is not None:
        saved["gap_threshold"] = cli.override_gap_threshold
    if cli.override_gap_temperature is not None:
        saved["gap_temperature"] = cli.override_gap_temperature

    saved.setdefault("model_name", cli.model_name)
    saved.setdefault("dtype", cli.dtype)
    saved.setdefault("attn_implementation", cli.attn_implementation)
    saved.setdefault("trust_remote_code", cli.trust_remote_code)
    saved.setdefault("use_chat_template", cli.use_chat_template)
    saved.setdefault("prompt_template", cli.prompt_template)
    saved.setdefault("append_eos", cli.append_eos)
    saved.setdefault("length_normalize", cli.length_normalize)
    saved.setdefault("max_prompt_len", cli.max_prompt_len)
    saved.setdefault("max_answer_len", cli.max_answer_len)
    saved.setdefault("max_seq_len", cli.max_seq_len)
    saved.setdefault("max_user_len", getattr(cli, "max_user_len", 512))
    saved.setdefault("user_dim", 256)
    # Keep fallback aligned with training-side model default to avoid
    # shape mismatches when older checkpoints lack this field.
    saved.setdefault("prompt_dim", 256)
    saved.setdefault("memory_dim", 512)
    saved.setdefault("memory_slots", 2048)
    saved.setdefault("memory_topk", 8)
    saved.setdefault("candidate_dim", 256)
    saved.setdefault("module_dropout", 0.05)
    saved.setdefault("memory_temperature", 0.5)
    saved.setdefault("support_min_k", 4)
    saved.setdefault("support_max_k", 48)
    saved.setdefault("support_top_p_min", 0.70)
    saved.setdefault("support_top_p_max", 0.95)
    saved.setdefault("entropy_threshold", 0.3)
    saved.setdefault("entropy_temperature", 0.08)
    saved.setdefault("score_scale", 1.0)
    saved.setdefault("score_scale_warmup_ratio", 0.10)
    saved.setdefault("zero_mean_scores", 1)
    saved.setdefault("score_clip", 3.0)
    saved.setdefault("pairwise_pref_weight", 1.0)
    saved.setdefault("pairwise_beta", 1.0)
    saved.setdefault("chosen_ce_weight", 0.35)
    saved.setdefault("chosen_gain_hinge_weight", 0.15)
    saved.setdefault("chosen_gain_margin", 0.0)
    saved.setdefault("anchor_kl_weight", 5e-4)
    saved.setdefault("hard_pair_margin", 0.0)
    saved.setdefault("hard_pair_temperature", 0.25)
    saved.setdefault("hard_pair_min_weight", 0.25)
    saved.setdefault("train_force_label_in_support", 1)
    saved.setdefault("train_force_rejected_label_in_support", 0)
    saved.setdefault("user_context_mode", "history_only")
    saved.setdefault("max_history_pairs", 4)
    saved.setdefault("max_history_prompt_len", 256)
    saved.setdefault("max_history_answer_len", 256)
    saved.setdefault("history_attn_temperature", 1.0)
    saved.setdefault("history_empty_fallback", "zero")
    saved.setdefault("context_pool", getattr(cli, "context_pool", "attn"))
    saved.setdefault("require_history", getattr(cli, "require_history", 0))
    saved.setdefault("pref_hash_buckets", getattr(cli, "pref_hash_buckets", 16))
    saved.setdefault("pref_hash_bits", getattr(cli, "pref_hash_bits", 10))
    saved.setdefault("personalize_eos", 0)
    saved.setdefault("ignore_eos_in_loss", 1)
    return SimpleNamespace(**saved)


def maybe_warn_runtime_mismatch(primary_name: str, primary_args: SimpleNamespace, other_name: str, other_args: SimpleNamespace) -> None:
    fields = ["model_name", "dtype", "attn_implementation", "max_prompt_len", "max_seq_len"]
    diffs = []
    for field in fields:
        if getattr(primary_args, field, None) != getattr(other_args, field, None):
            diffs.append(f"{field}={getattr(primary_args, field, None)} vs {getattr(other_args, field, None)}")
    if diffs:
        print(
            f"[warn] runtime args differ between {primary_name} and {other_name}; "
            f"using {primary_name} settings for shared prompt/tokenizer setup: " + ", ".join(diffs)
        )


def load_lora_sft_model(
    train_module: ModuleType,
    train_args: SimpleNamespace,
    checkpoint_dir: Path,
    device: torch.device,
) -> nn.Module:
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("peft is required to evaluate --lora_sft_checkpoint") from exc

    _, lora_handles = train_module.load_base_handles(train_args, device)
    lora_model = PeftModel.from_pretrained(lora_handles.base_model, str(checkpoint_dir), is_trainable=False)
    dt_name = str(getattr(train_args, "dtype", "float32"))
    if dt_name in ("bfloat16", "float16"):
        lora_model.to(train_module.str_to_dtype(dt_name))
    lora_model.to(device)
    lora_model.eval()
    return lora_model


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


# -----------------------------------------------------------------------------
# Lightweight retrieval / fallback metric helpers
# -----------------------------------------------------------------------------


def _simple_tokenize(text: str) -> List[str]:
    return [tok for tok in normalize_ws(text).lower().split() if tok]


def _hashed_bow(texts: Sequence[str], dim: int = 512) -> torch.Tensor:
    vecs = torch.zeros((len(texts), dim), dtype=torch.float32)
    for i, text in enumerate(texts):
        for tok in _simple_tokenize(text):
            vecs[i, hash(tok) % dim] += 1.0
    return F.normalize(vecs, dim=-1)


def _lexical_f1(cand: str, ref: str) -> float:
    cand_toks = _simple_tokenize(cand)
    ref_toks = _simple_tokenize(ref)
    if not cand_toks and not ref_toks:
        return 1.0
    if not cand_toks or not ref_toks:
        return 0.0
    cand_counts = defaultdict(int)
    ref_counts = defaultdict(int)
    for t in cand_toks:
        cand_counts[t] += 1
    for t in ref_toks:
        ref_counts[t] += 1
    overlap = sum(min(cand_counts[t], ref_counts[t]) for t in set(cand_counts) | set(ref_counts))
    precision = overlap / max(1, len(cand_toks))
    recall = overlap / max(1, len(ref_toks))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class _FallbackBERTScorer:
    def score(self, cands: Sequence[str], refs: Sequence[str], batch_size: int = 16, verbose: bool = False):
        f1 = torch.tensor([_lexical_f1(c, r) for c, r in zip(cands, refs)], dtype=torch.float32)
        return f1, f1, f1


class DenseRetriever:
    def __init__(self, model_name: str, device: torch.device):
        self.model = SentenceTransformer(model_name, device=str(device)) if SentenceTransformer is not None else None

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)
        if self.model is None:
            return _hashed_bow(texts)
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.float().cpu()


# -----------------------------------------------------------------------------
# Support-set construction
# -----------------------------------------------------------------------------


def support_sort_key(row: Dict[str, Any]) -> Tuple[str, str, int, float]:
    meta = row.get("meta", {}) or {}
    dt = str(meta.get("generated_datetime", ""))
    conv = str(meta.get("conversation_id", ""))
    turn = int(meta.get("turn", 0) or 0)
    gap = float(meta.get("score_gap", 0.0) or 0.0)
    return (dt, conv, turn, -gap)


def support_anchor_key(row: Dict[str, Any]) -> Tuple[str, str, int, str]:
    meta = row.get("meta", {}) or {}
    return (
        str(meta.get("user_id", row.get("user_id", ""))),
        str(meta.get("conversation_id", "")),
        int(meta.get("turn", 0) or 0),
        normalize_ws(row.get("chosen", "")),
    )


def group_rows_by_user(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta", {}) or {}
        user_id = str(meta.get("user_id", row.get("user_id", "unknown")))
        grouped[user_id].append(row)
    return grouped


def unique_support_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uniq: "OrderedDict[Tuple[str, str, int, str], Dict[str, Any]]" = OrderedDict()
    for row in sorted(rows, key=support_sort_key):
        key = support_anchor_key(row)
        if key not in uniq:
            uniq[key] = row
    return list(uniq.values())


def select_support_rows(rows: Sequence[Dict[str, Any]], budget: int, selection: str) -> List[Dict[str, Any]]:
    rows = list(rows)
    if budget <= 0:
        return []
    if selection == "recent":
        return rows[-budget:]
    return rows[:budget]


def extract_last_user_prompt(row: Dict[str, Any]) -> str:
    return shared_extract_last_user_prompt(row)


def build_rag_query_text(row: Dict[str, Any]) -> str:
    return extract_last_user_prompt(row)


def build_rag_support_text(row: Dict[str, Any], match_field: str) -> str:
    prompt = extract_last_user_prompt(row)
    chosen = normalize_ws(row.get("chosen", ""))
    if match_field == "prompt_plus_chosen":
        return normalize_ws(f"User request: {prompt}\nPreferred assistant response: {chosen}")
    return prompt


def synthesize_cautious_cos_history_from_support(
    support_rows: Sequence[Dict[str, Any]],
    mode: str,
    include_prompt: bool,
    max_chars: int,
) -> str:
    if not support_rows:
        return "No prior interaction history available."
    lines: List[str] = []
    for idx, row in enumerate(support_rows, start=1):
        prompt = clip_text(extract_last_user_prompt(row), max_chars)
        chosen = clip_text(row.get("chosen", ""), max_chars)
        rejected = clip_text(row.get("rejected", ""), max_chars)
        if mode == "chosen_only":
            if include_prompt and prompt:
                lines.append(f"Previously preferred interaction {idx}: User request: {prompt}")
            lines.append(f"Previously preferred answer style example: {chosen}")
        else:
            if include_prompt and prompt:
                lines.append(f"Support example {idx}: User request: {prompt}")
            lines.append(f"Preferred assistant response: {chosen}")
            if rejected:
                lines.append(f"Less preferred assistant response: {rejected}")
    return "\n".join(lines)


def build_cautious_cos_history_pairs_from_support(
    support_rows: Sequence[Dict[str, Any]],
    max_items: int,
    max_chars: int,
) -> List[Dict[str, str]]:
    return shared_build_history_pairs_from_support(
        support_rows=support_rows,
        max_items=max_items,
        max_chars=max_chars,
    )


def build_icl_prefix(
    query_row: Dict[str, Any],
    support_rows: Sequence[Dict[str, Any]],
    intro: str,
    icl_mode: str,
    include_prompt: bool,
    include_user_profile: bool,
    max_chars: int,
) -> str:
    history_pairs = shared_build_history_pairs_from_support(
        support_rows=support_rows,
        max_items=len(support_rows),
        max_chars=max_chars,
    )
    return build_history_exemplar_prefix(
        history_pairs=history_pairs,
        intro=intro,
        exemplar_mode=icl_mode,
        include_prompt=include_prompt,
        include_user_profile=include_user_profile,
        user_profile=query_row.get("user_profile_text", query_row.get("user_profile", "")),
        max_chars=max_chars,
        max_items=len(history_pairs),
    )


def build_icl_query_row(
    row: Dict[str, Any],
    train_module: ModuleType,
    tokenizer: Any,
    train_args: SimpleNamespace,
    support_rows: Sequence[Dict[str, Any]],
    cli: argparse.Namespace,
) -> Dict[str, Any]:
    base_prompt = train_module.render_prompt(row, tokenizer, train_args)
    icl_prefix = build_icl_prefix(
        query_row=row,
        support_rows=support_rows,
        intro=cli.icl_intro,
        icl_mode=cli.icl_mode,
        include_prompt=cli.icl_include_prompt,
        include_user_profile=cli.icl_include_user_profile,
        max_chars=cli.icl_example_max_chars,
    )
    new_row = dict(row)
    new_row["prompt_text"] = (icl_prefix + "\n\nCurrent interaction:\n" + base_prompt).strip()
    new_row.pop("messages", None)
    new_row.pop("prompt", None)
    meta = dict(row.get("meta", {}) or {})
    meta["icl_support_count"] = len(support_rows)
    new_row["meta"] = meta
    return new_row


def build_budget_rows(
    train_module: ModuleType,
    tokenizer: Any,
    train_args: SimpleNamespace,
    support_rows: Sequence[Dict[str, Any]],
    query_rows: Sequence[Dict[str, Any]],
    budget: int,
    strict_support_budget: bool,
    support_selection: str,
    cli: argparse.Namespace,
    retriever: Optional[DenseRetriever] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    support_by_user = group_rows_by_user(support_rows)
    query_by_user = group_rows_by_user(query_rows)

    base_eval_rows: List[Dict[str, Any]] = []
    icl_eval_rows: List[Dict[str, Any]] = []
    icl_rag_eval_rows: List[Dict[str, Any]] = []
    cos_eval_rows: List[Dict[str, Any]] = []
    steering_eval_rows: List[Dict[str, Any]] = []

    included_users = 0
    skipped_users = 0
    support_counts_used: List[int] = []

    for user_id, user_query_rows in sorted(query_by_user.items()):
        anchors = unique_support_rows(support_by_user.get(user_id, []))
        support_count = len(anchors)
        if support_count == 0:
            skipped_users += 1
            continue
        if strict_support_budget and support_count < budget:
            skipped_users += 1
            continue

        used_count = min(budget, support_count)
        selected_support = select_support_rows(anchors, used_count, selection=support_selection)
        if not selected_support:
            skipped_users += 1
            continue

        retrieval_candidates = selected_support if cli.icl_rag_scope == "selected_k" else anchors
        candidate_embs = None
        if retriever is not None:
            candidate_texts = [build_rag_support_text(r, cli.icl_rag_match_field) for r in retrieval_candidates]
            candidate_embs = retriever.encode(candidate_texts, batch_size=cli.icl_rag_batch_size)

        cautious_cos_history = synthesize_cautious_cos_history_from_support(
            support_rows=selected_support,
            mode=cli.cautious_cos_history_mode,
            include_prompt=cli.cautious_cos_history_include_prompt,
            max_chars=cli.cautious_cos_history_max_chars,
        )
        cos_history = synthesize_cautious_cos_history_from_support(
            support_rows=selected_support,
            mode=(cli.cos_history_mode or cli.cautious_cos_history_mode),
            include_prompt=(cli.cos_history_include_prompt or cli.cautious_cos_history_include_prompt),
            max_chars=(cli.cos_history_max_chars if cli.cos_history_max_chars > 0 else cli.cautious_cos_history_max_chars),
        )

        included_users += 1
        support_counts_used.append(len(selected_support))

        for row in sorted(user_query_rows, key=support_sort_key):
            meta = dict(row.get("meta", {}) or {})
            meta["support_budget"] = int(budget)
            meta["support_count_used"] = int(len(selected_support))
            meta["support_count_available"] = int(support_count)

            base_row = dict(row)
            base_row["meta"] = meta
            base_eval_rows.append(base_row)

            cos_row = dict(row)
            cos_row["user_history_text"] = cos_history
            cos_meta = dict(row.get("meta", {}) or {})
            cos_meta.update(meta)
            cos_meta["cos_history_support_count"] = len(selected_support)
            cos_row["meta"] = cos_meta
            cos_eval_rows.append(cos_row)

            icl_row = build_icl_query_row(
                row=row,
                train_module=train_module,
                tokenizer=tokenizer,
                train_args=train_args,
                support_rows=selected_support,
                cli=cli,
            )
            icl_meta = dict(icl_row.get("meta", {}) or {})
            icl_meta.update(meta)
            icl_row["meta"] = icl_meta
            icl_eval_rows.append(icl_row)

            if retriever is not None:
                query_text = build_rag_query_text(row)
                query_emb = retriever.encode([query_text], batch_size=1)[0]
                scores = torch.mv(candidate_embs, query_emb)
                topk = min(cli.icl_rag_topk, len(retrieval_candidates))
                top_idx = torch.topk(scores, k=topk).indices.tolist()
                rag_support = [retrieval_candidates[i] for i in top_idx]

                icl_rag_row = build_icl_query_row(
                    row=row,
                    train_module=train_module,
                    tokenizer=tokenizer,
                    train_args=train_args,
                    support_rows=rag_support,
                    cli=cli,
                )
                rag_meta = dict(icl_rag_row.get("meta", {}) or {})
                rag_meta.update(meta)
                rag_meta["icl_rag_support_count"] = len(rag_support)
                rag_meta["icl_rag_retrieval_scores"] = [float(scores[i]) for i in top_idx]
                icl_rag_row["meta"] = rag_meta
                icl_rag_eval_rows.append(icl_rag_row)

            cautious_cos_history_pairs = build_cautious_cos_history_pairs_from_support(
                support_rows=selected_support,
                max_items=int(getattr(train_args, "max_history_pairs", len(selected_support))),
                max_chars=cli.cautious_cos_history_max_chars,
            )

            steering_row = dict(row)
            if cli.steering_use_icl_prompt:
                icl_prompt_row = build_icl_query_row(
                    row=row,
                    train_module=train_module,
                    tokenizer=tokenizer,
                    train_args=train_args,
                    support_rows=selected_support,
                    cli=cli,
                )
                steering_row["prompt_text"] = icl_prompt_row["prompt_text"]
                steering_row.pop("messages", None)
                steering_row.pop("prompt", None)
            steering_row["user_history_text"] = cautious_cos_history
            steering_row["user_history_pairs"] = cautious_cos_history_pairs
            steering_meta = dict(row.get("meta", {}) or {})
            steering_meta.update(meta)
            if cli.steering_use_icl_prompt:
                steering_meta["steering_prompt_mode"] = "icl_exemplar"
            steering_row["meta"] = steering_meta
            steering_eval_rows.append(steering_row)

    info = {
        "budget": int(budget),
        "included_users": int(included_users),
        "skipped_users": int(skipped_users),
        "mean_support_count_used": float(sum(support_counts_used) / len(support_counts_used)) if support_counts_used else 0.0,
    }
    return base_eval_rows, icl_eval_rows, icl_rag_eval_rows, cos_eval_rows, steering_eval_rows, info


# -----------------------------------------------------------------------------
# Prompt/profile helpers
# -----------------------------------------------------------------------------


def get_prompt_text(row: Dict[str, Any], train_module: ModuleType, tokenizer: Any, train_args: SimpleNamespace) -> str:
    if "prompt_text" in row:
        return str(row["prompt_text"])
    return train_module.render_prompt(row, tokenizer, train_args)


def get_history_context_text(row: Dict[str, Any]) -> str:
    return normalize_ws(row.get("user_history_text", row.get("user_history", "")))


def build_history_conditioned_prompt(prompt_text: str, history_text: str, history_template: str) -> str:
    history_text = normalize_ws(history_text)
    if not history_text:
        return prompt_text
    return str(history_template).format(history=history_text, prompt=prompt_text).strip()


def build_history_conditioned_row(
    row: Dict[str, Any],
    train_module: ModuleType,
    tokenizer: Any,
    train_args: SimpleNamespace,
    history_template: str,
) -> Dict[str, Any]:
    base_prompt = get_prompt_text(row, train_module, tokenizer, train_args)
    history_text = get_history_context_text(row)
    new_row = dict(row)
    new_row["prompt_text"] = build_history_conditioned_prompt(base_prompt, history_text, history_template)
    new_row.pop("messages", None)
    new_row.pop("prompt", None)
    return new_row


def collect_single_token_ids(tokenizer: Any, texts: Sequence[str]) -> List[int]:
    ids = []
    for text in texts:
        tok_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(tok_ids) == 1:
            ids.append(int(tok_ids[0]))
    return sorted(set(ids))


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    cutoff = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < cutoff, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)
    remove = cumprobs > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return out


def sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True")
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)




def dominant_script(text: str) -> str:
    counts = defaultdict(int)
    for ch in str(text or ""):
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            continue
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            counts["cjk"] += 1
        elif 0x3040 <= code <= 0x30FF:
            counts["kana"] += 1
        elif 0xAC00 <= code <= 0xD7AF:
            counts["hangul"] += 1
        elif 0x0400 <= code <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x0600 <= code <= 0x06FF:
            counts["arabic"] += 1
        elif ch.isdigit():
            counts["digit"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["latin"] += 1
        else:
            counts["other"] += 1
    if not counts:
        return "empty"
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    if best == "kana":
        return "cjk"
    return best


def first_content_script(text: str) -> str:
    for ch in str(text or ""):
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            continue
        return dominant_script(ch)
    return "empty"


# -----------------------------------------------------------------------------
# Generation helpers
# -----------------------------------------------------------------------------


@torch.inference_mode()
def generate_base_responses_batch(
    base_model: nn.Module,
    tokenizer: Any,
    prompt_texts: Sequence[str],
    train_args: SimpleNamespace,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    enc = tokenizer(
        list(prompt_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(getattr(train_args, "max_prompt_len", 1024)),
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample and top_k > 0 else None,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    out = base_model.generate(**gen_kwargs)

    input_width = input_ids.size(1)
    results = []
    for i in range(out.size(0)):
        new_ids = out[i, input_width:]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    return results


@torch.inference_mode()
def generate_cos_responses_batch(
    base_model: nn.Module,
    tokenizer: Any,
    train_module: ModuleType,
    train_args: SimpleNamespace,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    cos_lambda: float,
    history_template: str,
) -> Tuple[List[str], Dict[str, float]]:
    prompt_texts = [get_prompt_text(r, train_module, tokenizer, train_args) for r in rows]
    history_texts = [get_history_context_text(r) for r in rows]
    ctx_prompt_texts = [build_history_conditioned_prompt(p, h, history_template) for p, h in zip(prompt_texts, history_texts)]

    base_enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(getattr(train_args, "max_prompt_len", 1024)),
        add_special_tokens=False,
    )
    ctx_enc = tokenizer(
        ctx_prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(getattr(train_args, "max_prompt_len", 1024)),
        add_special_tokens=False,
    )

    base_input = base_enc["input_ids"].to(device)
    base_attn = base_enc["attention_mask"].to(device)
    ctx_input = ctx_enc["input_ids"].to(device)
    ctx_attn = ctx_enc["attention_mask"].to(device)

    cur_base_in = base_input
    cur_ctx_in = ctx_input
    cur_base_attn = base_attn
    cur_ctx_attn = ctx_attn
    base_past = None
    ctx_past = None

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    finished = torch.zeros(base_input.size(0), dtype=torch.bool, device=device)
    generated: List[torch.Tensor] = []
    diag: Dict[str, List[float]] = defaultdict(list)

    for _ in range(max_new_tokens):
        base_out = base_model(
            input_ids=cur_base_in,
            attention_mask=cur_base_attn,
            past_key_values=base_past,
            use_cache=True,
            return_dict=True,
        )
        ctx_out = base_model(
            input_ids=cur_ctx_in,
            attention_mask=cur_ctx_attn,
            past_key_values=ctx_past,
            use_cache=True,
            return_dict=True,
        )
        base_past = base_out.past_key_values
        ctx_past = ctx_out.past_key_values

        base_next = base_out.logits[:, -1, :].float()
        ctx_next = ctx_out.logits[:, -1, :].float()
        steered = ctx_next + float(cos_lambda) * (ctx_next - base_next)

        if finished.any():
            steered[finished] = float("-inf")
            steered[finished, eos_id] = 0.0

        next_tok = sample_next_token(
            logits=steered,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        generated.append(next_tok)
        finished = finished | (next_tok == eos_id)

        delta = ctx_next - base_next
        diag["cos_delta_abs_mean"].append(float(delta.abs().mean().item()))
        diag["cos_delta_max_mean"].append(float(delta.abs().max(dim=-1).values.mean().item()))
        diag["cos_top1_change_rate"].append(float((base_next.argmax(dim=-1) != steered.argmax(dim=-1)).float().mean().item()))
        diag["cos_eos_top1_rate"].append(float((steered.argmax(dim=-1) == eos_id).float().mean().item()))

        if finished.all():
            break

        cur_base_in = next_tok.unsqueeze(-1)
        cur_ctx_in = next_tok.unsqueeze(-1)
        cur_base_attn = torch.cat([cur_base_attn, torch.ones((cur_base_attn.size(0), 1), dtype=cur_base_attn.dtype, device=device)], dim=1)
        cur_ctx_attn = torch.cat([cur_ctx_attn, torch.ones((cur_ctx_attn.size(0), 1), dtype=cur_ctx_attn.dtype, device=device)], dim=1)

    if generated:
        gen_ids = torch.stack(generated, dim=1)
    else:
        gen_ids = torch.empty((base_input.size(0), 0), dtype=torch.long, device=device)

    texts = [tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip() for i in range(gen_ids.size(0))]
    diag_summary = {k: float(sum(v) / len(v)) for k, v in diag.items() if v}
    return texts, diag_summary


def _encode_steering_for_generation(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    if hasattr(model, "_encode_context"):
        context, info = model._encode_context(batch)
        out = dict(context)
        out["info"] = info
        return out

    out = model._encode_user_prompt(batch)
    if not isinstance(out, tuple):
        raise RuntimeError("Unexpected _encode_user_prompt output type")
    if len(out) == 2:
        memory_vec, info = out
        return {"kind": "history_memory_only", "memory_vec": memory_vec, "info": info}
    if len(out) == 3:
        user_latent, prompt_latent, info = out
        return {"kind": "v3", "user_latent": user_latent, "prompt_latent": prompt_latent, "info": info}
    if len(out) == 4:
        profile_latent, history_latent, prompt_latent, info = out
        profile_available = batch.get("profile_present_mask", None)
        if profile_available is None:
            profile_available = torch.ones(profile_latent.size(0), dtype=torch.bool, device=profile_latent.device)
        else:
            profile_available = profile_available.bool()
        history_available = batch["history_pair_mask"].any(dim=-1)
        return {
            "kind": "v4",
            "profile_latent": profile_latent,
            "history_latent": history_latent,
            "prompt_latent": prompt_latent,
            "profile_available": profile_available,
            "history_available": history_available,
            "info": info,
        }
    raise RuntimeError(f"Unexpected _encode_user_prompt tuple length: {len(out)}")


def _find_batch_size_from_context(context: Dict[str, Any]) -> Optional[int]:
    for key in [
        "user_latent",
        "prompt_latent",
        "profile_latent",
        "history_latent",
        "memory_vec",
        "context_memory",
        "context_memory_mask",
        "has_profile",
        "has_history",
        "profile_available",
        "history_available",
    ]:
        val = context.get(key)
        if torch.is_tensor(val) and val.ndim >= 1:
            return int(val.size(0))
    return None


def _select_context_value(value: Any, example_idx: Optional[torch.Tensor], batch_size: Optional[int]) -> Any:
    if example_idx is None or batch_size is None or not torch.is_tensor(value):
        return value
    if value.ndim >= 1 and value.size(0) == batch_size:
        return value[example_idx]
    return value


def _normalize_availability_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None or not torch.is_tensor(x):
        return x
    return x.view(-1, 1) if x.ndim == 1 else x


def _build_steering_call_kwargs(
    fn: Any,
    last_hidden: torch.Tensor,
    context: Dict[str, Any],
    example_idx: Optional[torch.Tensor] = None,
    force_labels: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    batch_size = _find_batch_size_from_context(context)
    kwargs: Dict[str, Any] = {}

    def pick(name: str, *alts: str) -> Any:
        for key in (name,) + alts:
            if key in context:
                return _select_context_value(context[key], example_idx, batch_size)
        return None

    for pname in sig.parameters.keys():
        if pname == "self":
            continue
        if pname == "last_hidden":
            kwargs[pname] = last_hidden
        elif pname == "force_labels":
            kwargs[pname] = force_labels
        elif pname == "user_latent":
            kwargs[pname] = pick("user_latent", "profile_latent")
        elif pname == "prompt_latent":
            kwargs[pname] = pick("prompt_latent")
        elif pname == "fixed_memory_vec":
            kwargs[pname] = pick("memory_vec")
        elif pname == "context_memory":
            kwargs[pname] = pick("context_memory")
        elif pname == "context_memory_mask":
            kwargs[pname] = pick("context_memory_mask")
        elif pname == "has_profile":
            kwargs[pname] = pick("has_profile", "profile_available")
        elif pname == "has_history":
            kwargs[pname] = pick("has_history", "history_available")
        elif pname == "user_or_profile_latent":
            kwargs[pname] = pick("profile_latent", "user_latent")
        elif pname == "history_latent":
            kwargs[pname] = pick("history_latent")
        elif pname == "profile_available":
            kwargs[pname] = _normalize_availability_tensor(pick("profile_available", "has_profile"))
        elif pname == "history_available":
            kwargs[pname] = _normalize_availability_tensor(pick("history_available", "has_history"))

    return {k: v for k, v in kwargs.items() if v is not None}


@contextmanager
def patch_steering_forward(
    base_model: nn.Module,
    steering_model: nn.Module,
    enc_info: Dict[str, Any],
    skip_first_n_generated_tokens: int = 0,
    diagnostics: Optional[Dict[str, List[float]]] = None,
):
    original_forward = base_model.forward
    state = {"step_idx": 0}

    def _append_diag(name: str, value: float) -> None:
        if diagnostics is not None:
            diagnostics.setdefault(name, []).append(float(value))

    def patched_forward(*args, **kwargs):
        kwargs["return_dict"] = True
        kwargs["output_hidden_states"] = True
        outputs = original_forward(*args, **kwargs)

        if state["step_idx"] >= skip_first_n_generated_tokens:
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            p0 = next(steering_model.parameters(), None)
            if p0 is not None and last_hidden.dtype != p0.dtype:
                last_hidden = last_hidden.to(dtype=p0.dtype)
            bias_kwargs = _build_steering_call_kwargs(
                steering_model.compute_generation_logit_bias,
                last_hidden=last_hidden,
                context=enc_info,
                example_idx=None,
                force_labels=None,
            )
            delta_logits, step_diag = steering_model.compute_generation_logit_bias(**bias_kwargs)
            outputs.logits = outputs.logits.float()
            outputs.logits[:, -1, :] = outputs.logits[:, -1, :] + delta_logits.float()
            for key, value in step_diag.items():
                val = float(value.item()) if torch.is_tensor(value) else float(value)
                _append_diag(key, val)

        state["step_idx"] += 1
        return outputs

    base_model.forward = patched_forward
    try:
        yield
    finally:
        base_model.forward = original_forward


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def _merge_steering_diag_maps(
    diag_a: Dict[str, float], n_a: int, diag_b: Dict[str, float], n_b: int
) -> Dict[str, float]:
    if n_a == 0:
        return dict(diag_b)
    if n_b == 0:
        return dict(diag_a)
    keys = set(diag_a) | set(diag_b)
    out: Dict[str, float] = {}
    for k in keys:
        if k in diag_a and k in diag_b:
            out[k] = (diag_a[k] * n_a + diag_b[k] * n_b) / (n_a + n_b)
        elif k in diag_a:
            out[k] = float(diag_a[k])
        else:
            out[k] = float(diag_b[k])
    return out


@torch.inference_mode()
def generate_steering_responses_batch(
    model: nn.Module,
    tokenizer: Any,
    train_module: ModuleType,
    train_args: SimpleNamespace,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    skip_first_n_generated_tokens: int,
) -> Tuple[List[str], Dict[str, float]]:
    rows = list(rows)

    def _run_once(subrows: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, float]]:
        collator = train_module.PreferenceCollator(tokenizer, train_args)
        batch = collator(list(subrows))
        batch = move_batch_to_device(batch, device)
        enc_info = _encode_steering_for_generation(model, batch)

        prompt_texts = [get_prompt_text(row, train_module, tokenizer, train_args) for row in subrows]
        prompt_enc = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(getattr(train_args, "max_prompt_len", 1024)),
            add_special_tokens=False,
        )

        prompt_input_ids = prompt_enc["input_ids"].to(device)
        prompt_attention_mask = prompt_enc["attention_mask"].to(device)

        gen_kwargs = dict(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample and top_k > 0 else None,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        step_diags: Dict[str, List[float]] = {}
        with patch_steering_forward(
            model.base_model,
            model,
            enc_info=enc_info,
            skip_first_n_generated_tokens=skip_first_n_generated_tokens,
            diagnostics=step_diags,
        ):
            out = model.base_model.generate(**gen_kwargs)

        input_width = prompt_input_ids.size(1)
        texts = [
            tokenizer.decode(out[i, input_width:], skip_special_tokens=True).strip() for i in range(out.size(0))
        ]
        diag_summary = {f"steering_{k}": float(sum(v) / len(v)) for k, v in step_diags.items() if v}
        return texts, diag_summary

    try:
        return _run_once(rows)
    except BaseException as e:
        if not _is_cuda_oom(e) or len(rows) <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = len(rows) // 2
        texts_l, diag_l = generate_steering_responses_batch(
            model=model,
            tokenizer=tokenizer,
            train_module=train_module,
            train_args=train_args,
            rows=rows[:mid],
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            skip_first_n_generated_tokens=skip_first_n_generated_tokens,
        )
        texts_r, diag_r = generate_steering_responses_batch(
            model=model,
            tokenizer=tokenizer,
            train_module=train_module,
            train_args=train_args,
            rows=rows[mid:],
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            skip_first_n_generated_tokens=skip_first_n_generated_tokens,
        )
        merged_diag = _merge_steering_diag_maps(diag_l, mid, diag_r, len(rows) - mid)
        return texts_l + texts_r, merged_diag


def generate_rows_for_system(
    system_name: str,
    rows: Sequence[Dict[str, Any]],
    train_module: ModuleType,
    tokenizer: Any,
    train_args: SimpleNamespace,
    handles: Any,
    steering_model: Optional[nn.Module],
    lora_sft_model: Optional[nn.Module],
    device: torch.device,
    max_eval_rows: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    gen_batch_size: int,
    steering_skip_first_n: int,
    cos_lambda: float,
    cli: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows = list(rows)
    if max_eval_rows > 0:
        rows = rows[:max_eval_rows]
    if not rows:
        return [], {}

    rows = sorted(rows, key=lambda r: len(get_prompt_text(r, train_module, tokenizer, train_args).split()))
    row_batches = list(batched(rows, max(1, int(gen_batch_size))))

    preds: List[Dict[str, Any]] = []
    empty_so_far = 0
    done_so_far = 0
    diag_accum: Dict[str, float] = defaultdict(float)
    diag_count = 0
    gen_wall_time_total_sec = 0.0

    iterator = tqdm(row_batches, desc=f"Generate {system_name}", dynamic_ncols=True)
    for row_batch in iterator:
        prompt_texts = [get_prompt_text(row, train_module, tokenizer, train_args) for row in row_batch]
        _sync_device(device)
        t_gen0 = time.perf_counter()
        if system_name in {"base", "icl", "icl_rag"}:
            generated_batch = generate_base_responses_batch(
                base_model=handles.base_model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                train_args=train_args,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            batch_diag = {}
        elif system_name == "lora_sft":
            if lora_sft_model is None:
                raise ValueError("lora_sft_checkpoint is required for lora_sft generation")
            generated_batch = generate_base_responses_batch(
                base_model=lora_sft_model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                train_args=train_args,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            batch_diag = {}
        elif system_name in {"cos", "cos_history"}:
            generated_batch, batch_diag = generate_cos_responses_batch(
                base_model=handles.base_model,
                tokenizer=tokenizer,
                train_module=train_module,
                train_args=train_args,
                rows=row_batch,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                cos_lambda=cos_lambda,
                history_template=cli.cos_history_template,
            )
        elif system_name == "steer_distill":
            if steering_model is None:
                raise ValueError("steering_checkpoint is required for steer_distill generation")
            generated_batch, batch_diag = generate_steering_responses_batch(
                model=steering_model,
                tokenizer=tokenizer,
                train_module=train_module,
                train_args=train_args,
                rows=row_batch,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                skip_first_n_generated_tokens=steering_skip_first_n,
            )
        else:
            raise ValueError(f"Unknown system: {system_name}")
        _sync_device(device)
        batch_elapsed_sec = time.perf_counter() - t_gen0
        gen_wall_time_total_sec += batch_elapsed_sec
        n_batch = len(row_batch)
        gen_time_per_sample_sec = batch_elapsed_sec / float(max(1, n_batch))

        if batch_diag:
            for k, v in batch_diag.items():
                diag_accum[k] += float(v)
            diag_count += 1

        batch_empty = sum(1 for g in generated_batch if normalize_ws(g) == "")
        empty_so_far += batch_empty
        done_so_far += len(row_batch)
        iterator.set_postfix({"empty": f"{empty_so_far}/{done_so_far}", "empty_rate": f"{empty_so_far / max(1, done_so_far):.3f}"})

        for row, prompt_text, generated in zip(row_batch, prompt_texts, generated_batch):
            meta = row.get("meta", {}) or {}
            pred = {
                "user_id": meta.get("user_id", row.get("user_id", "unknown")),
                "conversation_id": meta.get("conversation_id", ""),
                "turn": meta.get("turn", None),
                "generated_datetime": meta.get("generated_datetime", ""),
                "support_budget": meta.get("support_budget", None),
                "system": system_name,
                "prompt_text": prompt_text,
                "user_history_text": row.get("user_history_text", ""),
                "user_history_pairs": row.get("user_history_pairs", []),
                "user_profile_text": row.get("user_profile_text", row.get("user_profile", "")),
                "generated": generated,
                "chosen": row.get("chosen", ""),
                "rejected": row.get("rejected", ""),
                "gen_time_sec": float(gen_time_per_sample_sec),
                "gen_wall_time_sec": float(gen_time_per_sample_sec),
            }
            preds.append(pred)

    diag_summary = {k: v / max(1, diag_count) for k, v in diag_accum.items()} if diag_count > 0 else {}
    n_preds = len(preds)
    diag_summary["gen_wall_time_total_sec"] = float(gen_wall_time_total_sec)
    diag_summary["gen_wall_time_mean_per_sample_sec"] = float(gen_wall_time_total_sec / max(1, n_preds))
    diag_summary["gen_time_sec"] = float(gen_wall_time_total_sec / max(1, n_preds))
    return preds, diag_summary


# -----------------------------------------------------------------------------
# Teacher-forced policy metrics
# -----------------------------------------------------------------------------


def flatten_target_positions(target_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nz = (target_ids != -100).nonzero(as_tuple=False)
    return nz[:, 0], nz[:, 1], target_ids[nz[:, 0], nz[:, 1]]


def compute_first_target_positions(target_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    example_idx: List[int] = []
    time_idx: List[int] = []
    for b in range(target_ids.size(0)):
        pos = (target_ids[b] != -100).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        example_idx.append(b)
        time_idx.append(int(pos[0, 0]))
    if not example_idx:
        empty = torch.empty((0,), dtype=torch.long, device=target_ids.device)
        return empty, empty
    return torch.tensor(example_idx, device=target_ids.device), torch.tensor(time_idx, device=target_ids.device)


def normalize_seq_logprobs(token_logprobs: torch.Tensor, example_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    token_logprobs = token_logprobs.float()
    seq_lp = torch.zeros(batch_size, device=token_logprobs.device, dtype=token_logprobs.dtype)
    seq_cnt = torch.zeros(batch_size, device=token_logprobs.device, dtype=token_logprobs.dtype)
    seq_lp.index_add_(0, example_idx, token_logprobs)
    seq_cnt.index_add_(0, example_idx, torch.ones_like(token_logprobs))
    return seq_lp / seq_cnt.clamp_min(1.0)


@torch.inference_mode()
def compute_base_side_policy_stats(
    base_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_ids: torch.Tensor,
) -> Dict[str, Any]:
    outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

    batch_size = input_ids.size(0)
    ex_idx, time_idx, labels = flatten_target_positions(target_ids)
    if labels.numel() == 0:
        return {
            "seq_lp": torch.zeros(batch_size, device=input_ids.device),
        }

    logits = outputs.logits[ex_idx, time_idx, :].float()
    token_lp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    return {
        "seq_lp": normalize_seq_logprobs(token_lp, ex_idx, batch_size),
    }


@torch.inference_mode()
def compute_cos_side_policy_stats(
    base_model: nn.Module,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    base_target_ids: torch.Tensor,
    ctx_input_ids: torch.Tensor,
    ctx_attention_mask: torch.Tensor,
    ctx_target_ids: torch.Tensor,
    cos_lambda: float,
) -> Dict[str, Any]:
    base_out = base_model(input_ids=base_input_ids, attention_mask=base_attention_mask, use_cache=False, return_dict=True)
    ctx_out = base_model(input_ids=ctx_input_ids, attention_mask=ctx_attention_mask, use_cache=False, return_dict=True)

    batch_size = base_input_ids.size(0)
    b_ex, b_time, b_labels = flatten_target_positions(base_target_ids)
    c_ex, c_time, c_labels = flatten_target_positions(ctx_target_ids)
    if b_labels.numel() == 0:
        return {
            "seq_lp": torch.zeros(batch_size, device=base_input_ids.device),
        }

    if b_labels.numel() != c_labels.numel() or not torch.equal(b_labels, c_labels):
        raise RuntimeError("Base/context target labels mismatch in CoS-history evaluation")

    base_logits = base_out.logits[b_ex, b_time, :].float()
    ctx_logits = ctx_out.logits[c_ex, c_time, :].float()
    steered_logits = ctx_logits + float(cos_lambda) * (ctx_logits - base_logits)
    token_lp = F.log_softmax(steered_logits, dim=-1).gather(-1, b_labels.unsqueeze(-1)).squeeze(-1)

    return {
        "seq_lp": normalize_seq_logprobs(token_lp, b_ex, batch_size),
    }


def _encode_steering_for_policy(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    if hasattr(model, "_encode_context"):
        context, info = model._encode_context(batch)
        out = dict(context)
        out["info"] = info
        return out

    out = model._encode_user_prompt(batch)
    if not isinstance(out, tuple):
        raise RuntimeError("Unexpected _encode_user_prompt output type")
    if len(out) == 2:
        memory_vec, info = out
        return {"kind": "history_memory_only", "memory_vec": memory_vec, "info": info}
    if len(out) == 3:
        user_latent, prompt_latent, info = out
        return {"kind": "v3", "user_latent": user_latent, "prompt_latent": prompt_latent, "info": info}
    if len(out) == 4:
        profile_latent, history_latent, prompt_latent, info = out
        profile_available = batch.get("profile_present_mask", None)
        if profile_available is None:
            profile_available = torch.ones(profile_latent.size(0), dtype=torch.bool, device=profile_latent.device)
        else:
            profile_available = profile_available.bool()
        history_available = batch["history_pair_mask"].any(dim=-1)
        return {
            "kind": "v4",
            "profile_latent": profile_latent,
            "history_latent": history_latent,
            "prompt_latent": prompt_latent,
            "profile_available": profile_available,
            "history_available": history_available,
            "info": info,
        }
    raise RuntimeError(f"Unexpected _encode_user_prompt tuple length: {len(out)}")


@torch.inference_mode()
def score_steering_side_full_policy(
    steering_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_ids: torch.Tensor,
    enc_info: Dict[str, Any],
) -> Dict[str, Any]:
    batch_size = input_ids.size(0)
    hidden = steering_model._backbone_forward(input_ids=input_ids, attention_mask=attention_mask)

    ex_idx, time_idx, labels = flatten_target_positions(target_ids)
    if labels.numel() == 0:
        return {
            "seq_lp": torch.zeros(batch_size, device=input_ids.device),
            "total_tokens": 0,
            "covered_tokens": 0,
            "support_k_sum": 0.0,
            "entropy_sum": 0.0,
        }

    selected_hidden = hidden[ex_idx, time_idx, :]
    call_kwargs = _build_steering_call_kwargs(
        steering_model.compute_support_reranking,
        last_hidden=selected_hidden,
        context=enc_info,
        example_idx=ex_idx,
        force_labels=None,
    )
    info = steering_model.compute_support_reranking(**call_kwargs)

    if hasattr(steering_model, "_exact_logprob_terms"):
        exact = steering_model._exact_logprob_terms(
            base_logp=info["base_logp"],
            support_idx=info["support_idx"],
            support_probs=info["support_probs"],
            support_mask=info["support_mask"],
            delta_support=info["delta_support"],
            labels=labels,
        )
        token_lp = exact["steered_label_lp"]
        covered = exact["has_label"]
    else:
        steered_logits = info["base_logits"].clone()
        steered_logits.scatter_add_(dim=-1, index=info["support_idx"], src=info["delta_support"].to(steered_logits.dtype))
        token_lp = F.log_softmax(steered_logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        label_match = ((info["support_idx"] == labels.unsqueeze(-1)) & info["support_mask"])
        covered = label_match.any(dim=-1)

    return {
        "seq_lp": normalize_seq_logprobs(token_lp, ex_idx, batch_size),
        "total_tokens": int(labels.numel()),
        "covered_tokens": int(covered.sum().item()),
        "support_k_sum": float(info["k_counts"].float().sum().item()),
        "entropy_sum": float(info["norm_entropy"].float().sum().item()),
    }


def evaluate_base_policy_rows(
    base_model: nn.Module,
    tokenizer: Any,
    train_module: ModuleType,
    train_args: SimpleNamespace,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if not rows:
        return {}
    collator = train_module.PreferenceCollator(tokenizer, train_args)
    loader = DataLoader(ListDataset(rows), batch_size=max(1, int(batch_size)), shuffle=False, collate_fn=collator)

    total_examples = 0
    total_pref_correct = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        chosen = compute_base_side_policy_stats(
            base_model=base_model,
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            target_ids=batch["chosen_target_ids"],
        )
        rejected = compute_base_side_policy_stats(
            base_model=base_model,
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            target_ids=batch["rejected_target_ids"],
        )
        total_examples += int(chosen["seq_lp"].numel())
        total_pref_correct += int((chosen["seq_lp"] > rejected["seq_lp"]).sum().item())

    return {
        "policy_preference_acc": float(total_pref_correct / max(1, total_examples)),
        "policy_coverage": 1.0,
    }


def evaluate_cos_policy_rows(
    base_model: nn.Module,
    tokenizer: Any,
    train_module: ModuleType,
    train_args: SimpleNamespace,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    cos_lambda: float,
    history_template: str,
) -> Dict[str, float]:
    if not rows:
        return {}
    collator = train_module.PreferenceCollator(tokenizer, train_args)

    total_examples = 0
    total_pref_correct = 0

    for row_batch in batched(list(rows), max(1, int(batch_size))):
        base_batch = collator(list(row_batch))
        ctx_rows = [build_history_conditioned_row(r, train_module, tokenizer, train_args, history_template) for r in row_batch]
        ctx_batch = collator(ctx_rows)
        base_batch = move_batch_to_device(base_batch, device)
        ctx_batch = move_batch_to_device(ctx_batch, device)

        chosen = compute_cos_side_policy_stats(
            base_model=base_model,
            base_input_ids=base_batch["chosen_input_ids"],
            base_attention_mask=base_batch["chosen_attention_mask"],
            base_target_ids=base_batch["chosen_target_ids"],
            ctx_input_ids=ctx_batch["chosen_input_ids"],
            ctx_attention_mask=ctx_batch["chosen_attention_mask"],
            ctx_target_ids=ctx_batch["chosen_target_ids"],
            cos_lambda=cos_lambda,
        )
        rejected = compute_cos_side_policy_stats(
            base_model=base_model,
            base_input_ids=base_batch["rejected_input_ids"],
            base_attention_mask=base_batch["rejected_attention_mask"],
            base_target_ids=base_batch["rejected_target_ids"],
            ctx_input_ids=ctx_batch["rejected_input_ids"],
            ctx_attention_mask=ctx_batch["rejected_attention_mask"],
            ctx_target_ids=ctx_batch["rejected_target_ids"],
            cos_lambda=cos_lambda,
        )
        total_examples += int(chosen["seq_lp"].numel())
        total_pref_correct += int((chosen["seq_lp"] > rejected["seq_lp"]).sum().item())

    return {
        "policy_preference_acc": float(total_pref_correct / max(1, total_examples)),
        "policy_coverage": 1.0,
    }


@torch.inference_mode()
def run_steering_rerank_sanity_check(
    steering_model: nn.Module,
    tokenizer: Any,
    train_module: ModuleType,
    train_args: SimpleNamespace,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if not rows:
        return {}
    collator = train_module.PreferenceCollator(tokenizer, train_args)
    sample_rows = list(rows)[: max(1, int(batch_size))]
    batch = move_batch_to_device(collator(sample_rows), device)
    enc_info = _encode_steering_for_policy(steering_model, batch)

    input_ids = batch["chosen_input_ids"]
    attention_mask = batch["chosen_attention_mask"]
    target_ids = batch["chosen_target_ids"]
    hidden = steering_model._backbone_forward(input_ids=input_ids, attention_mask=attention_mask)
    ex_idx, time_idx, labels = flatten_target_positions(target_ids)
    if labels.numel() == 0:
        return {}

    selected_hidden = hidden[ex_idx, time_idx, :]
    call_kwargs = _build_steering_call_kwargs(
        steering_model.compute_support_reranking,
        last_hidden=selected_hidden,
        context=enc_info,
        example_idx=ex_idx,
        force_labels=None,
    )
    info_a = steering_model.compute_support_reranking(**call_kwargs)
    info_b = steering_model.compute_support_reranking(**call_kwargs)

    delta_diff = (info_a["delta_support"] - info_b["delta_support"]).abs().max()
    steered_diff = (info_a["steered_support_logits"] - info_b["steered_support_logits"]).abs().max()

    exact_a = steering_model._exact_logprob_terms(
        base_logp=info_a["base_logp"],
        support_idx=info_a["support_idx"],
        support_probs=info_a["support_probs"],
        support_mask=info_a["support_mask"],
        delta_support=info_a["delta_support"],
        labels=labels,
    )
    exact_b = steering_model._exact_logprob_terms(
        base_logp=info_b["base_logp"],
        support_idx=info_b["support_idx"],
        support_probs=info_b["support_probs"],
        support_mask=info_b["support_mask"],
        delta_support=info_b["delta_support"],
        labels=labels,
    )
    eos_id = int(getattr(train_args, "eos_token_id", -1))
    content_mask = torch.ones_like(labels, dtype=torch.bool)
    if eos_id >= 0 and bool(int(getattr(train_args, "ignore_eos_in_loss", 1))):
        content_mask = labels != eos_id
    if content_mask.any():
        ce_a = -exact_a["steered_label_lp"][content_mask].mean()
        ce_b = -exact_b["steered_label_lp"][content_mask].mean()
        ce_diff = (ce_a - ce_b).abs()
    else:
        ce_diff = delta_diff.new_zeros(())

    return {
        "sanity_delta_support_max_abs_diff": float(delta_diff.item()),
        "sanity_steered_support_logits_max_abs_diff": float(steered_diff.item()),
        "sanity_chosen_ce_loss_abs_diff": float(ce_diff.item()),
    }


@torch.inference_mode()
def evaluate_steering_policy_rows(
    steering_model: nn.Module,
    tokenizer: Any,
    train_module: ModuleType,
    train_args: SimpleNamespace,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if not rows:
        return {}
    collator = train_module.PreferenceCollator(tokenizer, train_args)
    loader = DataLoader(ListDataset(rows), batch_size=max(1, int(batch_size)), shuffle=False, collate_fn=collator)

    total_tokens = 0
    total_examples = 0
    total_pref_correct = 0
    total_covered = 0
    total_support_k_sum = 0.0
    total_entropy_sum = 0.0

    steering_model.eval()
    if hasattr(steering_model, "set_schedule_fraction"):
        steering_model.set_schedule_fraction(1.0)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        enc_info = _encode_steering_for_policy(steering_model, batch)

        chosen = score_steering_side_full_policy(
            steering_model=steering_model,
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            target_ids=batch["chosen_target_ids"],
            enc_info=enc_info,
        )
        rejected = score_steering_side_full_policy(
            steering_model=steering_model,
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            target_ids=batch["rejected_target_ids"],
            enc_info=enc_info,
        )
        total_tokens += chosen["total_tokens"]
        total_examples += int(chosen["seq_lp"].numel())
        total_pref_correct += int((chosen["seq_lp"] > rejected["seq_lp"]).sum().item())
        total_covered += chosen["covered_tokens"]
        total_support_k_sum += chosen["support_k_sum"]
        total_entropy_sum += chosen["entropy_sum"]

    if total_examples == 0:
        return {}

    return {
        "policy_preference_acc": float(total_pref_correct / max(1, total_examples)),
        "policy_coverage": float(total_covered / max(1, total_tokens)),
        "policy_support_size_mean": float(total_support_k_sum / max(1, total_tokens)),
        "policy_entropy_mean": float(total_entropy_sum / max(1, total_tokens)),
    }


# -----------------------------------------------------------------------------
# Language drift / generation metrics
# -----------------------------------------------------------------------------


def summarize_generation_system(
    predictions: Sequence[Dict[str, Any]],
    bertscore_f1: Sequence[float],
    rouge1_f1: Sequence[float],
    rougeL_f1: Sequence[float],
) -> Dict[str, float]:
    if not predictions:
        return {"n_rows": 0, "n_users": 0}

    users: Dict[str, List[int]] = defaultdict(list)
    drift_flags, empty_flags = [], []

    for p in predictions:
        users[p["user_id"]].append(1)
        empty_flags.append(1.0 if normalize_ws(p["generated"]) == "" else 0.0)
        ref_lang = dominant_script(p.get("chosen", "") or p.get("prompt_text", ""))
        gen_lang = dominant_script(p.get("generated", ""))
        drift = float(ref_lang not in {"empty", "other"} and gen_lang not in {"empty", "other"} and ref_lang != gen_lang)
        drift_flags.append(drift)

    mean_bertscore_f1 = sum(float(x) for x in bertscore_f1) / len(bertscore_f1)
    mean_rouge1_f1 = sum(float(x) for x in rouge1_f1) / len(rouge1_f1)
    mean_rougeL_f1 = sum(float(x) for x in rougeL_f1) / len(rougeL_f1)
    mean_gen_len = sum(len(normalize_ws(p["generated"]).split()) for p in predictions) / len(predictions)

    return {
        "n_rows": int(len(predictions)),
        "n_users": int(len(users)),
        "bertscore_f1": float(mean_bertscore_f1),
        "rouge1_f1": float(mean_rouge1_f1),
        "rougeL_f1": float(mean_rougeL_f1),
        "mean_gen_len": float(mean_gen_len),
        "empty_rate": float(sum(empty_flags) / len(empty_flags)),
        "drift_rate": float(sum(drift_flags) / len(drift_flags)),
    }


def attach_generation_metrics(
    predictions: List[Dict[str, Any]],
    bert_scorer: Any,
    rouge_metric: Any,
    bertscore_batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if not predictions:
        return predictions, {"n_rows": 0}

    cands = [normalize_ws(p["generated"]) for p in predictions]
    pos_refs = [normalize_ws(p["chosen"]) for p in predictions]

    empty_idx = [i for i, c in enumerate(cands) if c == ""]
    nonempty_idx = [i for i, c in enumerate(cands) if c != ""]
    print(f"[metric] empty generations: {len(empty_idx)}/{len(cands)}")

    f1_list = [0.0] * len(cands)
    if nonempty_idx:
        score_cands = [cands[i] for i in nonempty_idx]
        score_refs = [pos_refs[i] for i in nonempty_idx]
        _, _, f1 = bert_scorer.score(score_cands, score_refs, batch_size=bertscore_batch_size, verbose=False)
        nonempty_f1 = f1.detach().float().cpu().tolist()
        for i, v in zip(nonempty_idx, nonempty_f1):
            f1_list[i] = float(v)

    rouge1_list: List[float] = []
    rougeL_list: List[float] = []
    for cand, ref in zip(cands, pos_refs):
        scores = rouge_metric.score(ref, cand)
        rouge1_list.append(float(scores["rouge1"].fmeasure))
        rougeL_list.append(float(scores["rougeL"].fmeasure))

    for i in range(len(predictions)):
        ref_lang = dominant_script(predictions[i].get("chosen", "") or predictions[i].get("prompt_text", ""))
        gen_lang = dominant_script(predictions[i]["generated"])
        predictions[i]["bertscore_f1"] = float(f1_list[i])
        predictions[i]["rouge1_f1"] = float(rouge1_list[i])
        predictions[i]["rougeL_f1"] = float(rougeL_list[i])
        predictions[i]["generated_language"] = gen_lang
        predictions[i]["reference_language"] = ref_lang
        predictions[i]["language_drifted"] = bool(ref_lang not in {"empty", "other"} and gen_lang not in {"empty", "other"} and ref_lang != gen_lang)

    metrics = summarize_generation_system(predictions, f1_list, rouge1_list, rougeL_list)
    return predictions, metrics


def print_summary(title: str, metrics: Dict[str, float]) -> None:
    pretty = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(f"\n[{title}]\n{pretty}\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    metric_device = torch.device(cli.metric_device) if cli.metric_device else device

    train_module = dynamic_import_module(cli.train_script)

    ckpt_path: Optional[Path] = None
    steering_train_args: Optional[SimpleNamespace] = None
    if cli.steering_checkpoint:
        ckpt_path, args_path = resolve_steering_checkpoint_paths(cli.steering_checkpoint)
        steering_train_args = load_train_namespace(args_path, cli)

    lora_ckpt_dir: Optional[Path] = None
    lora_train_args: Optional[SimpleNamespace] = None
    if cli.lora_sft_checkpoint:
        lora_ckpt_dir, lora_args_path = resolve_lora_checkpoint_dir(cli.lora_sft_checkpoint)
        lora_train_args = load_train_namespace(lora_args_path, cli)

    if steering_train_args is not None:
        train_args = steering_train_args
        if lora_train_args is not None:
            maybe_warn_runtime_mismatch("steer_distill", steering_train_args, "lora_sft", lora_train_args)
    elif lora_train_args is not None:
        train_args = lora_train_args
    else:
        train_args = build_namespace_from_cli(cli)

    support_rows = load_jsonl(cli.support_jsonl)
    query_rows = load_jsonl(cli.query_jsonl)
    save_dir = maybe_mkdir(cli.save_dir)

    tokenizer, handles = train_module.load_base_handles(train_args, device)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    steering_model = None
    if ckpt_path is not None:
        steering_cls = getattr(train_module, "ContextSteeringDistillModel", None)
        if steering_cls is None:
            raise AttributeError(f"{cli.train_script} must define ContextSteeringDistillModel")
        steering_model = steering_cls(handles, train_args).to(device)
        state = torch.load(ckpt_path, map_location="cpu")
        steering_model.load_steering_state_dict(state)
        dt_name = str(getattr(train_args, "dtype", "float32"))
        if dt_name in ("bfloat16", "float16"):
            steering_model.to(train_module.str_to_dtype(dt_name))
        if hasattr(steering_model, "set_schedule_fraction"):
            steering_model.set_schedule_fraction(1.0)
        steering_model.eval()
        print(f"[loaded] steering checkpoint: {ckpt_path}")

    lora_sft_model = None
    if lora_ckpt_dir is not None:
        lora_sft_model = load_lora_sft_model(
            train_module=train_module,
            train_args=lora_train_args or train_args,
            checkpoint_dir=lora_ckpt_dir,
            device=device,
        )
        print(f"[loaded] lora_sft checkpoint: {lora_ckpt_dir}")

    use_rag = cli.icl_rag or ("icl_rag" in cli.systems)
    rag_retriever = DenseRetriever(cli.icl_rag_retriever_model, metric_device) if use_rag else None

    if rouge_scorer is None:
        raise ImportError("rouge-score is not installed. Please install `rouge-score`.")
    rouge_metric = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    if BERTScorer is None:
        print("[warn] bert_score is not installed; using lexical-overlap fallback metrics.")
        bert_scorer = _FallbackBERTScorer()
    elif cli.bertscore_model_type:
        bert_scorer = BERTScorer(model_type=cli.bertscore_model_type, device=str(metric_device))
    else:
        bert_scorer = BERTScorer(lang=cli.bertscore_lang, device=str(metric_device))

    summary: Dict[str, Any] = {
        "support_jsonl": str(Path(cli.support_jsonl).resolve()),
        "query_jsonl": str(Path(cli.query_jsonl).resolve()),
        "steering_checkpoint": str(ckpt_path.resolve()) if (ckpt_path is not None and cli.steering_checkpoint) else "",
        "lora_sft_checkpoint": str(lora_ckpt_dir.resolve()) if lora_ckpt_dir is not None else "",
        "support_budgets": [1] if cli.no_support_budget_loop else list(cli.support_budgets),
        "no_support_budget_loop": bool(cli.no_support_budget_loop),
        "systems": list(cli.systems),
        "cos_lambda": float(cli.cos_lambda),
        "policy_metric_definition": "policy_preference_acc uses mean target-token logprob per response for base / icl / cos / steer_distill / lora_sft.",
        "bertscore_lang": cli.bertscore_lang,
        "bertscore_model_type": cli.bertscore_model_type,
        "budgets": {},
    }

    if cli.no_support_budget_loop:
        budget_list = [1]
        strict_for_build = False
    else:
        budget_list = list(cli.support_budgets)
        strict_for_build = cli.strict_support_budget

    for budget in budget_list:
        base_rows, icl_rows, icl_rag_rows, cos_rows, steering_rows, build_info = build_budget_rows(
            train_module=train_module,
            tokenizer=tokenizer,
            train_args=train_args,
            support_rows=support_rows,
            query_rows=query_rows,
            budget=budget,
            strict_support_budget=strict_for_build,
            support_selection=cli.support_selection,
            cli=cli,
            retriever=rag_retriever,
        )
        print_summary(f"budget={budget} build", build_info)

        row_map = {
            "base": base_rows,
            "icl": icl_rows,
            "icl_rag": icl_rag_rows,
            "cos": cos_rows,
            "cos_history": cos_rows,
            "steer_distill": steering_rows,
            "lora_sft": icl_rows,
        }
        steering_sanity_metrics: Dict[str, float] = {}
        if cli.sanity_check_rerank_repeat and steering_model is not None and steering_rows:
            steering_sanity_metrics = run_steering_rerank_sanity_check(
                steering_model=steering_model,
                tokenizer=tokenizer,
                train_module=train_module,
                train_args=train_args,
                rows=steering_rows[: cli.max_eval_rows] if cli.max_eval_rows > 0 else steering_rows,
                device=device,
                batch_size=cli.policy_eval_batch_size,
            )
            if steering_sanity_metrics:
                print_summary(f"budget={budget} steering_sanity", steering_sanity_metrics)

        budget_summary: Dict[str, Any] = {"build_info": build_info, "systems": {}}

        for system_name in cli.systems:
            if system_name == "icl_rag" and rag_retriever is None:
                print(f"[skip] {system_name} requested but retriever is disabled.")
                continue
            if system_name == "steer_distill" and steering_model is None:
                print(f"[skip] {system_name} requested but no steering checkpoint was given.")
                continue
            if system_name == "lora_sft" and lora_sft_model is None:
                print(f"[skip] {system_name} requested but no lora_sft checkpoint was given.")
                continue

            rows = row_map[system_name]
            predictions, gen_diag = generate_rows_for_system(
                system_name=system_name,
                rows=rows,
                train_module=train_module,
                tokenizer=tokenizer,
                train_args=train_args,
                handles=handles,
                steering_model=steering_model,
                lora_sft_model=lora_sft_model,
                device=device,
                max_eval_rows=cli.max_eval_rows,
                max_new_tokens=cli.max_new_tokens,
                do_sample=cli.do_sample,
                temperature=cli.temperature,
                top_p=cli.top_p,
                top_k=cli.top_k,
                gen_batch_size=cli.gen_batch_size,
                steering_skip_first_n=cli.steering_skip_first_n,
                cos_lambda=cli.cos_lambda,
                cli=cli,
            )
            predictions, gen_metrics = attach_generation_metrics(
                predictions=predictions,
                bert_scorer=bert_scorer,
                rouge_metric=rouge_metric,
                bertscore_batch_size=cli.bertscore_batch_size,
            )

            if system_name in {"base", "icl", "icl_rag"}:
                policy_metrics = evaluate_base_policy_rows(
                    base_model=handles.base_model,
                    tokenizer=tokenizer,
                    train_module=train_module,
                    train_args=train_args,
                    rows=rows[: cli.max_eval_rows] if cli.max_eval_rows > 0 else rows,
                    device=device,
                    batch_size=cli.policy_eval_batch_size,
                )
            elif system_name == "lora_sft":
                policy_metrics = evaluate_base_policy_rows(
                    base_model=lora_sft_model,
                    tokenizer=tokenizer,
                    train_module=train_module,
                    train_args=train_args,
                    rows=rows[: cli.max_eval_rows] if cli.max_eval_rows > 0 else rows,
                    device=device,
                    batch_size=cli.policy_eval_batch_size,
                )
            elif system_name in {"cos", "cos_history"}:
                policy_metrics = evaluate_cos_policy_rows(
                    base_model=handles.base_model,
                    tokenizer=tokenizer,
                    train_module=train_module,
                    train_args=train_args,
                    rows=rows[: cli.max_eval_rows] if cli.max_eval_rows > 0 else rows,
                    device=device,
                    batch_size=cli.policy_eval_batch_size,
                    cos_lambda=cli.cos_lambda,
                    history_template=cli.cos_history_template,
                )
            else:
                policy_metrics = evaluate_steering_policy_rows(
                    steering_model=steering_model,
                    tokenizer=tokenizer,
                    train_module=train_module,
                    train_args=train_args,
                    rows=rows[: cli.max_eval_rows] if cli.max_eval_rows > 0 else rows,
                    device=device,
                    batch_size=cli.policy_eval_batch_size,
                )

            metrics = {**gen_metrics, **policy_metrics, **gen_diag}
            if system_name == "steer_distill" and steering_sanity_metrics:
                metrics = {**metrics, **steering_sanity_metrics}
            budget_summary["systems"][system_name] = metrics
            print_summary(f"budget={budget} system={system_name}", metrics)

            if save_dir is not None:
                save_jsonl(save_dir / f"predictions_budget{budget}_{system_name}.jsonl", predictions)

            if cli.print_examples > 0 and predictions:
                print(f"\n[examples] budget={budget} system={system_name}")
                for row in predictions[: cli.print_examples]:
                    print("-" * 80)
                    print("PROMPT:", row["prompt_text"])
                    print("GENERATED:", row["generated"])
                    print("CHOSEN:", row["chosen"])
                    print("REJECTED:", row["rejected"])

        summary["budgets"][str(budget)] = budget_summary

    if save_dir is not None:
        with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary("overall summary", summary)


if __name__ == "__main__":
    main()
