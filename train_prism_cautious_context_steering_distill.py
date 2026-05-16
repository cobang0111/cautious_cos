#!/usr/bin/env python3
"""
Train the PRISM cautious context-steering distillation adapter.

The base LM stays frozen. ICL is used as a teacher only during training, and the
learned adapter steers only when the teacher appears helpful; otherwise it
preserves the base distribution and trains an explicit abstention gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm

from history_icl_utils import (
    build_history_exemplar_prefix,
    build_history_exemplar_prompt,
    extract_history_pairs,
    stringify_messages,
)

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


NEG_INF = -1e30


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train PRISM cautious context-steering distillation adapter")

    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--valid_jsonl", type=str, default="")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--resume_steering", type=str, default="")

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--prompt_template", type=str, default="User request:\n{prompt}\n\nAssistant:\n")
    ap.add_argument("--append_eos", action="store_true")

    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_context_len", type=int, default=2048)
    ap.add_argument("--max_answer_len", type=int, default=512)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_history_pairs", type=int, default=4)
    ap.add_argument("--history_example_max_chars", type=int, default=512)
    ap.add_argument("--require_history", type=int, default=0, choices=[0, 1])

    ap.add_argument("--teacher_icl_intro", type=str, default="")
    ap.add_argument("--teacher_icl_exemplar_mode", type=str, default="chosen_only", choices=["chosen_only", "pairwise"])
    ap.add_argument("--teacher_icl_include_prompt", type=int, default=1, choices=[0, 1])
    ap.add_argument("--teacher_icl_include_user_profile", type=int, default=0, choices=[0, 1])
    ap.add_argument("--context_intro", type=str, default="")

    ap.add_argument("--adapter_dim", type=int, default=512)
    ap.add_argument("--candidate_dim", type=int, default=512)
    ap.add_argument("--lambda_head_hidden", type=int, default=256)
    ap.add_argument("--module_dropout", type=float, default=0.05)
    ap.add_argument("--context_pool", type=str, default="last", choices=["last", "mean", "attn"])
    ap.add_argument("--cross_attn_memory_tokens", type=int, default=64)
    ap.add_argument("--cross_attn_heads", type=int, default=8)

    ap.add_argument("--support_min_k", type=int, default=4)
    ap.add_argument("--support_max_k", type=int, default=64)
    ap.add_argument("--support_top_p_min", type=float, default=0.70)
    ap.add_argument("--support_top_p_max", type=float, default=0.95)
    ap.add_argument("--distill_teacher_top_k", type=int, default=32)
    ap.add_argument("--oracle_lambda_min", type=float, default=-1.5)
    ap.add_argument("--oracle_lambda_max", type=float, default=1.5)
    ap.add_argument("--oracle_lambda_bisect_steps", type=int, default=24)
    ap.add_argument("--distill_min_delta_var", type=float, default=1e-5)
    ap.add_argument("--score_scale", type=float, default=1.0)
    ap.add_argument("--score_scale_warmup_ratio", type=float, default=0.10)
    ap.add_argument("--score_clip", type=float, default=3.0)
    ap.add_argument("--zero_mean_scores", type=int, default=1, choices=[0, 1])
    ap.add_argument("--train_force_label_in_support", type=int, default=1, choices=[0, 1])
    ap.add_argument("--train_force_rejected_label_in_support", type=int, default=0, choices=[0, 1])
    ap.add_argument("--personalize_eos", type=int, default=0, choices=[0, 1])
    ap.add_argument("--ignore_eos_in_loss", type=int, default=1, choices=[0, 1])
    ap.add_argument("--length_normalize", action="store_true")

    ap.add_argument("--distill_kl_weight", type=float, default=1.0)
    ap.add_argument("--distill_delta_weight", type=float, default=0.5)
    ap.add_argument("--distill_lambda_weight", type=float, default=0.2)
    ap.add_argument("--chosen_ce_weight", type=float, default=0.5)
    ap.add_argument("--pairwise_pref_weight", type=float, default=0.0)
    ap.add_argument("--pairwise_beta", type=float, default=1.0)
    ap.add_argument("--chosen_gain_hinge_weight", type=float, default=0.0)
    ap.add_argument("--chosen_gain_margin", type=float, default=0.0)
    ap.add_argument("--hard_pair_margin", type=float, default=0.0)
    ap.add_argument("--hard_pair_temperature", type=float, default=0.25)
    ap.add_argument("--hard_pair_min_weight", type=float, default=0.25)
    ap.add_argument("--adv_margin_tok", type=float, default=0.05)
    ap.add_argument("--adv_temp_tok", type=float, default=0.10)
    ap.add_argument("--gate_loss_weight", type=float, default=0.20)
    ap.add_argument("--base_preserve_weight", type=float, default=0.50)
    ap.add_argument("--gate_init_bias", type=float, default=-2.0)
    ap.add_argument("--gate_hidden", type=int, default=256)

    ap.add_argument("--per_device_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--optimizer", type=str, default="hybrid_muon", choices=["hybrid_muon", "adamw"])
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--muon_ns_steps", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--keep_last_n", type=int, default=3)

    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="context-steering-distill")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_run_name", type=str, default="")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()
    args.distill_variant = "cautious_context_steering"
    return args


def ddp_setup() -> Tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def ddp_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_scalar(value: float, device: torch.device, average: bool = True) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= dist.get_world_size()
    return float(tensor.item())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str_to_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(name)]


def render_prompt(record: Dict[str, Any], tokenizer: Any, args: argparse.Namespace | SimpleNamespace) -> str:
    if "prompt_text" in record:
        return str(record["prompt_text"])
    if "messages" in record:
        if not bool(getattr(args, "use_chat_template", False)):
            return stringify_messages(record["messages"]) + "\n[assistant] "
        try:
            return tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True)
    if "prompt" in record:
        return str(getattr(args, "prompt_template", "User request:\n{prompt}\n\nAssistant:\n")).format(
            prompt=str(record["prompt"])
        )
    raise KeyError("Each record must contain one of: prompt_text, messages, prompt")


def render_user_context(record: Dict[str, Any], args: argparse.Namespace | SimpleNamespace) -> str:
    pairs = extract_history_pairs(record)
    prefix = build_history_exemplar_prefix(
        history_pairs=pairs,
        intro=str(getattr(args, "context_intro", "")),
        exemplar_mode=str(getattr(args, "teacher_icl_exemplar_mode", "chosen_only")),
        include_prompt=bool(int(getattr(args, "teacher_icl_include_prompt", 1))),
        include_user_profile=bool(int(getattr(args, "teacher_icl_include_user_profile", 0))),
        user_profile=record.get("user_profile_text", record.get("user_profile", "")),
        max_chars=int(getattr(args, "history_example_max_chars", 512)),
        max_items=int(getattr(args, "max_history_pairs", 4)),
    )
    if prefix:
        return prefix
    raw = record.get("user_history_text", record.get("user_history", ""))
    if isinstance(raw, list):
        raw = stringify_messages(raw)
    return str(raw or "").strip()


def render_train_prompt(record: Dict[str, Any], tokenizer: Any, args: argparse.Namespace | SimpleNamespace) -> str:
    return render_prompt(record, tokenizer, args)


def render_teacher_prompt(record: Dict[str, Any], tokenizer: Any, args: argparse.Namespace | SimpleNamespace) -> str:
    base_prompt = render_prompt(record, tokenizer, args)
    pairs = extract_history_pairs(record)
    return build_history_exemplar_prompt(
        base_prompt=base_prompt,
        history_pairs=pairs,
        intro=str(getattr(args, "teacher_icl_intro", "")),
        exemplar_mode=str(getattr(args, "teacher_icl_exemplar_mode", "chosen_only")),
        include_prompt=bool(int(getattr(args, "teacher_icl_include_prompt", 1))),
        include_user_profile=bool(int(getattr(args, "teacher_icl_include_user_profile", 0))),
        user_profile=record.get("user_profile_text", record.get("user_profile", "")),
        max_chars=int(getattr(args, "history_example_max_chars", 512)),
        max_items=int(getattr(args, "max_history_pairs", 4)),
    )


class JsonlDataset(Dataset):
    def __init__(self, path: str, require_history: bool = False):
        self.records: List[Dict[str, Any]] = []
        skipped_no_history = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if require_history and not extract_history_pairs(rec):
                    skipped_no_history += 1
                    continue
                self.records.append(rec)
        if not self.records:
            raise ValueError(f"No records found in {path}")
        if skipped_no_history and is_main_process():
            print(f"[dataset] skipped_no_history={skipped_no_history} kept={len(self.records)} path={path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def pad_2d(seqs: Sequence[Sequence[int]], pad_value: int) -> torch.Tensor:
    max_len = max(max((len(x) for x in seqs), default=0), 1)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(seqs):
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


class PreferenceCollator:
    def __init__(self, tokenizer: Any, args: argparse.Namespace | SimpleNamespace):
        self.tokenizer = tokenizer
        self.args = args
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer needs either pad_token_id or eos_token_id")
        self.eos_text = tokenizer.eos_token if (bool(getattr(args, "append_eos", False)) and tokenizer.eos_token is not None) else ""

    def _encode_text(self, text: str, max_len: int) -> List[int]:
        ids = self.tokenizer.encode(str(text), add_special_tokens=False, truncation=True, max_length=max_len)
        return [int(x) for x in ids]

    def _build_seq(self, prompt_text: str, answer_text: str) -> Tuple[List[int], List[int], List[int]]:
        prompt_ids = self._encode_text(prompt_text, int(getattr(self.args, "max_prompt_len", 1024)))
        answer_ids = self._encode_text(str(answer_text) + self.eos_text, int(getattr(self.args, "max_answer_len", 512)))
        total_budget = int(getattr(self.args, "max_seq_len", 2048))
        if len(answer_ids) >= total_budget:
            answer_ids = answer_ids[: total_budget - 1]
        prompt_budget = max(1, total_budget - len(answer_ids))
        prompt_ids = prompt_ids[-prompt_budget:]
        full_ids = prompt_ids + answer_ids
        attn = [1] * len(full_ids)
        target_ids = [-100] * len(full_ids)
        answer_start = len(prompt_ids)
        for j in range(answer_start, len(full_ids)):
            target_ids[j - 1] = full_ids[j]
        return full_ids, attn, target_ids

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        base_prompts = [render_prompt(row, self.tokenizer, self.args) for row in batch]
        teacher_prompts = [render_teacher_prompt(row, self.tokenizer, self.args) for row in batch]
        context_texts = [render_user_context(row, self.args) for row in batch]
        chosen_texts = [str(row["chosen"]) for row in batch]
        rejected_texts = [str(row.get("rejected", "")) for row in batch]

        prompt_ids = [self._encode_text(text, int(getattr(self.args, "max_prompt_len", 1024))) for text in base_prompts]
        context_ids = [self._encode_text(text, int(getattr(self.args, "max_context_len", 2048))) for text in context_texts]
        has_history = [bool(extract_history_pairs(row)) or bool(context_texts[i].strip()) for i, row in enumerate(batch)]
        context_ids = [ids if ids else [self.pad_id] for ids in context_ids]
        context_attn = [[1] * len(ids) if has_history[i] else [0] * len(ids) for i, ids in enumerate(context_ids)]

        chosen_full, chosen_attn, chosen_tgt = [], [], []
        rejected_full, rejected_attn, rejected_tgt = [], [], []
        teacher_chosen_full, teacher_chosen_attn, teacher_chosen_tgt = [], [], []

        for base_prompt, teacher_prompt, chosen, rejected in zip(base_prompts, teacher_prompts, chosen_texts, rejected_texts):
            f, a, t = self._build_seq(base_prompt, chosen)
            chosen_full.append(f)
            chosen_attn.append(a)
            chosen_tgt.append(t)
            f, a, t = self._build_seq(base_prompt, rejected)
            rejected_full.append(f)
            rejected_attn.append(a)
            rejected_tgt.append(t)
            f, a, t = self._build_seq(teacher_prompt, chosen)
            teacher_chosen_full.append(f)
            teacher_chosen_attn.append(a)
            teacher_chosen_tgt.append(t)

        return {
            "prompt_input_ids": pad_2d(prompt_ids, self.pad_id),
            "prompt_attention_mask": pad_2d([[1] * len(x) for x in prompt_ids], 0),
            "context_input_ids": pad_2d(context_ids, self.pad_id),
            "context_attention_mask": pad_2d(context_attn, 0),
            "has_history": torch.tensor(has_history, dtype=torch.bool),
            "chosen_input_ids": pad_2d(chosen_full, self.pad_id),
            "chosen_attention_mask": pad_2d(chosen_attn, 0),
            "chosen_target_ids": pad_2d(chosen_tgt, -100),
            "rejected_input_ids": pad_2d(rejected_full, self.pad_id),
            "rejected_attention_mask": pad_2d(rejected_attn, 0),
            "rejected_target_ids": pad_2d(rejected_tgt, -100),
            "teacher_chosen_input_ids": pad_2d(teacher_chosen_full, self.pad_id),
            "teacher_chosen_attention_mask": pad_2d(teacher_chosen_attn, 0),
            "teacher_chosen_target_ids": pad_2d(teacher_chosen_tgt, -100),
        }


@dataclass
class BaseHandles:
    base_model: nn.Module
    text_backbone: nn.Module
    input_embeddings: nn.Module
    output_head: nn.Module
    hidden_size: int
    vocab_size: int


def get_hidden_size_and_vocab(config: Any) -> Tuple[int, int]:
    if hasattr(config, "text_config"):
        cfg = config.text_config
        return int(cfg.hidden_size), int(cfg.vocab_size)
    return int(config.hidden_size), int(config.vocab_size)


def locate_text_backbone(model: nn.Module) -> nn.Module:
    candidates: List[Any] = []
    if hasattr(model, "model"):
        candidates.append(model.model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
        if hasattr(model.language_model, "model"):
            candidates.append(model.language_model.model)
    if hasattr(model, "text_model"):
        candidates.append(model.text_model)
    if hasattr(model, "transformer"):
        candidates.append(model.transformer)
    for cand in candidates:
        if isinstance(cand, nn.Module):
            return cand
    return model


def locate_output_head(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head
    raise AttributeError("Could not locate LM output head")


def load_base_handles(args: argparse.Namespace | SimpleNamespace, device: torch.device) -> Tuple[Any, BaseHandles]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = str_to_dtype(str(getattr(args, "dtype", "bfloat16")))
    tokenizer = AutoTokenizer.from_pretrained(
        str(getattr(args, "model_name", "Qwen/Qwen3.5-0.8B-Base")),
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    common_kwargs = {
        "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    attn_impl = str(getattr(args, "attn_implementation", "sdpa") or "")
    if attn_impl:
        common_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(str(getattr(args, "model_name")), **common_kwargs)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise AttributeError("Base model does not expose input embeddings")
    text_backbone = locate_text_backbone(model)
    output_head = locate_output_head(model)
    hidden_size, vocab_size = get_hidden_size_and_vocab(model.config)
    return tokenizer, BaseHandles(
        base_model=model,
        text_backbone=text_backbone,
        input_embeddings=input_embeddings,
        output_head=output_head,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def flatten_target_positions(target_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nz = (target_ids != -100).nonzero(as_tuple=False)
    return nz[:, 0], nz[:, 1], target_ids[nz[:, 0], nz[:, 1]]


def scatter_sum_scalar(values: torch.Tensor, example_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    out = values.new_zeros((batch_size,))
    out.index_add_(0, example_idx, values)
    return out


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.bool()
    has_any = mask.any(dim=-1, keepdim=True)
    last_idx = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
    pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
    return torch.where(has_any, pooled, torch.zeros_like(pooled))


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask_b = mask.bool()
    safe = scores.masked_fill(~mask_b, NEG_INF)
    has_any = mask_b.any(dim=dim, keepdim=True)
    safe = torch.where(has_any, safe, torch.zeros_like(safe))
    probs = F.softmax(safe, dim=dim) * mask_b.to(scores.dtype)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)


def masked_log_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = masked_softmax(scores, mask, dim=dim)
    return probs.clamp_min(1e-30).log().masked_fill(~mask.bool(), NEG_INF)


def masked_row_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)


def masked_row_var(x: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    centered = x - mean.unsqueeze(-1)
    return (centered.pow(2) * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)


def weighted_masked_recenter(x: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    w = weight.to(dtype=x.dtype) * mask_f
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(eps)
    mean = (x * w).sum(dim=-1, keepdim=True)
    return (x - mean).masked_fill(~mask.bool(), 0.0)


def weighted_masked_standardize(
    x: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    centered = weighted_masked_recenter(x, weight, mask, eps=eps)
    mask_f = mask.to(dtype=x.dtype)
    w = weight.to(dtype=x.dtype) * mask_f
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(eps)
    var = (centered.pow(2) * w).sum(dim=-1)
    out = centered / torch.sqrt(var.unsqueeze(-1) + eps)
    return out.masked_fill(~mask.bool(), 0.0), var


def masked_abs_mean_2d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    return (x.abs() * mask_f).sum() / mask_f.sum().clamp_min(1.0)


@dataclass
class SideOutput:
    label_seq_logprob: torch.Tensor
    base_label_seq_logprob: torch.Tensor
    total_token_count: torch.Tensor
    covered_token_count: torch.Tensor
    chosen_ce_loss: torch.Tensor
    token_acc: torch.Tensor
    base_token_acc: torch.Tensor
    support_size_mean: torch.Tensor
    entropy_mean: torch.Tensor
    coverage: torch.Tensor
    label_logprob_mean: torch.Tensor
    base_label_logprob_mean: torch.Tensor
    distill_kl: torch.Tensor
    distill_delta_loss: torch.Tensor
    distill_lambda_loss: torch.Tensor
    oracle_lambda_mean: torch.Tensor
    lambda_pred_mean: torch.Tensor
    distill_valid_frac: torch.Tensor


class PoolProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float, pool_mode: str):
        super().__init__()
        self.pool_mode = pool_mode
        attn_hidden = max(64, min(input_dim, output_dim))
        self.attn_score = nn.Sequential(
            RMSNorm(input_dim),
            nn.Linear(input_dim, attn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1),
        )
        self.proj = nn.Sequential(
            RMSNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "last":
            pooled = last_token_pool(hidden, attention_mask)
        elif self.pool_mode == "mean":
            pooled = masked_mean_pool(hidden, attention_mask)
        elif self.pool_mode == "attn":
            scores = self.attn_score(hidden).squeeze(-1)
            alpha = masked_softmax(scores, attention_mask, dim=-1)
            pooled = torch.sum(alpha.unsqueeze(-1) * hidden, dim=1)
        else:
            raise ValueError(f"Unsupported pool_mode={self.pool_mode}")
        return self.proj(pooled)


class VectorProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleQueryCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"adapter_dim={dim} must be divisible by cross_attn_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Sequential(
            RMSNorm(dim * 2),
            nn.Linear(dim * 2, dim),
        )
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        current_latent: torch.Tensor,
        prompt_latent: torch.Tensor,
        context_memory: torch.Tensor,
        context_memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        n_rows, n_mem, dim = context_memory.shape
        q = self.query(torch.cat([current_latent, prompt_latent.to(dtype=current_latent.dtype)], dim=-1))
        k = self.key(context_memory.to(dtype=q.dtype))
        v = self.value(context_memory.to(dtype=q.dtype))

        q = q.view(n_rows, self.num_heads, self.head_dim).unsqueeze(2)
        k = k.view(n_rows, n_mem, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(n_rows, n_mem, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        mask = context_memory_mask.bool().unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~mask, NEG_INF)
        has_memory = mask.any(dim=-1, keepdim=True)
        scores = torch.where(has_memory, scores, torch.zeros_like(scores))
        attn = F.softmax(scores, dim=-1) * mask.to(dtype=scores.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attended = torch.matmul(attn, v).squeeze(2).reshape(n_rows, dim)
        attended = self.out(attended)
        return torch.where(context_memory_mask.bool().any(dim=-1, keepdim=True), attended, torch.zeros_like(attended))


def compress_context_memory(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_tokens <= 0 or hidden.size(1) <= max_tokens:
        return hidden, attention_mask.bool()

    bsz, seq_len, hidden_dim = hidden.shape
    chunk = math.ceil(seq_len / max_tokens)
    padded_len = chunk * max_tokens
    if padded_len != seq_len:
        pad_hidden = hidden.new_zeros((bsz, padded_len - seq_len, hidden_dim))
        pad_mask = attention_mask.new_zeros((bsz, padded_len - seq_len))
        hidden = torch.cat([hidden, pad_hidden], dim=1)
        attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

    hidden = hidden.view(bsz, max_tokens, chunk, hidden_dim)
    mask = attention_mask.bool().view(bsz, max_tokens, chunk)
    mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    memory = (hidden * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)
    memory_mask = mask.any(dim=2)
    return memory, memory_mask


class ContextSteeringDistillModel(nn.Module):
    def __init__(self, handles: BaseHandles, args: argparse.Namespace | SimpleNamespace):
        super().__init__()
        self.base_model = handles.base_model
        self.text_backbone = handles.text_backbone
        self.input_embeddings = handles.input_embeddings
        self.output_head = handles.output_head
        self.hidden_size = handles.hidden_size
        self.vocab_size = handles.vocab_size
        self.args = args

        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

        adapter_dim = int(getattr(args, "adapter_dim", 512))
        cand_dim = int(getattr(args, "candidate_dim", 512))
        dropout = float(getattr(args, "module_dropout", 0.05))
        pool_mode = str(getattr(args, "context_pool", "last"))

        self.prompt_encoder = PoolProjector(self.hidden_size, adapter_dim, dropout, pool_mode)
        self.context_encoder = PoolProjector(self.hidden_size, adapter_dim, dropout, pool_mode)
        self.current_encoder = VectorProjector(self.hidden_size, adapter_dim, dropout)
        self.context_memory_proj = nn.Sequential(
            RMSNorm(self.hidden_size),
            nn.Linear(self.hidden_size, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, adapter_dim),
        )
        self.cross_attention = SingleQueryCrossAttention(
            dim=adapter_dim,
            num_heads=int(getattr(args, "cross_attn_heads", 8)),
            dropout=dropout,
        )
        fusion_dim = adapter_dim * 3
        self.fusion = nn.Sequential(
            RMSNorm(fusion_dim),
            nn.Linear(fusion_dim, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, adapter_dim),
        )
        self.state_to_candidate = nn.Linear(adapter_dim, cand_dim, bias=False)

        self.hidden_norm = RMSNorm(self.hidden_size)
        self.token_norm = RMSNorm(self.hidden_size)
        self.token_proj = nn.Linear(self.hidden_size, cand_dim, bias=False)
        self.query_norm = RMSNorm(cand_dim)
        self.candidate_norm = RMSNorm(cand_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(cand_dim * 3 + 3, cand_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(cand_dim, cand_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(cand_dim // 2, 1),
        )
        lambda_hidden = int(getattr(args, "lambda_head_hidden", 256))
        self.lambda_head = nn.Sequential(
            RMSNorm(adapter_dim),
            nn.Linear(adapter_dim, lambda_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lambda_hidden, 1),
        )

        nn.init.normal_(self.token_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.state_to_candidate.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.context_memory_proj[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.context_memory_proj[-1].bias)
        nn.init.normal_(self.cross_attention.out[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.cross_attention.out[0].bias)
        nn.init.normal_(self.fusion[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fusion[-1].bias)
        nn.init.normal_(self.pair_mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.pair_mlp[-1].bias)
        nn.init.zeros_(self.lambda_head[-1].weight)
        nn.init.zeros_(self.lambda_head[-1].bias)

        self.runtime_score_scale = float(getattr(args, "score_scale", 1.0))

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def set_schedule_fraction(self, fraction: float) -> None:
        frac = max(0.0, min(1.0, float(fraction)))
        warmup_ratio = max(float(getattr(self.args, "score_scale_warmup_ratio", 0.10)), 0.0)
        warm = 1.0 if warmup_ratio <= 0 else min(1.0, frac / warmup_ratio)
        self.runtime_score_scale = float(getattr(self.args, "score_scale", 1.0)) * warm

    def steering_state_dict(self) -> Dict[str, torch.Tensor]:
        excluded = ("base_model.", "text_backbone.", "input_embeddings.", "output_head.")
        return {k: v for k, v in self.state_dict().items() if not k.startswith(excluded)}

    def load_steering_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.startswith(("base_model.", "text_backbone.", "input_embeddings.", "output_head."))]
        if missing and is_main_process():
            print(f"[load_steering] Missing keys: {missing}")
        if unexpected and is_main_process():
            print(f"[load_steering] Unexpected keys: {unexpected}")

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.float().sum() * 0.0

    def _backbone_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        with torch.no_grad():
            outputs = self.text_backbone(**kwargs)
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            elif isinstance(outputs, (tuple, list)):
                hidden = outputs[0]
            else:
                raise RuntimeError("Unsupported backbone output type")
        return hidden

    def _lookup_candidate_features(self, candidate_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(self.output_head, nn.Linear):
                return self.output_head.weight[candidate_ids]
            return self.input_embeddings(candidate_ids)

    def _encode_context(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        prompt_hidden = self._backbone_forward(batch["prompt_input_ids"], batch["prompt_attention_mask"])
        context_hidden = self._backbone_forward(batch["context_input_ids"], batch["context_attention_mask"])
        prompt_latent = self.prompt_encoder(prompt_hidden, batch["prompt_attention_mask"])
        context_latent = self.context_encoder(context_hidden, batch["context_attention_mask"])
        has_history = batch.get("has_history", batch["context_attention_mask"].bool().any(dim=-1)).bool()
        context_latent = torch.where(has_history.unsqueeze(-1), context_latent, torch.zeros_like(context_latent))
        memory_hidden, memory_mask = compress_context_memory(
            context_hidden,
            batch["context_attention_mask"],
            max_tokens=int(getattr(self.args, "cross_attn_memory_tokens", 64)),
        )
        context_memory = self.context_memory_proj(memory_hidden)
        memory_mask = memory_mask & has_history.unsqueeze(-1)
        context_memory = torch.where(memory_mask.unsqueeze(-1), context_memory, torch.zeros_like(context_memory))
        zero = prompt_latent.new_zeros(())
        info = {
            "prompt_latent_norm": prompt_latent.norm(dim=-1).mean(),
            "context_latent_norm": context_latent.norm(dim=-1).mean(),
            "history_present_frac": has_history.float().mean(),
            "history_pairs_used": zero,
            "history_attn_entropy": zero,
            "history_attn_max": zero,
            "history_latent_norm": context_latent.norm(dim=-1).mean(),
            "user_latent_norm": context_latent.norm(dim=-1).mean(),
            "context_memory_tokens": memory_mask.float().sum(dim=-1).mean(),
        }
        return {
            "prompt_latent": prompt_latent,
            "memory_vec": context_latent,
            "context_memory": context_memory,
            "context_memory_mask": memory_mask,
            "has_history": has_history,
        }, info

    def _encode_user_prompt(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        context, info = self._encode_context(batch)
        return context["memory_vec"], info

    def _build_fusion_state(
        self,
        last_hidden: torch.Tensor,
        prompt_latent: torch.Tensor,
        context_latent: torch.Tensor,
        has_history: Optional[torch.Tensor],
        context_memory: Optional[torch.Tensor] = None,
        context_memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_t = self.current_encoder(self.hidden_norm(last_hidden))
        z_x = prompt_latent.to(dtype=z_t.dtype)
        if context_memory is None or context_memory_mask is None:
            z_ca = context_latent.to(dtype=z_t.dtype)
        else:
            z_ca = self.cross_attention(
                current_latent=z_t,
                prompt_latent=z_x,
                context_memory=context_memory,
                context_memory_mask=context_memory_mask,
            )
        if has_history is not None:
            z_ca = torch.where(has_history.bool().view(-1, 1), z_ca, torch.zeros_like(z_ca))
        fusion_in = torch.cat([z_t, z_ca, z_t * z_ca], dim=-1)
        return self.fusion(fusion_in)

    def _build_support(
        self,
        base_logits: torch.Tensor,
        base_logp: torch.Tensor,
        base_probs: torch.Tensor,
        norm_entropy: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        extra_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = base_logits.device
        n_rows = base_logits.size(0)
        max_k = min(int(getattr(self.args, "support_max_k", 64)), base_logits.size(-1))
        min_k = min(int(getattr(self.args, "support_min_k", 4)), max_k)

        base_top_logits, base_top_idx = torch.topk(base_logits.detach(), k=max_k, dim=-1)
        base_top_probs = base_probs.gather(dim=-1, index=base_top_idx)
        cumulative = base_top_probs.cumsum(dim=-1)
        top_p = (
            float(getattr(self.args, "support_top_p_min", 0.70))
            + norm_entropy.squeeze(-1)
            * (float(getattr(self.args, "support_top_p_max", 0.95)) - float(getattr(self.args, "support_top_p_min", 0.70)))
        ).clamp(
            min=float(getattr(self.args, "support_top_p_min", 0.70)),
            max=float(getattr(self.args, "support_top_p_max", 0.95)),
        )
        meets = cumulative >= top_p.unsqueeze(-1)
        first_hit = torch.argmax(meets.float(), dim=-1)
        has_hit = meets.any(dim=-1)
        k_counts = torch.where(has_hit, first_hit + 1, torch.full_like(first_hit, max_k)).clamp(min=min_k, max=max_k)
        base_mask = torch.arange(max_k, device=device).unsqueeze(0) < k_counts.unsqueeze(-1)

        idx_parts = [base_top_idx]
        valid_parts = [base_mask]
        rank = torch.arange(max_k, device=device, dtype=base_logits.dtype).unsqueeze(0).expand_as(base_top_logits)
        denom = (k_counts - 1).clamp_min(1).to(base_logits.dtype).unsqueeze(-1)
        rank_parts: List[torch.Tensor] = [rank / denom]

        if labels is not None:
            idx_parts.append(labels.view(-1, 1))
            valid_parts.append(torch.ones((n_rows, 1), dtype=torch.bool, device=device))
            rank_parts.append(torch.ones((n_rows, 1), dtype=base_logits.dtype, device=device))
        if extra_ids is not None and extra_ids.numel() > 0:
            idx_parts.append(extra_ids.long())
            valid_parts.append(torch.ones_like(extra_ids, dtype=torch.bool))
            rank_parts.append(torch.ones_like(extra_ids, dtype=base_logits.dtype))

        support_idx = torch.cat(idx_parts, dim=-1)
        support_valid = torch.cat(valid_parts, dim=-1)
        rank_frac = torch.cat(rank_parts, dim=-1)
        total_k = support_idx.size(-1)
        pos = torch.arange(total_k, device=device)
        eq = support_idx.unsqueeze(-1) == support_idx.unsqueeze(-2)
        has_prev = (eq & (pos.view(1, total_k, 1) > pos.view(1, 1, total_k))).any(dim=-1)
        support_mask = support_valid & (~has_prev)

        support_logits = base_logits.gather(dim=-1, index=support_idx)
        support_logp = base_logp.gather(dim=-1, index=support_idx)
        support_probs = base_probs.gather(dim=-1, index=support_idx)

        top_ext_k = min(total_k + 1, base_logits.size(-1))
        top_ext_logits, top_ext_idx = torch.topk(base_logits.detach(), k=top_ext_k, dim=-1)
        top_gap = torch.zeros((n_rows, 1), device=device, dtype=base_logits.dtype)
        if top_ext_k >= 2:
            top_gap = (top_ext_logits[:, :1] - top_ext_logits[:, 1:2]).to(base_logits.dtype)

        return {
            "support_idx": support_idx,
            "support_mask": support_mask,
            "support_logits": support_logits,
            "support_logp": support_logp,
            "support_probs": support_probs,
            "rank_frac": rank_frac,
            "k_counts": support_mask.float().sum(dim=-1),
            "top_gap": top_gap,
            "top_ext_idx": top_ext_idx,
            "top_ext_logits": top_ext_logits.to(base_logits.dtype),
        }

    def _student_delta(
        self,
        last_hidden: torch.Tensor,
        prompt_latent: torch.Tensor,
        context_latent: torch.Tensor,
        has_history: Optional[torch.Tensor],
        context_memory: Optional[torch.Tensor],
        context_memory_mask: Optional[torch.Tensor],
        support_idx: torch.Tensor,
        support_mask: torch.Tensor,
        active_support_mask: torch.Tensor,
        support_logits: torch.Tensor,
        support_probs: torch.Tensor,
        rank_frac: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        state = self._build_fusion_state(
            last_hidden=last_hidden,
            prompt_latent=prompt_latent,
            context_latent=context_latent,
            has_history=has_history,
            context_memory=context_memory,
            context_memory_mask=context_memory_mask,
        )
        q = self.query_norm(self.state_to_candidate(state))
        q_exp = q.unsqueeze(1).expand(-1, support_idx.size(1), -1)

        token_features = self._lookup_candidate_features(support_idx).to(dtype=last_hidden.dtype)
        token_proj = self.token_proj(self.token_norm(token_features))
        token_n = self.candidate_norm(token_proj)
        inv_sqrt_d = 1.0 / math.sqrt(float(getattr(self.args, "candidate_dim", 512)))
        pair_dot = (q_exp * token_n).sum(dim=-1) * inv_sqrt_d

        support_mean = masked_row_mean(support_logits, support_mask)
        support_var = masked_row_var(support_logits, support_mask, support_mean)
        support_logit_z = (support_logits - support_mean.unsqueeze(-1)) / torch.sqrt(support_var.unsqueeze(-1) + 1e-6)
        scalar_feats = torch.stack([support_logit_z, support_probs, rank_frac], dim=-1).to(dtype=token_proj.dtype)
        pair_mlp_in = torch.cat([q_exp, token_proj, q_exp * token_proj, scalar_feats], dim=-1)
        pair_mlp = self.pair_mlp(pair_mlp_in).squeeze(-1)

        direction_raw = (pair_dot + pair_mlp).masked_fill(~active_support_mask, 0.0)
        direction_norm, direction_var = weighted_masked_standardize(direction_raw, weight=support_probs, mask=active_support_mask)

        max_abs_lambda = max(
            abs(float(getattr(self.args, "oracle_lambda_min", -1.5))),
            abs(float(getattr(self.args, "oracle_lambda_max", 1.5))),
        )
        lambda_pred = max_abs_lambda * torch.tanh(self.lambda_head(state).squeeze(-1))
        lambda_pred = lambda_pred.clamp(
            min=float(getattr(self.args, "oracle_lambda_min", -1.5)),
            max=float(getattr(self.args, "oracle_lambda_max", 1.5)),
        )
        if has_history is not None:
            lambda_pred = torch.where(has_history.bool().view(-1), lambda_pred, torch.zeros_like(lambda_pred))

        delta_support = lambda_pred.unsqueeze(-1) * direction_norm
        if float(getattr(self.args, "score_clip", 3.0)) > 0:
            c = float(getattr(self.args, "score_clip", 3.0))
            delta_support = c * torch.tanh(delta_support / c)
        delta_support = float(self.runtime_score_scale) * delta_support
        delta_support = delta_support.masked_fill(~active_support_mask, 0.0)
        if bool(int(getattr(self.args, "zero_mean_scores", 1))):
            delta_support = weighted_masked_recenter(delta_support, weight=support_probs, mask=active_support_mask)

        return {
            "direction_support": direction_norm,
            "direction_var": direction_var,
            "lambda_pred": lambda_pred,
            "delta_support": delta_support,
            "pair_dot": pair_dot,
            "pair_mlp": pair_mlp,
        }

    def compute_support_reranking(
        self,
        last_hidden: torch.Tensor,
        prompt_latent: torch.Tensor,
        fixed_memory_vec: torch.Tensor,
        has_history: Optional[torch.Tensor] = None,
        context_memory: Optional[torch.Tensor] = None,
        context_memory_mask: Optional[torch.Tensor] = None,
        force_labels: Optional[torch.Tensor] = None,
        extra_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base_logits = self.output_head(last_hidden).float()
        base_logp = F.log_softmax(base_logits, dim=-1)
        base_probs = base_logp.exp()
        entropy = -(base_probs * base_logp).sum(dim=-1, keepdim=True)
        norm_entropy = entropy / math.log(float(self.vocab_size))
        labels = force_labels if (force_labels is not None and bool(int(getattr(self.args, "train_force_label_in_support", 1)))) else None
        support = self._build_support(base_logits, base_logp, base_probs, norm_entropy, labels=labels, extra_ids=extra_ids)

        support_idx = support["support_idx"]
        support_mask = support["support_mask"]
        active_support_mask = support_mask
        eos_id = int(getattr(self.args, "eos_token_id", -1))
        if eos_id >= 0 and not bool(int(getattr(self.args, "personalize_eos", 0))):
            active_support_mask = active_support_mask & (support_idx != eos_id)

        student = self._student_delta(
            last_hidden=last_hidden,
            prompt_latent=prompt_latent,
            context_latent=fixed_memory_vec,
            has_history=has_history,
            context_memory=context_memory,
            context_memory_mask=context_memory_mask,
            support_idx=support_idx,
            support_mask=support_mask,
            active_support_mask=active_support_mask,
            support_logits=support["support_logits"],
            support_probs=support["support_probs"],
            rank_frac=support["rank_frac"],
        )
        delta_support = student["delta_support"]
        steered_support_logits = support["support_logits"] + delta_support
        return {
            "base_logits": base_logits,
            "base_logp": base_logp,
            "base_probs": base_probs,
            "norm_entropy": norm_entropy,
            "support_idx": support_idx,
            "support_mask": support_mask,
            "active_support_mask": active_support_mask,
            "support_logits": support["support_logits"],
            "support_logp": support["support_logp"],
            "support_probs": support["support_probs"],
            "steered_support_logits": steered_support_logits,
            "delta_support": delta_support,
            "direction_support": student["direction_support"],
            "lambda_pred": student["lambda_pred"],
            "k_counts": support["k_counts"],
            "top_gap": support["top_gap"],
            "top_ext_idx": support["top_ext_idx"],
            "top_ext_logits": support["top_ext_logits"],
            "rerank_dot_abs_mean": masked_abs_mean_2d(student["pair_dot"], active_support_mask).detach(),
            "rerank_mlp_abs_mean": masked_abs_mean_2d(student["pair_mlp"], active_support_mask).detach(),
            "delta_abs_mean": masked_abs_mean_2d(delta_support, active_support_mask).detach(),
        }

    @torch.inference_mode()
    def compute_generation_logit_bias(
        self,
        last_hidden: torch.Tensor,
        prompt_latent: torch.Tensor,
        fixed_memory_vec: torch.Tensor,
        has_history: Optional[torch.Tensor] = None,
        context_memory: Optional[torch.Tensor] = None,
        context_memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        info = self.compute_support_reranking(
            last_hidden=last_hidden,
            prompt_latent=prompt_latent,
            fixed_memory_vec=fixed_memory_vec,
            has_history=has_history,
            context_memory=context_memory,
            context_memory_mask=context_memory_mask,
        )
        delta_logits = torch.zeros_like(info["base_logits"])
        active_delta = info["delta_support"].masked_fill(~info["active_support_mask"], 0.0).to(dtype=delta_logits.dtype)
        delta_logits.scatter_add_(dim=-1, index=info["support_idx"], src=active_delta)
        diagnostics = {
            "entropy_mean": info["norm_entropy"].mean(),
            "support_size_mean": info["k_counts"].float().mean(),
            "top_gap_mean": info["top_gap"].mean(),
            "lambda_pred_mean": info["lambda_pred"].mean(),
            "delta_abs_mean": info["delta_abs_mean"],
            "rerank_dot_abs_mean": info["rerank_dot_abs_mean"],
            "rerank_mlp_abs_mean": info["rerank_mlp_abs_mean"],
        }
        return delta_logits, diagnostics

    def _exact_logprob_terms(
        self,
        base_logp: torch.Tensor,
        support_idx: torch.Tensor,
        support_probs: torch.Tensor,
        support_mask: torch.Tensor,
        delta_support: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mask_f = support_mask.to(dtype=delta_support.dtype)
        active_delta = delta_support.masked_fill(~support_mask.bool(), 0.0)
        correction = 1.0 + (support_probs * (active_delta.exp() - 1.0) * mask_f).sum(dim=-1)
        log_correction = correction.clamp_min(1e-8).log()
        row = torch.arange(labels.size(0), device=labels.device)
        base_label_lp = base_logp[row, labels]
        label_match = (support_idx == labels.unsqueeze(-1)) & support_mask.bool()
        has_label = label_match.any(dim=-1)
        label_pos = torch.argmax(label_match.long(), dim=-1)
        label_delta = torch.where(has_label, active_delta[row, label_pos], torch.zeros_like(base_label_lp))
        steered_label_lp = base_label_lp + label_delta - log_correction
        return {
            "base_label_lp": base_label_lp,
            "steered_label_lp": steered_label_lp,
            "log_correction": log_correction,
            "has_label": has_label,
        }

    def _full_vocab_pred_tokens(
        self,
        support_idx: torch.Tensor,
        support_mask: torch.Tensor,
        steered_support_logits: torch.Tensor,
        top_ext_idx: torch.Tensor,
        top_ext_logits: torch.Tensor,
    ) -> torch.Tensor:
        active_support_logits = steered_support_logits.masked_fill(~support_mask.bool(), NEG_INF)
        best_support_logit, best_support_pos = active_support_logits.max(dim=-1)
        best_support_token = support_idx.gather(dim=-1, index=best_support_pos.unsqueeze(-1)).squeeze(-1)
        ext_in_support = ((top_ext_idx.unsqueeze(-1) == support_idx.unsqueeze(1)) & support_mask.unsqueeze(1)).any(dim=-1)
        outside_logits = top_ext_logits.masked_fill(ext_in_support, NEG_INF)
        best_outside_logit, best_outside_pos = outside_logits.max(dim=-1)
        best_outside_token = top_ext_idx.gather(dim=-1, index=best_outside_pos.unsqueeze(-1)).squeeze(-1)
        return torch.where(best_support_logit >= best_outside_logit, best_support_token, best_outside_token)

    def _oracle_lambda(
        self,
        base_support_logp: torch.Tensor,
        direction_target: torch.Tensor,
        labels: torch.Tensor,
        support_idx: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row = torch.arange(labels.size(0), device=labels.device)
        label_match = (support_idx == labels.unsqueeze(-1)) & active_mask.bool()
        has_label = label_match.any(dim=-1)
        label_pos = torch.argmax(label_match.long(), dim=-1)
        d_gold = torch.where(has_label, direction_target[row, label_pos], torch.zeros_like(direction_target[:, 0]))
        lo_bound = torch.full_like(d_gold, float(getattr(self.args, "oracle_lambda_min", -1.5)))
        hi_bound = torch.full_like(d_gold, float(getattr(self.args, "oracle_lambda_max", 1.5)))

        def f(lam: torch.Tensor) -> torch.Tensor:
            logits = base_support_logp + lam.unsqueeze(-1) * direction_target
            probs = masked_softmax(logits, active_mask, dim=-1)
            return (probs * direction_target * active_mask.to(direction_target.dtype)).sum(dim=-1) - d_gold

        f_lo = f(lo_bound)
        f_hi = f(hi_bound)
        lo = lo_bound.clone()
        hi = hi_bound.clone()
        root_mask = (f_lo < 0) & (f_hi > 0) & has_label
        for _ in range(int(getattr(self.args, "oracle_lambda_bisect_steps", 24))):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            move_lo = (f_mid < 0) & root_mask
            lo = torch.where(move_lo, mid, lo)
            hi = torch.where((~move_lo) & root_mask, mid, hi)
        root = 0.5 * (lo + hi)
        lam = torch.where(f_lo >= 0, lo_bound, torch.where(f_hi <= 0, hi_bound, root))
        lam = torch.where(has_label, lam, torch.zeros_like(lam))
        return lam, has_label

    def _distill_terms(
        self,
        info: Dict[str, torch.Tensor],
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
        teacher_support_logp = teacher_logp.gather(dim=-1, index=info["support_idx"])
        raw_delta = teacher_support_logp - info["support_logp"]
        active_mask = info["active_support_mask"]
        direction_target, target_var = weighted_masked_standardize(raw_delta, weight=info["support_probs"], mask=active_mask)
        lambda_star, has_label = self._oracle_lambda(
            base_support_logp=info["support_logp"],
            direction_target=direction_target,
            labels=labels,
            support_idx=info["support_idx"],
            active_mask=active_mask,
        )
        valid = content_mask.bool() & has_label & (target_var > float(getattr(self.args, "distill_min_delta_var", 1e-5)))
        target_delta = lambda_star.unsqueeze(-1) * direction_target
        weight = info["support_probs"] * active_mask.to(info["support_probs"].dtype)
        row_delta_loss = ((info["direction_support"] - direction_target).pow(2) * weight).sum(dim=-1)
        row_delta_loss = row_delta_loss / weight.sum(dim=-1).clamp_min(1e-8)

        if valid.any():
            delta_loss = row_delta_loss[valid].mean()
            lambda_loss = F.smooth_l1_loss(info["lambda_pred"][valid], lambda_star[valid])
            target_logq = masked_log_softmax(info["support_logits"] + target_delta, active_mask, dim=-1)
            student_logq = masked_log_softmax(info["support_logits"] + info["delta_support"], active_mask, dim=-1)
            target_q = target_logq.exp() * active_mask.to(target_logq.dtype)
            kl_row = (target_q * (target_logq - student_logq)).sum(dim=-1)
            kl_loss = kl_row[valid].mean()
            oracle_lambda_mean = lambda_star[valid].mean()
            lambda_pred_mean = info["lambda_pred"][valid].mean()
        else:
            zero = self._zero(info["support_logits"])
            delta_loss = zero
            lambda_loss = zero
            kl_loss = zero
            oracle_lambda_mean = zero
            lambda_pred_mean = zero
        return {
            "distill_delta_loss": delta_loss,
            "distill_lambda_loss": lambda_loss,
            "distill_kl": kl_loss,
            "oracle_lambda_mean": oracle_lambda_mean,
            "lambda_pred_mean": lambda_pred_mean,
            "distill_valid_frac": valid.float().mean(),
        }

    def _score_side(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        context: Dict[str, torch.Tensor],
        positive: bool,
        teacher_input_ids: Optional[torch.Tensor] = None,
        teacher_attention_mask: Optional[torch.Tensor] = None,
        teacher_target_ids: Optional[torch.Tensor] = None,
    ) -> SideOutput:
        batch_size = input_ids.size(0)
        hidden = self._backbone_forward(input_ids=input_ids, attention_mask=attention_mask)
        ex_idx, time_idx, labels = flatten_target_positions(target_ids)
        if labels.numel() == 0:
            raise ValueError("A batch contained zero answer tokens after truncation.")

        selected_hidden = hidden[ex_idx, time_idx, :]
        prompt_sel = context["prompt_latent"][ex_idx]
        memory_sel = context["memory_vec"][ex_idx]
        context_memory_sel = context["context_memory"][ex_idx]
        context_memory_mask_sel = context["context_memory_mask"][ex_idx]
        has_sel = context["has_history"][ex_idx]

        teacher_logits = None
        teacher_extra_ids = None
        if positive and teacher_input_ids is not None and teacher_attention_mask is not None and teacher_target_ids is not None:
            teacher_hidden = self._backbone_forward(teacher_input_ids, teacher_attention_mask)
            t_ex, t_time, t_labels = flatten_target_positions(teacher_target_ids)
            if t_labels.numel() != labels.numel():
                n = min(t_labels.numel(), labels.numel())
                ex_idx = ex_idx[:n]
                labels = labels[:n]
                selected_hidden = selected_hidden[:n]
                prompt_sel = prompt_sel[:n]
                memory_sel = memory_sel[:n]
                context_memory_sel = context_memory_sel[:n]
                context_memory_mask_sel = context_memory_mask_sel[:n]
                has_sel = has_sel[:n]
                t_ex = t_ex[:n]
                t_time = t_time[:n]
            teacher_selected_hidden = teacher_hidden[t_ex, t_time, :]
            teacher_logits = self.output_head(teacher_selected_hidden).float()
            k_teacher = min(int(getattr(self.args, "distill_teacher_top_k", 32)), teacher_logits.size(-1))
            if k_teacher > 0:
                teacher_extra_ids = torch.topk(teacher_logits.detach(), k=k_teacher, dim=-1).indices

        force_chosen = self.training and positive and bool(int(getattr(self.args, "train_force_label_in_support", 1)))
        force_rejected = self.training and (not positive) and bool(int(getattr(self.args, "train_force_rejected_label_in_support", 0)))
        force_labels = labels if (force_chosen or force_rejected) else None
        info = self.compute_support_reranking(
            last_hidden=selected_hidden,
            prompt_latent=prompt_sel,
            fixed_memory_vec=memory_sel,
            has_history=has_sel,
            context_memory=context_memory_sel,
            context_memory_mask=context_memory_mask_sel,
            force_labels=force_labels,
            extra_ids=teacher_extra_ids,
        )
        exact = self._exact_logprob_terms(
            base_logp=info["base_logp"],
            support_idx=info["support_idx"],
            support_probs=info["support_probs"],
            support_mask=info["support_mask"],
            delta_support=info["delta_support"],
            labels=labels,
        )

        eos_id = int(getattr(self.args, "eos_token_id", -1))
        content_mask = torch.ones_like(labels, dtype=torch.bool)
        if eos_id >= 0 and bool(int(getattr(self.args, "ignore_eos_in_loss", 1))):
            content_mask = labels != eos_id

        ce_loss = -exact["steered_label_lp"][content_mask].mean() if (positive and content_mask.any()) else self._zero(selected_hidden)

        pred_token = self._full_vocab_pred_tokens(
            support_idx=info["support_idx"],
            support_mask=info["support_mask"],
            steered_support_logits=info["steered_support_logits"],
            top_ext_idx=info["top_ext_idx"],
            top_ext_logits=info["top_ext_logits"],
        )
        base_pred_token = info["top_ext_idx"][:, 0]

        content_f = content_mask.to(dtype=exact["steered_label_lp"].dtype)
        seq_lp = scatter_sum_scalar(exact["steered_label_lp"] * content_f, ex_idx, batch_size)
        base_seq_lp = scatter_sum_scalar(exact["base_label_lp"] * content_f, ex_idx, batch_size)
        total_counts = scatter_sum_scalar(content_f, ex_idx, batch_size)
        covered_counts = scatter_sum_scalar(exact["has_label"].to(content_f.dtype) * content_f, ex_idx, batch_size)
        if bool(getattr(self.args, "length_normalize", False)):
            denom = total_counts.clamp_min(1.0)
            seq_lp = seq_lp / denom
            base_seq_lp = base_seq_lp / denom

        if teacher_logits is not None:
            distill = self._distill_terms(info, teacher_logits=teacher_logits, labels=labels, content_mask=content_mask)
        else:
            zero = self._zero(selected_hidden)
            distill = {
                "distill_kl": zero,
                "distill_delta_loss": zero,
                "distill_lambda_loss": zero,
                "oracle_lambda_mean": zero,
                "lambda_pred_mean": info["lambda_pred"].mean(),
                "distill_valid_frac": zero,
            }

        return SideOutput(
            label_seq_logprob=seq_lp,
            base_label_seq_logprob=base_seq_lp,
            total_token_count=total_counts,
            covered_token_count=covered_counts,
            chosen_ce_loss=ce_loss,
            token_acc=(pred_token == labels).float().mean(),
            base_token_acc=(base_pred_token == labels).float().mean(),
            support_size_mean=info["k_counts"].float().mean(),
            entropy_mean=info["norm_entropy"].mean(),
            coverage=exact["has_label"].float().mean(),
            label_logprob_mean=exact["steered_label_lp"].mean(),
            base_label_logprob_mean=exact["base_label_lp"].mean(),
            distill_kl=distill["distill_kl"],
            distill_delta_loss=distill["distill_delta_loss"],
            distill_lambda_loss=distill["distill_lambda_loss"],
            oracle_lambda_mean=distill["oracle_lambda_mean"],
            lambda_pred_mean=distill["lambda_pred_mean"],
            distill_valid_frac=distill["distill_valid_frac"],
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context, user_info = self._encode_context(batch)
        chosen = self._score_side(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            target_ids=batch["chosen_target_ids"],
            context=context,
            positive=True,
            teacher_input_ids=batch.get("teacher_chosen_input_ids"),
            teacher_attention_mask=batch.get("teacher_chosen_attention_mask"),
            teacher_target_ids=batch.get("teacher_chosen_target_ids"),
        )
        rejected = self._score_side(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            target_ids=batch["rejected_target_ids"],
            context=context,
            positive=False,
        )

        zero = chosen.chosen_ce_loss * 0.0
        pair_valid = (chosen.total_token_count > 0) & (rejected.total_token_count > 0)
        pairwise_weight = float(getattr(self.args, "pairwise_pref_weight", 0.0))
        hinge_weight = float(getattr(self.args, "chosen_gain_hinge_weight", 0.0))
        do_pair = pair_valid.any() and (pairwise_weight > 0.0 or hinge_weight > 0.0)
        if do_pair:
            chosen_gain = chosen.label_seq_logprob - chosen.base_label_seq_logprob
            rejected_gain = rejected.label_seq_logprob - rejected.base_label_seq_logprob
            pair_margin = (chosen_gain - rejected_gain)[pair_valid]
            preference_acc = (pair_margin > 0).float().mean()
            absolute_preference_acc = (chosen.label_seq_logprob > rejected.label_seq_logprob).float().mean()
            chosen_gain_mean = chosen_gain.mean()
            rejected_gain_mean = rejected_gain.mean()
            margin_gain_mean = pair_margin.mean()
            if pairwise_weight > 0:
                base_margin = (chosen.base_label_seq_logprob - rejected.base_label_seq_logprob)[pair_valid]
                hardness = torch.sigmoid(
                    (float(getattr(self.args, "hard_pair_margin", 0.0)) - base_margin.detach())
                    / max(float(getattr(self.args, "hard_pair_temperature", 0.25)), 1e-6)
                )
                min_w = float(getattr(self.args, "hard_pair_min_weight", 0.25))
                hardness = min_w + (1.0 - min_w) * hardness
                raw = -F.logsigmoid(float(getattr(self.args, "pairwise_beta", 1.0)) * pair_margin)
                pairwise_loss = (hardness * raw).sum() / hardness.sum().clamp_min(1e-8)
                hard_weight_mean = hardness.mean()
                base_margin_mean = base_margin.mean()
            else:
                pairwise_loss = zero
                hard_weight_mean = zero
                base_margin_mean = zero
            chosen_gain_hinge = (
                F.relu(float(getattr(self.args, "chosen_gain_margin", 0.0)) - chosen_gain[pair_valid]).mean()
                if hinge_weight > 0
                else zero
            )
        else:
            pairwise_loss = zero
            preference_acc = zero
            absolute_preference_acc = zero
            hard_weight_mean = zero
            base_margin_mean = zero
            chosen_gain_mean = zero
            rejected_gain_mean = zero
            margin_gain_mean = zero
            chosen_gain_hinge = zero

        total_loss = (
            float(getattr(self.args, "distill_kl_weight", 1.0)) * chosen.distill_kl
            + float(getattr(self.args, "distill_delta_weight", 0.5)) * chosen.distill_delta_loss
            + float(getattr(self.args, "distill_lambda_weight", 0.2)) * chosen.distill_lambda_loss
            + float(getattr(self.args, "chosen_ce_weight", 0.5)) * chosen.chosen_ce_loss
            + pairwise_weight * pairwise_loss
            + hinge_weight * chosen_gain_hinge
        )
        entropy_mean = 0.5 * (chosen.entropy_mean + rejected.entropy_mean)
        support_size_mean = 0.5 * (chosen.support_size_mean + rejected.support_size_mean)
        return {
            "loss": total_loss,
            "distill_kl": chosen.distill_kl.detach(),
            "distill_delta_loss": chosen.distill_delta_loss.detach(),
            "distill_lambda_loss": chosen.distill_lambda_loss.detach(),
            "chosen_ce_loss": chosen.chosen_ce_loss.detach(),
            "pairwise_pref_loss": pairwise_loss.detach(),
            "chosen_gain_hinge": chosen_gain_hinge.detach(),
            "entropy_mean": entropy_mean.detach(),
            "support_size_mean": support_size_mean.detach(),
            "chosen_coverage": chosen.coverage.detach(),
            "rejected_coverage": rejected.coverage.detach(),
            "chosen_token_acc": chosen.token_acc.detach(),
            "chosen_base_token_acc": chosen.base_token_acc.detach(),
            "rejected_token_acc": rejected.token_acc.detach(),
            "rejected_base_token_acc": rejected.base_token_acc.detach(),
            "preference_acc": preference_acc.detach(),
            "absolute_preference_acc": absolute_preference_acc.detach(),
            "covered_pair_frac": pair_valid.float().mean().detach(),
            "chosen_gain_mean": chosen_gain_mean.detach(),
            "rejected_gain_mean": rejected_gain_mean.detach(),
            "margin_gain_mean": margin_gain_mean.detach(),
            "base_margin_mean": base_margin_mean.detach(),
            "hard_weight_mean": hard_weight_mean.detach(),
            "chosen_logprob_mean": chosen.label_logprob_mean.detach(),
            "chosen_base_logprob_mean": chosen.base_label_logprob_mean.detach(),
            "rejected_logprob_mean": rejected.label_logprob_mean.detach(),
            "rejected_base_logprob_mean": rejected.base_label_logprob_mean.detach(),
            "oracle_lambda_mean": chosen.oracle_lambda_mean.detach(),
            "lambda_pred_mean": chosen.lambda_pred_mean.detach(),
            "distill_valid_frac": chosen.distill_valid_frac.detach(),
            "runtime_score_scale": torch.tensor(self.runtime_score_scale, device=total_loss.device),
            "prompt_latent_norm": user_info["prompt_latent_norm"].detach(),
            "context_latent_norm": user_info["context_latent_norm"].detach(),
            "context_memory_tokens": user_info["context_memory_tokens"].detach(),
            "history_present_frac": user_info["history_present_frac"].detach(),
            "history_pairs_used": user_info["history_pairs_used"].detach(),
            "history_attn_entropy": user_info["history_attn_entropy"].detach(),
            "history_attn_max": user_info["history_attn_max"].detach(),
            "history_latent_norm": user_info["history_latent_norm"].detach(),
            "user_latent_norm": user_info["user_latent_norm"].detach(),
        }

# -----------------------------------------------------------------------------
# Cautious distillation variant
# -----------------------------------------------------------------------------

@dataclass
class CautiousSideOutput(SideOutput):
    gate_loss: torch.Tensor
    base_preserve_kl: torch.Tensor
    helpful_distill_kl: torch.Tensor
    gate_mean: torch.Tensor
    gate_target_mean: torch.Tensor
    teacher_advantage_mean: torch.Tensor
    helpful_frac: torch.Tensor
    signed_lambda_mean: torch.Tensor


class CautiousContextSteeringDistillModel(ContextSteeringDistillModel):
    def __init__(self, handles: BaseHandles, args: argparse.Namespace | SimpleNamespace):
        super().__init__(handles, args)
        adapter_dim = int(getattr(args, "adapter_dim", 512))
        hidden = int(getattr(args, "gate_hidden", 256))
        dropout = float(getattr(args, "module_dropout", 0.05))
        self.gate_head = nn.Sequential(
            RMSNorm(adapter_dim + 3),
            nn.Linear(adapter_dim + 3, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(getattr(args, "gate_init_bias", -2.0)))

    def _student_delta(
        self,
        last_hidden: torch.Tensor,
        prompt_latent: torch.Tensor,
        context_latent: torch.Tensor,
        has_history: Optional[torch.Tensor],
        context_memory: Optional[torch.Tensor],
        context_memory_mask: Optional[torch.Tensor],
        support_idx: torch.Tensor,
        support_mask: torch.Tensor,
        active_support_mask: torch.Tensor,
        support_logits: torch.Tensor,
        support_probs: torch.Tensor,
        rank_frac: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        state = self._build_fusion_state(
            last_hidden=last_hidden,
            prompt_latent=prompt_latent,
            context_latent=context_latent,
            has_history=has_history,
            context_memory=context_memory,
            context_memory_mask=context_memory_mask,
        )
        q = self.query_norm(self.state_to_candidate(state))
        q_exp = q.unsqueeze(1).expand(-1, support_idx.size(1), -1)

        token_features = self._lookup_candidate_features(support_idx).to(dtype=last_hidden.dtype)
        token_proj = self.token_proj(self.token_norm(token_features))
        token_n = self.candidate_norm(token_proj)
        inv_sqrt_d = 1.0 / float(getattr(self.args, "candidate_dim", 512)) ** 0.5
        pair_dot = (q_exp * token_n).sum(dim=-1) * inv_sqrt_d

        support_mean = masked_row_mean(support_logits, support_mask)
        support_var = masked_row_var(support_logits, support_mask, support_mean)
        support_logit_z = (support_logits - support_mean.unsqueeze(-1)) / torch.sqrt(support_var.unsqueeze(-1) + 1e-6)
        scalar_feats = torch.stack([support_logit_z, support_probs, rank_frac], dim=-1).to(dtype=token_proj.dtype)
        pair_mlp_in = torch.cat([q_exp, token_proj, q_exp * token_proj, scalar_feats], dim=-1)
        pair_mlp = self.pair_mlp(pair_mlp_in).squeeze(-1)

        direction_raw = (pair_dot + pair_mlp).masked_fill(~active_support_mask, 0.0)
        direction_norm, direction_var = weighted_masked_standardize(
            direction_raw,
            weight=support_probs,
            mask=active_support_mask,
        )

        max_abs_lambda = max(
            abs(float(getattr(self.args, "oracle_lambda_min", -1.5))),
            abs(float(getattr(self.args, "oracle_lambda_max", 1.5))),
        )
        signed_lambda = max_abs_lambda * torch.tanh(self.lambda_head(state).squeeze(-1))
        signed_lambda = signed_lambda.clamp(
            min=float(getattr(self.args, "oracle_lambda_min", -1.5)),
            max=float(getattr(self.args, "oracle_lambda_max", 1.5)),
        )

        mask_f = active_support_mask.to(dtype=support_probs.dtype)
        support_mass = (support_probs * mask_f).sum(dim=-1, keepdim=True)
        support_max = support_probs.masked_fill(~active_support_mask, 0.0).max(dim=-1, keepdim=True).values
        support_frac = mask_f.mean(dim=-1, keepdim=True)
        gate_in = torch.cat(
            [state, support_mass.to(state.dtype), support_max.to(state.dtype), support_frac.to(state.dtype)],
            dim=-1,
        )
        gate_logits = self.gate_head(gate_in).squeeze(-1)
        gate = torch.sigmoid(gate_logits)

        if has_history is not None:
            hist = has_history.bool().view(-1)
            signed_lambda = torch.where(hist, signed_lambda, torch.zeros_like(signed_lambda))
            gate = torch.where(hist, gate, torch.zeros_like(gate))
            gate_logits = torch.where(hist, gate_logits, torch.full_like(gate_logits, -30.0))

        lambda_pred = gate * signed_lambda
        delta_support = lambda_pred.unsqueeze(-1) * direction_norm
        if float(getattr(self.args, "score_clip", 3.0)) > 0:
            c = float(getattr(self.args, "score_clip", 3.0))
            delta_support = c * torch.tanh(delta_support / c)
        delta_support = float(self.runtime_score_scale) * delta_support
        delta_support = delta_support.masked_fill(~active_support_mask, 0.0)
        if bool(int(getattr(self.args, "zero_mean_scores", 1))):
            delta_support = weighted_masked_recenter(delta_support, weight=support_probs, mask=active_support_mask)

        return {
            "direction_support": direction_norm,
            "direction_var": direction_var,
            "lambda_pred": lambda_pred,
            "signed_lambda": signed_lambda,
            "gate": gate,
            "gate_logits": gate_logits,
            "delta_support": delta_support,
            "pair_dot": pair_dot,
            "pair_mlp": pair_mlp,
        }

    def compute_support_reranking(
        self,
        last_hidden: torch.Tensor,
        prompt_latent: torch.Tensor,
        fixed_memory_vec: torch.Tensor,
        has_history: Optional[torch.Tensor] = None,
        context_memory: Optional[torch.Tensor] = None,
        context_memory_mask: Optional[torch.Tensor] = None,
        force_labels: Optional[torch.Tensor] = None,
        extra_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base_logits = self.output_head(last_hidden).float()
        base_logp = F.log_softmax(base_logits, dim=-1)
        base_probs = base_logp.exp()
        entropy = -(base_probs * base_logp).sum(dim=-1, keepdim=True)
        norm_entropy = entropy / torch.log(torch.tensor(float(self.vocab_size), device=base_logits.device))
        labels = force_labels if (force_labels is not None and bool(int(getattr(self.args, "train_force_label_in_support", 1)))) else None
        support = self._build_support(base_logits, base_logp, base_probs, norm_entropy, labels=labels, extra_ids=extra_ids)

        support_idx = support["support_idx"]
        support_mask = support["support_mask"]
        active_support_mask = support_mask
        eos_id = int(getattr(self.args, "eos_token_id", -1))
        if eos_id >= 0 and not bool(int(getattr(self.args, "personalize_eos", 0))):
            active_support_mask = active_support_mask & (support_idx != eos_id)

        student = self._student_delta(
            last_hidden=last_hidden,
            prompt_latent=prompt_latent,
            context_latent=fixed_memory_vec,
            has_history=has_history,
            context_memory=context_memory,
            context_memory_mask=context_memory_mask,
            support_idx=support_idx,
            support_mask=support_mask,
            active_support_mask=active_support_mask,
            support_logits=support["support_logits"],
            support_probs=support["support_probs"],
            rank_frac=support["rank_frac"],
        )
        delta_support = student["delta_support"]
        steered_support_logits = support["support_logits"] + delta_support
        return {
            "base_logits": base_logits,
            "base_logp": base_logp,
            "base_probs": base_probs,
            "norm_entropy": norm_entropy,
            "support_idx": support_idx,
            "support_mask": support_mask,
            "active_support_mask": active_support_mask,
            "support_logits": support["support_logits"],
            "support_logp": support["support_logp"],
            "support_probs": support["support_probs"],
            "steered_support_logits": steered_support_logits,
            "delta_support": delta_support,
            "direction_support": student["direction_support"],
            "lambda_pred": student["lambda_pred"],
            "signed_lambda": student["signed_lambda"],
            "gate": student["gate"],
            "gate_logits": student["gate_logits"],
            "k_counts": support["k_counts"],
            "top_gap": support["top_gap"],
            "top_ext_idx": support["top_ext_idx"],
            "top_ext_logits": support["top_ext_logits"],
            "rerank_dot_abs_mean": masked_abs_mean_2d(student["pair_dot"], active_support_mask).detach(),
            "rerank_mlp_abs_mean": masked_abs_mean_2d(student["pair_mlp"], active_support_mask).detach(),
            "delta_abs_mean": masked_abs_mean_2d(delta_support, active_support_mask).detach(),
        }

    def _distill_terms(
        self,
        info: Dict[str, torch.Tensor],
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
        row = torch.arange(labels.size(0), device=labels.device)
        teacher_gold_logp = teacher_logp[row, labels]
        base_gold_logp = info["base_logp"][row, labels]
        advantage = teacher_gold_logp - base_gold_logp

        teacher_support_logp = teacher_logp.gather(dim=-1, index=info["support_idx"])
        raw_delta = teacher_support_logp - info["support_logp"]
        active_mask = info["active_support_mask"]
        direction_target, target_var = weighted_masked_standardize(
            raw_delta,
            weight=info["support_probs"],
            mask=active_mask,
        )
        lambda_star, has_label = self._oracle_lambda(
            base_support_logp=info["support_logp"],
            direction_target=direction_target,
            labels=labels,
            support_idx=info["support_idx"],
            active_mask=active_mask,
        )

        margin = float(getattr(self.args, "adv_margin_tok", 0.05))
        temp = max(float(getattr(self.args, "adv_temp_tok", 0.10)), 1e-6)
        reliable = content_mask.bool() & has_label & (target_var > float(getattr(self.args, "distill_min_delta_var", 1e-5)))
        helpful = reliable & (advantage > margin)
        preserve = content_mask.bool() & (~helpful)

        target_delta = lambda_star.unsqueeze(-1) * direction_target
        teacher_logq = masked_log_softmax(info["support_logits"] + target_delta, active_mask, dim=-1)
        teacher_q = teacher_logq.exp() * active_mask.to(teacher_logq.dtype)
        base_logq = masked_log_softmax(info["support_logits"], active_mask, dim=-1)
        base_q = base_logq.exp() * active_mask.to(base_logq.dtype)
        student_logq = masked_log_softmax(info["support_logits"] + info["delta_support"], active_mask, dim=-1)

        teacher_kl_row = (teacher_q * (teacher_logq - student_logq)).sum(dim=-1)
        base_kl_row = (base_q * (base_logq - student_logq)).sum(dim=-1)
        helpful_kl = teacher_kl_row[helpful].mean() if helpful.any() else self._zero(info["support_logits"])
        base_preserve_kl = base_kl_row[preserve].mean() if preserve.any() else self._zero(info["support_logits"])

        gate_target_soft = torch.sigmoid((advantage - margin) / temp).detach()
        gate_target = torch.where(helpful, gate_target_soft, torch.zeros_like(gate_target_soft))
        gate = info.get("gate", torch.ones_like(gate_target))
        gate_logits = info.get("gate_logits", torch.logit(gate.clamp(1e-6, 1.0 - 1e-6)))
        gate_loss_raw = F.binary_cross_entropy_with_logits(
            gate_logits.float(),
            gate_target.float(),
            reduction="none",
        )
        gate_mask = content_mask.bool()
        gate_loss = gate_loss_raw[gate_mask].mean() if gate_mask.any() else self._zero(info["support_logits"])

        return {
            "distill_kl": helpful_kl,
            "distill_delta_loss": self._zero(info["support_logits"]),
            "distill_lambda_loss": self._zero(info["support_logits"]),
            "oracle_lambda_mean": lambda_star[helpful].mean() if helpful.any() else self._zero(info["support_logits"]),
            "lambda_pred_mean": info["lambda_pred"][gate_mask].mean() if gate_mask.any() else self._zero(info["support_logits"]),
            "distill_valid_frac": helpful.float().mean(),
            "gate_loss": gate_loss,
            "base_preserve_kl": base_preserve_kl,
            "helpful_distill_kl": helpful_kl,
            "gate_mean": gate[gate_mask].mean() if gate_mask.any() else self._zero(info["support_logits"]),
            "gate_target_mean": gate_target[gate_mask].mean() if gate_mask.any() else self._zero(info["support_logits"]),
            "teacher_advantage_mean": advantage[gate_mask].mean() if gate_mask.any() else self._zero(info["support_logits"]),
            "helpful_frac": helpful.float().mean(),
            "signed_lambda_mean": info.get("signed_lambda", info["lambda_pred"])[gate_mask].mean()
            if gate_mask.any()
            else self._zero(info["support_logits"]),
        }

    def _score_side(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        context: Dict[str, torch.Tensor],
        positive: bool,
        teacher_input_ids: Optional[torch.Tensor] = None,
        teacher_attention_mask: Optional[torch.Tensor] = None,
        teacher_target_ids: Optional[torch.Tensor] = None,
    ) -> CautiousSideOutput:
        batch_size = input_ids.size(0)
        hidden = self._backbone_forward(input_ids=input_ids, attention_mask=attention_mask)
        ex_idx, time_idx, labels = flatten_target_positions(target_ids)
        if labels.numel() == 0:
            raise ValueError("A batch contained zero answer tokens after truncation.")

        selected_hidden = hidden[ex_idx, time_idx, :]
        prompt_sel = context["prompt_latent"][ex_idx]
        memory_sel = context["memory_vec"][ex_idx]
        context_memory_sel = context["context_memory"][ex_idx]
        context_memory_mask_sel = context["context_memory_mask"][ex_idx]
        has_sel = context["has_history"][ex_idx]

        teacher_logits = None
        teacher_extra_ids = None
        if positive and teacher_input_ids is not None and teacher_attention_mask is not None and teacher_target_ids is not None:
            teacher_hidden = self._backbone_forward(teacher_input_ids, teacher_attention_mask)
            t_ex, t_time, t_labels = flatten_target_positions(teacher_target_ids)
            if t_labels.numel() != labels.numel():
                n = min(t_labels.numel(), labels.numel())
                ex_idx = ex_idx[:n]
                labels = labels[:n]
                selected_hidden = selected_hidden[:n]
                prompt_sel = prompt_sel[:n]
                memory_sel = memory_sel[:n]
                context_memory_sel = context_memory_sel[:n]
                context_memory_mask_sel = context_memory_mask_sel[:n]
                has_sel = has_sel[:n]
                t_ex = t_ex[:n]
                t_time = t_time[:n]
            teacher_selected_hidden = teacher_hidden[t_ex, t_time, :]
            teacher_logits = self.output_head(teacher_selected_hidden).float()
            k_teacher = min(int(getattr(self.args, "distill_teacher_top_k", 32)), teacher_logits.size(-1))
            if k_teacher > 0:
                teacher_extra_ids = torch.topk(teacher_logits.detach(), k=k_teacher, dim=-1).indices

        force_chosen = self.training and positive and bool(int(getattr(self.args, "train_force_label_in_support", 1)))
        force_rejected = self.training and (not positive) and bool(int(getattr(self.args, "train_force_rejected_label_in_support", 0)))
        force_labels = labels if (force_chosen or force_rejected) else None
        info = self.compute_support_reranking(
            last_hidden=selected_hidden,
            prompt_latent=prompt_sel,
            fixed_memory_vec=memory_sel,
            has_history=has_sel,
            context_memory=context_memory_sel,
            context_memory_mask=context_memory_mask_sel,
            force_labels=force_labels,
            extra_ids=teacher_extra_ids,
        )
        exact = self._exact_logprob_terms(
            base_logp=info["base_logp"],
            support_idx=info["support_idx"],
            support_probs=info["support_probs"],
            support_mask=info["support_mask"],
            delta_support=info["delta_support"],
            labels=labels,
        )

        eos_id = int(getattr(self.args, "eos_token_id", -1))
        content_mask = torch.ones_like(labels, dtype=torch.bool)
        if eos_id >= 0 and bool(int(getattr(self.args, "ignore_eos_in_loss", 1))):
            content_mask = labels != eos_id

        ce_loss = -exact["steered_label_lp"][content_mask].mean() if (positive and content_mask.any()) else self._zero(selected_hidden)
        pred_token = self._full_vocab_pred_tokens(
            support_idx=info["support_idx"],
            support_mask=info["support_mask"],
            steered_support_logits=info["steered_support_logits"],
            top_ext_idx=info["top_ext_idx"],
            top_ext_logits=info["top_ext_logits"],
        )
        base_pred_token = info["top_ext_idx"][:, 0]

        content_f = content_mask.to(dtype=exact["steered_label_lp"].dtype)
        seq_lp = scatter_sum_scalar(exact["steered_label_lp"] * content_f, ex_idx, batch_size)
        base_seq_lp = scatter_sum_scalar(exact["base_label_lp"] * content_f, ex_idx, batch_size)
        total_counts = scatter_sum_scalar(content_f, ex_idx, batch_size)
        covered_counts = scatter_sum_scalar(exact["has_label"].to(content_f.dtype) * content_f, ex_idx, batch_size)
        if bool(getattr(self.args, "length_normalize", False)):
            denom = total_counts.clamp_min(1.0)
            seq_lp = seq_lp / denom
            base_seq_lp = base_seq_lp / denom

        if teacher_logits is not None:
            distill = self._distill_terms(info, teacher_logits=teacher_logits, labels=labels, content_mask=content_mask)
        else:
            zero = self._zero(selected_hidden)
            distill = {
                "distill_kl": zero,
                "distill_delta_loss": zero,
                "distill_lambda_loss": zero,
                "oracle_lambda_mean": zero,
                "lambda_pred_mean": info["lambda_pred"].mean(),
                "distill_valid_frac": zero,
                "gate_loss": zero,
                "base_preserve_kl": zero,
                "helpful_distill_kl": zero,
                "gate_mean": info.get("gate", info["lambda_pred"]).mean(),
                "gate_target_mean": zero,
                "teacher_advantage_mean": zero,
                "helpful_frac": zero,
                "signed_lambda_mean": info.get("signed_lambda", info["lambda_pred"]).mean(),
            }

        return CautiousSideOutput(
            label_seq_logprob=seq_lp,
            base_label_seq_logprob=base_seq_lp,
            total_token_count=total_counts,
            covered_token_count=covered_counts,
            chosen_ce_loss=ce_loss,
            token_acc=(pred_token == labels).float().mean(),
            base_token_acc=(base_pred_token == labels).float().mean(),
            support_size_mean=info["k_counts"].float().mean(),
            entropy_mean=info["norm_entropy"].mean(),
            coverage=exact["has_label"].float().mean(),
            label_logprob_mean=exact["steered_label_lp"].mean(),
            base_label_logprob_mean=exact["base_label_lp"].mean(),
            distill_kl=distill["distill_kl"],
            distill_delta_loss=distill["distill_delta_loss"],
            distill_lambda_loss=distill["distill_lambda_loss"],
            oracle_lambda_mean=distill["oracle_lambda_mean"],
            lambda_pred_mean=distill["lambda_pred_mean"],
            distill_valid_frac=distill["distill_valid_frac"],
            gate_loss=distill["gate_loss"],
            base_preserve_kl=distill["base_preserve_kl"],
            helpful_distill_kl=distill["helpful_distill_kl"],
            gate_mean=distill["gate_mean"],
            gate_target_mean=distill["gate_target_mean"],
            teacher_advantage_mean=distill["teacher_advantage_mean"],
            helpful_frac=distill["helpful_frac"],
            signed_lambda_mean=distill["signed_lambda_mean"],
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context, user_info = self._encode_context(batch)
        chosen = self._score_side(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            target_ids=batch["chosen_target_ids"],
            context=context,
            positive=True,
            teacher_input_ids=batch.get("teacher_chosen_input_ids"),
            teacher_attention_mask=batch.get("teacher_chosen_attention_mask"),
            teacher_target_ids=batch.get("teacher_chosen_target_ids"),
        )
        rejected = self._score_side(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            target_ids=batch["rejected_target_ids"],
            context=context,
            positive=False,
        )

        zero = chosen.chosen_ce_loss * 0.0
        pair_valid = (chosen.total_token_count > 0) & (rejected.total_token_count > 0)
        pairwise_weight = float(getattr(self.args, "pairwise_pref_weight", 0.0))
        hinge_weight = float(getattr(self.args, "chosen_gain_hinge_weight", 0.0))
        do_pair = pair_valid.any() and (pairwise_weight > 0.0 or hinge_weight > 0.0)
        if do_pair:
            chosen_gain = chosen.label_seq_logprob - chosen.base_label_seq_logprob
            rejected_gain = rejected.label_seq_logprob - rejected.base_label_seq_logprob
            pair_margin = (chosen_gain - rejected_gain)[pair_valid]
            preference_acc = (pair_margin > 0).float().mean()
            absolute_preference_acc = (chosen.label_seq_logprob > rejected.label_seq_logprob).float().mean()
            chosen_gain_mean = chosen_gain.mean()
            rejected_gain_mean = rejected_gain.mean()
            margin_gain_mean = pair_margin.mean()
            if pairwise_weight > 0:
                base_margin = (chosen.base_label_seq_logprob - rejected.base_label_seq_logprob)[pair_valid]
                hardness = torch.sigmoid(
                    (float(getattr(self.args, "hard_pair_margin", 0.0)) - base_margin.detach())
                    / max(float(getattr(self.args, "hard_pair_temperature", 0.25)), 1e-6)
                )
                min_w = float(getattr(self.args, "hard_pair_min_weight", 0.25))
                hardness = min_w + (1.0 - min_w) * hardness
                raw = -F.logsigmoid(float(getattr(self.args, "pairwise_beta", 1.0)) * pair_margin)
                pairwise_loss = (hardness * raw).sum() / hardness.sum().clamp_min(1e-8)
                hard_weight_mean = hardness.mean()
                base_margin_mean = base_margin.mean()
            else:
                pairwise_loss = zero
                hard_weight_mean = zero
                base_margin_mean = zero
            chosen_gain_hinge = (
                F.relu(float(getattr(self.args, "chosen_gain_margin", 0.0)) - chosen_gain[pair_valid]).mean()
                if hinge_weight > 0
                else zero
            )
        else:
            pairwise_loss = zero
            preference_acc = zero
            absolute_preference_acc = zero
            hard_weight_mean = zero
            base_margin_mean = zero
            chosen_gain_mean = zero
            rejected_gain_mean = zero
            margin_gain_mean = zero
            chosen_gain_hinge = zero

        total_loss = (
            float(getattr(self.args, "distill_kl_weight", 1.0)) * chosen.helpful_distill_kl
            + float(getattr(self.args, "base_preserve_weight", 0.5)) * chosen.base_preserve_kl
            + float(getattr(self.args, "gate_loss_weight", 0.2)) * chosen.gate_loss
            + float(getattr(self.args, "chosen_ce_weight", 0.0)) * chosen.chosen_ce_loss
            + pairwise_weight * pairwise_loss
            + hinge_weight * chosen_gain_hinge
        )
        entropy_mean = 0.5 * (chosen.entropy_mean + rejected.entropy_mean)
        support_size_mean = 0.5 * (chosen.support_size_mean + rejected.support_size_mean)
        return {
            "loss": total_loss,
            "distill_kl": chosen.helpful_distill_kl.detach(),
            "base_preserve_kl": chosen.base_preserve_kl.detach(),
            "gate_loss": chosen.gate_loss.detach(),
            "chosen_ce_loss": chosen.chosen_ce_loss.detach(),
            "pairwise_pref_loss": pairwise_loss.detach(),
            "chosen_gain_hinge": chosen_gain_hinge.detach(),
            "entropy_mean": entropy_mean.detach(),
            "support_size_mean": support_size_mean.detach(),
            "chosen_coverage": chosen.coverage.detach(),
            "rejected_coverage": rejected.coverage.detach(),
            "chosen_token_acc": chosen.token_acc.detach(),
            "chosen_base_token_acc": chosen.base_token_acc.detach(),
            "rejected_token_acc": rejected.token_acc.detach(),
            "rejected_base_token_acc": rejected.base_token_acc.detach(),
            "preference_acc": preference_acc.detach(),
            "absolute_preference_acc": absolute_preference_acc.detach(),
            "covered_pair_frac": pair_valid.float().mean().detach(),
            "chosen_gain_mean": chosen_gain_mean.detach(),
            "rejected_gain_mean": rejected_gain_mean.detach(),
            "margin_gain_mean": margin_gain_mean.detach(),
            "base_margin_mean": base_margin_mean.detach(),
            "hard_weight_mean": hard_weight_mean.detach(),
            "chosen_logprob_mean": chosen.label_logprob_mean.detach(),
            "chosen_base_logprob_mean": chosen.base_label_logprob_mean.detach(),
            "rejected_logprob_mean": rejected.label_logprob_mean.detach(),
            "rejected_base_logprob_mean": rejected.base_label_logprob_mean.detach(),
            "oracle_lambda_mean": chosen.oracle_lambda_mean.detach(),
            "lambda_pred_mean": chosen.lambda_pred_mean.detach(),
            "signed_lambda_mean": chosen.signed_lambda_mean.detach(),
            "gate_mean": chosen.gate_mean.detach(),
            "gate_target_mean": chosen.gate_target_mean.detach(),
            "teacher_advantage_mean": chosen.teacher_advantage_mean.detach(),
            "helpful_frac": chosen.helpful_frac.detach(),
            "distill_valid_frac": chosen.distill_valid_frac.detach(),
            "runtime_score_scale": torch.tensor(self.runtime_score_scale, device=total_loss.device),
            "prompt_latent_norm": user_info["prompt_latent_norm"].detach(),
            "context_latent_norm": user_info["context_latent_norm"].detach(),
            "history_present_frac": user_info["history_present_frac"].detach(),
            "history_pairs_used": user_info["history_pairs_used"].detach(),
            "history_attn_entropy": user_info["history_attn_entropy"].detach(),
            "history_attn_max": user_info["history_attn_max"].detach(),
            "history_latent_norm": user_info["history_latent_norm"].detach(),
            "user_latent_norm": user_info["user_latent_norm"].detach(),
        }


ContextSteeringDistillModel = CautiousContextSteeringDistillModel

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    if g.ndim != 2:
        raise ValueError("Muon zeropower update expects a 2D tensor")
    if g.numel() == 0:
        return g
    x = g.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / x.norm().clamp_min(1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(1, int(steps))):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon received a non-2D parameter")
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1.0 - momentum)
                update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                scale = math.sqrt(max(1.0, p.size(0) / max(1, p.size(1))))
                p.add_(update, alpha=-lr * scale)
        return loss


class HybridMuonAdamW:
    def __init__(
        self,
        muon_params: Sequence[torch.nn.Parameter],
        adamw_groups: Sequence[Dict[str, Any]],
        lr: float,
        weight_decay: float,
        muon_momentum: float,
        muon_ns_steps: int,
    ):
        self.muon = Muon(
            muon_params,
            lr=lr,
            momentum=muon_momentum,
            weight_decay=weight_decay,
            ns_steps=muon_ns_steps,
        )
        self.adamw = torch.optim.AdamW(adamw_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optimizer_name: str = "hybrid_muon",
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
) -> Any:
    muon_params, decay, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if optimizer_name == "hybrid_muon" and param.ndim == 2:
            muon_params.append(param)
            continue
        if param.ndim == 1 or name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    adamw_groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if optimizer_name == "adamw" or not muon_params:
        return torch.optim.AdamW(adamw_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    return HybridMuonAdamW(
        muon_params=muon_params,
        adamw_groups=adamw_groups,
        lr=lr,
        weight_decay=weight_decay,
        muon_momentum=muon_momentum,
        muon_ns_steps=muon_ns_steps,
    )


class CosineWithWarmup:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.step_num = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            scale = self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def set_model_schedule_fraction(model: nn.Module, fraction: float) -> None:
    core = model.module if isinstance(model, DDP) else model
    if hasattr(core, "set_schedule_fraction"):
        core.set_schedule_fraction(fraction)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    schedule_fraction: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    set_model_schedule_fraction(model, schedule_fraction)
    metrics_sum: Dict[str, float] = {}
    count = 0
    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if (use_amp and device.type == "cuda") else nullcontext()
        with amp_ctx:
            outputs = model(batch)
        for key, value in outputs.items():
            if torch.is_tensor(value):
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value.item())
        count += 1
    if count == 0:
        model.train()
        return {}
    metrics = {k: v / count for k, v in metrics_sum.items()}
    metrics = {k: all_reduce_scalar(v, device, average=True) for k, v in metrics.items()}
    model.train()
    return metrics


def _remove_old_checkpoints(output_dir: Path, keep_last_n: int) -> None:
    checkpoints = sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("step_")])
    while len(checkpoints) > keep_last_n:
        old = checkpoints.pop(0)
        for child in old.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(old.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        old.rmdir()


def save_checkpoint(model: nn.Module, tokenizer: Any, args: argparse.Namespace, step: int, output_dir: Path) -> None:
    if not is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / ("last" if step == 0 else f"step_{step:07d}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    core = model.module if isinstance(model, DDP) else model
    state = core.steering_state_dict()
    torch.save(state, ckpt_dir / "steering.pt")
    with open(ckpt_dir / "steering_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")
    _remove_old_checkpoints(output_dir, int(getattr(args, "keep_last_n", 3)))


def maybe_load_resume(model: nn.Module, resume_path: str) -> None:
    if not resume_path:
        return
    p = Path(resume_path)
    if p.is_dir():
        p = p / "steering.pt"
    state = torch.load(p, map_location="cpu")
    core = model.module if isinstance(model, DDP) else model
    core.load_steering_state_dict(state)
    if is_main_process():
        print(f"[resume] loaded steering weights from {p}")


def main() -> None:
    args = parse_args()
    ddp, rank, world_size, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(int(args.seed) + rank)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        print(f"[setup] ddp={ddp} rank={rank} world_size={world_size} device={device}")

    tokenizer, handles = load_base_handles(args, device)
    args.eos_token_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1
    collator = PreferenceCollator(tokenizer, args)

    wandb_run = None
    if args.use_wandb and wandb is None and is_main_process():
        print("[warn] wandb is not installed; disabling wandb logging.")
    if args.use_wandb and wandb is not None and is_main_process():
        init_kwargs = {"project": args.wandb_project, "config": vars(args), "mode": args.wandb_mode}
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        wandb_run = wandb.init(**init_kwargs)

    train_dataset = JsonlDataset(args.train_jsonl, require_history=bool(int(args.require_history)))
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.per_device_batch_size),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collator,
        num_workers=int(args.num_workers),
        pin_memory=False,
        drop_last=False,
    )

    valid_loader = None
    if args.valid_jsonl:
        valid_dataset = JsonlDataset(args.valid_jsonl, require_history=bool(int(args.require_history)))
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if ddp else None
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(args.per_device_batch_size),
            sampler=valid_sampler,
            shuffle=False,
            collate_fn=collator,
            num_workers=max(1, int(args.num_workers) // 2),
            pin_memory=False,
            drop_last=False,
        )

    model = ContextSteeringDistillModel(handles, args).to(device)
    if args.resume_steering:
        maybe_load_resume(model, args.resume_steering)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"[params] trainable={trainable:,} total_in_wrapper={total:,}")

    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, broadcast_buffers=False, find_unused_parameters=True)

    optimizer = build_optimizer(
        model,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        optimizer_name=str(args.optimizer),
        muon_momentum=float(args.muon_momentum),
        muon_ns_steps=int(args.muon_ns_steps),
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, int(args.gradient_accumulation_steps)))
    total_steps = int(args.max_steps) if int(args.max_steps) > 0 else steps_per_epoch * int(args.num_epochs)
    warmup_steps = max(1, int(total_steps * float(args.warmup_ratio)))
    scheduler = CosineWithWarmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    use_amp = args.dtype in {"bfloat16", "float16"}
    amp_dtype = str_to_dtype(args.dtype)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    running: Dict[str, float] = {}
    running_count = 0
    start_time = time.time()
    stop_training = False

    for epoch in range(int(args.num_epochs) if int(args.max_steps) <= 0 else 10**9):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}", dynamic_ncols=True) if is_main_process() else None
        for step_in_epoch, batch in enumerate(train_loader):
            progress = global_step / max(1, total_steps)
            set_model_schedule_fraction(model, progress)
            batch = move_batch_to_device(batch, device)
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if (use_amp and device.type == "cuda") else nullcontext()
            with amp_ctx:
                outputs = model(batch)
                loss = outputs["loss"] / int(args.gradient_accumulation_steps)
            loss.backward()

            for k, v in outputs.items():
                if torch.is_tensor(v):
                    running[k] = running.get(k, 0.0) + float(v.item())
            running_count += 1

            should_step = ((step_in_epoch + 1) % int(args.gradient_accumulation_steps) == 0) or ((step_in_epoch + 1) == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{outputs['loss'].item():.4f}",
                        "kl": f"{outputs['distill_kl'].item():.4f}",
                        "ce": f"{outputs['chosen_ce_loss'].item():.4f}",
                        "lam": f"{outputs['lambda_pred_mean'].item():.3f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })

                if global_step % int(args.log_every) == 0:
                    elapsed = time.time() - start_time
                    denom = max(1, running_count)
                    reduced = {k: all_reduce_scalar(v / denom, device, average=True) for k, v in running.items()}
                    if args.use_wandb and wandb is not None and is_main_process():
                        wandb.log(
                            {
                                "train/global_step": global_step,
                                "train/lr": scheduler.get_last_lr()[0],
                                "train/elapsed_min": elapsed / 60.0,
                                **{f"train/{k}": v for k, v in reduced.items()},
                            },
                            step=global_step,
                        )
                    running = {}
                    running_count = 0

                if valid_loader is not None and global_step % int(args.eval_every) == 0:
                    if ddp and isinstance(valid_loader.sampler, DistributedSampler):
                        valid_loader.sampler.set_epoch(global_step)
                    eval_metrics = evaluate(
                        model=model,
                        data_loader=valid_loader,
                        device=device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        schedule_fraction=global_step / max(1, total_steps),
                    )
                    if args.use_wandb and wandb is not None and is_main_process() and eval_metrics:
                        wandb.log({"eval/global_step": global_step, **{f"eval/{k}": v for k, v in eval_metrics.items()}}, step=global_step)
                    if is_main_process() and eval_metrics:
                        print("[eval] " + json.dumps(eval_metrics, ensure_ascii=False))

                if global_step % int(args.save_every) == 0:
                    save_checkpoint(model, tokenizer, args, global_step, output_dir)
                if global_step >= total_steps:
                    stop_training = True
                    break
        if pbar is not None:
            pbar.close()
        if stop_training:
            break

    if args.use_wandb and wandb is not None and is_main_process() and wandb_run is not None:
        wandb.finish()
    barrier()
    set_model_schedule_fraction(model, 1.0)
    save_checkpoint(model, tokenizer, args, 0, output_dir)
    if is_main_process():
        print(f"[done] training finished at step={global_step}")
    ddp_cleanup()


if __name__ == "__main__":
    main()
