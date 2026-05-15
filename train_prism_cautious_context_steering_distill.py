#!/usr/bin/env python3
"""
Cautious PRISM context-steering distillation.

This variant reuses the base trainer in train_prism_context_steering_distill.py,
but changes the learned steering objective:

- when the ICL/CoS teacher gives positive gold-token advantage over base, distill
  the oracle-steered teacher distribution;
- otherwise preserve the base distribution and train an explicit gate to abstain.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_prism_context_steering_distill as base


load_base_handles = base.load_base_handles
str_to_dtype = base.str_to_dtype
PreferenceCollator = base.PreferenceCollator
JsonlDataset = base.JsonlDataset
BaseHandles = base.BaseHandles
NEG_INF = base.NEG_INF
render_prompt = base.render_prompt
render_user_context = base.render_user_context
extract_history_pairs = base.extract_history_pairs
build_history_exemplar_prompt = base.build_history_exemplar_prompt
build_history_exemplar_prefix = base.build_history_exemplar_prefix

_base_parse_args = base.parse_args


def parse_args() -> argparse.Namespace:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--adv_margin_tok", type=float, default=0.05)
    wrapper.add_argument("--adv_temp_tok", type=float, default=0.10)
    wrapper.add_argument("--gate_loss_weight", type=float, default=0.20)
    wrapper.add_argument("--base_preserve_weight", type=float, default=0.50)
    wrapper.add_argument("--gate_init_bias", type=float, default=-2.0)
    wrapper.add_argument("--gate_hidden", type=int, default=256)
    known, remaining = wrapper.parse_known_args()

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + remaining
        args = _base_parse_args()
    finally:
        sys.argv = old_argv

    args.adv_margin_tok = float(known.adv_margin_tok)
    args.adv_temp_tok = float(known.adv_temp_tok)
    args.gate_loss_weight = float(known.gate_loss_weight)
    args.base_preserve_weight = float(known.base_preserve_weight)
    args.gate_init_bias = float(known.gate_init_bias)
    args.gate_hidden = int(known.gate_hidden)
    args.distill_variant = "cautious_context_steering"
    return args


@dataclass
class CautiousSideOutput(base.SideOutput):
    gate_loss: torch.Tensor
    base_preserve_kl: torch.Tensor
    helpful_distill_kl: torch.Tensor
    gate_mean: torch.Tensor
    gate_target_mean: torch.Tensor
    teacher_advantage_mean: torch.Tensor
    helpful_frac: torch.Tensor
    signed_lambda_mean: torch.Tensor


class CautiousContextSteeringDistillModel(base.ContextSteeringDistillModel):
    def __init__(self, handles: BaseHandles, args: argparse.Namespace | SimpleNamespace):
        super().__init__(handles, args)
        adapter_dim = int(getattr(args, "adapter_dim", 512))
        hidden = int(getattr(args, "gate_hidden", 256))
        dropout = float(getattr(args, "module_dropout", 0.05))
        self.gate_head = nn.Sequential(
            base.RMSNorm(adapter_dim + 3),
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

        support_mean = base.masked_row_mean(support_logits, support_mask)
        support_var = base.masked_row_var(support_logits, support_mask, support_mean)
        support_logit_z = (support_logits - support_mean.unsqueeze(-1)) / torch.sqrt(support_var.unsqueeze(-1) + 1e-6)
        scalar_feats = torch.stack([support_logit_z, support_probs, rank_frac], dim=-1).to(dtype=token_proj.dtype)
        pair_mlp_in = torch.cat([q_exp, token_proj, q_exp * token_proj, scalar_feats], dim=-1)
        pair_mlp = self.pair_mlp(pair_mlp_in).squeeze(-1)

        direction_raw = (pair_dot + pair_mlp).masked_fill(~active_support_mask, 0.0)
        direction_norm, direction_var = base.weighted_masked_standardize(
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
            delta_support = base.weighted_masked_recenter(delta_support, weight=support_probs, mask=active_support_mask)

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
            "rerank_dot_abs_mean": base.masked_abs_mean_2d(student["pair_dot"], active_support_mask).detach(),
            "rerank_mlp_abs_mean": base.masked_abs_mean_2d(student["pair_mlp"], active_support_mask).detach(),
            "delta_abs_mean": base.masked_abs_mean_2d(delta_support, active_support_mask).detach(),
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
        direction_target, target_var = base.weighted_masked_standardize(
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
        teacher_logq = base.masked_log_softmax(info["support_logits"] + target_delta, active_mask, dim=-1)
        teacher_q = teacher_logq.exp() * active_mask.to(teacher_logq.dtype)
        base_logq = base.masked_log_softmax(info["support_logits"], active_mask, dim=-1)
        base_q = base_logq.exp() * active_mask.to(base_logq.dtype)
        student_logq = base.masked_log_softmax(info["support_logits"] + info["delta_support"], active_mask, dim=-1)

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
        ex_idx, time_idx, labels = base.flatten_target_positions(target_ids)
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
            t_ex, t_time, t_labels = base.flatten_target_positions(teacher_target_ids)
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
        seq_lp = base.scatter_sum_scalar(exact["steered_label_lp"] * content_f, ex_idx, batch_size)
        base_seq_lp = base.scatter_sum_scalar(exact["base_label_lp"] * content_f, ex_idx, batch_size)
        total_counts = base.scatter_sum_scalar(content_f, ex_idx, batch_size)
        covered_counts = base.scatter_sum_scalar(exact["has_label"].to(content_f.dtype) * content_f, ex_idx, batch_size)
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


PengramModel = CautiousContextSteeringDistillModel
ContextSteeringDistillModel = CautiousContextSteeringDistillModel


def main() -> None:
    base.parse_args = parse_args
    base.ContextSteeringDistillModel = CautiousContextSteeringDistillModel
    base.PengramModel = CautiousContextSteeringDistillModel
    base.main()


if __name__ == "__main__":
    main()
