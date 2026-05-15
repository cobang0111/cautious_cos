#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

model_name="${1:-Qwen3-0.6B}"
version_name="${2:-cautious_context_steering}"
device="${3:-0}"

train_jsonl="${TRAIN_JSONL:-data/prism_pengram_splits/seen_train.jsonl}"
valid_jsonl="${VALID_JSONL:-data/prism_pengram_splits/seen_valid.jsonl}"
prism_support_jsonl="${SUPPORT_JSONL:-data/prism_pengram_splits/calib_unseen.jsonl}"
prism_query_jsonl="${QUERY_JSONL:-data/prism_pengram_splits/test_unseen.jsonl}"

run_dir="${RUN_DIR:-runs/prism_cautious_context_steering_distill_${model_name}_${version_name}}"
prism_eval_dir="${PRISM_EVAL_DIR:-runs/all_eval_prism_${model_name}_${version_name}_steer_distill}"
steering_checkpoint="${STEERING_CHECKPOINT:-${run_dir}/last}"

skip_train="${SKIP_TRAIN:-0}"
skip_prepare="${SKIP_PREPARE:-0}"

wandb_args=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  wandb_args=(
    --use_wandb
    --wandb_project "${WANDB_PROJECT:-pengram_distill}"
    --wandb_run_name "${model_name}_${version_name}"
  )
fi

uf_other_subsets="${UF_OTHER_SUBSETS:-single}"
uf_dataset_name="${UF_DATASET_NAME:-P_4}"
uf_survey_size="${UF_SURVEY_SIZE:-16}"

psoups_config="${PSOUPS_CONFIG:-default}"
tldr_top_workers="${TLDR_TOP_WORKERS:-40}"
personalllm_max_persons="${PERSONALLLM_MAX_PERSONS:-1000}"
personalllm_max_query_per_person="${PERSONALLLM_MAX_QUERY_PER_PERSON:-5}"

echo "[all-eval] checkpoint=${steering_checkpoint}"
echo "[all-eval] model=${model_name} version=${version_name} device=${device} skip_train=${skip_train} skip_prepare=${skip_prepare}"

if [[ "${skip_train}" != "1" ]]; then
  echo "[all-eval] train PRISM cautious context-steering distill"
  CUDA_VISIBLE_DEVICES=${device} python "${script_dir}/train_prism_cautious_context_steering_distill.py" \
    --model_name Qwen/${model_name} \
    --train_jsonl "${train_jsonl}" \
    --valid_jsonl "${valid_jsonl}" \
    --output_dir "${run_dir}" \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_epochs 2 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --optimizer adamw \
    --require_history 1 \
    --context_pool attn \
    --adapter_dim 512 \
    --candidate_dim 512 \
    --support_min_k 32 \
    --support_max_k 128 \
    --distill_teacher_top_k 32 \
    --oracle_lambda_min -2.0 \
    --oracle_lambda_max 2.0 \
    --oracle_lambda_bisect_steps 32 \
    --distill_min_delta_var 1e-5 \
    --score_scale 2.5 \
    --score_clip 5.0 \
    --zero_mean_scores 1 \
    --distill_kl_weight 1.0 \
    --distill_delta_weight 0.0 \
    --distill_lambda_weight 0.0 \
    --chosen_ce_weight 0.0 \
    --pairwise_pref_weight 0.0 \
    --chosen_gain_hinge_weight 0.0 \
    --length_normalize \
    --append_eos \
    --adv_margin_tok 0.15 \
    --adv_temp_tok 0.10 \
    --base_preserve_weight 0.70 \
    --gate_loss_weight 0.30 \
    --gate_init_bias -2.0 \
    --gate_hidden 256 \
    --eval_every 600 \
    "${wandb_args[@]}"
else
  echo "[all-eval] skip PRISM training"
fi

echo "[all-eval] PRISM steer_distill"
CUDA_VISIBLE_DEVICES=${device} python "${script_dir}/eval_cautious_context_steering_distill.py" \
  --dataset prism \
  --model_name Qwen/${model_name} \
  --steering_checkpoint "${steering_checkpoint}" \
  --support_jsonl "${prism_support_jsonl}" \
  --query_jsonl "${prism_query_jsonl}" \
  --support_budgets 4 \
  --strict_support_budget \
  --max_new_tokens 256 \
  --save_dir "${prism_eval_dir}" \
  --device cuda:0 \
  --metric_device cuda:0 \
  --gen_batch_size 32 \
  --policy_eval_batch_size 4 \
  --icl_mode chosen_only \
  --icl_include_prompt \
  --cos_history_mode chosen_only \
  --cos_history_include_prompt \
  --cos_lambda -0.1 \
  --pengram_history_mode chosen_only \
  --pengram_history_include_prompt \
  --systems steer_distill

echo "[all-eval] UltraFeedback steer_distill"
env -u SUPPORT_JSONL -u QUERY_JSONL -u TEST_JSONL \
  STEERING_CHECKPOINT="${steering_checkpoint}" \
  SKIP_PREPARE="${skip_prepare}" \
  SAVE_DIR="${UF_SAVE_DIR:-runs/all_eval_ultrafeedback_${uf_other_subsets}_${uf_dataset_name}_${model_name}_${version_name}_steer_distill}" \
  bash "${script_dir}/run_ultrafeedback_cautious_context_steering_distill.sh" \
  "${model_name}" \
  "${version_name}" \
  "${device}" \
  "${uf_other_subsets}" \
  "${uf_dataset_name}" \
  "${uf_survey_size}"

echo "[all-eval] PSOUPS steer_distill"
env -u SUPPORT_JSONL -u QUERY_JSONL -u TEST_JSONL \
  STEERING_CHECKPOINT="${steering_checkpoint}" \
  SYSTEMS="steer_distill" \
  SKIP_PREPARE="${skip_prepare}" \
  SAVE_DIR="${PSOUPS_SAVE_DIR:-runs/all_eval_psoups_${psoups_config}_${model_name}_${version_name}_steer_distill}" \
  bash "${script_dir}/run_psoups_cautious_context_steering_distill.sh" \
  "${model_name}" \
  "${version_name}" \
  "${device}" \
  "${psoups_config}"

echo "[all-eval] TLDR steer_distill"
env -u SUPPORT_JSONL -u QUERY_JSONL -u TEST_JSONL \
  STEERING_CHECKPOINT="${steering_checkpoint}" \
  SYSTEMS="steer_distill" \
  SKIP_PREPARE="${skip_prepare}" \
  SAVE_DIR="${TLDR_SAVE_DIR:-runs/all_eval_tldr_top${tldr_top_workers}_${model_name}_${version_name}_steer_distill}" \
  bash "${script_dir}/run_tldr_cautious_context_steering_distill.sh" \
  "${model_name}" \
  "${version_name}" \
  "${device}" \
  "${tldr_top_workers}"

echo "[all-eval] PersonalLLM steer_distill"
env -u SUPPORT_JSONL -u QUERY_JSONL -u TEST_JSONL \
  STEERING_CHECKPOINT="${steering_checkpoint}" \
  SYSTEMS="steer_distill" \
  SKIP_PREPARE="${skip_prepare}" \
  SAVE_DIR="${PERSONALLLM_SAVE_DIR:-runs/all_eval_personalllm_${model_name}_${version_name}_steer_distill}" \
  bash "${script_dir}/run_personalllm_cautious_context_steering_distill.sh" \
  "${model_name}" \
  "${version_name}" \
  "${device}" \
  "${personalllm_max_persons}" \
  "${personalllm_max_query_per_person}"

echo "[all-eval] done"
