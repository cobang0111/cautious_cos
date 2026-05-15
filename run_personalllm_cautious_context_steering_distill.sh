#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"

model_name="${1:-Qwen3-0.6B}"
version_name="${2:-cautious_context_steering}"
device="${3:-0}"
max_persons="${4:-1000}"
max_query_per_person="${5:-0}"

out_dir="${PERSONALLLM_OUT_DIR:-data/personalllm_cautious_cos_splits}"
support_jsonl="${SUPPORT_JSONL:-${out_dir}/calib_unseen.jsonl}"
query_jsonl="${QUERY_JSONL:-${out_dir}/test_unseen.jsonl}"

# By default this evaluates the checkpoint produced by run_prism_cautious_context_steering_distill.sh.
steering_checkpoint="${STEERING_CHECKPOINT:-runs/prism_cautious_context_steering_distill_${model_name}_${version_name}/last}"
save_dir="${SAVE_DIR:-runs/personalllm_cautious_context_steering_distill_eval_${model_name}_${version_name}}"
systems=(${SYSTEMS:-steer_distill})

prepare_args=(
  --out_dir "${out_dir}"
  --support_count 4
  --history_pairs 4
  --history_include_prompt
  --max_persons "${max_persons}"
  --max_query_per_person "${max_query_per_person}"
)

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  "${python_bin}" "${script_dir}/data_utils/personalllm_preprocessing.py" "${prepare_args[@]}"
fi

CUDA_VISIBLE_DEVICES=${device} "${python_bin}" "${script_dir}/eval_cautious_context_steering_distill.py" \
  --dataset personalllm \
  --model_name Qwen/${model_name} \
  --steering_checkpoint "${steering_checkpoint}" \
  --support_jsonl "${support_jsonl}" \
  --query_jsonl "${query_jsonl}" \
  --support_budgets 4 \
  --strict_support_budget \
  --max_new_tokens 256 \
  --save_dir "${save_dir}" \
  --device cuda:0 \
  --metric_device cuda:0 \
  --gen_batch_size 32 \
  --policy_eval_batch_size 4 \
  --icl_mode chosen_only \
  --icl_include_prompt \
  --cos_history_mode chosen_only \
  --cos_history_include_prompt \
  --cos_lambda -0.1 \
  --cautious_cos_history_mode chosen_only \
  --cautious_cos_history_include_prompt \
  --systems "${systems[@]}"
