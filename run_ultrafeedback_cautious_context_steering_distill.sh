#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

model_name="${1:-}"
version_name="${2:-cautious_context_steering}"
device="${3:-0}"
other_subsets="${4:-single}"
dataset_name="${5:-P_4}"
survey_size="${6:-16}"

if [ -z "${model_name}" ]; then
  echo "Usage: bash run_ultrafeedback_cautious_context_steering_distill.sh <model_name> [version_name] [device] [other_subsets] [dataset_name] [survey_size]"
  echo "Example: bash run_ultrafeedback_cautious_context_steering_distill.sh Qwen3-0.6B cautious_context_steering 0 single P_4 16"
  exit 1
fi

generated_data_dir="${GENERATED_DATA_DIR:-data/${dataset_name}_survey_${survey_size}}"
source_root="${SOURCE_ROOT:-data/UltraFeedback_${other_subsets}_${dataset_name}}"
out_dir="${UF_OUT_DIR:-data/ultrafeedback_${other_subsets}_${dataset_name}_history}"
support_jsonl="${SUPPORT_JSONL:-${out_dir}/calib_unseen.jsonl}"
query_jsonl="${QUERY_JSONL:-${out_dir}/test_unseen.jsonl}"
steering_checkpoint="${STEERING_CHECKPOINT:-runs/prism_cautious_context_steering_distill_${model_name}_${version_name}/last}"
save_dir="${SAVE_DIR:-runs/ultrafeedback_${other_subsets}_${dataset_name}_${model_name}_${version_name}_steer_distill_eval}"

prepare_args=(
  --other_subsets "${other_subsets}"
  --dataset_name "${dataset_name}"
  --survey_size "${survey_size}"
  --generated_data_dir "${generated_data_dir}"
  --source_root "${source_root}"
)

split_args=(
  --other_subsets "${other_subsets}"
  --dataset_name "${dataset_name}"
  --survey_size "${survey_size}"
  --generated_data_dir "${generated_data_dir}"
  --source_root "${source_root}"
  --out_dir "${out_dir}"
  --eval_support_ratio "${UF_EVAL_SUPPORT_RATIO:-1.0}"
)

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  if [[ "${SKIP_UF_DATASET:-0}" != "1" ]]; then
    python "${script_dir}/data_utils/get_uf_p_4_dataset.py" "${prepare_args[@]}"
  fi
  python "${script_dir}/data_utils/uf_p_4_preprocessing.py" "${split_args[@]}"
fi

python eval_ultrafeedback_cautious_context_steering_distill.py \
  --support_jsonl "${support_jsonl}" \
  --query_jsonl "${query_jsonl}" \
  --model_name Qwen/${model_name} \
  --steering_checkpoint "${steering_checkpoint}" \
  --support_budgets 4 \
  --strict_support_budget \
  --systems steer_distill \
  --icl_mode chosen_only \
  --icl_include_prompt \
  --cos_history_mode chosen_only \
  --cos_history_include_prompt \
  --cos_lambda -0.1 \
  --max_history_pairs 4 \
  --max_new_tokens 256 \
  --save_dir "${save_dir}" \
  --device cuda:${device} \
  --metric_device cuda:${device} \
  --gen_batch_size 32 \
  --policy_eval_batch_size "${POLICY_EVAL_BATCH_SIZE:-4}"
