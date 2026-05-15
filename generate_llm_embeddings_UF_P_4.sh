#!/usr/bin/env bash
set -euo pipefail

model_type="Qwen/Qwen3-0.6B"
other_subsets="single"
with_embeddings="${WITH_EMBEDDINGS:-False}"
build_eval_dataset="${BUILD_EVAL_DATASET:-1}"
history_items="${HISTORY_ITEMS:-4}"
survey_size=16
split_args=()
if [ "${build_eval_dataset}" != "1" ]; then
    split_args=(--skip_splits)
fi

python uf_p_4_preprocessing.py \
  --model_type "${model_type}" \
  --other_subsets "${other_subsets}" \
  --dataset_name P_4 \
  --survey_size "${survey_size}" \
  --history_items "${history_items}" \
  --with_embeddings "${with_embeddings}" \
  --source_root "data/UltraFeedback_${other_subsets}_P_4" \
  --generated_data_dir "data/P_4_survey_${survey_size}/${model_type}" \
  --out_dir "data/ultrafeedback_${other_subsets}_P_4_history" \
  --max_history_items "${history_items}" \
  "${split_args[@]}"

