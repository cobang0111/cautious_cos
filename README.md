### ✨Environment Setting
Clone this repository and run:

```bash
conda create -n cautious_cos python=3.11
conda activate cautious_cos
pip install -r requirements.txt
```
### ✨Experiments
```bash
# get PRISM raw data
python data_utils/get_prism_dataset.py

# PRISM preprocessing for context steering
python data_utils/prism_preprocessing.py \
  --survey_jsonl data/prism_raw/survey.jsonl \
  --conversations_jsonl data/prism_raw/conversations.jsonl \
  --out_dir data/prism_cautious_cos_splits \
  --only_english \
  --drop_flagged \
  --min_score_gap 5.0 \
  --unseen_user_frac 0.2 \
  --seen_valid_frac 0.15 \
  --unseen_support_frac 0.5 \
  --history_conversations 4 \
  --history_include_prompt

# Train on PRISM, then evaluate steer_distill across configured datasets
bash run_all_cautious_context_steering_distill_evals.sh model_name run_name

# Train and evaluate PRISM only
bash run_prism_cautious_context_steering_distill.sh model_name run_name

# For example
bash run_all_cautious_context_steering_distill_evals.sh Qwen3-0.6B v260318
```

### 주요 결과 지표 
Policy accuracy: `policy_preference_acc` 
<br>
Generation unigram similarity: `rouge1_f1`
<br>
Generation sequence similarity: `rougeL_f1`
<br>
Generation semantic similarity: `bertscore_f1`


### train 명령에서 조절해볼만한 param
```bash
# entropy를 soft gating 할 때 기준값
--entropy_threshold 0.3

# 작을수록 entropy gating을 더 예민하게 
--entropy_temperature 0.1

# steering 강도 조절
--score_scale 1.0

# KL divergence 비중 클수록 base policy에 묶임
--anchor_kl_weight 5e-4
```
