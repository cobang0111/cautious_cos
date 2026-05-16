<h1 align="center">Cautious Context Steering </h1>

<p align="center"> Gihoon Kim<sup>1 *</sup>, Jeyoung Lee<sup>1 *</sup>, Suhan Woo<sup>1</sup>, Sekwon Oh<sup>1</sup>, Minsu Jeon<sup>1</sup> and Euntai Kim<sup>1, 2 </sup> </p>

<p align="center"> <sup>1</sup> Yonsei University, <sup>2</sup> Korea Institute of Science and Technology </p>


## Introduction

이 Repo는 user별 과거 선호 history를 이용해 LLM 출력을 steering 하는 `cautious context steering` 실험 코드입니다. 핵심 아이디어는 PRISM 데이터로 학습한 작은 steering adapter가, 과거 이력이 실제로 도움이 될 때만 base model의 다음 토큰 분포를 조정하고, 도움이 불확실한 경우에는 base 분포를 최대한 보존하도록 학습하는 것입니다.

메인 실행 경로는 `run_all_cautious_context_steering_distill_evals.sh`입니다. 이 스크립트는 PRISM으로 cautious steering adapter를 학습한 뒤 PRISM(ID), UltraFeedback P_4, PSOUPS, TLDR, PersonalLLM (OOD) 평가를 한 번에 실행합니다.

## Environment

```bash
conda create -n cautious_cos python=3.11
conda activate cautious_cos
pip install -r requirements.txt
```

실행 스크립트는 bash 기준입니다. Windows에서는 Git Bash, WSL, 또는 bash가 포함된 환경에서 실행하세요.

## Data Preparation

`run_all_cautious_context_steering_distill_evals.sh`는 PRISM 학습 split이 이미 있다고 가정합니다. 처음 한 번은 PRISM 원본 데이터 다운로드와 전처리를 먼저 실행하세요.

```bash
python data_utils/get_prism_dataset.py

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
```

나머지 평가 데이터셋은 기본적으로 `run_all_cautious_context_steering_distill_evals.sh` 안에서 준비됩니다.

- UltraFeedback: `data_utils/get_uf_p_4_dataset.py`와 `data_utils/uf_p_4_preprocessing.py`가 실행됩니다. 기본 설정은 `single`, `P_4`, `survey_size=16`입니다.
- PSOUPS: `data_utils/psoups_preprocessing.py`가 실행됩니다.
- TLDR: `data_utils/tldr_preprocessing.py`가 실행됩니다.
- PersonalLLM: `data_utils/personalllm_preprocessing.py`가 실행됩니다.

이미 전처리된 파일을 가지고 있다면 `SKIP_PREPARE=1`로 나머지 데이터 준비를 생략할 수 있습니다.

```bash
SKIP_PREPARE=1 bash run_all_cautious_context_steering_distill_evals.sh Qwen3-0.6B cautious_context_steering 0
```

UltraFeedback의 원본 생성 단계만 건너뛰고 split 생성은 다시 하고 싶다면 `SKIP_UF_DATASET=1`을 사용하세요.

## Train & Evaluate All

기본 실행:

```bash
bash run_all_cautious_context_steering_distill_evals.sh Qwen3-0.6B cautious_context_steering 0
```

인자:

```bash
bash run_all_cautious_context_steering_distill_evals.sh <model_name> <version_name> <cuda_device>
```

위 명령은 다음 순서로 동작합니다.

1. `data/prism_cautious_cos_splits/seen_train.jsonl`과 `seen_valid.jsonl`로 PRISM cautious steering adapter를 학습합니다.
2. checkpoint를 `runs/prism_cautious_context_steering_distill_<model_name>_<version_name>/last`에 저장합니다.
3. 같은 checkpoint로 PRISM, UltraFeedback P_4, PSOUPS, TLDR, PersonalLLM 평가를 실행합니다.
4. 각 평가 결과를 `runs/all_eval_*_steer_distill` 아래에 저장합니다.

학습을 생략하고 기존 checkpoint만 평가하려면:

```bash
SKIP_TRAIN=1 \
STEERING_CHECKPOINT=runs/prism_cautious_context_steering_distill_Qwen3-0.6B_cautious_context_steering/last \
bash run_all_cautious_context_steering_distill_evals.sh Qwen3-0.6B cautious_context_steering 0
```

Weights & Biases를 끄려면:

```bash
USE_WANDB=0 bash run_all_cautious_context_steering_distill_evals.sh Qwen3-0.6B cautious_context_steering 0
```

## Main Outputs

학습 checkpoint:

```text
runs/prism_cautious_context_steering_distill_<model_name>_<version_name>/last/steering.pt
runs/prism_cautious_context_steering_distill_<model_name>_<version_name>/last/steering_args.json
```

평가 결과:

```text
runs/all_eval_prism_<model_name>_<version_name>_steer_distill/
runs/all_eval_ultrafeedback_single_P_4_<model_name>_<version_name>_steer_distill/
runs/all_eval_psoups_default_<model_name>_<version_name>_steer_distill/
runs/all_eval_tldr_top40_<model_name>_<version_name>_steer_distill/
runs/all_eval_personalllm_<model_name>_<version_name>_steer_distill/
```

각 평가 디렉터리에는 `summary.json`과 `predictions_budget*_<system>.jsonl`이 저장됩니다.

## Comparison with Baselines

평가 스크립트의 `--systems` 인자는 여러 시스템을 한 번에 받을 수 있습니다. `run_all_cautious_context_steering_distill_evals.sh`에서는 `SYSTEMS` 환경변수로 이를 넘깁니다.

```bash
SYSTEMS="base icl cos cautious-cos" \
bash run_all_cautious_context_steering_distill_evals.sh Qwen3-0.6B cautious_context_steering 0
```

각 system의 의미는 다음과 같습니다.

- `base`: 사용자 history 없이 base LM만 사용합니다.
- `icl`: support examples를 prompt에 직접 붙이는 in-context learning baseline입니다.
- `cos`: history-conditioned prompt와 base prompt의 logits를 섞는 CoS baseline입니다.
- `cautious-cos`: 학습된 cautious context steering adapter입니다. 내부 저장명은 기존 코드 호환을 위해 `steer_distill`로 정규화됩니다.

여러 system을 같이 돌리면 각 평가 디렉터리에 `predictions_budget<k>_<system>.jsonl` 파일이 system별로 저장되고, `summary.json` 안에도 system별 metric이 함께 기록됩니다. `cautious-cos` 결과 파일은 `predictions_budget<k>_steer_distill.jsonl` 이름으로 저장됩니다.

## Streamlit Comparison Dashboard

여러 baseline 결과를 시각적으로 비교하려면 Streamlit 대시보드를 실행하세요.

```bash
streamlit run streamlit_compare_evals.py
```

대시보드는 기본적으로 `runs/` 아래의 모든 `summary.json`과 `predictions_budget*.jsonl` 파일을 찾아서 사용합니다. 지원 기능은 다음과 같습니다.

- PRISM, UltraFeedback P_4, PSOUPS, TLDR, PersonalLLM 전체 dataset 결과 비교
- `base`, `icl`, `cos`, `cautious-cos` system별 metric table
- `rouge1_f1`, `rougeL_f1`, `bertscore_f1`, `gen_time_sec`, `policy_preference_acc` 등 지표별 grouped bar chart
- cautious-cos의 `rouge1_f1`이 selected baseline보다 높은 예시를 우선 보여주고, 이후 cautious-cos `rouge1_f1` 순서로 이어지는 generation examples
- 각 example에서 preference history, chosen/rejected reference, baseline별 generation을 나란히 확인

history 비교를 보려면 현재 코드로 evaluation을 다시 실행해야 합니다. 새 evaluation output에는 `user_history_text`와 `user_history_pairs`가 prediction JSONL에 함께 저장됩니다.

## PRISM Only

전체 평가 대신 PRISM 학습과 PRISM 평가만 실행하려면:

```bash
bash run_prism_cautious_context_steering_distill.sh Qwen3-0.6B cautious_context_steering 0
```

## Metrics

주요 평가 지표:

- `policy_preference_acc`: teacher-forced chosen/rejected preference accuracy
- `rouge1_f1`, `rougeL_f1`: lexical generation similarity
- `bertscore_f1`: semantic generation similarity

## Core Files

- `train_prism_cautious_context_steering_distill.py`: cautious context steering adapter 학습
- `eval_cautious_context_steering_distill.py`: 모든 데이터셋 공통 평가 엔트리포인트
- `run_all_cautious_context_steering_distill_evals.sh`: PRISM 학습 후 전체 데이터셋 평가
- `streamlit_compare_evals.py`: dataset/system별 평가 결과 비교 대시보드
- `data_utils/`: 데이터셋 다운로드와 전처리 스크립트
