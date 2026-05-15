# Cautious CoS

## Introduction

이 레포지토리는 사용자별 과거 선호 이력을 이용해 LLM 출력을 조심스럽게 조향하는 `cautious context steering` 실험 코드입니다. 핵심 아이디어는 PRISM 데이터로 학습한 작은 steering adapter가, 과거 이력이 실제로 도움이 될 때만 base model의 다음 토큰 분포를 조정하고, 도움이 불확실한 경우에는 base 분포를 최대한 보존하도록 학습하는 것입니다.

메인 실행 경로는 `run_all_cautious_context_steering_distill_evals.sh`입니다. 이 스크립트는 PRISM으로 cautious steering adapter를 학습한 뒤 PRISM, UltraFeedback P_4, PSOUPS, TLDR, PersonalLLM 평가를 한 번에 실행합니다.

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

## Train And Evaluate All

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

각 평가 디렉터리에는 `summary.json`과 `predictions_budget*_steer_distill.jsonl`이 저장됩니다.

## Useful Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `TRAIN_JSONL` | `data/prism_cautious_cos_splits/seen_train.jsonl` | PRISM train split |
| `VALID_JSONL` | `data/prism_cautious_cos_splits/seen_valid.jsonl` | PRISM validation split |
| `SUPPORT_JSONL` | `data/prism_cautious_cos_splits/calib_unseen.jsonl` | PRISM support split |
| `QUERY_JSONL` | `data/prism_cautious_cos_splits/test_unseen.jsonl` | PRISM query split |
| `RUN_DIR` | `runs/prism_cautious_context_steering_distill_<model>_<version>` | 학습 출력 디렉터리 |
| `STEERING_CHECKPOINT` | `${RUN_DIR}/last` | 평가에 사용할 checkpoint |
| `SKIP_TRAIN` | `0` | `1`이면 PRISM 학습 생략 |
| `SKIP_PREPARE` | `0` | `1`이면 non-PRISM 데이터 준비 생략 |
| `SKIP_UF_DATASET` | `0` | `1`이면 UltraFeedback 원본 생성 단계만 생략 |
| `USE_WANDB` | `1` | `0`이면 wandb 비활성화 |
| `UF_SURVEY_SIZE` | `16` | UltraFeedback survey size |
| `TLDR_TOP_WORKERS` | `40` | TLDR에서 사용할 상위 worker 수 |
| `PERSONALLLM_MAX_PERSONS` | `1000` | PersonalLLM 평가 person 수 |
| `PERSONALLLM_MAX_QUERY_PER_PERSON` | `5` | PersonalLLM person별 query 제한 |

## PRISM Only

전체 평가 대신 PRISM 학습과 PRISM 평가만 실행하려면:

```bash
bash run_prism_cautious_context_steering_distill.sh Qwen3-0.6B cautious_context_steering 0
```

## Metrics

주요 평가 지표:

- `policy_preference_acc`: teacher-forced chosen/rejected preference accuracy
- `policy_token_acc`: target token accuracy
- `policy_first_token_acc`: first target token accuracy
- `rouge1_f1`, `rougeL_f1`: lexical generation similarity
- `bertscore_f1`: semantic generation similarity
- `prefsim_margin`: generated response가 chosen에 rejected보다 더 가까운지 보는 preference similarity 차이

## Core Files

- `train_prism_cautious_context_steering_distill.py`: cautious context steering adapter 학습
- `eval_cautious_context_steering_distill.py`: 모든 데이터셋 공통 평가 엔트리포인트
- `run_all_cautious_context_steering_distill_evals.sh`: PRISM 학습 후 전체 데이터셋 평가
- `data_utils/`: 데이터셋 다운로드와 전처리 스크립트
