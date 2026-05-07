#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
DATASET_NAME="${DATASET_NAME:-datasets_pc_sft}"
RUN_NAME="${RUN_NAME:-qwen35_sft_run1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/finetuned_models/${RUN_NAME}}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  -m trl.scripts.sft \
  --model_name_or_path "${MODEL_NAME}" \
  --do_train True \
  --eval_strategy steps \
  --dataset_name "${DATASET_NAME}" \
  --dataset_text_field messages \
  --dataloader_num_workers 4 \
  --max_length 8192 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 0.0002 \
  --lr_scheduler_type cosine \
  --warmup_steps 100 \
  --num_train_epochs 3 \
  --eval_steps 250 \
  --save_steps 500 \
  --load_best_model_at_end True \
  --bf16 True \
  --use_peft True \
  --lora_r 32 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj v_proj k_proj o_proj gate_proj up_proj down_proj \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}"
