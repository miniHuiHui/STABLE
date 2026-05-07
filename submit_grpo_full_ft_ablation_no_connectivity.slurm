#!/usr/bin/env bash
#SBATCH --job-name=brickgpt-grpo-fullft-abl-noconn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --time=2-23:59:59
#SBATCH --output=/path/to/STABLE/training_log/%x-%j.out
#SBATCH --error=/path/to/STABLE/training_log/%x-%j.err

set -euo pipefail

module load cuda

PROJECT_ROOT="/path/to/STABLE"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/path/to/conda/env"

mkdir -p "${PROJECT_ROOT}/training_log"
cd "${PROJECT_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
export INIT_MODEL_PATH="${INIT_MODEL_PATH:-${PROJECT_ROOT}/finetuned_models/qwen35_sft_full_ft_run1_official}"
export DATASET_DIR="${DATASET_DIR:-datasets_pc}"
export OUTPUT_DIR="${OUTPUT_DIR:-finetuned_models/qwen_pc_grpo_full_ft_ablation_no_connectivity}"
export RUN_NAME="${RUN_NAME:-brickgpt-grpo-full-ft-ablation-no-connectivity}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-brickgpt-grpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

export TRAIN_LOG="${TRAIN_LOG:-${PROJECT_ROOT}/training_log/${RUN_NAME}-${SLURM_JOB_ID}.train.log}"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "run_name=${RUN_NAME}"
echo "model_name=${MODEL_NAME}"
echo "init_model_path=${INIT_MODEL_PATH}"
echo "dataset_dir=${DATASET_DIR}"
echo "output_dir=${OUTPUT_DIR}"
echo "attn_implementation=${ATTN_IMPLEMENTATION}"
echo "use_wandb=${USE_WANDB}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_entity=${WANDB_ENTITY}"
echo "wandb_mode=${WANDB_MODE}"
echo "train_log=${TRAIN_LOG}"
echo "host=$(hostname)"
echo "start_time=$(date '+%F %T')"

mkdir -p "$(dirname "${TRAIN_LOG}")"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
"${CONDA_ENV}/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" train_grpo_full_ft_ablation_no_connectivity.py > "${TRAIN_LOG}" 2>&1
