#!/usr/bin/env bash
#SBATCH --job-name=physical_stable 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --time=2-23:59:59
#SBATCH --output=/path/to/STABLE/training_log/%x-%j.out
#SBATCH --error=/path/to/STABLE/training_log/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/path/to/STABLE"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/path/to/conda/env"

mkdir -p "${PROJECT_ROOT}/training_log"
cd "${PROJECT_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Ensure CUDA tools are visible for DeepSpeed op builder
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda/13.1.1
fi

# DeepSpeed checks CUDA_HOME via torch.utils.cpp_extension.CUDA_HOME.
# If the module system didn't set it, try to derive it from `nvcc`.
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
echo "CUDA_HOME=${CUDA_HOME:-unset}"

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
export DATASET_NAME="${DATASET_NAME:-datasets_pc_sft}"
export RUN_NAME="${RUN_NAME:-qwen35_sft_run1_slurm}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/finetuned_models/${RUN_NAME}}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
export TRAIN_LOG="${TRAIN_LOG:-${PROJECT_ROOT}/training_log/${RUN_NAME}-${SLURM_JOB_ID}.train.log}"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "run_name=${RUN_NAME}"
echo "model_name=${MODEL_NAME}"
echo "dataset_name=${DATASET_NAME}"
echo "output_dir=${OUTPUT_DIR}"
echo "train_log=${TRAIN_LOG}"
echo "host=$(hostname)"
echo "start_time=$(date '+%F %T')"

mkdir -p "$(dirname "${TRAIN_LOG}")"
bash "${PROJECT_ROOT}/scripts/run_qwen35_sft.sh" > "${TRAIN_LOG}" 2>&1
