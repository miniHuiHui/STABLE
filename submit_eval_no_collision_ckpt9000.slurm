#!/usr/bin/env bash
#SBATCH --job-name=eval-no-coll-ck9k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --time=0-23:59:59
#SBATCH --output=/path/to/STABLE/training_log/%x-%j.out
#SBATCH --error=/path/to/STABLE/training_log/%x-%j.err

# =============================================================================
# Evaluate the no-collision ablation checkpoint-9000
# Submit:  sbatch submit_eval_no_collision_ckpt9000.slurm
# =============================================================================

set -euo pipefail

module load cuda || true

PROJECT_ROOT="/path/to/STABLE"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/path/to/conda/env"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/training_log"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft_ablation_no_collision/checkpoint-9000"
TOKENIZER_PATH="Qwen/Qwen3.5-2B"
RUN_NAME="grpo_no_collision_ckpt9000"
DATASET="${DATASET:-datasets_pc/test.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/eval_results/${SLURM_JOB_ID:-manual}}"

SAVE_GEN_FLAG=""
if [[ "${SAVE_GENERATIONS:-1}" == "1" || "${SAVE_GENERATIONS:-1}" == "true" ]]; then
    SAVE_GEN_FLAG="--save_generations"
fi

mkdir -p "${OUTPUT_DIR}"

echo "=========================================================================="
echo "job_id          = ${SLURM_JOB_ID:-manual}"
echo "host            = $(hostname)"
echo "start           = $(date '+%F %T')"
echo "model_path      = ${MODEL_PATH}"
echo "tokenizer_path  = ${TOKENIZER_PATH}"
echo "run_name        = ${RUN_NAME}"
echo "dataset         = ${DATASET}"
echo "max_samples     = ${MAX_SAMPLES}"
echo "output_dir      = ${OUTPUT_DIR}"
echo "=========================================================================="

RUN_LOG="${OUTPUT_DIR}/${RUN_NAME}.log"

"${CONDA_ENV}/bin/python" -u evaluate_models.py \
    --model_path "${MODEL_PATH}" \
    --model_type causal_lm \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --dataset "${DATASET}" \
    --max_samples "${MAX_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    ${SAVE_GEN_FLAG} \
    2>&1 | tee "${RUN_LOG}"

echo
echo "[done] evaluation complete"
echo "finish_time=$(date '+%F %T')"
