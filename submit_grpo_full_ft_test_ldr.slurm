#!/usr/bin/env bash
#SBATCH --job-name=grpo-fullft-ldr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --time=0-12:00:00
#SBATCH --output=/path/to/STABLE/training_log/%x-%j.out
#SBATCH --error=/path/to/STABLE/training_log/%x-%j.err

# =============================================================================
# Generate .ldr (and raw .txt) from GRPO full-FT weights on a JSONL split.
# Uses run_demo_grpo.py (BrickStructure.from_txt -> to_ldr).
#
# Submit:
#     sbatch submit_grpo_full_ft_test_ldr.slurm
#
# Overrides (examples):
#     NUM_SAMPLES=20 START_INDEX=0 sbatch submit_grpo_full_ft_test_ldr.slurm
#     GRPO_CHECKPOINT=/path/to/checkpoint-10651 DO_SAMPLE=1 sbatch ...
#     DATASET=datasets_pc/val.jsonl OUTPUT_DIR=/tmp/my_ldr sbatch ...
#
# Gated HF tokenizer base: export HF_TOKEN=... before sbatch (do not commit).
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
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

GRPO_CHECKPOINT="${GRPO_CHECKPOINT:-${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft}"
TOKENIZER_BASE="${TOKENIZER_BASE:-Qwen/Qwen3.5-2B}"
DATASET="${DATASET:-datasets_pc/test.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
START_INDEX="${START_INDEX:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/generated_ldr/grpo_full_ft_${SLURM_JOB_ID:-manual}}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"

mkdir -p "${OUTPUT_DIR}"

# BooleanOptionalAction uses --do_sample / --no-do_sample (underscore before sample)
DO_SAMPLE_ARGS=(--no-do_sample)
if [[ "${DO_SAMPLE}" == "1" || "${DO_SAMPLE}" == "true" ]]; then
  DO_SAMPLE_ARGS=(--do_sample)
fi

echo "=========================================================================="
echo "job_id           = ${SLURM_JOB_ID:-manual}"
echo "host             = $(hostname)"
echo "start            = $(date '+%F %T')"
echo "grpo_checkpoint  = ${GRPO_CHECKPOINT}"
echo "tokenizer_base   = ${TOKENIZER_BASE}"
echo "dataset          = ${DATASET}"
echo "start_index      = ${START_INDEX}"
echo "num_samples      = ${NUM_SAMPLES}"
echo "max_new_tokens   = ${MAX_NEW_TOKENS}"
echo "do_sample        = ${DO_SAMPLE}"
echo "output_dir       = ${OUTPUT_DIR}"
echo "=========================================================================="

python "${PROJECT_ROOT}/run_demo_grpo.py" \
  --grpo_checkpoint "${GRPO_CHECKPOINT}" \
  --model "${TOKENIZER_BASE}" \
  --dataset "${DATASET}" \
  --start_index "${START_INDEX}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  "${DO_SAMPLE_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}"

echo "Finished at $(date '+%F %T'). LDR files: ${OUTPUT_DIR}/sample_*_gen.ldr"
