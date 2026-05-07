#!/usr/bin/env bash
#SBATCH --job-name=grpo-no-coll-ckpt-ldr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --time=0-23:59:59
#SBATCH --output=/path/to/STABLE/training_log/%x-%j.out
#SBATCH --error=/path/to/STABLE/training_log/%x-%j.err

# =============================================================================
# 对 GRPO full-FT「no collision objective」消融模型做测试集生成（与 baseline_ablation
# 脚本相同流程：run_demo_grpo.py → sample_*.txt / .ldr）。
#
# 默认使用指定训练步的 checkpoint 目录（例如训练未跑完时先用 checkpoint-9000）。
#
# 提交示例:
#   sbatch submit_grpo_no_collision_ckpt_ldr.slurm
#
# 覆盖示例:
#   CHECKPOINT_NAME=checkpoint-9500 sbatch submit_grpo_no_collision_ckpt_ldr.slurm
#   GRPO_CHECKPOINT=/path/to/checkpoint-9000 sbatch submit_grpo_no_collision_ckpt_ldr.slurm
#   NUM_SAMPLES=100 START_INDEX=0 OUTPUT_ROOT=/tmp/out sbatch submit_grpo_no_collision_ckpt_ldr.slurm
#
# 量化评测（碰撞率、IoU 等）请在计算节点另跑 evaluate_models.py，例如:
#   python evaluate_models.py \
#     --model_path "${GRPO_CHECKPOINT}" --model_type causal_lm \
#     --run_name grpo_no_collision_ckpt9000 --dataset datasets_pc/test.jsonl \
#     --max_samples 100 --output_dir eval_results/grpo_no_collision_ckpt9000
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

CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint-9000}"
ABLATION_ROOT="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft_ablation_no_collision"
GRPO_CHECKPOINT="${GRPO_CHECKPOINT:-${ABLATION_ROOT}/${CHECKPOINT_NAME}}"

DATASET="${DATASET:-datasets_pc/test.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
START_INDEX="${START_INDEX:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
QWEN_BASE="${QWEN_BASE_MODEL:-Qwen/Qwen3.5-2B}"

OUT_KEY="${OUT_KEY:-grpo_no_collision_${CHECKPOINT_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/generated_ldr/${OUT_KEY}_${SLURM_JOB_ID:-manual}}"

DO_SAMPLE_ARGS=(--no-do_sample)
if [[ "${DO_SAMPLE}" == "1" || "${DO_SAMPLE}" == "true" ]]; then
  DO_SAMPLE_ARGS=(--do_sample)
fi

if [[ ! -d "${GRPO_CHECKPOINT}" ]]; then
  echo "[error] checkpoint 目录不存在: ${GRPO_CHECKPOINT}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
out_dir="${OUTPUT_ROOT}/${OUT_KEY}"
mkdir -p "${out_dir}"

echo "=========================================================================="
echo "job_id           = ${SLURM_JOB_ID:-manual}"
echo "host             = $(hostname)"
echo "start            = $(date '+%F %T')"
echo "grpo_checkpoint  = ${GRPO_CHECKPOINT}"
echo "tokenizer (base) = ${QWEN_BASE}"
echo "dataset          = ${DATASET}"
echo "num_samples      = ${NUM_SAMPLES}"
echo "start_index      = ${START_INDEX}"
echo "max_new_tokens   = ${MAX_NEW_TOKENS}"
echo "do_sample        = ${DO_SAMPLE}"
echo "output_dir       = ${out_dir}"
echo "=========================================================================="

set +e
"${CONDA_ENV}/bin/python" -u "${PROJECT_ROOT}/run_demo_grpo.py" \
  --grpo_checkpoint "${GRPO_CHECKPOINT}" \
  --model "${QWEN_BASE}" \
  --dataset "${DATASET}" \
  --start_index "${START_INDEX}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  "${DO_SAMPLE_ARGS[@]}" \
  --output_dir "${out_dir}" \
  2>&1 | tee "${out_dir}/generation.log"
rc=${PIPESTATUS[0]}
set -e

echo "finish_time=$(date '+%F %T')"
if [[ ${rc} -ne 0 ]]; then
  echo "[fail] run_demo_grpo exited with status ${rc}"
  exit "${rc}"
fi
echo "[done] generation complete → ${out_dir}"
