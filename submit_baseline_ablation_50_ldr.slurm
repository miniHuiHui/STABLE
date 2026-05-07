#!/usr/bin/env bash
#SBATCH --job-name=baseline-ablation-ldr50
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
# Generate 50 test samples (.txt + .ldr when parseable) for baseline + ablation
# models, using the same dataset prompt format as evaluate_models.py.
#
# Submit:
#   sbatch submit_baseline_ablation_50_ldr.slurm
#
# Optional overrides:
#   NUM_SAMPLES=50 START_INDEX=0 sbatch submit_baseline_ablation_50_ldr.slurm
#   MODELS="qwen_base sft_full_ft grpo_no_connectivity" sbatch ...
#   DATASET=datasets_pc/test.jsonl OUTPUT_ROOT=/tmp/ldr_runs sbatch ...
#
# Notes:
# - This script calls run_demo_grpo.py for each model key sequentially.
# - For gated HF models, export HF_TOKEN before sbatch (do not commit secrets).
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

DATASET="${DATASET:-datasets_pc/test.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
START_INDEX="${START_INDEX:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
QWEN_BASE="${QWEN_BASE_MODEL:-Qwen/Qwen3.5-2B}"
BRICKGPT_HF="${BRICKGPT_HF:-AvaLovelace/BrickGPT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/generated_ldr/baseline_ablation_${SLURM_JOB_ID:-manual}}"

# ---------------------------------------------------------------------------
# Model registry (aligned with submit_eval.slurm naming).
# run_demo_grpo.py currently supports plain causal-LM loading.
# ---------------------------------------------------------------------------
declare -A MODEL_PATH
declare -A TOKEN_PATH
declare -A EXTRA_ARGS

# baselines
MODEL_PATH[qwen_base]="${QWEN_BASE}"
TOKEN_PATH[qwen_base]="-"
EXTRA_ARGS[qwen_base]=""

MODEL_PATH[sft_full_ft]="${PROJECT_ROOT}/finetuned_models/qwen35_sft_full_ft_run1_official"
TOKEN_PATH[sft_full_ft]="-"
EXTRA_ARGS[sft_full_ft]=""

# SFT eval: force only 1x1 bricks and export filtered LDR
MODEL_PATH[sft_full_ft_only_1x1]="${PROJECT_ROOT}/finetuned_models/qwen35_sft_full_ft_run1_official"
TOKEN_PATH[sft_full_ft_only_1x1]="-"
EXTRA_ARGS[sft_full_ft_only_1x1]="--only_brick_size 1x1 --append_user_constraint 只使用1x1砖块生成乐高结构。不要输出任何非1x1砖块。"

MODEL_PATH[grpo_base_full_ft]="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_base_full_ft"
TOKEN_PATH[grpo_base_full_ft]="${QWEN_BASE}"
EXTRA_ARGS[grpo_base_full_ft]=""

MODEL_PATH[grpo_full_ft]="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft"
TOKEN_PATH[grpo_full_ft]="${QWEN_BASE}"
EXTRA_ARGS[grpo_full_ft]=""

MODEL_PATH[brickgpt_causal]="${BRICKGPT_HF}"
TOKEN_PATH[brickgpt_causal]="-"
EXTRA_ARGS[brickgpt_causal]=""

# ablations
MODEL_PATH[grpo_no_connectivity]="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft_ablation_no_connectivity"
TOKEN_PATH[grpo_no_connectivity]="${QWEN_BASE}"
EXTRA_ARGS[grpo_no_connectivity]=""

MODEL_PATH[grpo_no_structural]="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft_ablation_no_structural"
TOKEN_PATH[grpo_no_structural]="${QWEN_BASE}"
EXTRA_ARGS[grpo_no_structural]=""

MODEL_PATH[grpo_no_collision]="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft_ablation_no_collision"
TOKEN_PATH[grpo_no_collision]="${QWEN_BASE}"
EXTRA_ARGS[grpo_no_collision]=""

DEFAULT_MODELS="qwen_base sft_full_ft sft_full_ft_only_1x1 grpo_base_full_ft grpo_full_ft brickgpt_causal grpo_no_connectivity grpo_no_structural grpo_no_collision"
MODELS="${MODELS:-${DEFAULT_MODELS}}"

mkdir -p "${OUTPUT_ROOT}"

DO_SAMPLE_ARGS=(--no-do_sample)
if [[ "${DO_SAMPLE}" == "1" || "${DO_SAMPLE}" == "true" ]]; then
  DO_SAMPLE_ARGS=(--do_sample)
fi

echo "=========================================================================="
echo "job_id          = ${SLURM_JOB_ID:-manual}"
echo "host            = $(hostname)"
echo "start           = $(date '+%F %T')"
echo "dataset         = ${DATASET}"
echo "num_samples     = ${NUM_SAMPLES}"
echo "start_index     = ${START_INDEX}"
echo "max_new_tokens  = ${MAX_NEW_TOKENS}"
echo "do_sample       = ${DO_SAMPLE}"
echo "output_root     = ${OUTPUT_ROOT}"
echo "models          = ${MODELS}"
echo "=========================================================================="

FAILED=()
for key in ${MODELS}; do
  mp="${MODEL_PATH[$key]:-}"
  tp="${TOKEN_PATH[$key]:-}"
  ea="${EXTRA_ARGS[$key]:-}"
  if [[ -z "${mp}" ]]; then
    echo "[skip] unknown model key: ${key}"
    continue
  fi
  if [[ "${mp}" == /* && ! -d "${mp}" ]]; then
    echo "[warn] missing local model dir for ${key}: ${mp}"
    FAILED+=("${key} (missing checkpoint)")
    continue
  fi

  tok="${tp}"
  if [[ -z "${tok}" || "${tok}" == "-" ]]; then
    tok="${mp}"
  fi
  out_dir="${OUTPUT_ROOT}/${key}"
  mkdir -p "${out_dir}"

  echo "--------------------------------------------------------------------------"
  echo "==> generating [${key}] model=${mp}"
  echo "    tokenizer=${tok}"
  echo "    out_dir=${out_dir}"
  echo "--------------------------------------------------------------------------"

  set +e
  "${CONDA_ENV}/bin/python" -u "${PROJECT_ROOT}/run_demo_grpo.py" \
    --grpo_checkpoint "${mp}" \
    --model "${tok}" \
    --dataset "${DATASET}" \
    --start_index "${START_INDEX}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    "${DO_SAMPLE_ARGS[@]}" \
    --output_dir "${out_dir}" \
    ${ea} \
    2>&1 | tee "${out_dir}/generation.log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[fail] ${key} exited with status ${rc}"
    FAILED+=("${key}")
  else
    echo "[ok]   ${key} finished"
  fi
  echo
done

echo "finish_time=$(date '+%F %T')"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[done] completed with failures: ${FAILED[*]}"
  exit 1
fi
echo "[done] all baseline + ablation generations complete"

