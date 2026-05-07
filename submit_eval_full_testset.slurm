#!/usr/bin/env bash
#SBATCH --job-name=brickgpt-eval-full-ts
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --time=2-23:59:59
#SBATCH --output=/path/to/STABLE/training_log/%x-%j.out
#SBATCH --error=/path/to/STABLE/training_log/%x-%j.err

# =============================================================================
# 在完整 test 集上跑 evaluate_models（与 submit_eval.slurm 相同流程，但默认
# max_samples = 测试集行数，walltime = 2 天 23 小时 59 分）。
# ---
# Metrics: parse success, IoU, collision rate, connectivity, interlocking, etc.
#
# 提交:
#     sbatch submit_eval_full_testset.slurm
#
# 只评部分模型: MODELS="grpo_full_ft brickgpt_causal" sbatch ...
# 子集样本数:   MAX_SAMPLES=500 sbatch ...
# 其他: USE_GUROBI, DATASET, OUTPUT_DIR, SAVE_GENERATIONS 等同 submit_eval.slurm
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
# For gated Hugging Face models, set before sbatch: export HF_TOKEN=...  (do not commit secrets)
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

# -----------------------------------------------------------------------------
# Evaluation configuration
# -----------------------------------------------------------------------------
DATASET="${DATASET:-datasets_pc/test.jsonl}"
if [[ "${DATASET}" != /* ]]; then
    _DATASET_ABS="${PROJECT_ROOT}/${DATASET}"
else
    _DATASET_ABS="${DATASET}"
fi
# 默认跑满整个 DATASET；若需子集可显式设置 MAX_SAMPLES
if [[ -z "${MAX_SAMPLES:-}" ]]; then
    MAX_SAMPLES="$(wc -l < "${_DATASET_ABS}")"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/eval_results/${SLURM_JOB_ID:-manual}}"
USE_GUROBI_FLAG=""
if [[ "${USE_GUROBI:-0}" == "1" || "${USE_GUROBI:-0}" == "true" ]]; then
    USE_GUROBI_FLAG="--use_gurobi"
fi

SAVE_GEN_FLAG=""
if [[ "${SAVE_GENERATIONS:-0}" == "1" || "${SAVE_GENERATIONS:-0}" == "true" ]]; then
    SAVE_GEN_FLAG="--save_generations"
fi

mkdir -p "${OUTPUT_DIR}"
export OUTPUT_DIR
echo "=========================================================================="
echo "job_id          = ${SLURM_JOB_ID:-manual}"
echo "host            = $(hostname)"
echo "start           = $(date '+%F %T')"
echo "dataset         = ${DATASET}"
echo "max_samples     = ${MAX_SAMPLES}"
echo "use_gurobi      = ${USE_GUROBI:-0}"
echo "output_dir      = ${OUTPUT_DIR}"
echo "=========================================================================="

# -----------------------------------------------------------------------------
# Models to evaluate
# Each entry: "key|model_type|model_path|tokenizer_path|extra_args"
# tokenizer_path is required if the checkpoint dir lacks tokenizer files.
# Use '-' to leave tokenizer_path empty (defaults to model_path).
# -----------------------------------------------------------------------------
QWEN_BASE="${QWEN_BASE_MODEL:-Qwen/Qwen3.5-2B}"
BRICKGPT_HF="${BRICKGPT_HF:-AvaLovelace/BrickGPT}"

declare -A MODEL_TYPE
declare -A MODEL_PATH
declare -A TOKEN_PATH
declare -A EXTRA_ARGS

# --- our main trained model (GRPO full fine-tune starting from SFT) ---
MODEL_TYPE[grpo_full_ft]=causal_lm
MODEL_PATH[grpo_full_ft]="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft"
TOKEN_PATH[grpo_full_ft]="${QWEN_BASE}"
EXTRA_ARGS[grpo_full_ft]=""

# --- GRPO directly from the Qwen base (no SFT warm start) ---
MODEL_TYPE[grpo_base_full_ft]=causal_lm
MODEL_PATH[grpo_base_full_ft]="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_base_full_ft"
TOKEN_PATH[grpo_base_full_ft]="${QWEN_BASE}"
EXTRA_ARGS[grpo_base_full_ft]=""

# --- ablations ---
MODEL_TYPE[grpo_no_connectivity]=causal_lm
MODEL_PATH[grpo_no_connectivity]="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft_ablation_no_connectivity"
TOKEN_PATH[grpo_no_connectivity]="${QWEN_BASE}"
EXTRA_ARGS[grpo_no_connectivity]=""

MODEL_TYPE[grpo_no_structural]=causal_lm
MODEL_PATH[grpo_no_structural]="${PROJECT_ROOT}/finetuned_models/qwen_pc_grpo_full_ft_ablation_no_structural"
TOKEN_PATH[grpo_no_structural]="${QWEN_BASE}"
EXTRA_ARGS[grpo_no_structural]=""

# --- SFT-only baseline (our fine-tune before GRPO) ---
MODEL_TYPE[sft_full_ft]=causal_lm
MODEL_PATH[sft_full_ft]="${PROJECT_ROOT}/finetuned_models/qwen35_sft_full_ft_run1_official"
TOKEN_PATH[sft_full_ft]="-"
EXTRA_ARGS[sft_full_ft]=""

# --- Qwen base (zero-shot) ---
MODEL_TYPE[qwen_base]=causal_lm
MODEL_PATH[qwen_base]="${QWEN_BASE}"
TOKEN_PATH[qwen_base]="-"
EXTRA_ARGS[qwen_base]=""

# --- BrickGPT (AvaLovelace/BrickGPT) evaluated as a plain causal LM, using the
#     point-cloud prompts from our dataset for a like-for-like comparison ---
MODEL_TYPE[brickgpt_causal]=causal_lm
MODEL_PATH[brickgpt_causal]="${BRICKGPT_HF}"
TOKEN_PATH[brickgpt_causal]="-"
EXTRA_ARGS[brickgpt_causal]=""

# --- BrickGPT with its full inference pipeline (rejection sampling + rollback)
#     using the BrickGPT class and point_cloud instruction format ---
MODEL_TYPE[brickgpt_full]=brickgpt
MODEL_PATH[brickgpt_full]="${BRICKGPT_HF}"
TOKEN_PATH[brickgpt_full]="-"
EXTRA_ARGS[brickgpt_full]="--brickgpt_use_logit_masking --brickgpt_max_brick_rejections 500 --brickgpt_instruction_format point_cloud"

# -----------------------------------------------------------------------------
# Which models to run (default: all of the above).
# Override with: MODELS="grpo_full_ft brickgpt_causal" sbatch submit_eval.slurm
# -----------------------------------------------------------------------------
DEFAULT_MODELS="grpo_full_ft grpo_base_full_ft grpo_no_connectivity grpo_no_structural sft_full_ft qwen_base brickgpt_causal brickgpt_full"
MODELS="${MODELS:-${DEFAULT_MODELS}}"

echo "models_to_eval  = ${MODELS}"
echo

# -----------------------------------------------------------------------------
# Run each model sequentially on the single allocated GPU.
# -----------------------------------------------------------------------------
FAILED=()
for key in ${MODELS}; do
    mt="${MODEL_TYPE[$key]:-}"
    mp="${MODEL_PATH[$key]:-}"
    tp="${TOKEN_PATH[$key]:-}"
    ea="${EXTRA_ARGS[$key]:-}"

    if [[ -z "${mt}" || -z "${mp}" ]]; then
        echo "[skip] unknown model key: ${key}"
        continue
    fi

    if [[ "${mp}" == /* && ! -d "${mp}" ]]; then
        echo "[warn] local checkpoint dir not found: ${mp} -- skipping ${key}"
        FAILED+=("${key} (missing checkpoint)")
        continue
    fi

    TOK_ARG=""
    if [[ -n "${tp}" && "${tp}" != "-" ]]; then
        TOK_ARG="--tokenizer_path ${tp}"
    fi

    RUN_LOG="${OUTPUT_DIR}/${key}.log"
    echo "--------------------------------------------------------------------------"
    echo "==> evaluating [${key}] type=${mt} path=${mp} $(date '+%F %T')"
    echo "    log: ${RUN_LOG}"
    echo "--------------------------------------------------------------------------"

    set +e
    "${CONDA_ENV}/bin/python" -u evaluate_models.py \
        --model_path "${mp}" \
        --model_type "${mt}" \
        ${TOK_ARG} \
        --dataset "${DATASET}" \
        --max_samples "${MAX_SAMPLES}" \
        --output_dir "${OUTPUT_DIR}" \
        --run_name "${key}" \
        ${USE_GUROBI_FLAG} \
        ${SAVE_GEN_FLAG} \
        ${ea} \
        2>&1 | tee "${RUN_LOG}"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ ${rc} -ne 0 ]]; then
        echo "[fail] evaluation for ${key} exited with status ${rc}"
        FAILED+=("${key}")
    else
        echo "[ok]   ${key} finished"
    fi
    echo
done

# -----------------------------------------------------------------------------
# Roll all per-model summary JSONs into one comparison CSV for convenience.
# -----------------------------------------------------------------------------
python - <<'PYEOF'
import json
import csv
import os
from pathlib import Path

out_dir = Path(os.environ["OUTPUT_DIR"])
summaries = []
for p in sorted(out_dir.glob("*_summary.json")):
    try:
        payload = json.loads(p.read_text())
    except Exception as e:
        print(f"[warn] failed to read {p}: {e}")
        continue
    summaries.append(payload)

if not summaries:
    print("[warn] no *_summary.json files found; nothing to compare")
    raise SystemExit(0)

keys = ["run_name"]
metric_keys: list[str] = []
for s in summaries:
    for k in s.get("summary", {}).keys():
        if k not in metric_keys:
            metric_keys.append(k)
keys += metric_keys

comparison_csv = out_dir / "comparison.csv"
with open(comparison_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(keys)
    for s in summaries:
        row = [s.get("run_name", "?")]
        for k in metric_keys:
            v = s.get("summary", {}).get(k, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            row.append(v)
        writer.writerow(row)
print(f"[ok] wrote comparison CSV: {comparison_csv}")
print()
print("=== MODEL COMPARISON ===")
header_row = ["run_name"] + [k for k in metric_keys if "rate" in k or "mean" in k]
print(",".join(header_row))
for s in summaries:
    row = [s.get("run_name", "?")]
    for k in header_row[1:]:
        v = s.get("summary", {}).get(k, "")
        if isinstance(v, float):
            v = f"{v:.4f}"
        row.append(str(v))
    print(",".join(row))
PYEOF

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[done] finished with failures: ${FAILED[*]}"
    echo "finish_time=$(date '+%F %T')"
    exit 1
fi
echo "[done] all evaluations complete"
echo "finish_time=$(date '+%F %T')"
