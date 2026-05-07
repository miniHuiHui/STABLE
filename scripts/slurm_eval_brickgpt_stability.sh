#!/usr/bin/env bash
# Slurm: evaluate BrickGPT paper metrics (mean-stability, min-stability) on
# pre-generated LDraw samples under generated_ldr/baseline_ablation_16088.
#
# Usage (from repo root, after editing account/partition as needed):
#   mkdir -p logs
#   sbatch scripts/slurm_eval_brickgpt_stability.sh
#
# Optional env overrides:
#   EVAL_ROOT   — default: $PROJECT_ROOT/generated_ldr/baseline_ablation_16088
#   EVAL_OUT    — default: $EVAL_ROOT/brickgpt_stability_eval.json
#   ONLY_EXP    — space-separated experiment subdir names to run (else all)
#
# Requires a valid Gurobi license on the compute node (same as training stability checks).

#SBATCH --job-name=brickgpt-stab
#SBATCH --output=logs/brickgpt_stab_%j.out
#SBATCH --error=logs/brickgpt_stab_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/generated_ldr/baseline_ablation_16088}"
EVAL_OUT="${EVAL_OUT:-${EVAL_ROOT}/brickgpt_stability_eval.json}"
mkdir -p "$(dirname "${EVAL_OUT}")" logs

# Python: use active conda/venv if present, else `python3`
PY="${PY:-python3}"
if command -v micromamba >/dev/null 2>&1 && [[ -f "${PROJECT_ROOT}/.python-version" ]]; then
  : # user can `micromamba run -n ...` by setting PY
fi

ONLY_ARGS=()
if [[ -n "${ONLY_EXP:-}" ]]; then
  # shellcheck disable=SC2206
  ONLY_ARGS=(--only ${ONLY_EXP})
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "EVAL_ROOT=${EVAL_ROOT}"
echo "EVAL_OUT=${EVAL_OUT}"
echo "Starting: $(date -Is)"

${PY} scripts/eval_ldr_brickgpt_stability.py \
  --root "${EVAL_ROOT}" \
  --output_json "${EVAL_OUT}" \
  "${ONLY_ARGS[@]:-}"

echo "Done: $(date -Is)"
