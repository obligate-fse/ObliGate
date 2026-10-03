#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/validate_configs.py" adaptive
CONTROL_PYTHON="${OBLIGATE_CONTROL_PYTHON:-${OBLIGATE_AGENTDOJO_PYTHON:-python}}"
OUT="${OUTPUT_ROOT:-$ROOT/reproduced/adaptive}"
PHASE="${ADAPTIVE_PHASE:-main}"
case "$PHASE" in
  dry|main|stress|all) ;;
  *) echo "ADAPTIVE_PHASE must be dry, main, stress, or all" >&2; exit 2 ;;
esac
if [[ "${RUN_EXTERNAL:-0}" != "1" ]]; then
  echo "PLAN ONLY: phase=$PHASE; five rounds; main=1,739 cases/model; stress=three attacker seeds."
  echo "Required: OBLIGATE_ADAPTIVE_INPUT_ROOT, OBLIGATE_MECHANISM_PREFLIGHT,"
  echo "SAFETYBENCH_UPSTREAM, ASB_UPSTREAM, and three benchmark Python variables."
  "$CONTROL_PYTHON" -m evaluation.adaptive_closed_loop_v2.pipeline --help
  exit 0
fi
for name in OBLIGATE_ADAPTIVE_INPUT_ROOT OBLIGATE_MECHANISM_PREFLIGHT SAFETYBENCH_UPSTREAM ASB_UPSTREAM OBLIGATE_AGENTDOJO_PYTHON OBLIGATE_SAFETYBENCH_PYTHON OBLIGATE_ASB_PYTHON; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
export OBLIGATE_RESULT_ROOT="$OUT"
RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi
"$CONTROL_PYTHON" -m evaluation.adaptive_closed_loop_v2.pipeline --phase "$PHASE" --max-rounds 5 --models deepseek-v4-flash qwen-plus "${RESUME_ARGS[@]}"
