#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/validate_configs.py" ablation
OUT="${OUTPUT_ROOT:-$ROOT/reproduced/ablation_v2}"
PYTHON_BIN="${OBLIGATE_AGENTDOJO_PYTHON:-python}"
STAGE="${ABLATION_STAGE:-${RQ4_STAGE:-all}}"
SOURCE_PLAN="${OBLIGATE_AGENTDOJO_CASE_PLAN:-$ROOT/data/agentdojo_v1.2.2_949_case_plan.json}"
SOURCE_SHA="${OBLIGATE_AGENTDOJO_CASE_PLAN_SHA256:-4637e3a436ee5b8be6550871954cab654ca93ecf3a1956a85d3a0a28bb588eb7}"
if [[ "${RUN_EXTERNAL:-0}" != "1" ]]; then
  echo "PLAN ONLY: 8 configurations x 949 cases x 2 models = 15,184 episodes."
  echo "Stages: prepare -> paid 10-case preflight -> formal run."
  "$PYTHON_BIN" -m evaluation.obligate_ablation_v2.reproduce --help
  exit 0
fi
RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi
"$PYTHON_BIN" -m evaluation.obligate_ablation_v2.reproduce "$STAGE" --output-root "$OUT" --source-plan "$SOURCE_PLAN" --source-plan-sha256 "$SOURCE_SHA" "${RESUME_ARGS[@]}"
