#!/usr/bin/env bash
# Cross-benchmark reproduction entrypoint.
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/validate_configs.py" cross-benchmark
SAFETYBENCH_PYTHON="${OBLIGATE_SAFETYBENCH_PYTHON:-python}"
ASB_PYTHON="${OBLIGATE_ASB_PYTHON:-python}"
SAFETYBENCH_WORKERS="${SAFETYBENCH_WORKERS:-1}"
ASB_WORKERS="${ASB_WORKERS:-4}"
MODELS="${MODELS:-deepseek-v4-flash qwen-plus}"
OUT="${OUTPUT_ROOT:-$ROOT/reproduced/main_cross_benchmark}"
ASB_CASE_PLAN="${OBLIGATE_ASB_CASE_PLAN:-$ROOT/data/asb_iclr2025_8160_case_plan.json}"
if [[ "${RUN_EXTERNAL:-0}" != "1" ]]; then
  echo "PLAN ONLY. Separate benchmark interpreters are recorded below."
fi
if [[ "${RUN_EXTERNAL:-0}" == "1" ]]; then
  [[ -n "${SAFETYBENCH_UPSTREAM:-}" && -n "${ASB_UPSTREAM:-}" ]] || { echo "SAFETYBENCH_UPSTREAM and ASB_UPSTREAM are required" >&2; exit 2; }
fi
for model in $MODELS; do
  for defense in none obligate_visible_fair; do
    args=(run agent_safetybench --python "$SAFETYBENCH_PYTHON" --repo-root "$ROOT" --output-root "$OUT" --model "$model" --seed 0 --limit 2000 --runner-arg=--defense --runner-arg="$defense" --runner-arg=--temperature --runner-arg=0 --runner-arg=--max-tokens --runner-arg=2048 --runner-arg=--timeout --runner-arg=180 --runner-arg=--max-rounds --runner-arg=10 --runner-arg=--workers --runner-arg="$SAFETYBENCH_WORKERS")
    if [[ "${RUN_EXTERNAL:-0}" != "1" ]]; then
      python -m obligate.eval.benchmark_cli "${args[@]}" --dry-run
    else
      python -m obligate.eval.benchmark_cli "${args[@]}" --upstream-dir "$SAFETYBENCH_UPSTREAM"
    fi
  done
  for mode in no_defense obligate_registry_blind; do
    args=(run asb_iclr2025 --python "$ASB_PYTHON" --repo-root "$ROOT" --output-root "$OUT" --model "$model" --seed 0 --runner-arg=--modes --runner-arg="$mode" --runner-arg=--attack-modes --runner-arg=dpi --runner-arg=opi --runner-arg=mixed --runner-arg=memory --runner-arg=--attack-types --runner-arg=naive --runner-arg=context_ignoring --runner-arg=combined_attack --runner-arg=--attack-tool-type --runner-arg=all --runner-arg=--task-num --runner-arg=all --runner-arg=--case-ids-file --runner-arg="$ASB_CASE_PLAN" --runner-arg=--temperature --runner-arg=0 --runner-arg=--max-tokens --runner-arg=512 --runner-arg=--timeout --runner-arg=60 --runner-arg=--workers --runner-arg="$ASB_WORKERS")
    if [[ "${RUN_EXTERNAL:-0}" != "1" ]]; then
      python -m obligate.eval.benchmark_cli "${args[@]}" --dry-run
    else
      python -m obligate.eval.benchmark_cli "${args[@]}" --upstream-dir "$ASB_UPSTREAM"
    fi
  done
done
