#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/validate_configs.py" agentdojo
AGENTDOJO_PYTHON="${OBLIGATE_AGENTDOJO_PYTHON:-python}"
export AGENTDOJO_LLM_RETRY_ATTEMPTS=4
export AGENTDOJO_LLM_RETRY_INITIAL_SEC=2
export OBLIGATE_LLM_TIMEOUT=300
MODELS="${MODELS:-deepseek-v4-flash qwen-plus}"
SUITES="${SUITES:-banking slack travel workspace}"
ATTACKS="${ATTACKS:-none important_instructions}"
OUT="${OUTPUT_ROOT:-$ROOT/reproduced/main_agentdojo}"
for model in $MODELS; do
  for suite in $SUITES; do
    for attack in $ATTACKS; do
      for defense in none obligate; do
        args=(run agentdojo --python "$AGENTDOJO_PYTHON" --repo-root "$ROOT" --output-root "$OUT" --model "$model" --suite "$suite" --seed 0 --attack "$attack" --defense "$defense" --runner-arg=--max-iters --runner-arg=24 --runner-arg=--confirmation-mode --runner-arg=strict_eval --runner-arg=--mode --runner-arg=fair --runner-arg=--save-full-trace)
        if [[ "${RUN_EXTERNAL:-0}" != "1" ]]; then
          python -m obligate.eval.benchmark_cli "${args[@]}" --dry-run
        else
          python -m obligate.eval.benchmark_cli "${args[@]}"
        fi
      done
    done
  done
done
