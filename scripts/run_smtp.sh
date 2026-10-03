#!/usr/bin/env bash
# Paper RQ4 / Tables 6 and 7: real SMTP execution boundary and warm latency.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${OBLIGATE_SMTP_PYTHON:-${PYTHON:-python}}"
"$PYTHON_BIN" "$ROOT/scripts/validate_configs.py" smtp
if [[ "$#" -gt 0 ]]; then
  exec "$PYTHON_BIN" -m experiments.smtp_runtime.run "$@"
fi
if [[ "${RUN_LOCAL:-0}" != "1" ]]; then
  exec "$PYTHON_BIN" -m experiments.smtp_runtime.run --mode plan
fi
STAGE="${SMTP_STAGE:-smoke}"
OUT_ARGS=()
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  OUT_ARGS=(--output-root "$OUTPUT_ROOT")
fi
exec "$PYTHON_BIN" -m experiments.smtp_runtime.run --mode "$STAGE" "${OUT_ARGS[@]}"
