# ObliGate

ObliGate checks evidence and authorization obligations before an agent commits a tool action. This repository contains the gate and runtime implementation, benchmark adapters, adaptive attacks, component and policy ablations, and a real SMTP testbed.

## Paper results and experiment entrypoints

| Paper result | Research question | Entrypoint |
| --- | --- | --- |
| Table 1: AgentDojo | RQ1: cross-benchmark generalization | [run_main_agentdojo.sh](scripts/run_main_agentdojo.sh) |
| Table 1: Agent-SafetyBench and Agent Security Bench (ASB) | RQ1: cross-benchmark generalization | [run_main_cross_benchmark.sh](scripts/run_main_cross_benchmark.sh) |
| Table 2: AgentDyn | RQ1: cross-benchmark generalization | [run_agentdyn.sh](scripts/run_agentdyn.sh) / [run_agentdyn.py](scripts/run_agentdyn.py) |
| Table 3: adaptive attacks | RQ2: adaptive attacks | [run_adaptive.sh](scripts/run_adaptive.sh) / [pipeline.py](evaluation/adaptive_closed_loop_v2/pipeline.py), main phase |
| Table 4: mechanism stress | RQ2: adaptive attacks | [run_adaptive.sh](scripts/run_adaptive.sh) / [pipeline.py](evaluation/adaptive_closed_loop_v2/pipeline.py), stress phase |
| Table 5: component and policy ablations | RQ3: component ablation | [run_ablation.sh](scripts/run_ablation.sh) |
| Table 6: SMTP commit faults | RQ4: real SMTP and runtime | [run_smtp.sh](scripts/run_smtp.sh), `--only table6` |
| Table 7: SMTP warm-path latency | RQ4: real SMTP and runtime | [run_smtp.sh](scripts/run_smtp.sh), `--only table7` |

## Repository layout

```text
src/obligate/theory/                 Evidence, obligations, certificates and runtime gate
src/obligate/eval/                   Benchmark registry and AgentDojo/AgentDyn integration
experiments/agent_safetybench/       Agent-SafetyBench adapter
experiments/agent_security_bench/    ASB adapter
experiments/agentdyn/                AgentDyn evaluation and model transport
experiments/smtp_runtime/            SMTP server, adapter and fault/latency driver
experiments/adaptive_ablation/       Adaptive inputs and mechanism preflight
evaluation/adaptive_closed_loop_v2/  Adaptive attack controller and analysis
evaluation/obligate_ablation_v2/     Eight-configuration ablation pipeline
configs/                            Experiment protocols and SMTP fixtures
scripts/                            Setup, planning, execution and analysis commands
tests/                              Regression checks
```

## Quick start

Use Python 3.11 for the benchmark environments. The shell commands below use Bash on Linux, macOS, WSL or Git Bash. Run commands from the repository root.

```bash
git clone https://github.com/obligate-fse/ObliGate.git
cd ObliGate
python3.11 -m venv .venv-control
source .venv-control/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Inspect the SMTP protocol without model calls or network connections.
python -m experiments.smtp_runtime.run --mode plan

# Run the local SMTP smoke experiment; no model API key is required.
python -m experiments.smtp_runtime.run --mode smoke
```

For Windows PowerShell, the same SMTP driver can be run directly:

```powershell
py -3.11 -m venv .venv-control
.\.venv-control\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src;$PWD"
.\.venv-control\Scripts\python.exe -m experiments.smtp_runtime.run --mode smoke
```

## Benchmark environments and upstream inputs

Create separate environments for AgentDojo, Agent-SafetyBench, ASB and AgentDyn. Their dependency versions differ, and AgentDyn uses the same Python package name, `agentdojo`, as AgentDojo.

From the control environment:

```bash
python3.11 -m venv .venv-agentdojo
.venv-agentdojo/bin/python -m pip install -r requirements-agentdojo.txt
python3.11 -m venv .venv-safetybench
.venv-safetybench/bin/python -m pip install -r requirements-safetybench.txt
python3.11 -m venv .venv-asb
.venv-asb/bin/python -m pip install -r requirements-asb.txt

export OBLIGATE_AGENTDOJO_PYTHON="$PWD/.venv-agentdojo/bin/python"
export OBLIGATE_SAFETYBENCH_PYTHON="$PWD/.venv-safetybench/bin/python"
export OBLIGATE_ASB_PYTHON="$PWD/.venv-asb/bin/python"
export OBLIGATE_CONTROL_PYTHON="$OBLIGATE_AGENTDOJO_PYTHON"

git clone https://github.com/thu-coai/Agent-SafetyBench.git third_party/Agent-SafetyBench
git -C third_party/Agent-SafetyBench checkout 74feea8de601b3a1449a93fcf70017fe61556f73
git clone https://github.com/agiresearch/ASB.git third_party/ASB
git -C third_party/ASB checkout 1f561dccf92d55302368fa67679b4ba9d9c8fdc4
export SAFETYBENCH_UPSTREAM="$PWD/third_party/Agent-SafetyBench"
export ASB_UPSTREAM="$PWD/third_party/ASB"

# Generate the case plans used by the ablation and ASB entrypoints.
"$OBLIGATE_AGENTDOJO_PYTHON" scripts/build_agentdojo_case_plan.py
python scripts/build_asb_case_plan.py --upstream "$ASB_UPSTREAM"

# Check dependencies and input paths before running cases.
python -m obligate.eval.benchmark_cli doctor agentdojo --python "$OBLIGATE_AGENTDOJO_PYTHON" --strict
python -m obligate.eval.benchmark_cli doctor agent_safetybench --python "$OBLIGATE_SAFETYBENCH_PYTHON" --upstream-dir "$SAFETYBENCH_UPSTREAM" --strict
python -m obligate.eval.benchmark_cli doctor asb_iclr2025 --python "$OBLIGATE_ASB_PYTHON" --upstream-dir "$ASB_UPSTREAM" --strict
```

AgentDojo is installed as `agentdojo==0.1.35` and evaluated with benchmark data version `v1.2.2`. The generated plans are `data/agentdojo_v1.2.2_949_case_plan.json` and `data/asb_iclr2025_8160_case_plan.json`.

Agent-SafetyBench generation produces inputs for its upstream ShieldAgent scorer. Computing official safety scores requires the scorer, its weights and a separate Linux/CUDA environment. ASB judge-based metrics use its upstream evaluator.

## Model configuration

The main, adaptive and ablation protocols use `deepseek-v4-flash` and `qwen-plus`. AgentDyn defaults to `gpt-4o-2024-08-06`.

| Route | API key variable | Endpoint variable |
| --- | --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| Qwen / DashScope | `DASHSCOPE_API_KEY` | `DASHSCOPE_BASE_URL` |
| AgentDojo compatible runtime override | `OBLIGATE_LLM_API_KEY` | `OBLIGATE_LLM_BASE_URL` |
| AgentDyn / OpenAI | `OPENAI_API_KEY` | `--base-url` or `victim.base_url` in [agentdyn.yaml](configs/agentdyn.yaml) |

Set the required API keys in your environment before enabling provider inference. Main and adaptive runners support the corresponding endpoint overrides; the ablation protocol freezes its canonical DeepSeek and DashScope endpoints. AgentDyn also accepts `--model`, `--provider`, `--base-url` and `--api-key-env` overrides; its transport uses OpenAI-compatible Chat Completions.

## RQ1: main and cross-benchmark evaluation

Print the planned commands and manifests:

```bash
bash scripts/run_main_agentdojo.sh
bash scripts/run_main_cross_benchmark.sh
```

Enable model calls:

```bash
RUN_EXTERNAL=1 bash scripts/run_main_agentdojo.sh
RUN_EXTERNAL=1 bash scripts/run_main_cross_benchmark.sh
```

These entrypoints run both the baseline and ObliGate. AgentDojo includes clean tasks and 949 attack cases per model; Agent-SafetyBench uses 2,000 cases; ASB uses 8,160 cases across four attack protocols. Main-run settings are in [standard_protocols.yaml](configs/standard_protocols.yaml).

Use `MODELS` and `OUTPUT_ROOT` to select models and output paths. AgentDojo also accepts `SUITES` and `ATTACKS`:

```bash
RUN_EXTERNAL=1 MODELS=qwen-plus SUITES=banking ATTACKS=none \
  OUTPUT_ROOT="$PWD/reproduced/agentdojo_banking_clean" \
  bash scripts/run_main_agentdojo.sh
```

Default outputs are `reproduced/main_agentdojo/` and `reproduced/main_cross_benchmark/`. Cross-benchmark concurrency is controlled by `SAFETYBENCH_WORKERS` and `ASB_WORKERS`.

### AgentDyn / Table 2

Create its environment and fetch the pinned native benchmark:

```bash
python3.11 -m venv .venv-agentdyn
.venv-agentdyn/bin/python scripts/setup_agentdyn.py --install
.venv-agentdyn/bin/python scripts/setup_agentdyn.py --check
```

The setup command uses [SaFo-Lab/AgentDyn](https://github.com/SaFo-Lab/AgentDyn) at commit `5353cf7615b135cace8d07c8f12dac53a16b6db3`. The runner loads AgentDyn's `agentdojo` package from that checkout and evaluates the native `shopping`, `github` and `dailylife` suites, with clean tasks and `important_instructions` attacks.

```bash
# Freeze the run plan without API calls.
.venv-agentdyn/bin/python scripts/run_agentdyn.py \
  --plan-only --output results/agentdyn

# Run the paired No Defense / ObliGate evaluation with OPENAI_API_KEY set.
.venv-agentdyn/bin/python scripts/run_agentdyn.py \
  --run --output results/agentdyn

# Summarize completed episodes.
.venv-agentdyn/bin/python scripts/analyze_agentdyn.py results/agentdyn \
  --upstream third_party/AgentDyn --config configs/agentdyn.yaml
```

The configuration is [configs/agentdyn.yaml](configs/agentdyn.yaml). Use `--upstream` or `AGENTDYN_ROOT` for another checkout location. `--limit`, `--injection-limit` and `--workers` support smaller runs. Keep the same configuration when resuming an output directory; use a new output directory when changing the model, case selection or source code. Analysis writes `summary.json`, `analysis.json` and `metrics.csv`.

## RQ2: adaptive attacks and mechanism stress

The adaptive controller consumes completed AgentDojo, Agent-SafetyBench and ASB runs, plus the Full ablation's snapshots and raw runs. Prepare these inputs for one selected model:

```bash
MODEL=qwen-plus
INPUTS="$PWD/reproduced/adaptive_inputs_$MODEL"
FULL="$PWD/reproduced/ablation_v2/e2e/$MODEL/full"

# Set the AD_*_RUN, SB_OBLIGATE_RUN and ASB_OBLIGATE_RUN variables to
# completed run directories containing manifest.json for this model.
"$OBLIGATE_AGENTDOJO_PYTHON" scripts/prepare_adaptive_inputs.py \
  --model "$MODEL" \
  --agentdojo-run-dirs "$AD_BANKING_RUN" "$AD_SLACK_RUN" "$AD_TRAVEL_RUN" "$AD_WORKSPACE_RUN" \
  --safetybench-run-dir "$SB_OBLIGATE_RUN" \
  --asb-run-dir "$ASB_OBLIGATE_RUN" \
  --full-snapshot-dir "$FULL/snapshots" \
  --full-raw-dir "$FULL/raw_runs" \
  --output-dir "$INPUTS"
source "$INPUTS/adaptive_environment.sh"

bash scripts/run_adaptive.sh

# Execute only the model whose feedback and snapshots were prepared above.
OBLIGATE_RESULT_ROOT="$PWD/reproduced/adaptive_$MODEL" \
  "$OBLIGATE_CONTROL_PYTHON" -m evaluation.adaptive_closed_loop_v2.pipeline \
  --phase main --max-rounds 5 --models "$MODEL"
OBLIGATE_RESULT_ROOT="$PWD/reproduced/adaptive_stress_$MODEL" \
  "$OBLIGATE_CONTROL_PYTHON" -m evaluation.adaptive_closed_loop_v2.pipeline \
  --phase stress --max-rounds 5 --models "$MODEL"
```

Repeat input preparation and execution separately for `deepseek-v4-flash`. The per-model commands above keep each model's feedback and Full snapshots matched; the shell execution wrapper selects both models. The current adaptive preflight requires both `DEEPSEEK_API_KEY` and `DASHSCOPE_API_KEY`, including for a single-model invocation.

The main protocol runs five rounds of public-feedback attacks. The stress phase uses three attacker seeds and mechanism activation inputs. Add `--resume` to a direct pipeline command to continue the same sealed run. `--phase all` executes the dry, main and stress phases; the pipeline's dry phase also makes model calls. The shell wrapper without `RUN_EXTERNAL=1` only prints planning information and help. Settings are in [adaptive_closed_loop_v2.yaml](configs/adaptive_closed_loop_v2.yaml).

## RQ3: eight ablation configurations

[ablation_variants.yaml](configs/ablation_variants.yaml) defines all eight Table 5 configurations:

| Paper configuration | Configuration key |
| --- | --- |
| Full | `full` |
| scalar-average | `scalar-average` |
| unbound-evidence | `unbound-evidence` |
| gap-blind | `gap-blind` |
| single-remedy | `single-remedy` |
| DAG-Single | `dag-single` |
| PolicyPromptOnly | `policy-prompt-only` |
| block-all-guarded | `block-all-guarded` |

The formal protocol is **8 configurations × 949 attack cases × 2 models = 15,184 attack episodes**.

```bash
bash scripts/run_ablation.sh
RUN_EXTERNAL=1 ABLATION_STAGE=prepare bash scripts/run_ablation.sh
RUN_EXTERNAL=1 ABLATION_STAGE=preflight bash scripts/run_ablation.sh
RUN_EXTERNAL=1 ABLATION_STAGE=formal bash scripts/run_ablation.sh
```

Alternatively, `ABLATION_STAGE=all` runs the stages in sequence. The pipeline prepares the fixed population, checks the controls, runs the 10-case preflight per model/configuration, and then executes the formal cells. Results default to `reproduced/ablation_v2/`; use `OUTPUT_ROOT` and `RESUME=1` to continue the same sealed protocol.

## RQ4: real SMTP and runtime

The testbed runs a loopback TCP SMTP server and captures transmitted messages and envelopes. It exercises commit binding, confirmation, mutation, audit, redaction and lost-DATA-reply conditions using five mail fixtures in [smtp_mail_seeds.json](configs/smtp_mail_seeds.json).

```bash
# Print the plan.
bash scripts/run_smtp.sh

# Run the smaller local experiment.
RUN_LOCAL=1 SMTP_STAGE=smoke bash scripts/run_smtp.sh

# Run both formal tables.
RUN_LOCAL=1 SMTP_STAGE=formal bash scripts/run_smtp.sh

# Run one table through the direct CLI.
python -m experiments.smtp_runtime.run --mode formal --only table6
python -m experiments.smtp_runtime.run --mode formal --only table7
```

Formal mode runs 50 trials per condition and method (`NoGate` / `Full`), and 500 warm samples for each latency path (`P0` / `P1_confirmed`). Warmups are excluded; confirmation latency excludes human response time. The SMTP experiment requires no model API key or external mail service.

Settings are in [smtp_runtime.json](configs/smtp_runtime.json). Outputs include `summary.json`, `table6_trials.jsonl`, `table7_latency.jsonl`, environment and service metadata, and message captures under `results/smtp_runtime/`. An explicit `--output-root` must name a new or empty directory for each run.

## Local checks

With the control environment and `PYTHONPATH` configured:

```bash
python scripts/validate_configs.py agentdojo
python scripts/validate_configs.py cross-benchmark
python scripts/validate_configs.py adaptive
python scripts/validate_configs.py ablation
python scripts/validate_configs.py smtp
python -m unittest discover -s tests -p 'test_smtp_runtime.py'
```
