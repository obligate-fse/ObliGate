"""Frozen paired AgentDyn experiments with native benchmark scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from dataclasses import asdict
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
PINNED_REVISION = "5353cf7615b135cace8d07c8f12dac53a16b6db3"
SUITES = ("shopping", "github", "dailylife")


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _digest(value):
    return hashlib.sha256(_encoded(value).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    try:
        temporary.replace(path)
    except PermissionError:
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            temporary.unlink()
        except PermissionError:
            pass


def _positive(value, name, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer" + (" or null" if optional else ""))


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return validate_config(config)


def validate_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("AgentDyn config requires schema_version: 1")
    if config.get("upstream", {}).get("revision") != PINNED_REVISION:
        raise ValueError(f"AgentDyn revision must be {PINNED_REVISION}")
    benchmark = config.get("benchmark", {})
    if benchmark.get("version") != "v1.2.2" or benchmark.get("attack") != "important_instructions":
        raise ValueError("AgentDyn requires v1.2.2 and important_instructions")
    suites = benchmark.get("suites")
    if not isinstance(suites, list) or not suites or len(set(suites)) != len(suites) or any(s not in SUITES for s in suites):
        raise ValueError(f"benchmark.suites must contain unique names from {SUITES}")
    methods = config.get("methods")
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) or any(m not in {"none", "obligate"} for m in methods):
        raise ValueError("methods must contain unique names from none and obligate")
    victim = config.get("victim", {})
    for field in ("provider", "model", "api_key_env"):
        if not isinstance(victim.get(field), str) or not victim[field].strip():
            raise ValueError(f"victim.{field} must be a nonempty string")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", victim["api_key_env"]):
        raise ValueError("victim.api_key_env must be an environment variable name")
    if any(key in victim for key in ("api_key", "authorization", "headers", "extra_headers")):
        raise ValueError("Configure only an API key environment variable")
    victim.setdefault("base_url", None)
    victim.setdefault("temperature", 0)
    victim.setdefault("timeout", 120)
    victim.setdefault("max_retries", 0)
    victim.setdefault("extra_body", {})
    victim.setdefault("developer_role", "preserve")
    if victim["base_url"] is not None:
        if not isinstance(victim["base_url"], str):
            raise ValueError("victim.base_url must be null or an HTTP(S) URL")
        endpoint = urlsplit(victim["base_url"])
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError("victim.base_url must be an HTTP(S) URL without credentials or query parameters")
    if victim["temperature"] is not None and (isinstance(victim["temperature"], bool) or not isinstance(victim["temperature"], (int, float)) or not 0 <= victim["temperature"] <= 2):
        raise ValueError("victim.temperature must be null or a number from 0 to 2")
    if victim["max_retries"] != 0:
        raise ValueError("victim.max_retries must be 0 for complete API call accounting")
    if victim["developer_role"] not in {"preserve", "system"}:
        raise ValueError("victim.developer_role must be preserve or system")
    if not isinstance(victim["timeout"], (int, float)) or isinstance(victim["timeout"], bool) or victim["timeout"] <= 0:
        raise ValueError("victim.timeout must be positive")
    if not isinstance(victim["extra_body"], dict):
        raise ValueError("victim.extra_body must be a mapping")
    if any(key in victim["extra_body"] for key in ("api_key", "authorization", "headers", "extra_headers")):
        raise ValueError("victim.extra_body cannot contain credentials")
    execution = config.setdefault("execution", {})
    for key, default in (("limit", None), ("injection_limit", None), ("max_iters", 24), ("workers", 8)):
        execution.setdefault(key, default)
        _positive(execution[key], f"execution.{key}", key in {"limit", "injection_limit"})
    return config


def _git(upstream, *arguments):
    return subprocess.check_output(["git", "-C", str(upstream), *arguments], stderr=subprocess.PIPE).decode("utf-8").strip()


def bootstrap_upstream(path, revision=PINNED_REVISION):
    upstream = Path(path).resolve()
    if not (upstream / "src/agentdojo/__init__.py").is_file():
        raise RuntimeError(f"AgentDyn checkout is missing at {upstream}; run scripts/setup_agentdyn.py first")
    if _git(upstream, "rev-parse", "HEAD") != revision:
        raise RuntimeError(f"AgentDyn checkout must be at revision {revision}")
    if _git(upstream, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("AgentDyn checkout has tracked modifications")
    for directory in (ROOT, ROOT / "src", upstream / "src"):
        location = str(directory)
        if location in sys.path:
            sys.path.remove(location)
        sys.path.insert(0, location)
    importlib.invalidate_caches()
    package = importlib.import_module("agentdojo")
    imported = Path(package.__file__).resolve()
    expected = (upstream / "src/agentdojo/__init__.py").resolve()
    if imported != expected:
        raise RuntimeError(f"Wrong agentdojo import: {imported}; start a fresh process using {expected}")
    return upstream


def local_source_hashes():
    local_paths = set((ROOT / "src").rglob("*.py")) | set((ROOT / "src").rglob("*.yaml"))
    local_paths.update((ROOT / "experiments/agentdyn").glob("*.py"))
    local_paths.add(ROOT / "scripts/run_agentdyn.py")
    local_paths.add(ROOT / "scripts/run_agentdyn.sh")
    local_paths.add(ROOT / "scripts/setup_agentdyn.py")
    local_paths.add(ROOT / "scripts/analyze_agentdyn.py")
    local_paths.update(ROOT.glob("requirements*.txt"))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(local_paths) if "__pycache__" not in p.parts}


def source_snapshot(upstream, config_path):
    files = subprocess.check_output(["git", "-C", str(upstream), "ls-files", "-z"]).decode("utf-8").split("\0")
    execution_files = [name for name in files if name.startswith("src/") or name in {"pyproject.toml", "uv.lock", "poetry.lock"}
                       or (name.startswith("requirements") and name.endswith(".txt") and "/" not in name)]
    upstream_hashes = {name: hashlib.sha256((Path(upstream) / name).read_bytes()).hexdigest()
                       for name in sorted(execution_files) if (Path(upstream) / name).is_file()}
    return {"local": local_source_hashes(), "upstream": upstream_hashes,
            "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest()}


def episode_id(case, method):
    components = (case["suite"], case["user_task"], case["injection_task"] or "clean", method)
    if any(not isinstance(part, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in components):
        raise ValueError("Episode identifiers must be safe path components")
    return "__".join(components)


def validate_resume(existing, candidate):
    old = {key: value for key, value in existing.items() if key != "created_utc"}
    new = {key: value for key, value in candidate.items() if key != "created_utc"}
    if old != new:
        raise RuntimeError("AgentDyn run configuration, cases, source or upstream changed; select a new output directory")


def prepare(output, upstream, config, config_path):
    upstream = bootstrap_upstream(upstream, config["upstream"]["revision"])
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
    from agentdojo.models import MODEL_NAMES
    from obligate.eval.agentdyn.tool_properties import AgentDynToolTaxonomy
    model = config["victim"]["model"]
    if model not in MODEL_NAMES:
        MODEL_NAMES[model] = config["victim"].get("attack_model_name", model)
    taxonomy = AgentDynToolTaxonomy()
    cases, coverage, schemas = [], {}, {}
    for name in config["benchmark"]["suites"]:
        suite = get_suite(config["benchmark"]["version"], name)
        coverage[name] = taxonomy.coverage([tool.name for tool in suite.tools], suite=name)
        schemas[name] = {tool.name: {"description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
            "properties": asdict(taxonomy.classify(tool.name, suite=name))} for tool in suite.tools}
        attack = ImportantInstructionsAttack(suite, SimpleNamespace(name=model))
        tasks = list(suite.user_tasks.values())[:config["execution"]["limit"]]
        injections = list(suite.injection_tasks.values())[:config["execution"]["injection_limit"]]
        for task in tasks:
            cases.append({"suite": name, "user_task": task.ID, "injection_task": None, "injections": {}})
            for injection in injections:
                cases.append({"suite": name, "user_task": task.ID, "injection_task": injection.ID,
                              "injections": attack.attack(task, injection)})
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "methods": config["methods"], "upstream_revision": config["upstream"]["revision"],
                "source_hashes": source_snapshot(upstream, config_path), "coverage": coverage,
                "tool_schemas_sha256": _digest(schemas), "base_cases": len(cases),
                "paired_episodes": len(cases) * len(config["methods"]), "cases": cases}
    ids = [episode_id(case, method) for case in cases for method in config["methods"]]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Generated duplicate AgentDyn episode identifiers")
    manifest["run_id"] = _digest({k: v for k, v in manifest.items() if k != "created_utc"})
    output = Path(output)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume(existing, manifest)
        stored_schemas = json.loads((output / "tool_schemas.json").read_text(encoding="utf-8"))
        if _digest(stored_schemas) != existing["tool_schemas_sha256"]:
            raise RuntimeError("Frozen AgentDyn tool schemas changed")
        return existing
    if any((output / directory).exists() and any((output / directory).iterdir()) for directory in ("episodes", "traces", "audit", "model_journal")):
        raise RuntimeError("Output contains experiment files but has no frozen manifest")
    write_json(manifest_path, manifest)
    write_json(output / "tool_schemas.json", schemas)
    return manifest


def _safe_error(exc, api_key_env=None):
    text = f"{type(exc).__name__}: {exc}"
    for key, value in os.environ.items():
        if ("API_KEY" in key.upper() or key == api_key_env) and len(value) >= 8:
            text = text.replace(value, "[REDACTED_CREDENTIAL]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED_CREDENTIAL]", text)


def run_case(job, llm_factory=None):
    config, case, method = job["config"], job["case"], job["method"]
    bootstrap_upstream(job["upstream"], config["upstream"]["revision"])
    from agentdojo.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.basic_elements import SystemMessage, InitQuery
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
    from agentdojo.task_suite.load_suites import get_suite
    from obligate.eval.agentdojo.gate.tool_firewall import AgentDojoToolFirewall
    from obligate.eval.agentdojo.gate.runtime_wrapper import AgentDojoFirewallPipeline, AgentDojoGuardedFunctionsRuntime
    from obligate.eval.agentdyn.tool_properties import ACTIVE_CALL, AgentDynToolTaxonomy
    from obligate.eval.model_accounting import begin_case, snapshot
    from experiments.agentdyn.transport import create_transport

    class PropertyRuntime(AgentDojoGuardedFunctionsRuntime):
        def run_function(self, env, function, kwargs, raise_on_error=False):
            token = ACTIVE_CALL.set((env, kwargs))
            try:
                return super().run_function(env, function, kwargs, raise_on_error)
            finally:
                ACTIVE_CALL.reset(token)

    class PropertyInjector(BasePipelineElement):
        def __init__(self, firewall, context):
            self.firewall, self.context = firewall, context

        def query(self, query, runtime, env=None, messages=(), extra_args=None):
            if not isinstance(runtime, PropertyRuntime):
                runtime = PropertyRuntime(runtime, self.firewall, task_context=self.context, suite=case["suite"])
            return query, runtime, env, messages, extra_args or {}

    traces = []
    class RecordingElement(BasePipelineElement):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def query(self, query, runtime, env=None, messages=(), extra_args=None):
            traces[-1]["messages"] = list(messages)
            result = self.wrapped.query(query, runtime, env, messages, extra_args or {})
            traces[-1]["messages"] = list(result[3])
            return result

    class RecordingPipeline(BasePipelineElement):
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.name = config["victim"]["model"] + "-" + method

        def query(self, *args, **kwargs):
            traces.append({"attempt": len(traces) + 1, "messages": []})
            return self.wrapped.query(*args, **kwargs)

    identifier = episode_id(case, method)
    output = Path(job["output"])
    os.environ["OBLIGATE_MODEL_JOURNAL_DIR"] = str(output / "model_journal")
    begin_case(identifier)
    started = time.perf_counter()
    result = {"id": identifier, "run_id": job["run_id"], "suite": case["suite"],
              "user_task": case["user_task"], "injection_task": case["injection_task"],
              "method": method, "defense": method, "model": config["victim"]["model"],
              "status": "ERROR", "user_success": None, "attack_success": None}
    llm, firewall = None, None
    try:
        suite = get_suite(config["benchmark"]["version"], case["suite"])
        task = suite.user_tasks[case["user_task"]]
        injection = suite.injection_tasks[case["injection_task"]] if case["injection_task"] else None
        llm = (llm_factory or create_transport)(config["victim"])
        base_system = load_system_message(None)
        context = {"suite": case["suite"], "user_task": task.PROMPT, "user_task_id": task.ID,
                   "run_id": job["run_id"], "sample_id": identifier, "defense_mode": "fair"}
        if method == "obligate":
            firewall = AgentDojoToolFirewall(taxonomy=AgentDynToolTaxonomy(), confirmation_mode="strict_eval")
            system = AgentDojoFirewallPipeline._compose_system_message(None, base_system)
            executor_elements = [PropertyInjector(firewall, context), RecordingElement(ToolsExecutor()), RecordingElement(llm)]
        elif method == "none":
            system, executor_elements = base_system, [RecordingElement(ToolsExecutor()), RecordingElement(llm)]
        else:
            raise ValueError(f"Unsupported AgentDyn method {method}")
        pipeline = RecordingPipeline(AgentPipeline(elements=[SystemMessage(system), InitQuery(), RecordingElement(llm),
            ToolsExecutionLoop(executor_elements, max_iters=config["execution"]["max_iters"])]))
        utility, attack_success = suite.run_task_with_pipeline(pipeline, task, injection, case["injections"])
        result.update(status="OK", user_success=bool(utility), attack_success=bool(attack_success) if injection else None)
    except Exception as exc:
        result["error"] = _safe_error(exc, config["victim"]["api_key_env"])
    finally:
        if llm is not None:
            try:
                if hasattr(llm, "close"):
                    llm.close()
                elif hasattr(llm, "client"):
                    llm.client.close()
            except Exception as exc:
                result.update(status="ERROR", error=_safe_error(exc, config["victim"]["api_key_env"]), user_success=None, attack_success=None)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["usage"] = snapshot()
    events = firewall.audit_events if firewall else []
    result["decisions"] = dict(Counter(event["decision"] for event in events if event.get("event_type") == "agentdojo_tool_gate_decision"))
    result["gate_dispatches"] = sum(bool(event.get("gate_dispatched")) for event in events)
    messages = [message for trace in traces for message in trace["messages"]]
    result["tool_calls"] = sum(len(message.get("tool_calls") or []) for message in messages if message["role"] == "assistant")
    payloads = list(case["injections"].values())
    tool_text = [str(block.get("content", "")) for message in messages if message["role"] == "tool" for block in (message.get("content") or [])]
    result["attack_payload_visible"] = any(payload in text for payload in payloads for text in tool_text)
    result["hit_iteration_limit"] = any(trace["messages"] and trace["messages"][-1]["role"] == "assistant" and bool(trace["messages"][-1].get("tool_calls")) for trace in traces)
    result["pipeline_attempts"] = len(traces)
    write_json(output / "traces" / (identifier + ".json"), traces)
    write_json(output / "audit" / (identifier + ".json"), events)
    result["trace_sha256"] = hashlib.sha256((output / "traces" / (identifier + ".json")).read_bytes()).hexdigest()
    result["audit_sha256"] = hashlib.sha256((output / "audit" / (identifier + ".json")).read_bytes()).hexdigest()
    write_json(output / "episodes" / (identifier + ".json"), result)
    return result


def _episode_rows(output, manifest):
    output = Path(output)
    expected = {episode_id(case, method): (case, method) for case in manifest["cases"] for method in manifest["config"]["methods"]}
    rows = []
    for path in sorted((output / "episodes").glob("*.json")):
        if not path.is_file() or path.suffix != ".json" or path.stem not in expected:
            raise RuntimeError(f"Unexpected AgentDyn episode file {path.name}")
        row = json.loads(path.read_text(encoding="utf-8"))
        case, method = expected[path.stem]
        if (row.get("id") != path.stem or row.get("run_id") != manifest["run_id"] or row.get("suite") != case["suite"]
                or row.get("user_task") != case["user_task"] or row.get("injection_task") != case["injection_task"]
                or row.get("method") != method or row.get("model") != manifest["config"]["victim"]["model"]):
            raise RuntimeError(f"Episode does not belong to frozen AgentDyn run: {path.name}")
        if row.get("status") not in {"OK", "ERROR"}:
            raise RuntimeError(f"Invalid episode status: {path.name}")
        if row["status"] == "OK" and (not isinstance(row.get("user_success"), bool) or
                (case["injection_task"] is not None and not isinstance(row.get("attack_success"), bool))):
            raise RuntimeError(f"Invalid native episode scores: {path.name}")
        if row["status"] == "ERROR" and (row.get("user_success") is not None or row.get("attack_success") is not None):
            raise RuntimeError(f"Error episode contains native scores: {path.name}")
        for kind in ("trace", "audit"):
            directory = "traces" if kind == "trace" else "audit"
            artifact = output / directory / path.name
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != row.get(kind + "_sha256"):
                raise RuntimeError(f"Episode {kind} changed or missing: {path.name}")
        rows.append(row)
    return rows


def summarize(output, manifest):
    if local_source_hashes() != manifest["source_hashes"]["local"]:
        raise RuntimeError("Frozen local AgentDyn experiment source changed")
    rows = _episode_rows(output, manifest)
    summary = {"run_id": manifest["run_id"], "expected": manifest["paired_episodes"], "completed": len(rows),
               "pending": manifest["paired_episodes"] - len(rows), "errors": sum(row["status"] == "ERROR" for row in rows), "groups": {}}
    for suite in ("all", *manifest["config"]["benchmark"]["suites"]):
        for method in manifest["config"]["methods"]:
            group = [row for row in rows if row["method"] == method and (suite == "all" or row["suite"] == suite)]
            scheduled = [case for case in manifest["cases"] if suite == "all" or case["suite"] == suite]
            clean_expected = sum(case["injection_task"] is None for case in scheduled)
            attacked_expected = len(scheduled) - clean_expected
            valid = [row for row in group if row["status"] == "OK"]
            clean = [row for row in valid if row["injection_task"] is None]
            attacked = [row for row in valid if row["injection_task"] is not None]
            ratio = lambda records, field: sum(bool(row[field]) for row in records) / len(records) if records else None
            bounds = lambda records, field, expected: (
                sum(bool(row[field]) for row in records) / expected,
                (sum(bool(row[field]) for row in records) + expected - len(records)) / expected) if expected else (None, None)
            clean_bounds = bounds(clean, "user_success", clean_expected)
            attacked_bounds = bounds(attacked, "user_success", attacked_expected)
            attack_bounds = bounds(attacked, "attack_success", attacked_expected)
            summary["groups"][suite + "/" + method] = {"episodes": len(group), "valid": len(valid), "errors": len(group) - len(valid),
                "expected": len(scheduled), "missing": len(scheduled) - len(group),
                "clean_expected": clean_expected, "attacked_expected": attacked_expected,
                "clean_n": len(clean), "attacked_n": len(attacked), "clean_valid": len(clean), "attacked_valid": len(attacked),
                "clean_utility": ratio(clean, "user_success"),
                "attacked_utility": ratio(attacked, "user_success"), "attack_success_rate": ratio(attacked, "attack_success"),
                "clean_utility_lower": clean_bounds[0], "clean_utility_upper": clean_bounds[1],
                "attacked_utility_lower": attacked_bounds[0], "attacked_utility_upper": attacked_bounds[1],
                "attack_success_lower": attack_bounds[0], "attack_success_upper": attack_bounds[1],
                "iteration_limit_n": sum(row["hit_iteration_limit"] for row in group),
                "zero_tool_call_n": sum(row["tool_calls"] == 0 for row in group),
                "payload_visible_n": sum(row["attack_payload_visible"] for row in attacked),
                "input_tokens": sum(row["usage"]["input_tokens"] for row in group),
                "output_tokens": sum(row["usage"]["output_tokens"] for row in group),
                "api_errors": sum(row["usage"]["errors"] for row in group),
                "usage_unknown_calls": sum(row["usage"]["usage_unknown_calls"] for row in group),
                "decisions": dict(sum((Counter(row["decisions"]) for row in group), Counter()))}
    if set(manifest["config"]["methods"]) == {"none", "obligate"}:
        by_id = {row["id"]: row for row in rows}
        pairs = [(by_id.get(episode_id(case, "none")), by_id.get(episode_id(case, "obligate"))) for case in manifest["cases"]]
        valid_pairs = [(a, b) for a, b in pairs if a and b and a["status"] == b["status"] == "OK"]
        attacked_pairs = [(a, b) for a, b in valid_pairs if a["injection_task"] is not None]
        summary["paired"] = {"valid_pairs": len(valid_pairs), "attacked_pairs": len(attacked_pairs),
            "baseline_success_defense_failure": sum(a["attack_success"] and not b["attack_success"] for a, b in attacked_pairs),
            "baseline_failure_defense_success": sum(not a["attack_success"] and b["attack_success"] for a, b in attacked_pairs)}
    write_json(Path(output) / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired native AgentDyn evaluation")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agentdyn.yaml")
    parser.add_argument("--upstream", type=Path, default=Path(os.getenv("AGENTDYN_ROOT", str(ROOT / "third_party/AgentDyn"))))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--injection-limit", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--model")
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        for key in ("limit", "injection_limit", "max_iters", "workers"):
            value = getattr(args, key)
            if value is not None:
                _positive(value, key)
                config["execution"][key] = value
        for key in ("model", "provider", "base_url", "api_key_env"):
            if getattr(args, key) is not None:
                config["victim"][key] = getattr(args, key)
        validate_config(config)
        output = args.output or ROOT / config.get("output", "results/agentdyn")
        upstream = bootstrap_upstream(args.upstream, config["upstream"]["revision"])
        manifest = prepare(output, upstream, config, args.config)
        summary = summarize(output, manifest)
        if args.run:
            if not os.environ.get(config["victim"]["api_key_env"]):
                raise RuntimeError(f"Set {config['victim']['api_key_env']} before --run")
            completed = {row["id"] for row in _episode_rows(output, manifest)}
            jobs = [{"case": case, "method": method, "output": str(output), "config": config,
                     "upstream": str(upstream), "run_id": manifest["run_id"]}
                    for case in manifest["cases"] for method in config["methods"] if episode_id(case, method) not in completed]
            if config["execution"]["workers"] == 1:
                for job in jobs:
                    row = run_case(job)
                    print(json.dumps({"id": row["id"], "status": row["status"]}), flush=True)
            elif jobs:
                with ProcessPoolExecutor(max_workers=config["execution"]["workers"]) as pool:
                    futures = [pool.submit(run_case, job) for job in jobs]
                    for future in as_completed(futures):
                        row = future.result()
                        print(json.dumps({"id": row["id"], "status": row["status"]}), flush=True)
            summary = summarize(output, manifest)
            if source_snapshot(upstream, args.config) != manifest["source_hashes"]:
                raise RuntimeError("Frozen AgentDyn experiment source or configuration changed during execution")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 1 if args.run and summary["errors"] else 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(_safe_error(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
