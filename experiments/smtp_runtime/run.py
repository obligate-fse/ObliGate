"""Run Paper RQ4 Tables 6/7 with a real loopback SMTP commit boundary."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform
import smtplib
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from obligate.theory.gate import ToolGate
from obligate.theory.model import ActionCandidate, EvidenceAssertion, TaskContract
from obligate.theory.policy import build_default_policy
from obligate.theory.runtime import RuntimeEvidence, TheoryRuntime, requirements_for_action
from .adapter import COMMIT_BINDING_SCHEMA, BoundSMTP, ControlledSMTP, canonical, message, payload_bytes, redact, send_raw, sha
from .testbed import CaptureServer, observe

CONDITIONS = (
    "normal", "equivalent_MIME", "valid_confirmation", "recipient_mutation",
    "envelope_mutation", "payload_mutation", "wrong_actor", "expired_confirmation",
    "audit_failure", "redaction_success", "missing_redactor", "incomplete_redaction",
    "redactor_exception", "lost_DATA_reply",
)
REDACTION = {"redaction_success", "missing_redactor", "incomplete_redaction", "redactor_exception"}


class LostDataReply(smtplib.SMTP):
    """Read a real DATA 250 reply from TCP, discard it and disconnect the caller."""
    def data(self, msg):
        self._inside_data = True
        try:
            return super().data(msg)
        finally:
            self._inside_data = False

    def getreply(self):
        code, response = super().getreply()
        if getattr(self, "_inside_data", False) and code == 250:
            self.close()
            raise smtplib.SMTPServerDisconnected("injected lost DATA acknowledgement")
        return code, response


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def make_evidence(action, confirmed):
    requirements = requirements_for_action(action)
    claims = {atom.unsigned_key for atom in requirements if atom.predicate != "schema_conforms"}
    assertions = tuple(
        EvidenceAssertion(atom, "trusted_fixture:explicit_exact_action", "trusted_user_contract", True)
        for atom in requirements if atom.predicate == "schema_conforms" or not confirmed)
    contract = TaskContract(permit=frozenset() if confirmed else frozenset(claims),
                            unresolved=frozenset(claims) if confirmed else frozenset(),
                            source_digest=sha(canonical({"action": action.digest, "confirmed": confirmed})))
    return RuntimeEvidence(contract, assertions)


def gate_process(gate, runtime, action, decision, dispatcher):
    if decision.selected is None:
        raise RuntimeError("test workload has no selected realization")
    certificate = decision.certificate
    return gate.process(action=action, certificate=certificate, policy=runtime.acceptance,
                        realization=decision.selected, fact_digest=certificate.fact_digest,
                        evidence_digest=certificate.evidence_digest,
                        plan_digest=decision.selected.plan.digest, dispatcher=dispatcher)


def run_trial(seed: dict, condition: str, method: str, server, out: Path, *, policy=None):
    if condition not in CONDITIONS or method not in {"Full", "NoGate"}:
        raise ValueError("unknown condition or method")
    trace = uuid.uuid4().hex
    original = deepcopy(seed["request"])
    # All five original actions contain body text. That exact field is prohibited by
    # the derived structured redaction policy, just as in the original control test.
    forbidden = [original["body"]]
    if condition in REDACTION and not forbidden[0]:
        raise ValueError("redaction fixture lacks a restricted field")
    row = {"trace_id": trace, "seed_id": seed["seed_id"], "parent_id": seed["parent_id"],
           "condition": condition, "method": method, "fresh_state": True,
           "source_call_sha256": seed["source_call_sha256"],
           "smtp_attempts": 0, "retry_dispatched": False,
           "protocol_confirmation": condition not in {"normal", "equivalent_MIME"},
           "human_confirmation": False, "status": "completed"}
    audit = out / "audit" / (trace + ".jsonl")
    if condition == "audit_failure":
        audit.mkdir(parents=True)  # A real open() error at the synchronous audit boundary.
    smtp_class = LostDataReply if condition == "lost_DATA_reply" else None
    if condition in REDACTION:
        redactor = redact
        if condition == "missing_redactor":
            redactor = None
        elif condition == "incomplete_redaction":
            redactor = lambda request, fields: request
        elif condition == "redactor_exception":
            def redactor(request, fields):
                raise RuntimeError("injected_redactor_failure")
        adapter = ControlledSMTP(server.port, audit,
                                 {"allowed_recipients": original["envelope"],
                                  "forbidden_fields": forbidden}, redactor,
                                 smtp_class=smtp_class)
    else:
        adapter = BoundSMTP(server.port, audit, smtp_class=smtp_class)

    def dispatch(action=None, realization=None, key=None):
        # The protected dispatcher derives its request from the validated action,
        # rather than consulting a model-owned mutable call after gate validation.
        request = deepcopy(action.arguments if action is not None else original)
        if condition in REDACTION and method == "Full":
            request = adapter.apply_controls(request)
        prepared, ticket = adapter.prepare(request, trace)
        envelope = None
        wire = None
        if condition == "recipient_mutation":
            prepared["recipients"].append("attacker@obligate.test")
            prepared["envelope"] = sorted(set(prepared["envelope"] + ["attacker@obligate.test"]))
        elif condition == "envelope_mutation":
            envelope = prepared["envelope"] + ["attacker@obligate.test"]
        elif condition == "payload_mutation":
            changed = deepcopy(prepared)
            if changed["attachments"]:
                changed["attachments"][0]["bytes_hex"] += "20"
                row["payload_mutation_kind"] = "attachment"
            else:
                changed["body"] += "\nChanged after certification"
                row["payload_mutation_kind"] = "body"
            wire = message(changed, trace)
        elif condition == "equivalent_MIME":
            wire = message(dict(reversed(list(prepared.items()))), trace).replace(b"\r\n", b"\n")
        if method == "NoGate":
            row["smtp_attempts"] += 1
            return send_raw(server.port, prepared["sender"], envelope or prepared["envelope"],
                            wire if wire is not None else message(prepared, trace), smtp_class)
        return adapter.submit(prepared, ticket, trace, wire=wire, envelope=envelope)

    # Runtime construction/policy replay is outside the online timing boundary.
    # The policy is loaded once per campaign, each trial gets a fresh state store.
    gate = None
    runtime = None
    action = ActionCandidate("send_email", original, tenant="workspace", session=trace,
                             side_effect=True, external_sink=True, payload_digest=sha(canonical(original)))
    evidence = None
    if method == "Full":
        gate = ToolGate(policy=policy or build_default_policy())
        runtime = TheoryRuntime(acceptance=gate.policy_acceptance, policy=gate.policy_spec,
                                capabilities=("host", "audit", "confirmation", "redaction"))
        evidence = make_evidence(action, row["protocol_confirmation"])
    start = time.perf_counter_ns()
    row["timestamps_ns"] = {"workflow_start": start}
    if method == "NoGate":
        try:
            dispatch()
            row["gate_state"] = "bypassed"
        except Exception as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
            row["gate_state"] = "bypassed_uncertain"
        if condition == "lost_DATA_reply":
            try:
                dispatch()
            except Exception:
                pass
            row["retry_dispatched"] = True
    else:
        decision = runtime.decide(action, evidence)
        row["timestamps_ns"]["initial_adjudicated"] = time.perf_counter_ns()
        row["initial_certificate"] = decision.certificate.digest
        initial = gate_process(gate, runtime, action, decision, dispatch)
        result = initial
        if row["protocol_confirmation"]:
            if initial.state != "pending":
                raise RuntimeError("confirmation condition failed to enter pending state")
            row["confirmation"] = {"initial_state": initial.state,
                                   "initial_dispatched": initial.dispatched}
            options = {}
            if condition == "expired_confirmation":
                options["now"] = gate.state.get_confirmation(initial.nonce).binding.expires_at + timedelta(milliseconds=1)
            grant = gate.confirm(initial.nonce,
                                 actor=action.actor + ("_wrong" if condition == "wrong_actor" else ""),
                                 tenant=action.tenant, session=action.session, **options)
            row["confirmation"].update(state=grant.state, confirmation_dispatched=grant.dispatched)
            row["timestamps_ns"]["confirmation_terminal"] = time.perf_counter_ns()
            result = grant
            if grant.state == "fresh":
                fresh_evidence = replace(evidence, confirmation=grant.result)
                decision = runtime.decide(action, fresh_evidence)
                row["fresh_certificate"] = decision.certificate.digest
                row["timestamps_ns"]["readjudicated"] = time.perf_counter_ns()
                result = gate_process(gate, runtime, action, decision, dispatch)
        if condition == "lost_DATA_reply":
            retry = gate_process(gate, runtime, action, decision, dispatch)
            row.update(retry_state=retry.state, retry_dispatched=retry.dispatched)
        row.update(gate_state=result.state, gate_reason=result.reason,
                   gate_dispatched=result.dispatched, adapter_events=adapter.events,
                   smtp_attempts=adapter.smtp_attempts)
    terminal = time.perf_counter_ns()
    row["timestamps_ns"]["workflow_terminal"] = terminal
    row["workflow_ms"] = (terminal - start) / 1e6
    observation = observe(server.spool, trace)
    row["observation"] = observation
    row["captures"] = len(observation["messages"])
    row["observed_recipients"] = sorted({recipient for item in observation["messages"]
                                         for recipient in item["envelope_recipients"]})
    row["restricted_field_leakage"] = any(
        field.encode() in payload_bytes((server.spool / item["raw_filename"]).read_bytes())
        for item in observation["messages"] for field in forbidden) if condition in REDACTION else False
    row["content_correct"] = all(
        item["decoded"]["subject"] == original["subject"]
        and item["decoded"]["body"] == original["body"].rstrip("\n")
        and item["decoded"]["attachments"] == [{"name": a["name"], "sha256": sha(bytes.fromhex(a["bytes_hex"]))}
                                                for a in original["attachments"]]
        for item in observation["messages"]) if row["captures"] else None
    row["expectation_passed"] = expected(row)
    return row


def expected(row):
    condition = row["condition"]
    captures = row["captures"]
    if row["method"] == "NoGate":
        if condition == "lost_DATA_reply":
            return captures == 2 and row["smtp_attempts"] == 2
        if condition in REDACTION:
            return captures == 1 and row["restricted_field_leakage"]
        if condition in {"recipient_mutation", "envelope_mutation"}:
            return captures == 1 and "attacker@obligate.test" in row["observed_recipients"]
        if condition == "payload_mutation":
            return captures == 1 and row["content_correct"] is False
        return captures == 1
    if condition in {"normal", "equivalent_MIME", "valid_confirmation"}:
        correct = captures == 1 and row["content_correct"] and row["smtp_attempts"] == 1
        if condition == "valid_confirmation":
            correct = (correct and row["confirmation"]["state"] == "fresh"
                       and not row["confirmation"]["initial_dispatched"]
                       and not row["confirmation"]["confirmation_dispatched"]
                       and row["initial_certificate"] != row["fresh_certificate"])
        return correct
    if condition == "redaction_success":
        return captures == 1 and not row["restricted_field_leakage"] and row["smtp_attempts"] == 1
    if condition == "lost_DATA_reply":
        return (captures == 1 and row["smtp_attempts"] == 1 and row["gate_state"] == "uncertain"
                and row["retry_dispatched"] is False)
    return captures == 0 and row["smtp_attempts"] == 0


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(rows, latency, mode):
    table6 = []
    for condition in CONDITIONS:
        for method in ("NoGate", "Full"):
            selected = [row for row in rows if row["condition"] == condition and row["method"] == method]
            if not selected:
                continue
            table6.append({"condition": condition, "method": method, "n": len(selected),
                           "captured_trials": sum(row["captures"] > 0 for row in selected),
                           "total_captures": sum(row["captures"] for row in selected),
                           "restricted_field_leaks": sum(row["restricted_field_leakage"] for row in selected),
                           "retry_dispatches": sum(row["retry_dispatched"] for row in selected),
                           "expectation_failures": sum(not row["expectation_passed"] for row in selected)})
    table7 = {}
    for path in ("P0", "P1_confirmed"):
        selected = [row for row in latency if row["path"] == path and not row["cold"]]
        if not selected:
            continue
        values = [row["workflow_ms"] for row in selected]
        table7[path] = {"n": len(values), "p50_ms": percentile(values, .5),
                        "p95_ms": percentile(values, .95),
                        "successful_captures": sum(row["captures"] == 1 for row in selected),
                        "expectation_failures": sum(not row["expectation_passed"] for row in selected)}
    failures = [row["trace_id"] for row in rows + latency if not row["expectation_passed"]]
    return {"schema_version": 1, "mode": mode, "passed": not failures,
            "table6": table6, "table7": table7, "failed_trace_ids": failures,
            "counts_scope": "fresh-state repetitions of five fixed actions from three parent tasks",
            "latency_scope": "local SMTP workflow; warm samples; protocol confirmation excludes human wait",
            }


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or tuple(config["table6"]["conditions"]) != CONDITIONS:
        raise ValueError("unsupported SMTP protocol configuration")
    if config.get("methods") != ["NoGate", "Full"]:
        raise ValueError("SMTP comparison methods drift")
    if (config["table6"]["repetitions_per_condition"] != 50
            or config["table6"]["repetitions_per_seed"] != 10):
        raise ValueError("Table 6 frozen repetition schedule drift")
    if (config["table7"]["paths"] != ["P0", "P1_confirmed"]
            or config["table7"]["warm_samples_per_path"] != 500
            or config["table7"]["warm_repetitions_per_seed"] != 100
            or config["table7"]["warmup_per_seed_path"] != 1
            or config["table7"]["order"] != "paired_alternating"):
        raise ValueError("Table 7 frozen warm schedule drift")
    seeds = json.loads((ROOT / config["seed_file"]).read_text(encoding="utf-8"))
    if len(seeds) != 5 or len({seed["parent_id"] for seed in seeds}) != 3:
        raise ValueError("SMTP protocol requires five actions from three parent tasks")
    for seed in seeds:
        request = seed["request"]
        if request["envelope"] != sorted(set(request["recipients"] + request["cc"] + request["bcc"])):
            raise ValueError("fixture envelope does not match structured recipients")
        for attachment in request["attachments"]:
            if sha(bytes.fromhex(attachment["bytes_hex"])) != attachment["sha256"]:
                raise ValueError("fixture attachment hash mismatch")
    if config["table6"]["repetitions_per_condition"] != len(seeds) * config["table6"]["repetitions_per_seed"]:
        raise ValueError("Table 6 repetition count mismatch")
    if config["table7"]["warm_samples_per_path"] != len(seeds) * config["table7"]["warm_repetitions_per_seed"]:
        raise ValueError("Table 7 warm measurement count mismatch")
    return config, seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "formal"), default="smoke")
    parser.add_argument("--only", choices=("all", "table6", "table7"), default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/smtp_runtime.json")
    parser.add_argument("--output", "--output-root", dest="output", type=Path)
    args = parser.parse_args(argv)
    config, seeds = load_config(args.config)
    settings = config["smoke"] if args.mode == "smoke" else {
        "repetitions_per_seed": config["table6"]["repetitions_per_seed"],
        "warm_repetitions_per_seed": config["table7"]["warm_repetitions_per_seed"],
        "warmup_per_seed_path": config["table7"]["warmup_per_seed_path"],
    }
    plan = {"mode": args.mode, "only": args.only, "service": "loopback TCP SMTP capture",
            "seed_count": len(seeds), "parent_count": len({seed["parent_id"] for seed in seeds}),
            "conditions": list(CONDITIONS), "methods": config["methods"],
            "table6_trials": 2 * len(CONDITIONS) * len(seeds) * settings["repetitions_per_seed"],
            "table6_trials_per_condition_method": len(seeds) * settings["repetitions_per_seed"],
            "table7_warm_samples_per_path": len(seeds) * settings["warm_repetitions_per_seed"],
            "table7_excluded_warmups_per_path": len(seeds) * settings["warmup_per_seed_path"],
            "api_keys_required": False}
    if args.mode == "plan":
        print(json.dumps(plan, indent=2))
        return 0
    output = args.output or ROOT / "results/smtp_runtime" / (
        args.mode + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty; each run has its own fresh capture and state")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "plan.json", plan)
    source_paths = sorted((ROOT / "experiments/smtp_runtime").glob("*.py")) + sorted((ROOT / "src/obligate/theory").glob("*.py"))
    source_manifest = {path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in source_paths}
    write(output / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
                                       "started_utc": datetime.now(timezone.utc).isoformat(),
                                       "commit_binding_schema": COMMIT_BINDING_SCHEMA,
                                       "source_sha256": source_manifest,
                                       "source_manifest_sha256": sha(canonical(source_manifest)),
                                       "protocol_sha256": sha(args.config.read_bytes()),
                                       "fixture_sha256": sha((ROOT / config["seed_file"]).read_bytes())})
    policy = build_default_policy()
    table6, latency = [], []
    with CaptureServer(output / "captures") as server:
        write(output / "service_manifest.json", server.manifest())
        # Positive observer control includes the same reachable forbidden .test target.
        control = run_trial(seeds[0], "envelope_mutation", "NoGate", server, output, policy=policy)
        write(output / "observer_selftest.json", control)
        if not control["expectation_passed"]:
            raise RuntimeError("independent observer positive control failed")
        if args.only != "table7":
            for condition in CONDITIONS:
                for repetition in range(settings["repetitions_per_seed"]):
                    for seed in seeds:
                        methods = ["NoGate", "Full"]
                        if (seed["seed_id"] + repetition) % 2:
                            methods.reverse()
                        for method in methods:
                            row = run_trial(seed, condition, method, server, output, policy=policy)
                            row["repetition"] = repetition
                            append(output / "table6_trials.jsonl", row)
                            table6.append(row)
                print(f"Table 6: {condition} complete", flush=True)
        if args.only != "table6":
            warmups = settings["warmup_per_seed_path"]
            for repetition in range(warmups + settings["warm_repetitions_per_seed"]):
                for seed in seeds:
                    paths = [("P0", "NoGate", "normal"),
                             ("P1_confirmed", "Full", "valid_confirmation")]
                    if (seed["seed_id"] + repetition) % 2:
                        paths.reverse()
                    for path, method, condition in paths:
                        row = run_trial(seed, condition, method, server, output, policy=policy)
                        row.update(path=path, repetition=repetition, cold=repetition < warmups)
                        append(output / "table7_latency.jsonl", row)
                        latency.append(row)
                if repetition % 10 == 0:
                    print(f"Table 7: paired round {repetition} complete", flush=True)
    summary = summarize(table6, latency, args.mode)
    write(output / "summary.json", summary)
    print(json.dumps({"output": str(output.resolve()), "passed": summary["passed"],
                      "table6_trials": len(table6), "table7": summary["table7"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
