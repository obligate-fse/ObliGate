"""Consolidate Full snapshots and derive mechanism-active manifests."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from obligate.theory.lattice import realization_satisfies, satisfies_all
from obligate.theory.model import (
    AuditContract,
    AuditUnit,
    BehaviorPlan,
    ImplementationConstraint,
    Realization,
    SemanticObligation,
    sha256_json,
)
from obligate.theory.policy import build_default_policy
from experiments.adaptive_ablation.snapshots import deserialize_action, deserialize_evidence

from .common import atomic_write_json, sha256_file, utc_now
from .variant_runtime import (
    PARAMETER_SOURCE_PREDICATE,
    BoundEvidenceRecord,
    bound_records_for_action,
)


def iter_logical_snapshots(directory: Path, *, model: str) -> Iterable[dict[str, Any]]:
    """Yield one record per boundary decision, merging fresh-rebind duplicates."""

    for path in sorted(directory.glob("*.jsonl"), key=lambda item: item.name):
        buckets: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                sid = str(row.get("snapshot_id") or "")
                if not sid:
                    continue
                if sid not in buckets:
                    buckets[sid] = {}
                    order.append(sid)
                buckets[sid][str(row.get("event"))] = row
        logical: list[dict[str, Any]] = []
        for sid in order:
            bucket = buckets[sid]
            source = bucket.get("input")
            decision = bucket.get("decision")
            if source is None or decision is None:
                continue
            input_digest = sha256_json({"action": source["action"], "evidence": source["evidence"]})
            current = {
                "snapshot_id": sid,
                "model": model,
                "case_id": str(source.get("case_id") or path.stem),
                "input_digest": input_digest,
                "input": source,
                "decision": decision["deterministic"],
                "gate": (bucket.get("tool_gate") or {}).get("gate"),
                "gate_event": bucket.get("tool_gate"),
                "online_snapshot_ids": [sid],
            }
            if (
                logical
                and logical[-1]["input_digest"] == input_digest
                and logical[-1].get("gate") is None
                and current["gate"] is not None
            ):
                # A production dispatch has an explicit two-event shape: the
                # candidate decision has no ToolGate event and the immediately
                # following fresh re-adjudication does.  Do not collapse merely
                # because two adjacent model actions happen to be identical;
                # repeated blocked actions are distinct trajectory steps.
                logical[-1]["online_snapshot_ids"].append(sid)
                if current["gate"] is not None:
                    logical[-1]["gate"] = current["gate"]
                    logical[-1]["gate_event"] = current["gate_event"]
                logical[-1]["fresh_rebind_observed"] = True
                continue
            current["fresh_rebind_observed"] = False
            logical.append(current)
        yield from logical


def _witness_partition(closure: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    values: dict[str, str] = {}
    for claim, row in sorted((closure.get("witnesses") or {}).items()):
        values[str(claim)] = str(row.get("value") or "unknown")
        for witness in row.get("support") or row.get("positive") or row.get("positive_witnesses") or ():
            positive.append({"claim": claim, "witness": witness})
        for witness in row.get("refute") or row.get("negative") or row.get("negative_witnesses") or ():
            negative.append({"claim": claim, "witness": witness})
    return positive, negative, values


def _plan_digest(plan: Mapping[str, Any]) -> str:
    return sha256_json(dict(plan))


def _audit_unit(value: Mapping[str, Any]) -> AuditUnit:
    return AuditUnit(
        field=str(value["field"]),
        value_constraint=str(value["value_constraint"]),
        transform=str(value["transform"]),
        sink=str(value["sink"]),
    )


def _behavior_plan(value: Mapping[str, Any]) -> BehaviorPlan:
    audit = value.get("audit") or {}
    return BehaviorPlan(
        execution_env=str(value.get("execution_env") or "host"),  # type: ignore[arg-type]
        network_scope=str(value.get("network_scope") or "allow"),  # type: ignore[arg-type]
        data_scope=str(value.get("data_scope") or "raw"),  # type: ignore[arg-type]
        human_gate=str(value.get("human_gate") or "none"),  # type: ignore[arg-type]
        audit=AuditContract(
            must=frozenset(_audit_unit(item) for item in audit.get("must") or ()),
            allowed=frozenset(_audit_unit(item) for item in audit.get("allowed") or ()),
        ),
    )


def _semantic_obligation(value: Mapping[str, Any]) -> SemanticObligation:
    return SemanticObligation(
        obligation_id=str(value["obligation_id"]),
        trigger_id=str(value["trigger_id"]),
        description=str(value.get("description") or ""),
        forbid_dispatch=bool(value.get("forbid_dispatch")),
        forbid_confirmation=bool(value.get("forbid_confirmation")),
        min_execution_env=value.get("min_execution_env"),
        min_network_scope=value.get("min_network_scope"),
        min_data_scope=value.get("min_data_scope"),
        require_approval=bool(value.get("require_approval")),
        audit_must=frozenset(_audit_unit(item) for item in value.get("audit_must") or ()),
        bad_effect=str(value.get("bad_effect") or "unsafe_default_dispatch"),
    )


def _implementation_constraint(value: Mapping[str, Any]) -> ImplementationConstraint:
    return ImplementationConstraint(
        constraint_id=str(value["constraint_id"]),
        trigger_id=str(value["trigger_id"]),
        capability=str(value["capability"]),
    )


def _materialized_policy_realizations(capabilities: Iterable[str]) -> tuple[Realization, ...]:
    available_capabilities = frozenset(str(item) for item in capabilities)
    values: list[Realization] = []
    for template in build_default_policy().realizations:
        required = set(template.capabilities)
        values.append(
            replace(
                template,
                realization_id=f"snapshot-analysis:{template.realization_id}",
                capabilities=frozenset(required & available_capabilities),
                actual_audit=(
                    template.actual_audit
                    if "audit" in available_capabilities
                    else frozenset()
                ),
                available=template.available and required <= available_capabilities,
            )
        )
    return tuple(values)


def _requirement_claims(decision: Mapping[str, Any]) -> frozenset[str]:
    eoc = (decision.get("trace") or {}).get("eoc") or {}
    claims: set[str] = set()
    for item in eoc.get("requirements") or ():
        if isinstance(item, str):
            claims.add(item.removeprefix("+").removeprefix("-"))
            continue
        if not isinstance(item, Mapping):
            continue
        predicate = str(item.get("predicate") or "")
        if not predicate:
            continue
        arguments = tuple(str(value) for value in item.get("arguments") or ())
        claims.add(predicate if not arguments else f"{predicate}({','.join(arguments)})")
    return frozenset(claims)


def _multi_obligation_evaluation(
    decision: Mapping[str, Any],
    *,
    runtime_capabilities: Iterable[str] | None = None,
) -> dict[str, Any]:
    triggers = decision.get("triggers") or ()
    eoc = (decision.get("trace") or {}).get("eoc") or {}
    rem0 = build_default_policy().default_plan.to_dict()
    local_plans: dict[str, Mapping[str, Any]] = {_plan_digest(rem0): rem0}
    for remedies in (eoc.get("remedies") or {}).values():
        for item in remedies:
            if isinstance(item, Mapping) and isinstance(item.get("plan"), Mapping):
                local_plans[_plan_digest(item["plan"])] = item["plan"]

    obligations = tuple(
        _semantic_obligation(item)
        for item in decision.get("semantic_obligations") or ()
        if isinstance(item, Mapping)
    )
    constraints = tuple(
        _implementation_constraint(item)
        for item in decision.get("implementation_constraints") or ()
        if isinstance(item, Mapping)
    )
    if runtime_capabilities is None:
        runtime_capabilities = {
            str(capability)
            for item in decision.get("feasible_realizations") or ()
            if isinstance(item, Mapping)
            for capability in item.get("capabilities") or ()
        }
    realizations = _materialized_policy_realizations(runtime_capabilities)
    local_checks: list[dict[str, Any]] = []
    local_satisfying: list[str] = []
    for digest, payload in sorted(local_plans.items()):
        plan = _behavior_plan(payload)
        semantic_ok = satisfies_all(plan, obligations)
        feasible_ids = sorted(
            item.realization_id
            for item in realizations
            if realization_satisfies(item, plan, constraints)
        )
        satisfies_everything = semantic_ok and bool(feasible_ids)
        if satisfies_everything:
            local_satisfying.append(digest)
        local_checks.append(
            {
                "plan_digest": digest,
                "is_rem0": digest == _plan_digest(rem0),
                "semantic_obligations_satisfied": semantic_ok,
                "implementation_and_backend_feasible": bool(feasible_ids),
                "feasible_realization_ids": feasible_ids,
                "satisfies_all_semantic_and_implementation_obligations": satisfies_everything,
            }
        )
    ideal_digests = {
        _plan_digest(item)
        for item in decision.get("ideal") or ()
        if isinstance(item, Mapping)
    }
    lifted_join_pairs: list[tuple[str, str]] = []
    for item in decision.get("ideal") or ():
        if not isinstance(item, Mapping):
            continue
        plan = _behavior_plan(item)
        if plan.block_like or not satisfies_all(plan, obligations):
            continue
        lifted_join_pairs.extend(
            (plan.digest, realization.realization_id)
            for realization in realizations
            if realization_satisfies(realization, plan, constraints)
        )
    active = len(triggers) >= 2 and not local_satisfying and bool(lifted_join_pairs)
    return {
        "active": active,
        "trigger_count": len(triggers),
        "rem0": rem0,
        "local_candidate_count_including_rem0": len(local_plans),
        "local_candidate_digests": sorted(local_plans),
        "local_candidate_checks": local_checks,
        "single_candidate_satisfying_all_count": len(local_satisfying),
        "single_candidate_satisfying_all_digests": local_satisfying,
        "ideal_lifted_join_digests": sorted(ideal_digests),
        "non_blocking_feasible_lifted_join_pair_count": len(lifted_join_pairs),
        "non_blocking_feasible_lifted_join_realization_ids": sorted(
            realization_id for _, realization_id in lifted_join_pairs
        ),
    }


def _multi_obligation_active(decision: Mapping[str, Any]) -> bool:
    return bool(_multi_obligation_evaluation(decision)["active"])


def _constrained_realization_active(decision: Mapping[str, Any]) -> bool:
    if decision.get("public_decision") != "execute_with_constraints":
        return False
    selected = decision.get("selected") or {}
    return any(str(item) != "host" for item in selected.get("capabilities") or ())


def _history_projection(facts: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in facts.items() if key.startswith("history.")}


def _observation_projection(facts: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = ("agentdojo.arg_", "agentdojo.field_", "agentdojo.tool_call_", "agentdojo.source_")
    return {key: value for key, value in facts.items() if key.startswith(prefixes)}


def _expanded(
    record: Mapping[str, Any],
    *,
    relation_reasons: list[str],
    current_bound_records: Iterable[BoundEvidenceRecord] = (),
    relation_history_records: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    source = record["input"]
    evidence = source["evidence"]
    decision = record["decision"]
    trace = decision.get("trace") or {}
    eoc = trace.get("eoc") or {}
    closure = trace.get("witness_closure") or {}
    positive, negative, values = _witness_partition(closure)
    requirement_claims = _requirement_claims(decision)
    conflicting_requirements = sorted(
        claim for claim in requirement_claims if values.get(claim) == "conflict"
    )
    multi_evaluation = _multi_obligation_evaluation(
        decision,
        runtime_capabilities=(source.get("runtime") or {}).get("capabilities") or (),
    )
    facts = evidence.get("facts") or {}
    selected = decision.get("selected")
    gate = record.get("gate") or {}
    return {
        "schema_version": "obligate-full-guarded-action-snapshot-v1",
        "snapshot_id": record["snapshot_id"],
        "online_snapshot_ids": record["online_snapshot_ids"],
        "fresh_rebind_observed": record.get("fresh_rebind_observed", False),
        "case_id": record["case_id"],
        "model": record["model"],
        "step": None,
        "U": evidence.get("contract"),
        "H_lt": _history_projection(facts),
        "O_lt": _observation_projection(facts),
        "X_t": {
            "graph_nodes": evidence.get("graph_nodes") or [],
            "graph_edges": evidence.get("graph_edges") or [],
            "visible_runtime_facts": facts,
        },
        "sigma_t": _history_projection(facts),
        "a_t": source["action"],
        "kappa": (source.get("runtime") or {}).get("policy_acceptance_digest"),
        "G_t": trace.get("action_graph"),
        "S_t": trace.get("action_graph"),
        "Req_a_t": eoc.get("requirements") or [],
        "W_plus": positive,
        "W_minus": negative,
        "Val_t": values,
        "Gap_t": trace.get("gap") or [],
        "Haz_t": trace.get("hazard") or [],
        "Overflow_t": trace.get("overflow") or [],
        "Trig_t": decision.get("triggers") or [],
        "Q_sem_t": decision.get("semantic_obligations") or [],
        "Q_impl_t": decision.get("implementation_constraints") or [],
        "Ideal_t": decision.get("ideal") or [],
        "SafePlan_t": decision.get("safe_candidates") or [],
        "Pair_t": decision.get("feasible_realizations") or [],
        "NonBlock_t": [
            item
            for item in decision.get("feasible_realizations") or []
            if (item.get("plan") or {}).get("execution_env") != "no_execute"
        ],
        "Front_t": decision.get("frontier") or [],
        "b_star_t": (selected or {}).get("plan"),
        "r_star_t": selected,
        "eta_t": decision.get("certificate"),
        "eta_digest": decision.get("certificate_digest"),
        "d_t": decision.get("public_decision"),
        "e_t": "ok" if selected is not None or decision.get("public_decision") == "block" else "compliance_error",
        "gate": gate,
        "guarded": bool((trace.get("guarded"))),
        "relation_active": bool(relation_reasons),
        "relation_active_reasons": relation_reasons,
        "bound_evidence_records": [item.to_dict() for item in current_bound_records],
        "relation_history_records": [dict(item) for item in relation_history_records],
        "conflict_active": bool(conflicting_requirements),
        "conflicting_requirement_claims": conflicting_requirements,
        "pure_gap_active": bool(trace.get("gap"))
        and not bool(trace.get("hazard"))
        and not bool(trace.get("overflow"))
        and not bool(evidence.get("explicit_denies")),
        "multi_obligation_active": bool(multi_evaluation["active"]),
        "multi_obligation_evaluation": multi_evaluation,
        "constrained_realization_active": _constrained_realization_active(decision),
        "input": source,
        "decision": {
            "event": "decision",
            "snapshot_id": record["snapshot_id"],
            "deterministic": decision,
        },
        "deterministic_decision": decision,
    }


def _relation_evaluation(
    current_records: Iterable[BoundEvidenceRecord],
    prior_entries: Iterable[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Evaluate all five preregistered relation-active conditions.

    ``prior_entries`` includes observations from every preceding action in the
    case, including unguarded public reads.  Only records relevant to the
    current projection are copied into the replay row.
    """

    current = tuple(current_records)
    prior = tuple(prior_entries)
    current_parameters = tuple(
        item for item in current if item.predicate == PARAMETER_SOURCE_PREDICATE
    )
    prior_parameters = tuple(
        entry
        for entry in prior
        if isinstance(entry.get("record"), BoundEvidenceRecord)
        and entry["record"].predicate == PARAMETER_SOURCE_PREDICATE
    )
    reasons: list[str] = []
    for item in current_parameters:
        same_key = [entry for entry in prior_parameters if entry["record"].parameter_key == item.parameter_key]
        if any(entry["record"].value != item.value for entry in same_key):
            reasons.append(f"same_parameter_key_multiple_values:{item.parameter_key}")
        if any(
            entry["record"].value == item.value
            and entry["record"].source_type != item.source_type
            for entry in same_key
        ):
            reasons.append(f"same_value_multiple_sources:{item.parameter_key}")
        same_family_key = [
            entry
            for entry in same_key
            if entry["record"].tool_family == item.tool_family
        ]
        if any(entry["record"].source_step != item.source_step for entry in same_family_key):
            reasons.append(f"same_tool_family_key_different_step:{item.tool_family}:{item.parameter_key}")
        if any(
            entry["record"].supports
            and entry["record"].trusted_for_authorization
            and entry["record"].object_role == "destination_or_object"
            and item.object_role == "destination_or_object"
            and entry["record"].object_id != item.object_id
            for entry in same_family_key
        ):
            reasons.append(f"historical_evidence_current_object_mismatch:{item.tool_family}:{item.parameter_key}")

    prior_objects = {
        entry["record"].object_id
        for entry in prior_parameters
        if entry["record"].object_role == "destination_or_object"
    }
    current_objects = {
        item.object_id
        for item in current_parameters
        if item.object_role == "destination_or_object"
    }
    if len(prior_objects | current_objects) >= 2:
        reasons.append("multiple_destination_or_object")

    relevant: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in prior:
        historical = entry.get("record")
        if not isinstance(historical, BoundEvidenceRecord):
            continue
        if not any(
            historical.projection == item.projection and historical.mismatches(item)
            for item in current
        ):
            continue
        if historical.digest in seen:
            continue
        seen.add(historical.digest)
        relevant.append(
            {
                **historical.to_dict(),
                "source_action_guarded": bool(entry.get("source_action_guarded")),
            }
        )
    return sorted(set(reasons)), sorted(relevant, key=lambda item: str(item.get("evidence_ref") or ""))


def consolidate(model_snapshot_dirs: Mapping[str, Path], output_root: Path) -> dict[str, Any]:
    output_dir = output_root / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": output_dir / "full_guarded_actions.jsonl",
        "relation": output_dir / "relation_active.jsonl",
        "conflict": output_dir / "conflict_active.jsonl",
        "pure_gap": output_dir / "pure_gap_active.jsonl",
        "multi": output_dir / "multi_obligation_active.jsonl",
        "constrained": output_dir / "constrained_realization_active.jsonl",
    }
    handles = {name: path.open("w", encoding="utf-8", newline="\n") for name, path in paths.items()}
    counts = defaultdict(int)
    cases = defaultdict(set)
    try:
        for model, directory in sorted(model_snapshot_dirs.items()):
            history: dict[str, list[dict[str, Any]]] = defaultdict(list)
            step_by_case: dict[str, int] = defaultdict(int)
            for record in iter_logical_snapshots(directory, model=model):
                counts["history_all_boundary_observations"] += 1
                case_id = record["case_id"]
                step_by_case[case_id] += 1
                step = step_by_case[case_id]
                action = record["input"]["action"]
                facts = record["input"]["evidence"].get("facts") or {}
                family = str(facts.get("agentdojo.tool_group") or action.get("tool_name"))
                action_value = deserialize_action(action)
                evidence_value = deserialize_evidence(record["input"]["evidence"])
                current_records = bound_records_for_action(
                    action_value,
                    evidence_value,
                    tool_family=family,
                    source_step=f"{case_id}:step:{step}",
                )
                reasons, replay_history = _relation_evaluation(current_records, history[case_id])
                guarded = action_value.guarded
                if not guarded:
                    counts["history_unguarded_public_read_observations"] += 1
                    counts["history_unguarded_public_read_records"] += len(current_records)
                    history[case_id].extend(
                        {"record": item, "source_action_guarded": False}
                        for item in current_records
                    )
                    continue
                expanded = _expanded(
                    record,
                    relation_reasons=reasons,
                    current_bound_records=current_records,
                    relation_history_records=replay_history,
                )
                expanded["step"] = step
                expanded["relation_history_record_count"] = len(history[case_id])
                expanded["relation_history_unguarded_public_read_record_count"] = sum(
                    not bool(item.get("source_action_guarded"))
                    for item in history[case_id]
                )
                expanded["relation_history_includes_unguarded_public_read"] = any(
                    not bool(item.get("source_action_guarded")) for item in replay_history
                )
                payload = json.dumps(expanded, ensure_ascii=False, sort_keys=True)
                handles["all"].write(payload + "\n")
                counts["all"] += 1
                cases["all"].add(case_id)
                for name, field in (
                    ("relation", "relation_active"),
                    ("conflict", "conflict_active"),
                    ("pure_gap", "pure_gap_active"),
                    ("multi", "multi_obligation_active"),
                    ("constrained", "constrained_realization_active"),
                ):
                    if expanded[field]:
                        handles[name].write(payload + "\n")
                        counts[name] += 1
                        cases[name].add(case_id)
                history[case_id].extend(
                    {"record": item, "source_action_guarded": True}
                    for item in current_records
                )
    finally:
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    manifest = {
        "schema_version": "obligate-snapshot-manifest-v1",
        "created_at": utc_now(),
        "models": sorted(model_snapshot_dirs),
        "files": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "snapshot_count": counts[name],
                "unique_case_count": len(cases[name]),
            }
            for name, path in paths.items()
        },
        "empty_active_sets_are_not_fabricated": True,
        "relation_history_accounting": {
            "all_boundary_observation_count": counts["history_all_boundary_observations"],
            "unguarded_public_read_observation_count": counts[
                "history_unguarded_public_read_observations"
            ],
            "unguarded_public_read_bound_record_count": counts[
                "history_unguarded_public_read_records"
            ],
            "unguarded_public_reads_are_committed_before_guarded_active_set_classification": True,
        },
        "raw_U_H_O_limitation": (
            "Production RuntimeEvidence stores the frozen TaskContract and observable facts/provenance projection, "
            "not a second raw copy of the full AgentDojo prompt/history. U/H/O fields above are the exact pre-policy "
            "observable projections available to ObliGate."
        ),
    }
    atomic_write_json(output_root / "manifests" / "snapshot_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepseek-snapshots", type=Path, required=True)
    parser.add_argument("--qwen-snapshots", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    value = consolidate(
        {"deepseek-v4-flash": args.deepseek_snapshots, "qwen-plus": args.qwen_snapshots},
        args.output_root,
    )
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["consolidate", "iter_logical_snapshots"]
