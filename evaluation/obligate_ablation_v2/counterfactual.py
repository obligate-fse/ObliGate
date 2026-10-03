"""Replay mechanism-active Full snapshots through isolated research variants."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from obligate.theory.model import ActionCandidate
from experiments.adaptive_ablation.snapshots import deserialize_action, deserialize_evidence

from .common import atomic_write_json, sha256_file, utc_now
from .variant_runtime import (
    BOUND_RECORD_FACT_KEY,
    BoundEvidenceRecord,
    ResearchRuntime,
    ResearchVariant,
    bound_records_for_action,
)

VARIANT_ACTIVE_FIELD = {
    ResearchVariant.UNBOUND_EVIDENCE: "relation_active",
    ResearchVariant.CONFLICT_TO_SUPPORT: "conflict_active",
    ResearchVariant.CONFLICT_TO_REFUTE: "conflict_active",
    ResearchVariant.GAP_BLIND: "pure_gap_active",
    ResearchVariant.SINGLE_REMEDY: "multi_obligation_active",
}


def _bound_records(action: ActionCandidate, evidence: Any, row: Mapping[str, Any]) -> tuple[BoundEvidenceRecord, ...]:
    facts = evidence.facts or {}
    family = str(facts.get("agentdojo.tool_group") or action.tool_name)
    step = str(row.get("step") or facts.get("graph.sequence_depth") or 0)
    return bound_records_for_action(
        action,
        evidence,
        tool_family=family,
        source_step=f"{row.get('case_id')}:step:{step}",
    )


def _plan_strength(plan: Mapping[str, Any] | None) -> tuple[int, int, int, int, int] | None:
    if not plan:
        return None
    env = {"host": 0, "sandbox": 1, "no_execute": 2}
    net = {"allow": 0, "allowlist": 1, "deny": 2}
    data = {"raw": 0, "redact": 1, "no_sensitive": 2}
    human = {"none": 0, "approval_required": 1}
    audit = plan.get("audit") or {}
    return (
        env[str(plan.get("execution_env") or "host")],
        net[str(plan.get("network_scope") or "allow")],
        data[str(plan.get("data_scope") or "raw")],
        human[str(plan.get("human_gate") or "none")],
        len(audit.get("must") or ()),
    )


def _weaker(variant: Mapping[str, Any] | None, full: Mapping[str, Any] | None) -> bool:
    left = _plan_strength(variant)
    right = _plan_strength(full)
    if left is None or right is None:
        return False
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def _decision_flags(full: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, bool]:
    full_name = str(full.get("public_decision"))
    variant_name = str(variant.get("public_decision"))
    full_plan = (full.get("selected") or {}).get("plan")
    variant_plan = (variant.get("selected") or {}).get("plan")
    full_dispatch = full_name in {"allow", "execute_with_constraints"} and bool(full.get("execute"))
    variant_dispatch = variant_name in {"allow", "execute_with_constraints"} and bool(variant.get("execute"))
    full_non_blocking = bool(full_plan) and full_plan.get("execution_env") != "no_execute"
    variant_block = variant_name in {"block", "block_with_compliance_error"} or (
        bool(variant_plan) and variant_plan.get("execution_env") == "no_execute"
    )
    return {
        "unsafe_flip": (variant_name == "allow" and full_name != "allow")
        or (variant_dispatch and not full_dispatch and full_name in {"block", "block_with_compliance_error", "require_confirmation"}),
        "over_conservative_flip": full_non_blocking
        and (
            variant_block
            or (variant_name == "require_confirmation" and full_name != "require_confirmation")
        ),
        "constraint_downgrade": full_name == "execute_with_constraints"
        and (variant_name == "allow" or _weaker(variant_plan, full_plan)),
        "non_blocking_plan_lost": full_non_blocking and variant_block,
    }


def _changed_ids(full_items: Iterable[Mapping[str, Any]], variant_items: Iterable[Mapping[str, Any]], key: str) -> bool:
    return sorted(str(item.get(key)) for item in full_items) != sorted(str(item.get(key)) for item in variant_items)


def replay(input_path: Path, output_path: Path, summary_path: Path | None = None) -> dict[str, Any]:
    source_rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    runtimes: dict[tuple[str, str, ResearchVariant], ResearchRuntime] = {}
    counts: dict[str, dict[str, Any]] = {
        variant.value: {"active_snapshots": 0, "active_cases": set(), "first_decision_divergences": 0, "unsafe_flips": 0, "over_conservative_flips": 0, "constraint_downgrades": 0, "non_blocking_plans_lost": 0}
        for variant in VARIANT_ACTIVE_FIELD
    }
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in source_rows:
            action = deserialize_action(row["input"]["action"])
            evidence = deserialize_evidence(row["input"]["evidence"])
            records = _bound_records(action, evidence, row)
            evidence = replace(
                evidence,
                facts={**dict(evidence.facts or {}), BOUND_RECORD_FACT_KEY: [item.to_dict() for item in records]},
            )
            capabilities = tuple((row["input"].get("runtime") or {}).get("capabilities") or ())
            for variant, active_field in VARIANT_ACTIVE_FIELD.items():
                key = (str(row["model"]), str(row["case_id"]), variant)
                runtime = runtimes.get(key)
                if runtime is None:
                    runtime = ResearchRuntime(
                        variant,
                        research_counterfactual_mode=True,
                        capabilities=capabilities,
                        case_id=str(row["case_id"]),
                    )
                    runtimes[key] = runtime
                if variant is ResearchVariant.UNBOUND_EVIDENCE:
                    runtime.bound_history.commit(
                        BoundEvidenceRecord.from_dict(item)
                        for item in row.get("relation_history_records") or ()
                        if isinstance(item, Mapping)
                    )
                decision = runtime.decide(action, evidence)
                if not bool(row.get(active_field)):
                    continue
                full = row["deterministic_decision"]
                variant_value = decision.to_dict()
                flags = _decision_flags(full, variant_value)
                changed_decision = (
                    full.get("public_decision") != variant_value.get("public_decision")
                    or (full.get("selected") or {}).get("plan") != (variant_value.get("selected") or {}).get("plan")
                )
                full_trace = full.get("trace") or {}
                variant_trace = variant_value.get("trace") or {}
                result = {
                    "schema_version": "obligate-counterfactual-result-v1",
                    "snapshot_id": row["snapshot_id"],
                    "case_id": row["case_id"],
                    "model": row["model"],
                    "step": row.get("step"),
                    "variant": variant.value,
                    "active_set": active_field,
                    "research_only": True,
                    "verified": False,
                    "research_counterfactual_mode": True,
                    "full_decision": full.get("public_decision"),
                    "variant_decision": variant_value.get("public_decision"),
                    "full_plan": (full.get("selected") or {}).get("plan"),
                    "variant_plan": (variant_value.get("selected") or {}).get("plan"),
                    **flags,
                    "first_decision_divergence": changed_decision,
                    "changed_triggers": _changed_ids(full.get("triggers") or (), variant_value.get("triggers") or (), "trigger_id"),
                    "changed_obligations": _changed_ids(full.get("semantic_obligations") or (), variant_value.get("semantic_obligations") or (), "obligation_id"),
                    "changed_realization": (full.get("selected") or {}).get("realization_id")
                    != (variant_value.get("selected") or {}).get("realization_id"),
                    "unbound_evidence_misuse": variant_trace.get("unbound_evidence_misuse") or [],
                    "explanation_root": {
                        "full_trigger_ids": [item.get("trigger_id") for item in full.get("triggers") or ()],
                        "variant_trigger_ids": [item.get("trigger_id") for item in variant_value.get("triggers") or ()],
                        "full_gap": full_trace.get("gap") or [],
                        "variant_gap": variant_trace.get("gap") or [],
                    },
                }
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                bucket = counts[variant.value]
                bucket["active_snapshots"] += 1
                bucket["active_cases"].add(str(row["case_id"]))
                bucket["first_decision_divergences"] += int(changed_decision)
                bucket["unsafe_flips"] += int(flags["unsafe_flip"])
                bucket["over_conservative_flips"] += int(flags["over_conservative_flip"])
                bucket["constraint_downgrades"] += int(flags["constraint_downgrade"])
                bucket["non_blocking_plans_lost"] += int(flags["non_blocking_plan_lost"])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    serializable = {
        variant: {**value, "active_cases": len(value["active_cases"])}
        for variant, value in counts.items()
    }
    summary = {
        "schema_version": "obligate-counterfactual-summary-v1",
        "created_at": utc_now(),
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "variants": serializable,
        "empty_active_sets_are_reported_not_fabricated": True,
    }
    atomic_write_json(summary_path or output_path.with_suffix(".summary.json"), summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(replay(args.snapshots, args.output, args.summary), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["replay"]
