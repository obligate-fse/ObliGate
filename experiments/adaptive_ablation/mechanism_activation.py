"""Research-only evidence activation profile for mechanism ablations.

The profile is deliberately benchmark-label blind.  It refines already
observable, pre-policy hazards into independent bipolar evidence and the
least local control named by the checked policy.  Full and every research
variant receive the identical transformed evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from obligate.theory.benchmark_adapter import CanonicalBenchmarkInput
from obligate.theory.model import EvidenceAssertion, SignedAtom, canonical_json
from obligate.theory.runtime import RuntimeEvidence

ACTIVATION_PROFILE_VERSION = "mechanism-activation-profile-v1"
ACTIVATION_CAPABILITIES = frozenset(
    {"host", "sandbox", "network", "redaction", "confirmation", "audit"}
)

# These are observable policy/evidence signals, not scorer labels.  A soft
# local control still pauses or constrains dispatch through the unchanged
# checked lattice and ToolGate.
_CONTROL_BY_SIGNAL = {
    "observable_injection_to_side_effect": "sandbox",
    "observable_private_to_external": "redact",
    "observable_untrusted_target": "require_confirmation",
    "unauthorized_critical_action": "require_confirmation",
    "ADJ-BANKING-UNTRUSTED-FINANCIAL-COMMIT": "require_confirmation",
    "ADJ-GRAPH-001": "redact",
    "ADJ-GRAPH-002": "sandbox",
    "ADJ-GRAPH-003": "require_confirmation",
    "ADJ-FIN-001": "require_confirmation",
    "ADJ-SLACK-CONFIRM-MEMBERSHIP-EXPANSION": "require_confirmation",
    "ADJ-DATA-001": "redact",
    "ADJ-MSG-001": "require_confirmation",
    "ADJ-GOAL-001": "require_confirmation",
    "ADJ-TRAVEL-INJECTED-SEND": "sandbox",
    "ADJ-WORKSPACE-INJECTED-CALENDAR": "sandbox",
    "ADJ-WORKSPACE-ADAPTIVE-EXTERNAL-SEND": "sandbox",
    "ADJ-DATA-002": "redact",
    "ADJ-STATE-001": "require_confirmation",
}

# Independent low-trust path observations refute a claim even when a trusted
# task-contract source supports it.  Keeping both assertions is the intended
# bipolar semantics; neither source erases the other.
_REFUTED_CLAIM_BY_SIGNAL = {
    "observable_injection_to_side_effect": "flow_authorized",
    "observable_private_to_external": "flow_authorized",
    "observable_untrusted_target": "action_authorized",
    "unauthorized_critical_action": "action_authorized",
    "ADJ-BANKING-UNTRUSTED-FINANCIAL-COMMIT": "action_authorized",
    "ADJ-GRAPH-001": "flow_authorized",
    "ADJ-GRAPH-002": "flow_authorized",
    "ADJ-GRAPH-003": "action_authorized",
    "ADJ-FIN-001": "action_authorized",
    "ADJ-SLACK-CONFIRM-MEMBERSHIP-EXPANSION": "action_authorized",
    "ADJ-DATA-001": "flow_authorized",
    "ADJ-MSG-001": "action_authorized",
    "ADJ-GOAL-001": "action_authorized",
    "ADJ-TRAVEL-INJECTED-SEND": "flow_authorized",
    "ADJ-WORKSPACE-INJECTED-CALENDAR": "action_authorized",
    "ADJ-WORKSPACE-ADAPTIVE-EXTERNAL-SEND": "flow_authorized",
    "ADJ-DATA-002": "flow_authorized",
    "ADJ-STATE-001": "action_authorized",
}


def activate_runtime_evidence(action_tool_name: str, evidence: RuntimeEvidence) -> RuntimeEvidence:
    """Return the label-blind mechanism-activation view of pre-policy evidence."""

    assertions = list(evidence.assertions)
    existing_atoms = {item.atom for item in assertions}
    transformed_hazards = []
    transformations: list[dict[str, Any]] = []
    for signal in evidence.hazards:
        control = _CONTROL_BY_SIGNAL.get(signal.signal_id)
        transformed = signal
        if control is not None:
            transformed = replace(signal, hard=False, preferred_control=control)
            transformations.append(
                {
                    "signal_id": signal.signal_id,
                    "from": "hard" if signal.hard else signal.preferred_control,
                    "to": control,
                }
            )
        transformed_hazards.append(transformed)

        predicate = _REFUTED_CLAIM_BY_SIGNAL.get(signal.signal_id)
        if predicate is None:
            continue
        atom = SignedAtom(predicate, (action_tool_name,), "-")
        if atom in existing_atoms:
            continue
        assertions.append(
            EvidenceAssertion(
                atom=atom,
                evidence_ref=f"activation:{signal.signal_id}",
                source_kind="observable_low_trust_path",
                trusted_for_authorization=False,
            )
        )
        existing_atoms.add(atom)

    facts = dict(evidence.facts or {})
    facts["adaptive_ablation.mechanism_activation_profile"] = ACTIVATION_PROFILE_VERSION
    facts["adaptive_ablation.mechanism_activation_transformations"] = transformations
    return replace(
        evidence,
        assertions=tuple(assertions),
        hazards=tuple(transformed_hazards),
        facts=facts,
    )


def activate_canonical_input(value: CanonicalBenchmarkInput) -> CanonicalBenchmarkInput:
    return replace(
        value,
        evidence=activate_runtime_evidence(value.action.tool_name, value.evidence),
    )


def install_activation_bridge(firewall_module: Any) -> tuple[Any, Any]:
    """Install the profile in one adaptive-run process and return originals."""

    original_adapter = firewall_module.from_tool_boundary
    original_init = firewall_module.AgentDojoToolFirewall.__init__

    def activated_adapter(*args: Any, **kwargs: Any) -> CanonicalBenchmarkInput:
        return activate_canonical_input(original_adapter(*args, **kwargs))

    def activated_init(instance: Any, *args: Any, **kwargs: Any) -> None:
        supplied = kwargs.get("theory_capabilities") or ()
        kwargs["theory_capabilities"] = tuple(
            sorted(set(supplied) | ACTIVATION_CAPABILITIES)
        )
        original_init(instance, *args, **kwargs)

    firewall_module.from_tool_boundary = activated_adapter
    firewall_module.AgentDojoToolFirewall.__init__ = activated_init
    return original_adapter, original_init


def uninstall_activation_bridge(firewall_module: Any, originals: tuple[Any, Any]) -> None:
    firewall_module.from_tool_boundary, firewall_module.AgentDojoToolFirewall.__init__ = originals


def activation_manifest() -> dict[str, Any]:
    path = Path(__file__).resolve()
    material = {
        "profile_version": ACTIVATION_PROFILE_VERSION,
        "capabilities": sorted(ACTIVATION_CAPABILITIES),
        "control_by_signal": dict(sorted(_CONTROL_BY_SIGNAL.items())),
        "refuted_claim_by_signal": dict(sorted(_REFUTED_CLAIM_BY_SIGNAL.items())),
        "code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "benchmark_labels_or_scorer_fields_used": False,
    }
    return {
        **material,
        "profile_sha256": hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest(),
    }


__all__ = [
    "ACTIVATION_CAPABILITIES",
    "ACTIVATION_PROFILE_VERSION",
    "activate_canonical_input",
    "activate_runtime_evidence",
    "activation_manifest",
    "install_activation_bridge",
    "uninstall_activation_bridge",
]
