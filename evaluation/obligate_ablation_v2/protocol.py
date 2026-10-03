"""Paper Table 5 ordering and the distinct comparator execution contracts."""

E2E_VARIANTS = (
    "full", "scalar-average", "unbound-evidence", "gap-blind",
    "single-remedy", "dag-single", "policy-prompt-only", "block-all-guarded",
)
POLICY_CONTROLS = frozenset({"dag-single", "policy-prompt-only"})
COUNTERFACTUAL_VARIANTS = frozenset(set(E2E_VARIANTS) - POLICY_CONTROLS - {"full"})
COMMON_PROMPT_VARIANTS = tuple(v for v in E2E_VARIANTS if v not in {"full", "policy-prompt-only"})


def execution_contract(variant):
    if variant not in E2E_VARIANTS:
        raise ValueError(f"unknown Table 5 configuration: {variant}")
    return {
        "research_only": variant != "full",
        "verified": variant == "full",
        "research_counterfactual_mode": variant in COUNTERFACTUAL_VARIANTS,
        "policy_control": variant in POLICY_CONTROLS,
        "experiment_role": ("full" if variant == "full" else "policy_control" if variant in POLICY_CONTROLS else "component_ablation"),
        "formal_action_certificate": variant == "full",
        "production_toolgate": variant == "full",
    }
