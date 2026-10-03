"""Fail closed when an experiment entrypoint drifts from its frozen YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, Any]:
    import yaml

    path = ROOT / "configs" / name
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"configuration is not a mapping: {path}")
    return value


def validate(entrypoint: str) -> dict[str, Any]:
    if entrypoint in {"agentdojo", "cross-benchmark"}:
        value = _load("standard_protocols.yaml")
        if value.get("models") != ["deepseek-v4-flash", "qwen-plus"]:
            raise RuntimeError("standard protocol model list drift")
        expected = {
            "agentdojo": {
                "cases_per_model": 949,
                "attack": "important_instructions",
                "temperature": 0.0,
                "max_iters": 24,
                "timeout_seconds": 300,
                "max_attempts": 4,
                "initial_backoff_seconds": 2,
            },
            "agent_safetybench": {
                "cases_per_model": 2000,
                "temperature": 0.0,
                "max_tokens": 2048,
                "timeout_seconds": 180,
                "max_rounds": 10,
                "case_attempts": 1,
            },
            "agent_security_bench": {
                "cases_per_model": 8160,
                "cases_per_attack_type": 2040,
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout_seconds": 60,
                "seed": 0,
            },
        }
        for name, fields in expected.items():
            section_value = value.get(name) or {}
            drift = {
                key: (section_value.get(key), wanted)
                for key, wanted in fields.items()
                if section_value.get(key) != wanted
            }
            if drift:
                raise RuntimeError(f"{name} protocol drift: {drift}")
        protocols = (value.get("agent_security_bench") or {}).get(
            "attack_protocols"
        )
        if protocols != [
            ["dpi", "naive"],
            ["opi", "context_ignoring"],
            ["mixed", "combined_attack"],
            ["memory", "combined_attack"],
        ]:
            raise RuntimeError("Agent Security Bench attack protocol drift")
        section = value["agentdojo"] if entrypoint == "agentdojo" else value
    elif entrypoint == "adaptive":
        value = _load("adaptive_closed_loop_v2.yaml")
        if value.get("max_rounds") != 5 or value.get("main_attacker_seeds") != [20260716]:
            raise RuntimeError("Adaptive frozen round/seed settings drift")
        if set((value.get("workers") or {})) != {
            "attacker",
            "agentdojo",
            "agent_safetybench",
            "agent_security_bench",
        }:
            raise RuntimeError("Adaptive worker configuration is incomplete")
        section = value
    elif entrypoint == "ablation":
        value = _load("ablation_variants.yaml")
        expected = [
            "full",
            "scalar-average",
            "unbound-evidence",
            "gap-blind",
            "single-remedy",
            "dag-single",
            "policy-prompt-only",
            "block-all-guarded",
        ]
        if value.get("e2e") != expected:
            raise RuntimeError("Table 5 eight-variant order drift")
        protocol = value.get("protocol") or {}
        if protocol.get("cases_per_cell") != 949 or protocol.get("models") != ["deepseek-v4-flash", "qwen-plus"]:
            raise RuntimeError("Table 5 population/model drift")
        section = value
    elif entrypoint == "smtp":
        value = json.loads((ROOT / "configs" / "smtp_runtime.json").read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise RuntimeError("SMTP protocol schema drift")
        if (value.get("table6") or {}).get("repetitions_per_condition") != 50:
            raise RuntimeError("Table 6 replay count drift")
        if (value.get("table7") or {}).get("warm_samples_per_path") != 500:
            raise RuntimeError("Table 7 warm sample count drift")
        if (value.get("table7") or {}).get("paths") != ["P0", "P1_confirmed"]:
            raise RuntimeError("Table 7 path drift")
        if value.get("methods") != ["NoGate", "Full"]:
            raise RuntimeError("SMTP method drift")
        if (value.get("table6") or {}).get("conditions") != [
            "normal", "equivalent_MIME", "valid_confirmation",
            "recipient_mutation", "envelope_mutation", "payload_mutation",
            "wrong_actor", "expired_confirmation", "audit_failure",
            "redaction_success", "missing_redactor", "incomplete_redaction",
            "redactor_exception", "lost_DATA_reply",
        ]:
            raise RuntimeError("Table 6 fault-condition drift")
        if (value.get("table6") or {}).get("repetitions_per_seed") != 10:
            raise RuntimeError("Table 6 per-seed repeat drift")
        if (value.get("table7") or {}).get("warm_repetitions_per_seed") != 100 or (value.get("table7") or {}).get("warmup_per_seed_path") != 1:
            raise RuntimeError("Table 7 per-seed warm schedule drift")
        section = value
    else:
        raise ValueError(f"Unknown experiment: {entrypoint}")
    return {
        "entrypoint": entrypoint,
        "schema_version": value.get("schema_version"),
        "validated": True,
        "section": section,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint", choices=("agentdojo", "cross-benchmark", "adaptive", "ablation", "smtp"))
    args = parser.parse_args()
    result = validate(args.entrypoint)
    print(
        json.dumps(
            {
                "entrypoint": result["entrypoint"],
                "schema_version": result["schema_version"],
                "validated": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
