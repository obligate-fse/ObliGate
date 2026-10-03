"""Freeze code/config hashes and read-only protected result inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .common import atomic_write_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = (
    ROOT / "results" / "adaptive_ablation",
    ROOT / "results" / "adaptive_closed_loop_v2",
    ROOT / "results" / "full",
)
CODE_ROOTS = (
    ROOT / "src" / "obligate" / "eval" / "agentdojo",
    ROOT / "src" / "obligate" / "theory",
    ROOT / "evaluation" / "obligate_ablation_v2",
)
CODE_INPUT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml"}
EXTRA_CODE_FILES = (
    ROOT / "experiments" / "adaptive_ablation" / "safety.py",
    ROOT / "experiments" / "adaptive_ablation" / "snapshots.py",
)
STATIC_ARTIFACT_FILES = (
    Path("config") / "experiment.yaml",
    Path("config") / "model_configs.json",
    Path("config") / "variants.yaml",
    Path("manifests") / "agentdojo_949.json",
    Path("manifests") / "fold_assignment.json",
    Path("manifests") / "sampling_audit.json",
    Path("preflight") / "dry_run_case_ids.txt",
)
DYNAMIC_CONFIG_FILES = {"config/scalar_crossfit_thresholds.json", "manifests/snapshot_manifest.json"}


def _hash_stream(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_tree(root: Path, output: Path) -> dict[str, Any]:
    files = sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    tree = hashlib.sha256()
    count = 0
    total_bytes = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            stat = path.stat()
            row = {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _hash_stream(path),
            }
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write(payload + "\n")
            tree.update(payload.encode("utf-8"))
            tree.update(b"\n")
            count += 1
            total_bytes += stat.st_size
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return {
        "root": str(root.resolve()),
        "manifest": str(output.resolve()),
        "manifest_sha256": sha256_file(output),
        "tree_sha256": tree.hexdigest(),
        "file_count": count,
        "total_bytes": total_bytes,
    }


def code_hashes() -> dict[str, Any]:
    discovered = {
        path.resolve()
        for code_root in CODE_ROOTS
        for path in code_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in CODE_INPUT_SUFFIXES
    }
    discovered.update(path.resolve() for path in EXTRA_CODE_FILES if path.is_file())
    files = sorted(discovered, key=lambda item: item.relative_to(ROOT).as_posix())
    rows = {
        str(path.relative_to(ROOT).as_posix()): sha256_file(path)
        for path in files
        if path.is_file()
    }
    aggregate = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "obligate-ablation-v2-code-freeze-v1",
        "created_at": utc_now(),
        "files": rows,
        "aggregate_sha256": aggregate,
        "production_files_modified_by_experiment": False,
    }


def bundle_fingerprint(output_root: Path) -> dict[str, Any]:
    """Hash every static input that may affect a formal episode or its analysis.

    Cross-fit thresholds are generated from the frozen Full trajectories during
    the formal run and are therefore phase-sealed separately in the cell config.
    All other configuration and manifest files are part of this preflight seal.
    """

    root = Path(output_root).resolve()
    code = code_hashes()
    artifact_paths = {root / relative for relative in STATIC_ARTIFACT_FILES}
    missing = sorted(
        path.relative_to(root).as_posix() for path in artifact_paths if not path.is_file()
    )
    if missing:
        raise FileNotFoundError("static formal bundle inputs are missing: " + ", ".join(missing))
    rows = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(artifact_paths, key=lambda item: item.as_posix())
        if path.is_file()
    }
    aggregate_material = {
        "code_aggregate_sha256": code["aggregate_sha256"],
        "static_artifacts": rows,
    }
    aggregate = hashlib.sha256(
        json.dumps(aggregate_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "obligate-ablation-v2-bundle-freeze-v1",
        "created_at": utc_now(),
        "code": code,
        "static_artifacts": rows,
        "aggregate_sha256": aggregate,
        "dynamic_artifact_exclusions": sorted(DYNAMIC_CONFIG_FILES),
    }


def freeze_before(output_root: Path) -> dict[str, Any]:
    bundle = bundle_fingerprint(output_root)
    preflight_gate = output_root / "preflight" / "preflight.json"
    if not preflight_gate.is_file():
        raise FileNotFoundError(preflight_gate)
    bundle["preflight_gate_sha256"] = sha256_file(preflight_gate)
    hashes = bundle["code"]
    atomic_write_json(output_root / "hashes" / "code_hashes.json", hashes)
    atomic_write_json(output_root / "hashes" / "bundle_fingerprint.json", bundle)
    protected: dict[str, Any] = {}
    for root in PROTECTED:
        protected[root.name] = inventory_tree(
            root,
            output_root / "hashes" / "protected_before" / f"{root.name}.jsonl",
        )
    value = {
        "schema_version": "obligate-protected-results-before-v1",
        "created_at": utc_now(),
        "protected": protected,
    }
    atomic_write_json(output_root / "hashes" / "protected_before.json", value)
    return {"code": hashes, "bundle": bundle, "protected": value}


def verify_after(output_root: Path) -> dict[str, Any]:
    before = json.loads((output_root / "hashes" / "protected_before.json").read_text(encoding="utf-8"))
    current_code = code_hashes()
    frozen_code = json.loads((output_root / "hashes" / "code_hashes.json").read_text(encoding="utf-8"))
    current_bundle = bundle_fingerprint(output_root)
    frozen_bundle = json.loads(
        (output_root / "hashes" / "bundle_fingerprint.json").read_text(encoding="utf-8")
    )
    preflight_gate = output_root / "preflight" / "preflight.json"
    preflight_gate_unchanged = (
        preflight_gate.is_file()
        and sha256_file(preflight_gate) == frozen_bundle.get("preflight_gate_sha256")
    )
    protected: dict[str, Any] = {}
    for root in PROTECTED:
        protected[root.name] = inventory_tree(
            root,
            output_root / "hashes" / "protected_after" / f"{root.name}.jsonl",
        )
    comparisons: dict[str, Any] = {}
    for name, current in protected.items():
        prior = before["protected"][name]
        comparisons[name] = {
            "tree_sha256_equal": prior["tree_sha256"] == current["tree_sha256"],
            "file_count_equal": prior["file_count"] == current["file_count"],
            "total_bytes_equal": prior["total_bytes"] == current["total_bytes"],
        }
        comparisons[name]["unchanged"] = all(comparisons[name].values())
    value = {
        "schema_version": "obligate-protected-results-after-v1",
        "created_at": utc_now(),
        "protected": protected,
        "comparisons": comparisons,
        "all_protected_results_unchanged": all(item["unchanged"] for item in comparisons.values()),
        "formal_code_hash_unchanged": current_code["aggregate_sha256"] == frozen_code["aggregate_sha256"],
        "formal_bundle_hash_unchanged": current_bundle["aggregate_sha256"]
        == frozen_bundle["aggregate_sha256"],
        "preflight_gate_hash_unchanged": preflight_gate_unchanged,
        "frozen_code_aggregate_sha256": frozen_code["aggregate_sha256"],
        "current_code_aggregate_sha256": current_code["aggregate_sha256"],
        "frozen_bundle_aggregate_sha256": frozen_bundle["aggregate_sha256"],
        "current_bundle_aggregate_sha256": current_bundle["aggregate_sha256"],
    }
    atomic_write_json(output_root / "hashes" / "protected_after.json", value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("before", "after"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    value = freeze_before(args.output_root) if args.mode == "before" else verify_after(args.output_root)
    print(json.dumps(value if args.mode == "after" else {"code": value["code"]}, ensure_ascii=False))
    if args.mode == "after":
        return (
            0
            if value["all_protected_results_unchanged"]
            and value["formal_code_hash_unchanged"]
            and value["formal_bundle_hash_unchanged"]
            and value["preflight_gate_hash_unchanged"]
            else 4
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "bundle_fingerprint",
    "code_hashes",
    "freeze_before",
    "inventory_tree",
    "verify_after",
]
