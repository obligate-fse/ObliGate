from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.agentdyn import run


def local_integrity(manifest):
    changed = []
    for name, expected in manifest["source_hashes"]["local"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid relative source path in AgentDyn manifest")
        source = ROOT / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            changed.append(name)
    return changed


def cluster_interval(rows, manifest, field, attacked, samples):
    lookup = {row["id"]: row for row in rows}
    clusters = defaultdict(list)
    for case in manifest["cases"]:
        if (case["injection_task"] is not None) != attacked:
            continue
        baseline = lookup.get(run.episode_id(case, "none"))
        guarded = lookup.get(run.episode_id(case, "obligate"))
        if baseline and guarded and baseline["status"] == guarded["status"] == "OK":
            clusters[(case["suite"], case["user_task"])].append(int(baseline[field]) - int(guarded[field]))
    if not clusters:
        return None
    counts = [(len(values), sum(values)) for values in clusters.values()]
    rng = random.Random(20261003)
    distribution = []
    for _ in range(samples):
        drawn = rng.choices(counts, k=len(counts))
        distribution.append(sum(total for _, total in drawn) / sum(n for n, _ in drawn))
    distribution.sort()
    return {"baseline_minus_obligate": sum(total for _, total in counts) / sum(n for n, _ in counts),
            "ci95": [distribution[int(samples * .025)], distribution[min(samples - 1, int(samples * .975))]],
            "clusters": len(counts), "pairs": sum(n for n, _ in counts), "seed": 20261003,
            "bootstrap_samples": samples}


def analyze(output, upstream=None, config_path=None, samples=10000):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    changed = local_integrity(manifest)
    if config_path is not None:
        actual = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        if actual != manifest["source_hashes"]["config_sha256"]:
            changed.append("config")
    if upstream is not None:
        checkout = run.bootstrap_upstream(upstream, manifest["upstream_revision"])
        snapshot = run.source_snapshot(checkout, config_path or ROOT / "configs/agentdyn.yaml")
        if snapshot["upstream"] != manifest["source_hashes"]["upstream"]:
            changed.append("upstream")
    if changed:
        raise RuntimeError("Frozen AgentDyn source changed: " + ", ".join(changed))
    rows = run._episode_rows(output, manifest)
    summary = run.summarize(output, manifest)
    summary["status"] = ("INCOMPLETE" if summary["pending"] else
                         "COMPLETE_WITH_ERRORS" if summary["errors"] else "COMPLETE")
    summary["source_verified"] = True
    summary["upstream_verified"] = upstream is not None
    if set(manifest["config"]["methods"]) == {"none", "obligate"}:
        summary["paired_asr_difference"] = cluster_interval(rows, manifest, "attack_success", True, samples)
        summary["paired_clean_utility_difference"] = cluster_interval(rows, manifest, "user_success", False, samples)
        summary["paired_attacked_utility_difference"] = cluster_interval(rows, manifest, "user_success", True, samples)
    run.write_json(output / "analysis.json", summary)
    fields = ["suite", "method"] + sorted({key for group in summary["groups"].values() for key in group})
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group_key, metrics in summary["groups"].items():
            suite, method = group_key.split("/", 1)
            values = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                      for key, value in metrics.items()}
            writer.writerow({"suite": suite, "method": method, **values})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    try:
        summary = analyze(args.output, args.upstream, args.config, args.bootstrap_samples)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(run._safe_error(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
