from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/SaFo-Lab/AgentDyn.git"
REVISION = "5353cf7615b135cace8d07c8f12dac53a16b6db3"


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True, encoding="utf-8"
    ).strip()


def verify(checkout: Path) -> None:
    origin = git(checkout, "remote", "get-url", "origin").rstrip("/").removesuffix(".git")
    if origin not in {"https://github.com/SaFo-Lab/AgentDyn", "git@github.com:SaFo-Lab/AgentDyn"}:
        raise RuntimeError("AgentDyn checkout origin must be SaFo-Lab/AgentDyn")
    if git(checkout, "rev-parse", "HEAD") != REVISION:
        raise RuntimeError(f"AgentDyn checkout must be at {REVISION}")
    if git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("AgentDyn checkout has modified tracked files")
    if not (checkout / "src/agentdojo/task_suite/load_suites.py").is_file():
        raise RuntimeError("AgentDyn suite loader is missing")
    if not (checkout / "src/agentdojo/data").is_dir():
        raise RuntimeError("AgentDyn benchmark data is missing")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path,
                        default=Path(os.getenv("AGENTDYN_ROOT", str(ROOT / "third_party/AgentDyn"))))
    parser.add_argument("--check", action="store_true", help="Verify an existing checkout without fetching")
    parser.add_argument("--install", action="store_true", help="Install benchmark dependencies in this interpreter")
    args = parser.parse_args(argv)
    checkout = args.upstream.expanduser().resolve()
    if args.check:
        verify(checkout)
    else:
        if checkout.exists():
            if not (checkout / ".git").exists():
                raise RuntimeError(f"Destination already exists and is not a Git checkout: {checkout}")
            verify(checkout)
        else:
            checkout.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", REVISION], check=True)
            verify(checkout)
    if args.install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"),
                        "-e", str(checkout)], check=True)
    print(json.dumps({"upstream": str(checkout), "revision": REVISION,
                      "installed": args.install}))


if __name__ == "__main__":
    main()
