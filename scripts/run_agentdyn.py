"""AgentDyn experiment entry point."""
from pathlib import Path
import os
import subprocess
import sys

if os.name == "nt" and not sys.flags.utf8_mode:
    os.environ["PYTHONUTF8"] = "1"
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.agentdyn.run import main

if __name__ == "__main__":
    raise SystemExit(main())
