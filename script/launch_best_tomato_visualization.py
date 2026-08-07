"""Wait for a completed evaluation manifest, then visualize its best checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()

    print(f"[WAIT] evaluation manifest: {manifest_path}", flush=True)
    while not manifest_path.is_file():
        time.sleep(args.poll_seconds)
    manifest = json.loads(manifest_path.read_text())
    checkpoint = Path(manifest["best_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(f"[VISUALIZE] best checkpoint: {checkpoint}", flush=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "script" / "play_tomato_policy.py"),
        str(checkpoint),
        "--num-envs",
        "1",
        "--seed",
        str(args.seed),
        "--render",
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
