"""Preserve intermediate checkpoints without changing the running trainer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--milestones", type=int, nargs="+", required=True)
    p.add_argument("--pid", type=int, required=True)
    a = p.parse_args()
    checkpoint = a.run / "checkpoints/last_agent.pt"
    progress = a.run / "progress.json"
    for milestone in sorted(a.milestones):
        target = a.output / f"milestone_{milestone}"
        if (target / "PROVENANCE.json").exists():
            continue
        while True:
            alive = Path(f"/proc/{a.pid}").exists()
            if progress.exists() and checkpoint.exists():
                try:
                    before = progress.read_bytes()
                    info = json.loads(before)
                    stat = checkpoint.stat()
                    if (info["training_transitions"] >= milestone
                            and stat.st_mtime_ns <= progress.stat().st_mtime_ns):
                        target.mkdir(parents=True, exist_ok=True)
                        temporary = target / "agent.pt.tmp"
                        shutil.copy2(checkpoint, temporary)
                        after_stat = checkpoint.stat()
                        if (before == progress.read_bytes()
                                and (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                                == (after_stat.st_ino, after_stat.st_mtime_ns, after_stat.st_size)):
                            temporary.replace(target / "agent.pt")
                            info.update(requested_milestone=milestone,
                                        snapshot_sha256=hashlib.sha256((target / "agent.pt").read_bytes()).hexdigest(),
                                        stable_source_and_progress_verified=True)
                            (target / "PROVENANCE.json").write_text(json.dumps(info, indent=2) + "\n")
                            print(json.dumps(info), flush=True)
                            break
                except (json.JSONDecodeError, FileNotFoundError):
                    pass  # A concurrently replaced progress file is retried.
            if not alive:
                raise RuntimeError(f"Training ended before a stable snapshot at {milestone}")
            time.sleep(15)


if __name__ == "__main__":
    main()
