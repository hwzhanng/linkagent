"""Keep the final PhysX PPO job alive across terminal/session loss.

This watchdog does not touch the simulator while the trainer PID is alive.  If
the trainer exits before the requested update count, it resumes from the most
recent checkpoint.  A normally completed run is never restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def read_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
        return value if value > 0 else None
    except (OSError, ValueError):
        return None


def alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def latest_update(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return 0
    try:
        with metrics_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(max(0, end - 256 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            if line.strip():
                return int(json.loads(line).get("update", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return 0


def log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-updates", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=160)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    launcher = root / "simulation/ppo_tomato_grasp/launch_physx_sphere_final.sh"
    trainer_pid = output_dir / "trainer.pid"
    metrics = output_dir / "metrics.jsonl"
    checkpoint = output_dir / "latest.pt"
    watchdog_log = output_dir / "watchdog.log"

    while True:
        pid = read_pid(trainer_pid)
        if alive(pid):
            time.sleep(max(5.0, args.poll_seconds))
            continue

        update = latest_update(metrics)
        if update >= args.target_updates:
            log(watchdog_log, f"completed update={update}; watchdog exiting")
            return
        if not checkpoint.exists():
            log(watchdog_log, "trainer absent but no latest.pt; waiting")
            time.sleep(max(10.0, args.poll_seconds))
            continue

        remaining = args.target_updates - update
        env = os.environ.copy()
        env.update(
            {
                "PHYSX_RUN_DIR": str(output_dir),
                "WANDB_MODE": "offline",
                "MONITOR_INTERVAL_SECONDS": "3600",
                "NUM_ENVS": str(args.num_envs),
                "HORIZON": str(args.horizon),
                "UPDATES": str(remaining),
                "PPO_EPOCHS": str(args.epochs),
                "MINIBATCH_SIZE": str(args.minibatch_size),
                "PHYSX_DEVICE": args.device,
                "MPLCONFIGDIR": f"/tmp/linkbot-physx-watchdog-{os.getpid()}",
                "WANDB_RUN_NAME": f"physx-sphere-final-v2-resume-{update}",
            }
        )
        command = [
            str(launcher),
            "--resume",
            str(checkpoint),
            "--save-interval",
            "10",
            "--eval-interval",
            str(args.eval_interval),
            "--eval-steps",
            str(args.eval_steps),
        ]
        log(watchdog_log, f"trainer absent; resuming from update={update} remaining={remaining}")
        result = subprocess.run(command, cwd=root, env=env, check=False)
        log(watchdog_log, f"resume process exited returncode={result.returncode}")
        time.sleep(10.0)


if __name__ == "__main__":
    main()
