"""Wait for bounded training, evaluate candidate checkpoints, and freeze the best."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "logs/rsl_rl/tomato_pick_physics_v2_learning"
OUTPUT = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/eight_hour_evaluation"
STATUS = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/eight_hour_finalizer_status.json"
BEST = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/best_eight_hour_checkpoint.pt"


def _write(state: str, **values: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "state": state,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hardware_boundary": "simulation_only_no_robot_connections",
        **values,
    }
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def _active(unit: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit], check=False
    ).returncode == 0


def _gpu_compute_processes() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else "query_failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-unit", default="linkagent-physics-v2-eight-hour.service")
    parser.add_argument("--training-log", type=Path, default=Path("/tmp/tomato_physics_v2_eight_hour.log"))
    parser.add_argument("--max-wait-hours", type=float, default=9.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.max_wait_hours * 3600.0
    while _active(args.training_unit):
        if time.monotonic() >= deadline:
            _write("stopped_wait_timeout")
            return 3
        _write("waiting_for_training")
        time.sleep(60)

    # Allow W&B/Isaac cleanup to release CUDA before starting deterministic evaluation.
    for _ in range(30):
        processes = _gpu_compute_processes()
        if not processes:
            break
        _write("waiting_for_gpu_release", observed_compute_processes=processes)
        time.sleep(20)
    else:
        _write("stopped_gpu_not_released")
        return 4

    runs = sorted(
        (
            path
            for path in RUN_ROOT.glob("*_eight_hour_learning_seed43_from_scratch")
            if path.is_dir()
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not runs:
        _write("stopped_no_run_directory")
        return 5
    run = runs[-1]
    checkpoints = sorted(
        run.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1])
    )
    if not checkpoints:
        _write("stopped_no_checkpoints", run_directory=str(run))
        return 6
    candidates = checkpoints[-3:]
    iterations = [int(path.stem.split("_")[-1]) for path in candidates]
    training_text = args.training_log.read_text(encoding="utf-8", errors="replace")
    wandb_urls = re.findall(r"https://wandb\.ai/[^\s\x1b]+/runs/([a-zA-Z0-9]+)", training_text)
    if not wandb_urls:
        _write("stopped_wandb_run_identity_missing", training_log=str(args.training_log))
        return 8
    wandb_run_id = wandb_urls[-1]
    wandb_url = (
        "https://wandb.ai/miobarnacle-china-agricultural-university/"
        f"linkagent-tomato-physics-v2/runs/{wandb_run_id}"
    )
    _write(
        "evaluating",
        run_directory=str(run),
        candidate_iterations=iterations,
        wandb_run_id=wandb_run_id,
        wandb_url=wandb_url,
    )

    environment = dict(os.environ)
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "script/evaluate_tomato_sweep.py"),
        "--run-dir",
        str(run),
        "--output-dir",
        str(OUTPUT),
        "--iterations",
        *[str(value) for value in iterations],
        "--seeds",
        "17",
        "42",
        "2026",
        "--num-envs",
        "64",
        "--steps",
        "160",
        "--workers",
        "1",
        "--wandb-run-id",
        wandb_run_id,
        "--wandb-url",
        wandb_url,
        "--task",
        "LinkAgent-TomatoPick-Physics-v2",
        "--external-callback",
        "simulation.tomato_pick.isaac.physics_v2.register_task",
    ]
    log = OUTPUT / "sweep.log"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        _write("stopped_evaluation_failed", returncode=result.returncode, log=str(log))
        return result.returncode

    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(manifest["best_checkpoint"])
    shutil.copy2(source, BEST)
    digest = hashlib.sha256(BEST.read_bytes()).hexdigest()
    if digest != manifest["best_checkpoint_sha256"]:
        _write("stopped_best_copy_hash_mismatch")
        return 7
    _write(
        "completed",
        run_directory=str(run),
        candidate_iterations=iterations,
        best_source_checkpoint=str(source),
        frozen_best_checkpoint=str(BEST),
        best_checkpoint_sha256=digest,
        best_summary=manifest["best_summary"],
        evaluation_manifest=str(manifest_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
