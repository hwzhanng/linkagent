"""Wait for an idle GPU, then run and summarize the bounded Physics-v2 PPO probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/unattended_probe_status.json"
LOG = Path("/tmp/tomato_physics_v2_ppo_256x4.log")
REPORT = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/ppo_256x4_probe.json"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_status(state: str, **details: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": 1,
        "state": state,
        "updated_at": _now(),
        "hardware_boundary": "simulation_only_no_robot_connections",
        "formal_training_authorized": False,
        **details,
    }
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def _compute_processes() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi compute-process query failed")
    processes = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3 and fields[0].isdigit():
            processes.append({"pid": fields[0], "name": fields[1], "memory_mib": fields[2]})
    return processes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wait-hours", type=float, default=12.0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--idle-checks", type=int, default=3)
    args = parser.parse_args()
    if args.max_wait_hours <= 0 or args.poll_seconds < 5 or args.idle_checks < 2:
        raise ValueError("invalid unattended wait policy")

    deadline = time.monotonic() + args.max_wait_hours * 3600.0
    idle_count = 0
    try:
        while time.monotonic() < deadline:
            processes = _compute_processes()
            if processes:
                idle_count = 0
                _write_status("waiting_for_gpu", observed_compute_processes=processes)
            else:
                idle_count += 1
                _write_status(
                    "confirming_gpu_idle",
                    consecutive_idle_checks=idle_count,
                    required_idle_checks=args.idle_checks,
                )
                if idle_count >= args.idle_checks:
                    break
            time.sleep(args.poll_seconds)
        else:
            _write_status("stopped_wait_timeout", max_wait_hours=args.max_wait_hours)
            return 3

        _write_status("running_numerical_probe", log=str(LOG), report=str(REPORT))
        environment = dict(os.environ)
        with LOG.open("w", encoding="utf-8") as handle:
            probe = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "script/probe_tomato_physics_v2_ppo.py")],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if probe.returncode != 0:
            _write_status("stopped_probe_failed", returncode=probe.returncode, log=str(LOG))
            return probe.returncode

        summary = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "script/summarize_tomato_physics_v2_probe.py"),
                "--log",
                str(LOG),
                "--output",
                str(REPORT),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if summary.returncode != 0:
            _write_status(
                "stopped_summary_failed",
                returncode=summary.returncode,
                error=summary.stderr[-2000:],
                log=str(LOG),
            )
            return summary.returncode
        result = json.loads(REPORT.read_text(encoding="utf-8"))
        _write_status(
            "completed_probe_passed",
            report=str(REPORT),
            wandb_run_url=result.get("wandb_run_url"),
            final_checkpoint=result.get("final_checkpoint"),
        )
        return 0
    except Exception as exc:
        _write_status("stopped_orchestrator_error", error=f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
