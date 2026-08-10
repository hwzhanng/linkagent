"""Monitor the bounded Physics-v2 run and persist hourly learning audits."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import re
from pathlib import Path
import re
import shutil
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/eight_hour_training_status.json"
AUDITS = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/hourly_audits"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
BLOCK = re.compile(r"Learning iteration (\d+)/(\d+)")
MIN_FREE_BYTES = 8 * 1024**3


def _metric(block: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}:\s+([-+0-9.eE]+)", block)
    return float(match.group(1)) if match else None


def _iterations(log: Path) -> list[dict[str, float | int | None]]:
    if not log.is_file():
        return []
    text = ANSI.sub("", log.read_text(encoding="utf-8", errors="replace"))
    matches = list(BLOCK.finditer(text))
    values = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start():stop]
        if "Iteration time:" not in block:
            continue
        values.append(
            {
                "iteration": int(match.group(1)),
                "steps_per_second": _metric(block, "Steps per second"),
                "value_loss": _metric(block, "Mean value loss"),
                "surrogate_loss": _metric(block, "Mean surrogate loss"),
                "mean_reward": _metric(block, "Mean reward"),
                "action_std": _metric(block, "Mean action std"),
                "success_rate": _metric(block, "Metrics/success_rate"),
                "contact": _metric(block, "Episode_Reward/contact"),
                "drop_rate": _metric(block, "Metrics/drop_rate"),
                "detach_rate": _metric(block, "Metrics/detach_rate"),
                "retained_rate": _metric(block, "Metrics/retained_rate"),
                "max_force_failure": _metric(block, "Episode_Reward/excess_force_failure"),
            }
        )
    return values


def _gpu() -> dict[str, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    memory, utilization, temperature = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
    return {"memory_used_mib": memory, "utilization_percent": utilization, "temperature_c": temperature}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _service_active(unit: str) -> bool:
    result = subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit], check=False)
    return result.returncode == 0


def _service_started_epoch(unit: str) -> float | None:
    """Return the service start time in the same wall-clock domain as time.time().

    systemd's ``ActiveEnterTimestampMonotonic`` is based on the host boot clock,
    while this Python process may run in a container with a different monotonic
    epoch.  Comparing the two can make a restarted monitor reset the run age.
    """
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=ActiveEnterTimestamp", "--value"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value:
        return None
    try:
        # systemd emits local time, e.g. ``Mon 2026-08-10 02:25:45 CST``.
        parsed = datetime.strptime(value.rsplit(" ", 1)[0], "%a %Y-%m-%d %H:%M:%S")
        return parsed.replace(tzinfo=datetime.now().astimezone().tzinfo).timestamp()
    except ValueError:
        return None


def _stop(unit: str, reason: str) -> None:
    subprocess.run(["systemctl", "--user", "kill", "--signal=SIGINT", unit], check=False)
    _write(STATUS, {"state": "automatic_stop_requested", "reason": reason, "updated_at": datetime.now().astimezone().isoformat()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", default="linkagent-physics-v2-eight-hour.service")
    parser.add_argument("--log", type=Path, default=Path("/tmp/tomato_physics_v2_eight_hour.log"))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--hour-seconds", type=int, default=3600)
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()
    if args.max_hours <= 0.0:
        raise ValueError("max-hours must be positive")
    started = _service_started_epoch(args.unit) or time.time()
    if started > time.time():
        started = time.time()
    last_progress = started
    last_iteration = -1
    existing_audits = []
    for audit_path in AUDITS.glob("hour_*.json"):
        match = re.fullmatch(r"hour_(\d+)\.json", audit_path.name)
        if match:
            existing_audits.append(int(match.group(1)))
    next_audit = started + (max(existing_audits) + 1 if existing_audits else 0) * args.hour_seconds
    hot_polls = 0
    while _service_active(args.unit):
        now = time.time()
        metrics = _iterations(args.log)
        if metrics and int(metrics[-1]["iteration"]) > last_iteration:
            last_iteration = int(metrics[-1]["iteration"])
            last_progress = now
        gpu = _gpu()
        hot_polls = hot_polls + 1 if gpu["temperature_c"] >= 90 else 0
        free_bytes = shutil.disk_usage(PROJECT_ROOT).free
        log_text = args.log.read_text(encoding="utf-8", errors="replace") if args.log.is_file() else ""
        error = re.search(
            r"Traceback|CUDA out of memory|PhysX[^\n]*overflow|:\s*(?:nan|[-+]?inf)\b",
            log_text,
            flags=re.IGNORECASE,
        )
        reason = None
        if error:
            reason = f"numerical_or_runtime_error:{error.group(0)}"
        elif free_bytes < MIN_FREE_BYTES:
            reason = "disk_free_below_8_gib"
        elif hot_polls >= 3:
            reason = "gpu_temperature_at_least_90c_for_three_polls"
        elif now - last_progress > 1800:
            reason = "no_completed_iteration_for_30_minutes"
        elif now - started >= args.max_hours * 3600.0:
            reason = "bounded_training_time_elapsed"
        if reason:
            _stop(args.unit, reason)
            return 2

        status = {
            "schema_version": 1,
            "state": "training",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_hours": round((now - started) / 3600.0, 3),
            "latest_iteration": last_iteration,
            "completed_updates": len(metrics),
            "gpu": gpu,
            "disk_free_gib": round(free_bytes / 1024**3, 2),
            "latest_metrics": metrics[-1] if metrics else None,
            "hardware_boundary": "simulation_only_no_robot_connections",
        }
        _write(STATUS, status)
        if now >= next_audit:
            hour = int((now - started) // args.hour_seconds)
            recent = metrics[-72:]
            finite = all(
                value is None or math.isfinite(value)
                for item in recent
                for key, value in item.items()
                if key != "iteration"
            )
            audit = {
                **status,
                "audit_hour": hour,
                "recent_window": recent,
                "decision": "continue" if finite else "stop_non_finite",
                "automatic_parameter_change": None,
            }
            _write(AUDITS / f"hour_{hour:02d}.json", audit)
            next_audit += args.hour_seconds
        time.sleep(args.poll_seconds)

    metrics = _iterations(args.log)
    final_state = {
        "schema_version": 1,
        "state": "training_process_finished",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completed_updates": len(metrics),
        "latest_iteration": int(metrics[-1]["iteration"]) if metrics else -1,
        "latest_metrics": metrics[-1] if metrics else None,
        "hardware_boundary": "simulation_only_no_robot_connections",
    }
    _write(STATUS, final_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
