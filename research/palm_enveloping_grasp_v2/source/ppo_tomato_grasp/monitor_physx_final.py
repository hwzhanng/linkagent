"""Hourly, bounded monitor for the long-running PhysX PPO job.

The monitor never touches the simulator or hardware.  It reads the trainer's
JSONL metrics and writes a small control file that the trainer consumes at the
next PPO update.  Changes are deliberately conservative and require two
consecutive observations before being applied.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


DEFAULT_OVERRIDES = {
    "proximity_reward_weight": 2.0,
    "contact_reward_weight": 2.0,
    "secure_reward_weight": 2.0,
    "force_penalty_weight": 0.10,
    "speed_penalty_weight": 0.50,
    "success_bonus": 10.0,
    "target_force_n": 0.65,
    "learning_rate": 3e-4,
    "entropy_coef": 1e-3,
    "policy_std_scale": 1.0,
}


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def last_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(max(0, end - 128 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            if line.strip():
                value = json.loads(line)
                return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return None


def trainer_alive(pid_path: Path) -> bool:
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def f(record: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(record.get(key, default))
        return value if value == value else default
    except (TypeError, ValueError):
        return default


def adapt(overrides: dict, current: dict, previous: dict | None, stagnant_checks: int) -> tuple[dict, int, str]:
    """Return new bounded overrides, streak count and an audit explanation."""

    success = max(f(current, "eval/success_rate"), f(current, "rollout/success_rate"))
    reward = f(current, "rollout/reward_mean")
    contact = f(current, "eval/contact_force_mean_n", f(current, "contact_force_mean_n"))
    peak = max(f(current, "eval/contact_force_max_n"), f(current, "contact_force_max_n"))
    damage = max(f(current, "eval/damage_score"), f(current, "damage_score"))
    prev_success = max(
        f(previous or {}, "eval/success_rate"), f(previous or {}, "rollout/success_rate")
    )
    improving = success > prev_success + 0.01
    if success < 0.01 and contact < 0.05 and not improving:
        stagnant_checks += 1
    else:
        stagnant_checks = 0

    reason = "hold: metrics are improving or inside the current envelope"
    if stagnant_checks >= 2:
        # Exploration/shaping only: do not weaken force or termination safety.
        overrides["proximity_reward_weight"] = min(3.5, float(overrides["proximity_reward_weight"]) + 0.5)
        overrides["contact_reward_weight"] = min(5.0, float(overrides["contact_reward_weight"]) + 0.75)
        overrides["secure_reward_weight"] = min(5.0, float(overrides["secure_reward_weight"]) + 0.5)
        overrides["entropy_coef"] = min(0.0025, float(overrides["entropy_coef"]) * 1.25)
        overrides["policy_std_scale"] = min(1.25, float(overrides["policy_std_scale"]) * 1.08)
        reason = "two stagnant checks: increased bounded approach/contact shaping and exploration"
        stagnant_checks = 0
    elif success > 0.10 and (peak > 4.0 or damage > 0.20):
        overrides["force_penalty_weight"] = min(1.0, float(overrides["force_penalty_weight"]) * 1.35)
        overrides["target_force_n"] = max(0.40, float(overrides["target_force_n"]) * 0.90)
        overrides["policy_std_scale"] = max(0.85, float(overrides["policy_std_scale"]) * 0.95)
        reason = "success with excessive force/damage: strengthened force shaping and reduced target"
    elif success > 0.30 and peak < 2.0 and damage < 0.05:
        overrides["force_penalty_weight"] = max(0.05, float(overrides["force_penalty_weight"]) * 0.90)
        reason = "stable low-force success: gently relaxed force penalty for exploration"
    return overrides, stagnant_checks, reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=3600.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    metrics_path = args.output_dir / "metrics.jsonl"
    control_path = args.output_dir / "adaptive_overrides.json"
    state_path = args.output_dir / "monitor_state.json"
    audit_path = args.output_dir / "monitor_audit.jsonl"
    pid_path = args.output_dir / "trainer.pid"
    overrides = dict(DEFAULT_OVERRIDES)
    previous = None
    stagnant_checks = 0
    started = time.time()
    if control_path.exists():
        try:
            loaded = json.loads(control_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                overrides.update({key: loaded[key] for key in DEFAULT_OVERRIDES if key in loaded})
        except (OSError, json.JSONDecodeError):
            pass
    atomic_write_json(control_path, overrides)

    while True:
        record = last_record(metrics_path)
        if record is not None and (previous is None or record.get("update") != previous.get("update")):
            overrides, stagnant_checks, reason = adapt(overrides, record, previous, stagnant_checks)
            atomic_write_json(control_path, overrides)
            audit = {
                "timestamp": time.time(),
                "update": record.get("update"),
                "success_rate": max(f(record, "eval/success_rate"), f(record, "rollout/success_rate")),
                "reason": reason,
                "overrides": overrides,
            }
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, sort_keys=True) + "\n")
            atomic_write_json(
                state_path,
                {"last_check": audit, "stagnant_checks": stagnant_checks},
            )
            print(f"[physx-monitor] {reason}; update={record.get('update')}", flush=True)
            previous = record
        if args.once:
            return
        if not trainer_alive(pid_path) and time.time() - started > 300.0:
            print("[physx-monitor] trainer is no longer alive; exiting", flush=True)
            return
        time.sleep(max(5.0, args.interval_seconds))


if __name__ == "__main__":
    main()
