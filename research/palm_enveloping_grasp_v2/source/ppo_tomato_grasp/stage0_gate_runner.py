"""Orchestrate and audit the formal 1024-scene Stage-0 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/miobarnacle/anaconda3/envs/isaacsim/bin/python")
CODE_FILES = (
    "physx_tomato_env.py",
    "stage0_scripted_controller.py",
    "stage0_gate_shard.py",
    "tomato_stem_physics.py",
    "hand_manifold.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(record: dict) -> str:
    code = int(record["termination_code"])
    if code == 1:
        return "success"
    if code == 2:
        return "drop"
    if code == 3:
        return "severe_overload"
    if code == 4:
        return "damage"
    if code == 5:
        return "episode_timeout"
    if code == 6:
        return "gate_budget_no_terminal"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=1024)
    parser.add_argument("--envs-per-shard", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=21000)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--residual-span-fraction", type=float, default=0.12)
    parser.add_argument("--position-jitter-xy-m", type=float, default=0.002)
    parser.add_argument("--friction", type=float, default=0.55)
    parser.add_argument("--palm-rim-height-m", type=float, default=0.015)
    parser.add_argument("--retention-seek", type=float, default=0.002)
    parser.add_argument("--retention-all-digit-increment", type=float, default=0.0)
    parser.add_argument("--prelatch-seek", type=float, default=0.003)
    parser.add_argument("--hold-seek", type=float, default=0.001)
    parser.add_argument("--missing-hysteresis", type=int, default=4)
    parser.add_argument("--trim-limit", type=float, default=0.8)
    parser.add_argument("--pick-speed", type=float, default=0.02)
    parser.add_argument("--stem-tensile-limit", type=float, default=0.70)
    parser.add_argument("--stem-critical-duration", type=float, default=0.20)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Allow a sub-1024 diagnostic run that cannot sign off Stage 0")
    parser.add_argument("--resume", action="store_true", help="Reuse only validated, complete shard JSONL files")
    args = parser.parse_args()
    if args.scenes < 1024 and not args.preflight:
        raise ValueError("formal Stage-0 gate requires at least 1024 scenes")
    if args.scenes % args.envs_per_shard:
        raise ValueError("--scenes must be divisible by --envs-per-shard")

    output = args.output_dir.resolve()
    shards_dir = output / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "Y",
            "MPLCONFIGDIR": "/tmp/mpl",
            "PYTHONPATH": str(HERE.parents[1]),
        }
    )
    all_records: list[dict] = []
    shard_count = args.scenes // args.envs_per_shard
    for shard in range(shard_count):
        seed = args.base_seed + shard
        shard_path = shards_dir / f"seed_{seed}.jsonl"
        if args.resume and shard_path.exists():
            with shard_path.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            valid = (
                len(records) == args.envs_per_shard
                and all(int(record.get("seed", -1)) == seed for record in records)
                and {int(record.get("env", -1)) for record in records}
                == set(range(args.envs_per_shard))
            )
            if valid:
                all_records.extend(records)
                print(
                    f"[{shard + 1}/{shard_count}] seed={seed} "
                    f"success={sum(bool(r['success']) for r in records)}/{len(records)} (validated resume)",
                    flush=True,
                )
                continue
        command = [
            str(args.python),
            str(HERE / "stage0_gate_shard.py"),
            "--headless",
            "--device",
            args.device,
            "--num-envs",
            str(args.envs_per_shard),
            "--seed",
            str(seed),
            "--max-steps",
            str(args.max_steps),
            "--settle-steps",
            str(args.settle_steps),
            "--residual-span-fraction",
            str(args.residual_span_fraction),
            "--position-jitter-xy-m",
            str(args.position_jitter_xy_m),
            "--friction", str(args.friction),
            "--palm-rim-height-m", str(args.palm_rim_height_m),
            "--retention-seek", str(args.retention_seek),
            "--retention-all-digit-increment", str(args.retention_all_digit_increment),
            "--prelatch-seek", str(args.prelatch_seek),
            "--hold-seek", str(args.hold_seek),
            "--missing-hysteresis", str(args.missing_hysteresis),
            "--trim-limit", str(args.trim_limit),
            "--pick-speed", str(args.pick_speed),
            "--stem-tensile-limit", str(args.stem_tensile_limit),
            "--stem-critical-duration", str(args.stem_critical_duration),
            "--output",
            str(shard_path),
        ]
        completed = subprocess.run(
            command,
            cwd=HERE,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (shards_dir / f"seed_{seed}.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        if completed.returncode != 0 or not shard_path.exists():
            raise RuntimeError(
                f"Stage-0 shard seed {seed} failed with code {completed.returncode}; "
                f"see {shards_dir / f'seed_{seed}.log'}"
            )
        with shard_path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if len(records) != args.envs_per_shard:
            raise RuntimeError(f"seed {seed} returned {len(records)} records")
        all_records.extend(records)
        print(
            f"[{shard + 1}/{shard_count}] seed={seed} "
            f"success={sum(bool(r['success']) for r in records)}/{len(records)}",
            flush=True,
        )

    records_path = output / "stage0_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            record["failure_class"] = classify(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    successes = sum(bool(record["success"]) for record in all_records)
    success_rate = successes / len(all_records)
    failures = Counter(classify(record) for record in all_records if not record["success"])
    nearest_failure = max(
        (record for record in all_records if not record["success"]),
        key=lambda record: record["post_break_hold_steps"],
        default=None,
    )
    representative_success = next(
        (record for record in all_records if record["success"]), None
    )
    snapshot = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "stage0_scripted_physx_preflight" if args.preflight else "stage0_scripted_physx_v1",
        "scenes": args.scenes,
        "envs_per_shard": args.envs_per_shard,
        "base_seed": args.base_seed,
        "max_steps": args.max_steps,
        "settle_steps": args.settle_steps,
        "residual_span_fraction": args.residual_span_fraction,
        "device": args.device,
        "axis_range_m": [0.032, 0.035],
        "isotropic_sphere_first_version": True,
        "position_jitter_m": [
            args.position_jitter_xy_m,
            args.position_jitter_xy_m,
            min(0.001, 0.5 * args.position_jitter_xy_m),
        ],
        "friction": args.friction,
        "palm_rim_height_m": args.palm_rim_height_m,
        "retention_seek": args.retention_seek,
        "retention_all_digit_increment": args.retention_all_digit_increment,
        "prelatch_seek": args.prelatch_seek,
        "hold_seek": args.hold_seek,
        "missing_hysteresis": args.missing_hysteresis,
        "trim_limit": args.trim_limit,
        "pick_speed_m_s": args.pick_speed,
        "stem_tensile_limit_n": args.stem_tensile_limit,
        "stem_critical_duration_s": args.stem_critical_duration,
        "physics_hz": 120,
        "policy_hz": 20,
        "sensor_fault_injection": False,
        "required_success_rate": 0.95,
        "code_sha256": {name: sha256(HERE / name) for name in CODE_FILES},
    }
    summary = {
        "passed": (not args.preflight) and success_rate >= 0.95,
        "preflight_threshold_met": success_rate >= 0.95,
        "successes": successes,
        "scenes": len(all_records),
        "success_rate": success_rate,
        "required_success_rate": 0.95,
        "failure_counts": dict(sorted(failures.items())),
        "records_sha256": sha256(records_path),
        "representative_success": representative_success,
        "nearest_failure": nearest_failure,
    }
    (output / "config_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("STAGE0_GATE=" + json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
