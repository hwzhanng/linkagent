"""Run the complete Physics-v2 analytic qualification with CUDA hidden.

The command set is an allow-list.  It intentionally excludes Isaac/PhysX
integration, PPO training/evaluation, W&B, robot SDKs, and live monitoring.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "logs/qualification/tomato_physics_v2/cpu_offline"
TEST_FILES = (
    "tests/test_tomato_physics_v2.py",
    "tests/test_tomato_physics_v2_models.py",
    "tests/test_tomato_physics_v2_artifact_audit.py",
    "tests/test_tomato_physics_v2_sensitivity.py",
    "tests/test_tomato_physics_v2_curriculum.py",
    "tests/test_tomato_physics_v2_experiment.py",
)


def qualification_commands(python: Path, output: Path) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    for step in range(1, 8):
        commands.append(
            (
                f"step_{step}",
                [
                    str(python),
                    str(ROOT / f"script/check_tomato_physics_v2_step{step}.py"),
                    "--output",
                    str(output / f"step_{step}.json"),
                ],
            )
        )
    commands.extend(
        (
            (
                "parameter_sensitivity",
                [
                    str(python),
                    str(ROOT / "script/audit_tomato_physics_v2_parameter_sensitivity.py"),
                    "--samples",
                    "50000",
                    "--seed",
                    "20260810",
                    "--output",
                    str(output / "parameter_sensitivity.json"),
                    "--markdown",
                    str(output / "parameter_sensitivity.md"),
                ],
            ),
            (
                "reward_counterfactuals",
                [
                    str(python),
                    str(ROOT / "script/audit_tomato_physics_v2_reward_counterfactuals.py"),
                    "--output",
                    str(output / "reward_counterfactuals.json"),
                ],
            ),
            (
                "pytest_cpu",
                [str(python), "-m", "pytest", "-q", *TEST_FILES],
            ),
        )
    )
    return commands


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if args.timeout_seconds < 30:
        raise ValueError("timeout-seconds must be at least 30")
    # Do not resolve this path: managed virtual environments may expose their
    # interpreter as a symlink, and dereferencing it loses the environment's
    # site-packages identity (including its CPU-capable Torch installation).
    python = args.python.expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python}")
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["WANDB_MODE"] = "disabled"
    environment["OMNI_KIT_ACCEPT_EULA"] = "NO"
    # These variables are not required by analytic gates and are removed so a
    # locally configured robot endpoint cannot accidentally leak into a child.
    for name in tuple(environment):
        if name.startswith(("LBOT_", "LINKERHAND_", "ROS_")):
            environment.pop(name, None)

    results = []
    all_passed = True
    for name, command in qualification_commands(python, output):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        log = output / f"{name}.log"
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        passed = completed.returncode == 0
        all_passed &= passed
        results.append(
            {
                "name": name,
                "command": command,
                "returncode": completed.returncode,
                "passed": passed,
                "log": str(log.resolve()),
                "log_sha256": _sha256(log),
            }
        )
        if not passed:
            break

    artifacts = []
    for path in sorted(output.glob("*.json")):
        if path.name == "manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "qualification_id": "tomato_pick_physics_v2_cpu_offline",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "boundary": {
            "cuda_visible_devices": "empty",
            "wandb_disabled": True,
            "isaac_or_physx_commands": False,
            "ppo_training_or_evaluation": False,
            "robot_sdk_or_hardware": False,
        },
        "python": str(python),
        "commands": results,
        "artifacts": artifacts,
        "passed": all_passed and len(results) == len(qualification_commands(python, output)),
        "gpu_training_authorized": False,
        "curriculum_activation_authorized": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "passed": manifest["passed"]}))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
