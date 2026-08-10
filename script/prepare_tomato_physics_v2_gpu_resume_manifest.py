"""Prepare a CPU-only, fail-closed manifest for a future powered GPU session."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.tomato_pick.curriculum_v2 import DEFAULT_CURRICULUM_PATH, load_curriculum_v2
from simulation.tomato_pick.data_protocol_v2 import (
    DEFAULT_DATA_PROTOCOL_PATH,
    load_data_protocol_v2,
)
from simulation.tomato_pick.experiment_v2 import (
    DEFAULT_EXPERIMENT_MATRIX_PATH,
    load_experiment_matrix_v2,
)
from simulation.tomato_pick.physics_v2 import PHYSICS_V2_PROFILE_PATH, load_physics_v2_profile


SOURCE_FILES = (
    PHYSICS_V2_PROFILE_PATH,
    DEFAULT_CURRICULUM_PATH,
    DEFAULT_DATA_PROTOCOL_PATH,
    DEFAULT_EXPERIMENT_MATRIX_PATH,
    ROOT / "simulation/tomato_pick/physics_v2.py",
    ROOT / "simulation/tomato_pick/physics_v2_models.py",
    ROOT / "simulation/tomato_pick/physics_v2_scenarios.py",
    ROOT / "simulation/tomato_pick/isaac/physics_v2/env.py",
    ROOT / "simulation/tomato_pick/isaac/physics_v2/env_cfg.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_resume_manifest(cpu_manifest: Path | None = None) -> dict[str, object]:
    profile = load_physics_v2_profile()
    curriculum = load_curriculum_v2()
    data_protocol = load_data_protocol_v2()
    matrix = load_experiment_matrix_v2()
    cpu = None
    if cpu_manifest is not None:
        cpu = json.loads(cpu_manifest.read_text(encoding="utf-8"))
    source_identity = {
        str(path.relative_to(ROOT)): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in SOURCE_FILES
    }
    cpu_passed = isinstance(cpu, dict) and cpu.get("passed") is True
    clean = _git("status", "--porcelain") == ""
    checks = {
        "physics_profile_valid": profile["profile_id"] == "tomato_pick_physics_v2",
        "formal_training_fail_closed": profile["implementation_boundary"]["training_allowed"] is False,
        "curriculum_is_inactive": curriculum["status"] == "draft_not_runtime_active",
        "data_protocol_is_valid_and_inactive": data_protocol["status"] == "cpu_authored_inactive",
        "experiment_matrix_is_inactive": matrix["status"] == "inactive_cpu_authored_protocol",
        "all_source_files_present": all(path.is_file() for path in SOURCE_FILES),
        "cpu_qualification_manifest_passed": cpu_passed,
        "git_worktree_clean": clean,
    }
    return {
        "schema_version": 1,
        "manifest_id": "tomato_pick_physics_v2_future_gpu_resume",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "CPU preparation only; no GPU query, Isaac launch, W&B connection, or hardware",
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "worktree_clean": clean,
        },
        "source_identity": source_identity,
        "cpu_qualification": {
            "manifest": str(cpu_manifest.resolve()) if cpu_manifest else None,
            "manifest_sha256": _sha256(cpu_manifest) if cpu_manifest else None,
            "passed": cpu_passed,
        },
        "static_checks": checks,
        "static_preparation_complete": all(checks.values()),
        "external_checks_pending": [
            "high_power_supply_connected_and_stable",
            "gpu_driver_and_cuda_runtime_available",
            "minimum_disk_space_available",
            "wandb_identity_verified_if_online_logging_is_requested",
            "64_environment_integrated_smoke_passes_on_exact_runtime",
            "256_environment_four_update_numerical_probe_passes",
            "independent_test_split_metrics_and_checkpoint_hash_are_recorded",
        ],
        "gpu_preflight_authorized": False,
        "ppo_training_authorized": False,
        "curriculum_activation_authorized": False,
        "hardware_connection_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_resume_manifest(args.cpu_manifest)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "static_preparation_complete": report["static_preparation_complete"]}))
    # Missing a CPU manifest or a dirty tree is a failed static handoff, even
    # though no GPU action would be authorized in either case.
    return 0 if report["static_preparation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
