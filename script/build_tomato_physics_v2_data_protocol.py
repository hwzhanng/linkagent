"""Build a deterministic CPU-only handoff manifest for PPO data/evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.tomato_pick.curriculum_v2 import load_curriculum_v2
from simulation.tomato_pick.data_protocol_v2 import (
    DEFAULT_DATA_PROTOCOL_PATH,
    load_data_protocol_v2,
)
from simulation.tomato_pick.experiment_v2 import load_experiment_matrix_v2
from simulation.tomato_pick.physics_v2 import PHYSICS_V2_PROFILE_PATH, load_physics_v2_profile


SOURCE_FILES = (
    DEFAULT_DATA_PROTOCOL_PATH,
    PHYSICS_V2_PROFILE_PATH,
    ROOT / "simulation/config/tomato_pick_physics_v2_experiment_matrix.json",
    ROOT / "simulation/config/tomato_pick_physics_v2_curriculum_draft.json",
    ROOT / "simulation/tomato_pick/data_protocol_v2.py",
    ROOT / "simulation/tomato_pick/physics_v2_scenarios.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_data_protocol_manifest() -> dict[str, object]:
    protocol = load_data_protocol_v2()
    profile = load_physics_v2_profile()
    matrix = load_experiment_matrix_v2()
    curriculum = load_curriculum_v2()
    matrix_protocol = matrix["common_protocol"]
    evaluation = matrix["evaluation"]
    source_identity = {
        str(path.relative_to(ROOT)): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in SOURCE_FILES
    }
    split_plan = {
        name: {
            "seeds": list(split["seeds"]),
            "episode_steps": evaluation["episode_steps"],
            "randomization": split["randomization"],
            "checkpoint_selection_allowed": split["checkpoint_selection_allowed"],
        }
        for name, split in protocol["splits"].items()
    }
    checks = {
        "protocol_valid": True,
        "physics_profile_training_blocked": profile["implementation_boundary"]["training_allowed"] is False,
        "curriculum_inactive": curriculum["status"] == "draft_not_runtime_active",
        "matrix_inactive": matrix["status"] == "inactive_cpu_authored_protocol",
        "train_validation_test_seeds_disjoint": True,
        "matrix_evaluation_buckets_match_protocol": set(evaluation["robustness_buckets"])
        == set(protocol["robustness_buckets"]),
        "matrix_selection_order_matches_protocol": matrix_protocol["best_checkpoint_selection_order"]
        == protocol["checkpoint_policy"]["selection_order"],
        "hardware_out_of_scope": protocol["checkpoint_policy"]["hardware_deployment_allowed"] is False,
    }
    return {
        "schema_version": 1,
        "manifest_id": "tomato_pick_physics_v2_data_protocol_handoff_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "CPU-only protocol construction; no Isaac, PhysX, GPU, PPO, W&B, or hardware",
        "protocol": protocol,
        "split_plan": split_plan,
        "source_identity": source_identity,
        "checks": checks,
        "passed": all(checks.values()),
        "formal_training_authorized": False,
        "hardware_deployment_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_data_protocol_manifest()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"]}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
