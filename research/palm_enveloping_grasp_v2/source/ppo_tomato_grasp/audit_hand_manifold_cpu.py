#!/usr/bin/env python3
"""Audit the migrated hand action space without Isaac Sim or a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    MAPPING_ID,
    SOURCE_LEFT_CALIBRATION,
    SOURCE_LINKAGENT_MAPPER_SHA256,
    VALIDATION_STATUS,
    hybrid_action_to_joint_targets,
    synergy_to_joint_targets,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    action = 2.0 * torch.rand((args.samples, 10), generator=generator) - 1.0
    coarse_targets = synergy_to_joint_targets(action[:, :5])
    targets = hybrid_action_to_joint_targets(action)
    lower = torch.tensor(LEFT_SAFE_LOWER_RAD)
    upper = torch.tensor(LEFT_SAFE_UPPER_RAD)
    tolerance = 1.0e-7
    below = targets < lower - tolerance
    above = targets > upper + tolerance
    coarse_at_limit = ((coarse_targets - lower).abs() < 1.0e-6) | (
        (coarse_targets - upper).abs() < 1.0e-6
    )
    hybrid_at_limit = ((targets - lower).abs() < 1.0e-6) | (
        (targets - upper).abs() < 1.0e-6
    )

    origin = torch.zeros((1, 5))
    base = synergy_to_joint_targets(origin)
    columns = []
    for coordinate in range(5):
        probe = origin.clone()
        probe[0, coordinate] = 1.0e-3
        columns.append(((synergy_to_joint_targets(probe) - base) / 1.0e-3)[0])
    jacobian = torch.stack(columns, dim=1)

    report = {
        "schema_version": 1,
        "mapping_id": MAPPING_ID,
        "validation_status": VALIDATION_STATUS,
        "device": "cpu",
        "samples": args.samples,
        "seed": args.seed,
        "source": {
            "linkagent_mapper_sha256": SOURCE_LINKAGENT_MAPPER_SHA256,
            "left_motion_calibration": SOURCE_LEFT_CALIBRATION,
        },
        "checks": {
            "all_finite": bool(torch.isfinite(targets).all()),
            "inside_measured_safe_ranges": not bool((below | above).any()),
            "local_five_dimensional_rank": int(torch.linalg.matrix_rank(jacobian).item()),
            "maximum_coarse_limit_saturation_fraction": float(
                coarse_at_limit.float().mean(dim=0).max()
            ),
            "maximum_hybrid_limit_saturation_fraction": float(
                hybrid_at_limit.float().mean(dim=0).max()
            ),
        },
        "formal_training_allowed": False,
        "blocking_gates": [
            "left thumb side-swing and roll direction/zero visual audit",
            "URDF fingertip pose versus real hand landmark audit",
            "self-collision sweep with the left-hand meshes",
            "Isaac PhysX contact-force and mimic-joint validation",
        ],
    }
    report["passed_cpu_scope"] = (
        report["checks"]["all_finite"]
        and report["checks"]["inside_measured_safe_ranges"]
        and report["checks"]["local_five_dimensional_rank"] == 5
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed_cpu_scope"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
