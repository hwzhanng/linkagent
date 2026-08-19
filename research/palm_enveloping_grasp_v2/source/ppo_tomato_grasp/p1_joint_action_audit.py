"""Audit the hand action manifold before palm-supported enveloping PPO.

This is a CPU/GPU-independent structural audit.  It does not start Isaac Sim
or send any hardware command.  It measures which of the 10 policy channels
can move the four straight fingers' spread, root-flexion and distal-flexion
joints, and explicitly detects the current coupling where one digit residual
moves both root and distal joints together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_ACTIVE_JOINT_ORDER,
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    hybrid_action_to_joint_targets,
)


FINGER_NAMES = ("index", "middle", "ring", "little")
FINGER_GROUPS = {
    name: {
        # Active order is [thumb/finger joint1] then [spread joint0] then
        # [thumb joint1/3 and finger joint2].  joint1 is the main root-flexion
        # axis, joint2 is the distal/fingertip flexion axis, and joint0 is the
        # lateral spread axis for each straight finger.
        "root_flexion": 1 + index,
        "spread": 6 + index,
        "distal_flexion": 12 + index,
    }
    for index, name in enumerate(FINGER_NAMES)
}


def _rank(matrix: torch.Tensor, tolerance: float = 1.0e-7) -> int:
    singular = torch.linalg.svdvals(matrix)
    if singular.numel() == 0:
        return 0
    threshold = float(singular.max()) * tolerance
    return int((singular > threshold).sum().item())


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, int, bool, str)) or value is None:
        return value
    raise TypeError(type(value).__name__)


def audit(*, samples: int, seed: int, device: str) -> dict[str, object]:
    if samples < 128:
        raise ValueError("samples must be at least 128")
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    action_samples = torch.empty((samples, 10), device=torch_device).uniform_(-1.0, 1.0, generator=generator)
    targets = hybrid_action_to_joint_targets(action_samples)
    lower = torch.tensor(LEFT_SAFE_LOWER_RAD, device=torch_device)
    upper = torch.tensor(LEFT_SAFE_UPPER_RAD, device=torch_device)
    baseline_action = torch.zeros((1, 10), device=torch_device)
    baseline = hybrid_action_to_joint_targets(baseline_action)[0]

    # Local finite-difference Jacobian at neutral action.  The map is
    # piecewise affine; this exposes both independent channels and structural
    # coupling without needing Isaac Sim.
    epsilon = 1.0e-4
    jacobian = torch.zeros((16, 10), device=torch_device)
    for column in range(10):
        plus = baseline_action.clone()
        minus = baseline_action.clone()
        plus[0, column] += epsilon
        minus[0, column] -= epsilon
        jacobian[:, column] = (
            hybrid_action_to_joint_targets(plus)[0]
            - hybrid_action_to_joint_targets(minus)[0]
        ) / (2.0 * epsilon)

    rows: dict[str, object] = {}
    for joint_index, name in enumerate(LEFT_ACTIVE_JOINT_ORDER):
        rows[name] = {
            "index": joint_index,
            "safe_range_rad": [float(lower[joint_index]), float(upper[joint_index])],
            "sampled_target_min_rad": float(targets[:, joint_index].min()),
            "sampled_target_max_rad": float(targets[:, joint_index].max()),
            "sampled_target_span_rad": float(targets[:, joint_index].max() - targets[:, joint_index].min()),
            "jacobian_abs_by_action": jacobian[joint_index].abs(),
            "jacobian_l2": float(torch.linalg.vector_norm(jacobian[joint_index])),
        }

    digit_report: dict[str, object] = {}
    for digit, group in FINGER_GROUPS.items():
        root = group["root_flexion"]
        distal = group["distal_flexion"]
        spread = group["spread"]
        # Action layout is [five coarse synergies, thumb, index, middle,
        # ring, little residual].  The straight-finger residuals therefore
        # start at column 6; column 5 is the thumb residual.
        residual_column = 6 + FINGER_NAMES.index(digit)
        root_delta = (targets[:, root] - baseline[root]).abs()
        distal_delta = (targets[:, distal] - baseline[distal]).abs()
        residual_jacobian = jacobian[[root, distal], residual_column]
        digit_report[digit] = {
            "joint_indices": group,
            "root_jacobian_l2": float(torch.linalg.vector_norm(jacobian[root])),
            "distal_jacobian_l2": float(torch.linalg.vector_norm(jacobian[distal])),
            "spread_jacobian_l2": float(torch.linalg.vector_norm(jacobian[spread])),
            "sampled_root_abs_delta_mean_rad": float(root_delta.mean()),
            "sampled_distal_abs_delta_mean_rad": float(distal_delta.mean()),
            "sampled_distal_to_root_delta_ratio": float(
                distal_delta.mean() / root_delta.mean().clamp_min(1.0e-12)
            ),
            "residual_action_column": residual_column,
            "residual_root_distal_jacobian": residual_jacobian,
            "residual_moves_root_and_distal_together": bool(
                residual_jacobian.abs().min() > 1.0e-8
            ),
            "distal_only_residual_sensitivity": 0.0,
            "distal_action_columns_rank": _rank(jacobian[distal : distal + 1, 5:10]),
        }

    four_root_distal_rows = [
        row
        for digit in FINGER_NAMES
        for row in (FINGER_GROUPS[digit]["root_flexion"], FINGER_GROUPS[digit]["distal_flexion"])
    ]
    distal_rows = [FINGER_GROUPS[digit]["distal_flexion"] for digit in FINGER_NAMES]
    root_rows = [FINGER_GROUPS[digit]["root_flexion"] for digit in FINGER_NAMES]
    residual_block = jacobian[distal_rows, 5:10]
    report = {
        "schema_version": 1,
        "audit": "p1_joint_action_manifold",
        "device": str(torch_device),
        "samples": samples,
        "seed": seed,
        "action_dim": 10,
        "joint_dim": 16,
        "active_joint_order": list(LEFT_ACTIVE_JOINT_ORDER),
        "global_jacobian_rank": _rank(jacobian),
        "four_finger_root_distal_jacobian_rank": _rank(jacobian[four_root_distal_rows]),
        "four_finger_distal_jacobian_rank": _rank(jacobian[distal_rows]),
        "four_finger_root_jacobian_rank": _rank(jacobian[root_rows]),
        "four_finger_distal_residual_block_rank": _rank(residual_block),
        "current_mapping_structural_finding": (
            "Each per-digit residual moves the corresponding root-flexion and distal-flexion "
            "target together; it provides no distal-only residual channel. The global 10D map "
            "can move distal joints through global curl-shape, but cannot independently decide "
            "'keep root, curl fingertip' for each straight finger."
        ),
        "minimal_candidate_action_upgrade": {
            "action_dim": 14,
            "layout": "5 coarse synergies + 5 existing digit residuals + 4 straight-finger distal-only residuals",
            "new_channels": [f"{digit}_distal_residual" for digit in FINGER_NAMES],
            "reason": "Preserve old semantics while giving index/middle/ring/little distal joints independent tactile control.",
            "must_be_validated_before_training": True,
        },
        "joints": rows,
        "digits": digit_report,
        "hardware_command_allowed": False,
        "isaac_or_cuda_required": False,
    }
    return _jsonable(report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(samples=args.samples, seed=args.seed, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
