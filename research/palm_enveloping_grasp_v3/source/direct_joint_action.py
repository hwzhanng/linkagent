"""V3 14-D action contract with independent thumb and fingertip flexion."""

from __future__ import annotations

import torch

from simulation.ppo_tomato_grasp.hand_manifold import LEFT_SAFE_LOWER_RAD, LEFT_SAFE_UPPER_RAD


# Widened joint envelope for the simulation-only ``direct_flexion14_v1``
# contract.  The vendor URDF (and the imported PhysX USD) allow proximal
# flexion (joint1) to 1.66 rad and distal flexion (joint2) to 1.36 rad, but
# the inherited LEFT_SAFE_* ranges are the *hardware-calibrated* conservative
# envelope (~5% inside the device's validated raw ranges), which truncates
# finger curl by ~0.3 rad per flexion layer.  For the offline PhysX task
# (hardware_command_allowed=false) the flexion envelope is widened back toward
# the URDF limits with ~5% margin so the fingers can wrap a 90 mm fruit.
# This is a simulation-only kinematic prior and must never be sent to hardware.
V3_DIRECT_LOWER_RAD = list(LEFT_SAFE_LOWER_RAD)
_V3_DIRECT_UPPER = list(LEFT_SAFE_UPPER_RAD)
for _joint_id in (1, 2, 3, 4):
    _V3_DIRECT_UPPER[_joint_id] = 1.58
for _joint_id in (12, 13, 14, 15):
    _V3_DIRECT_UPPER[_joint_id] = 1.29
# The four ab/adduction axes (joint0) are also widened from the narrow
# hardware envelope toward the vendor URDF [-0.26, 0.26] so the fingers can
# adduct/abduct around a 90 mm sphere (index adduction is the binding limit).
for _joint_id in (6, 7, 8, 9):
    V3_DIRECT_LOWER_RAD[_joint_id] = -0.24
    _V3_DIRECT_UPPER[_joint_id] = 0.24
V3_DIRECT_LOWER_RAD = tuple(V3_DIRECT_LOWER_RAD)
V3_DIRECT_UPPER_RAD = tuple(_V3_DIRECT_UPPER)


V3_DIRECT_ACTION_DIM = 14
V3_DIRECT_ACTION_ORDER = (
    "thumb_joint2", "thumb_joint0", "thumb_joint1", "thumb_joint3",
    "index_joint1", "middle_joint1", "ring_joint1", "little_joint1",
    "index_joint2", "middle_joint2", "ring_joint2", "little_joint2",
    "finger_fan", "outer_inner_spread",
)
_DIRECT_JOINT_IDS = (0, 5, 10, 11, 1, 2, 3, 4, 12, 13, 14, 15)
_SPREAD_JOINT_IDS = (6, 7, 8, 9)
_FAN_PATTERN = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
_OUTER_INNER_PATTERN = (1.0, -1.0, -1.0, 1.0)


def v3_direct_action_to_joint_targets(action: torch.Tensor) -> torch.Tensor:
    """Map 14 normalized coordinates to the canonical 16 active joints.

    Twelve high-impact flexion axes are independent.  The remaining four
    narrow ab/adduction axes use two explicit, linearly independent patterns.
    This removes the inherited global-flexion coupling that made the restored
    four-axis thumb target unrepresentable.
    """

    if action.ndim < 1 or action.shape[-1] != V3_DIRECT_ACTION_DIM:
        raise ValueError("v3 direct action must end in 14 values")
    if not torch.isfinite(action).all():
        raise ValueError("v3 direct action contains non-finite values")
    bounded = action.clamp(-1.0, 1.0)
    lower = bounded.new_tensor(V3_DIRECT_LOWER_RAD)
    upper = bounded.new_tensor(V3_DIRECT_UPPER_RAD)
    span = upper - lower
    result = (0.5 * (lower + upper)).expand(*bounded.shape[:-1], 16).clone()
    direct_ids = list(_DIRECT_JOINT_IDS)
    unit = 0.5 * (bounded[..., :12] + 1.0)
    result[..., direct_ids] = lower[direct_ids] + unit * span[direct_ids]
    fan = bounded[..., 12].unsqueeze(-1)
    outer_inner = bounded[..., 13].unsqueeze(-1)
    spread_unit = 0.5 + 0.32 * (
        fan * bounded.new_tensor(_FAN_PATTERN)
        + outer_inner * bounded.new_tensor(_OUTER_INNER_PATTERN)
    )
    spread_ids = list(_SPREAD_JOINT_IDS)
    result[..., spread_ids] = lower[spread_ids] + spread_unit.clamp(0.0, 1.0) * span[spread_ids]
    return torch.maximum(lower, torch.minimum(upper, result))


def joint_targets_to_v3_direct_action(targets: torch.Tensor) -> torch.Tensor:
    """Least-squares inverse used to seed the event teacher."""

    if targets.ndim < 1 or targets.shape[-1] != 16:
        raise ValueError("joint targets must end in 16 values")
    if not torch.isfinite(targets).all():
        raise ValueError("joint targets contain non-finite values")
    lower = targets.new_tensor(V3_DIRECT_LOWER_RAD)
    upper = targets.new_tensor(V3_DIRECT_UPPER_RAD)
    bounded = torch.maximum(lower, torch.minimum(upper, targets))
    direct_ids = list(_DIRECT_JOINT_IDS)
    direct = 2.0 * (bounded[..., direct_ids] - lower[direct_ids]) / (upper[direct_ids] - lower[direct_ids]) - 1.0
    spread_ids = list(_SPREAD_JOINT_IDS)
    normalized = (bounded[..., spread_ids] - lower[spread_ids]) / (upper[spread_ids] - lower[spread_ids])
    rhs = (normalized - 0.5) / 0.32
    design = targets.new_tensor((_FAN_PATTERN, _OUTER_INNER_PATTERN)).T
    solution = torch.linalg.lstsq(design.expand(*rhs.shape[:-1], 4, 2), rhs.unsqueeze(-1)).solution.squeeze(-1)
    return torch.cat((direct, solution.clamp(-1.0, 1.0)), dim=-1)


__all__ = [
    "V3_DIRECT_ACTION_DIM", "V3_DIRECT_ACTION_ORDER",
    "V3_DIRECT_LOWER_RAD", "V3_DIRECT_UPPER_RAD",
    "joint_targets_to_v3_direct_action", "v3_direct_action_to_joint_targets",
]
