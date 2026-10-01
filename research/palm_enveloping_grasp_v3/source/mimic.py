"""URDF-sourced terminal-joint coupling lost during USD conversion."""

from __future__ import annotations

import torch


URDF_MIMIC_SPECS = {
    "thumb_joint4": ("thumb_joint3", 1.11, -0.04545),
    "index_joint3": ("index_joint2", 0.94, -0.04545),
    "middle_joint3": ("middle_joint2", 0.94, -0.04545),
    "ring_joint3": ("ring_joint2", 0.94, -0.04545),
    "little_joint3": ("little_joint2", 0.94, -0.04545),
}


def apply_urdf_mimic_targets(
    targets: torch.Tensor,
    joint_names: tuple[str, ...] | list[str],
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Return full articulation targets with the five URDF mimic equations."""

    if targets.ndim != 2 or targets.shape[-1] != len(joint_names):
        raise ValueError("targets and joint_names do not match")
    if lower.shape[-1] != len(joint_names) or upper.shape[-1] != len(joint_names):
        raise ValueError("joint limit tensors do not match joint_names")
    result = targets.clone()
    name_to_id = {name: index for index, name in enumerate(joint_names)}
    required = {name for child, spec in URDF_MIMIC_SPECS.items() for name in (child, spec[0])}
    missing = sorted(required - name_to_id.keys())
    if missing:
        raise ValueError(f"missing URDF mimic joints: {missing}")
    for child, (parent, multiplier, offset) in URDF_MIMIC_SPECS.items():
        child_id, parent_id = name_to_id[child], name_to_id[parent]
        value = float(multiplier) * result[:, parent_id] + float(offset)
        result[:, child_id] = value.clamp(lower[:, child_id], upper[:, child_id])
    return result


def advance_mimic_positions(
    current: torch.Tensor,
    desired: torch.Tensor,
    mimic_joint_ids: list[int] | tuple[int, ...],
    max_step_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance only mimic joints toward their coupled pose with a hard step bound.

    The imported USD gives the five terminal joints independent zero-position
    drives.  Merely writing their targets is therefore not equivalent to the
    vendor URDF mimic constraint.  This helper supplies the same bounded
    position-servo mechanism already used by the v2 active joints while leaving
    every non-mimic joint untouched.
    """

    if current.shape != desired.shape or current.ndim != 2:
        raise ValueError("current and desired must be matching [env, joint] tensors")
    if max_step_rad <= 0.0:
        raise ValueError("max_step_rad must be positive")
    result = current.clone()
    joint_ids = list(mimic_joint_ids)
    delta = (desired[:, joint_ids] - current[:, joint_ids]).clamp(
        -float(max_step_rad), float(max_step_rad)
    )
    result[:, joint_ids] = current[:, joint_ids] + delta
    return result, delta


__all__ = ["URDF_MIMIC_SPECS", "apply_urdf_mimic_targets", "advance_mimic_positions"]
