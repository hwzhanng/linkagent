"""Frozen nominal reset parameters for the local tomato-pick skill.

This module deliberately has no Isaac, Torch, SDK, or Agent dependency.  The
values describe the state *after* global planning has brought the robot to the
local skill workspace.  They are not a whole-workspace exploration range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# Preserve enough elbow extension reserve for the local 6--9 cm search region.
# With R4=-0.8 the arm straightened before reaching the fruit and the position
# Jacobian became singular.  Increasing R4 flexion while compensating R6 keeps
# the hand orientation useful and leaves PPO genuine Cartesian authority.
# The left parking pose is a kinematic mirror, not a component-wise sign
# inversion because corresponding joint axes differ.
RIGHT_ARM_PREGRASP_JOINTS_RAD = (-0.40, -0.60, 0.0, -1.20, 0.0, 0.80, 0.0)
LEFT_ARM_PARK_JOINTS_RAD = (0.40, 0.60, 0.0, -1.20, 0.0, 0.80, 0.0)


@dataclass(frozen=True, slots=True)
class PreGraspResetSpec:
    """Single C2 pre-grasp reset contract; no alternate reset variants."""

    # The global planner has already aligned the grasp.  The local policy
    # starts 5 mm before first distal-pad contact.
    fruit_clearance_m: float = 0.005
    collision_retry_clearance_increment_m: float = 0.020
    maximum_collision_retries: int = 2
    small_fruit_lateral_radius_m: float = 0.025
    large_fruit_lateral_radius_m: float = 0.035
    small_fruit_closure: float = 0.22
    large_fruit_closure: float = 0.12
    thumb_opposition: float = 0.0
    finger_spread: float = 0.0
    minimum_robot_settle_steps: int = 4
    required_stable_steps: int = 4
    maximum_reset_steps: int = 150
    arm_joint_position_tolerance_rad: float = 0.030
    hand_joint_position_tolerance_rad: float = 0.250
    arm_joint_velocity_tolerance_rad_s: float = 0.100
    hand_joint_step_delta_tolerance_rad: float = 0.010
    fruit_linear_velocity_tolerance_m_s: float = 0.010
    fruit_angular_velocity_tolerance_rad_s: float = 0.100
    maximum_initial_contact_force_n: float = 0.10
    maximum_anchor_displacement_m: float = 0.020
    minimum_arm_joint_limit_margin_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.fruit_clearance_m <= 0.0:
            raise ValueError("fruit_clearance_m must be > 0")
        if self.collision_retry_clearance_increment_m <= 0.0:
            raise ValueError("collision_retry_clearance_increment_m must be > 0")
        if self.maximum_collision_retries < 0:
            raise ValueError("maximum_collision_retries must be >= 0")
        if not 0.0 < self.small_fruit_lateral_radius_m < self.large_fruit_lateral_radius_m:
            raise ValueError("fruit radius interval must be positive and increasing")
        if not 0.0 <= self.large_fruit_closure <= self.small_fruit_closure <= 1.0:
            raise ValueError("pre-grasp closures must be ordered inside [0, 1]")
        if self.minimum_robot_settle_steps < 1 or self.required_stable_steps < 1:
            raise ValueError("settle windows must be positive")
        if self.maximum_reset_steps <= self.minimum_robot_settle_steps + self.required_stable_steps:
            raise ValueError("maximum_reset_steps is too short for the settle windows")
        if not 0.0 < self.minimum_arm_joint_limit_margin_fraction < 0.5:
            raise ValueError("joint limit margin fraction must be in (0, 0.5)")

    def closure_for_lateral_radius(self, lateral_radius_m: float) -> float:
        """Return a conservative pre-shape: smaller fruit, slightly more closure."""

        radius = float(lateral_radius_m)
        unit = (radius - self.small_fruit_lateral_radius_m) / (
            self.large_fruit_lateral_radius_m - self.small_fruit_lateral_radius_m
        )
        unit = min(1.0, max(0.0, unit))
        return self.small_fruit_closure + unit * (
            self.large_fruit_closure - self.small_fruit_closure
        )


def minimum_joint_limit_margin_fraction(
    positions_rad: Sequence[float],
    lower_rad: Sequence[float],
    upper_rad: Sequence[float],
) -> float:
    """Return the smallest normalized distance from any hard joint limit."""

    if not (len(positions_rad) == len(lower_rad) == len(upper_rad)) or not positions_rad:
        raise ValueError("joint position and limit sequences must have equal non-zero length")
    margins: list[float] = []
    for position, lower, upper in zip(positions_rad, lower_rad, upper_rad):
        if upper <= lower:
            raise ValueError("joint upper limit must exceed lower limit")
        margins.append(min(position - lower, upper - position) / (upper - lower))
    return min(margins)


PREGRASP_RESET_SPEC = PreGraspResetSpec()


__all__ = [
    "LEFT_ARM_PARK_JOINTS_RAD",
    "PREGRASP_RESET_SPEC",
    "RIGHT_ARM_PREGRASP_JOINTS_RAD",
    "PreGraspResetSpec",
    "minimum_joint_limit_margin_fraction",
]
