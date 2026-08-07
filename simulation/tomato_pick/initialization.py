"""One pre-grasp reset distribution for the local PPO skill."""

from __future__ import annotations

from dataclasses import dataclass


# Right-arm nominal pose obtained with the SDK-compatible 7-DoF IK model.
# The wrist is about 38 cm forward, 36 cm to the right and 12 cm below the
# shoulder mount, with a bent elbow and the open palm facing the task area.
RIGHT_ARM_PREGRASP_JOINTS_RAD = (0.859, -1.096, -1.045, -1.426, 1.285, -0.165, 0.930)
# The left arm is outside the single-arm PPO action space.  Keep every joint
# exactly at the SDK/robot zero position instead of inventing a second pose.
LEFT_ARM_PARK_JOINTS_RAD = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class PreGraspResetSpec:
    """The global planner has already produced a reachable local workspace."""

    # Fruit centre enclosed by a verified spherical G20 grasp, expressed in
    # the reset grasp frame relative to the palm.  This calibration comes from
    # a collision-enabled hand-shape grid, not a scripted policy action.
    closed_grasp_fruit_center_m: tuple[float, float, float] = (
        0.148,
        -0.0607,
        0.0535,
    )
    # Start outside that verified closure envelope.  PPO still owns the full
    # final approach and hand shape, and reset begins with no contact.
    fruit_distance_outside_grasp_m: tuple[float, float] = (0.055, 0.075)
    lateral_offset_range_m: tuple[float, float] = (0.005, 0.020)
    axial_noise_m: float = 0.005
    orientation_noise_deg: float = 7.0
    arm_joint_noise_deg: float = 3.0
    # G20's authored collision proxies are slimmer than its rendered shells.
    # A 25 mm conservative collider margin keeps the visible meshes separated.
    minimum_clearance_m: float = 0.015
    joint_limit_margin_fraction: float = 0.10

    def __post_init__(self) -> None:
        lower, upper = self.fruit_distance_outside_grasp_m
        if not 0.0 < lower < upper:
            raise ValueError("fruit distance range must be positive and increasing")
        if len(self.closed_grasp_fruit_center_m) != 3:
            raise ValueError("closed_grasp_fruit_center_m must contain xyz")
        lateral_lower, lateral_upper = self.lateral_offset_range_m
        if not 0.0 < lateral_lower < lateral_upper:
            raise ValueError("lateral offset range must be positive and increasing")
        if self.minimum_clearance_m <= 0.0:
            raise ValueError("minimum_clearance_m must be positive")
        if not 0.0 < self.joint_limit_margin_fraction < 0.5:
            raise ValueError("joint_limit_margin_fraction must lie in (0, 0.5)")


PREGRASP_RESET_SPEC = PreGraspResetSpec()


__all__ = [
    "LEFT_ARM_PARK_JOINTS_RAD",
    "PREGRASP_RESET_SPEC",
    "RIGHT_ARM_PREGRASP_JOINTS_RAD",
    "PreGraspResetSpec",
]
