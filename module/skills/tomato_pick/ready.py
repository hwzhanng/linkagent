"""Training-distribution admission evidence for the PPO handoff pose."""

from __future__ import annotations

import math

from proto import Pose, TomatoGraspPlan, TomatoTarget

from .geometry import quaternion_conjugate, rotate_vector


def evaluate_ppo_ready(
    *,
    target: TomatoTarget,
    plan: TomatoGraspPlan,
    actual_palm_pose: Pose,
    closed_grasp_fruit_center_m: tuple[float, ...] = (0.148, -0.0607, 0.0535),
) -> dict[str, object]:
    """Prove that the actual handoff lies inside the policy's reset envelope."""

    if len(
        {
            target.center_pose.frame_id,
            plan.pregrasp_pose.frame_id,
            actual_palm_pose.frame_id,
        }
    ) != 1:
        raise ValueError("target, plan, and actual palm pose must share a frame")
    palm_to_fruit_world = tuple(
        fruit - palm
        for fruit, palm in zip(
            target.center_pose.position_m, actual_palm_pose.position_m
        )
    )
    fruit_in_palm = rotate_vector(
        quaternion_conjugate(actual_palm_pose.quaternion_wxyz),
        palm_to_fruit_world,
    )
    fruit_in_grasp = rotate_vector(
        quaternion_conjugate(plan.grasp_frame.quaternion_wxyz),
        palm_to_fruit_world,
    )
    closed = tuple(float(value) for value in closed_grasp_fruit_center_m)
    approach_distance = fruit_in_grasp[0] - closed[0]
    lateral_radius = math.sqrt(
        (fruit_in_grasp[1] - closed[1]) ** 2
        + (fruit_in_grasp[2] - closed[2]) ** 2
    )
    orientation_dot = abs(
        sum(
            a * b
            for a, b in zip(
                actual_palm_pose.quaternion_wxyz,
                plan.pregrasp_pose.quaternion_wxyz,
            )
        )
    )
    orientation_error_deg = math.degrees(
        2.0 * math.acos(max(-1.0, min(1.0, orientation_dot)))
    )
    position_error_m = math.sqrt(
        sum(
            (actual - expected) ** 2
            for actual, expected in zip(
                actual_palm_pose.position_m, plan.pregrasp_pose.position_m
            )
        )
    )
    checks = {
        "fruit_axis_training_range": all(
            0.025 <= value <= 0.035 for value in target.semi_axes_m
        ),
        # Isaac's collision-free reset starts with 55--75 mm sampled clearance
        # and may move farther only along the same approach ray to clear the
        # authored hand shells. The released policy was trained on that final,
        # collision-expanded state, whose calibrated upper bound is 160 mm.
        "approach_distance_training_range": 0.055 <= approach_distance <= 0.160,
        "lateral_offset_training_range": 0.005 <= lateral_radius <= 0.020,
        "orientation_training_range": orientation_error_deg <= 7.0,
        "position_tracking_error": position_error_m <= 0.005,
        "target_plan_sequence_bound": (
            target.observation_sequence == plan.source_observation_sequence
        ),
    }
    return {
        "admitted": all(checks.values()),
        "checks": checks,
        "fruit_center_in_palm_m": fruit_in_palm,
        "fruit_center_in_grasp_m": fruit_in_grasp,
        "fruit_semi_axes_m": target.semi_axes_m,
        "pull_axis_in_palm": rotate_vector(
            quaternion_conjugate(actual_palm_pose.quaternion_wxyz),
            target.stem_direction_xyz,
        ),
        "approach_distance_m": approach_distance,
        "lateral_offset_m": lateral_radius,
        "orientation_error_deg": orientation_error_deg,
        "position_error_m": position_error_m,
        "ppo_visual_input_layout": (
            "fruit_center_in_palm[3]",
            "fruit_semi_axes[3]",
            "pull_axis_in_palm[3]",
        ),
    }


__all__ = ["evaluate_ppo_ready"]
