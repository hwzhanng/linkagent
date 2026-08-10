"""Isaac adapter for the exact frozen PPO reset/reference contract."""

from __future__ import annotations

import math

from proto import (
    ArmState,
    HandState,
    HarvestContainer,
    Pose,
    TomatoGraspPlan,
    TomatoTarget,
)

from .robotics import IsaacHarvestRoboticsBundle


def _basis_quaternion(matrix) -> tuple[float, float, float, float]:
    m00, m01, m02 = (float(matrix[0, index]) for index in range(3))
    m10, m11, m12 = (float(matrix[1, index]) for index in range(3))
    m20, m21, m22 = (float(matrix[2, index]) for index in range(3))
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)
    norm = math.sqrt(sum(value * value for value in q))
    return tuple(value / norm for value in q)  # type: ignore[return-value]


class IsaacTrainingDistributionGraspPlanner:
    """Bind the wrapper runner to the same reference used during PPO training.

    The simulated reset reference stands in for a collision-aware global
    planner/hardware calibration. No policy action, reward, or outcome state is
    consumed here.
    """

    def __init__(
        self,
        bundle: IsaacHarvestRoboticsBundle,
        *,
        plan_ttl_s: float = 30.0,
        mapping_calibration_id: str = "g20_eigengrasp_5d_v3_thumb_roll_fixed",
    ) -> None:
        self._bundle = bundle
        self._ttl_s = float(plan_ttl_s)
        self._mapping_id = mapping_calibration_id
        if self._ttl_s <= 0.0:
            raise ValueError("plan_ttl_s must be > 0")

    def plan(
        self,
        target: TomatoTarget,
        arm_state: ArmState,
        hand_state: HandState,
    ) -> TomatoGraspPlan:
        if arm_state.side is not hand_state.side or arm_state.side is not self._bundle.side:
            raise ValueError("Isaac plan requires matching configured arm and hand")
        basis = self._bundle.env._grasp_basis_w[0]
        grasp_q = _basis_quaternion(basis)
        reference = self._bundle.reference_pose
        grasp_frame = Pose(target.center_pose.position_m, grasp_q, "world")
        pull = tuple(float(value) for value in basis[:, 2])
        retreat = Pose(
            tuple(
                value + 0.08 * axis
                for value, axis in zip(reference.position_m, pull, strict=True)
            ),
            reference.quaternion_wxyz,
            "world",
        )
        now_ns = self._bundle.clock.monotonic_ns()
        return TomatoGraspPlan(
            grasp_plan_id=(
                f"plan-{target.target_id}-{arm_state.side.value}-"
                f"{target.observation_sequence}"
            ),
            target_id=target.target_id,
            source_observation_sequence=target.observation_sequence,
            arm_side=arm_state.side,
            grasp_frame=grasp_frame,
            pregrasp_pose=reference,
            # The frozen policy clamps residual control around the reset EE
            # reference; this field is that reference, despite its legacy name.
            contact_pose=reference,
            retreat_pose=retreat,
            max_residual_translation_m=tuple(
                float(value) for value in self._bundle.env.cfg.max_residual_translation_m
            ),
            max_residual_rotation_rad=tuple(
                float(value) for value in self._bundle.env.cfg.max_residual_rotation_rad
            ),
            mapping_calibration_id=self._mapping_id,
            planner_version="ppo_training_distribution_ready_v1",
            expires_monotonic_ns=max(1, now_ns + int(self._ttl_s * 1.0e9)),
        )


def plan_container_for_retained_payload(
    bundle: IsaacHarvestRoboticsBundle,
    *,
    container_id: str = "isaac-harvest-bin-01",
    capacity_remaining: int = 20,
) -> HarvestContainer:
    """Place the live fruit, rather than the palm, over the bin center."""

    arm = bundle.arm.get_state(bundle.side)
    fruit = bundle.env._tomato.data.root_pos_w.torch[0]
    fruit_offset = tuple(
        float(fruit[index]) - arm.ee_pose.position_m[index] for index in range(3)
    )
    desired_fruit = bundle.env.container_placement_position_w
    desired_palm = tuple(
        desired_fruit[index] - fruit_offset[index] for index in range(3)
    )
    return HarvestContainer(
        container_id=container_id,
        placement_pose=Pose(
            desired_palm, arm.ee_pose.quaternion_wxyz, "world"
        ),
        available=True,
        capacity_remaining=capacity_remaining,
    )


__all__ = [
    "IsaacTrainingDistributionGraspPlanner",
    "plan_container_for_retained_payload",
]
