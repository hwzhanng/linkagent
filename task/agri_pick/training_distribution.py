"""Planner and admission checks matching the PPO pre-grasp reset distribution."""

from __future__ import annotations

from module.robotics import Clock
from module.skills.tomato_pick.geometry import rotate_vector
from module.skills.tomato_pick.ready import evaluate_ppo_ready
from proto import ArmState, HandState, Pose, TomatoGraspPlan, TomatoTarget


def _add(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _sub(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


class TrainingDistributionTomatoGraspPlanner:
    """Place the palm at a deterministic point inside the trained reset envelope."""

    def __init__(
        self,
        clock: Clock,
        *,
        mapping_calibration_id: str = "g20_eigengrasp_5d_v3_thumb_roll_fixed",
        plan_ttl_s: float = 5.0,
        closed_grasp_fruit_center_m: tuple[float, ...] = (0.148, -0.0607, 0.0535),
        ready_approach_distance_m: float = 0.065,
        ready_lateral_offset_m: float = 0.0125,
        retreat_distance_m: float = 0.12,
    ) -> None:
        if not isinstance(clock, Clock):
            raise TypeError("clock must implement Clock")
        if not isinstance(mapping_calibration_id, str) or not mapping_calibration_id.strip():
            raise ValueError("mapping_calibration_id must be a non-empty string")
        if plan_ttl_s <= 0.0:
            raise ValueError("plan_ttl_s must be > 0")
        closed = tuple(float(value) for value in closed_grasp_fruit_center_m)
        if len(closed) != 3:
            raise ValueError("closed_grasp_fruit_center_m must contain xyz")
        if not 0.055 <= ready_approach_distance_m <= 0.075:
            raise ValueError("ready_approach_distance_m must stay in the PPO reset range")
        if not 0.005 <= ready_lateral_offset_m <= 0.020:
            raise ValueError("ready_lateral_offset_m must stay in the PPO reset range")
        if retreat_distance_m <= 0.0:
            raise ValueError("retreat_distance_m must be > 0")
        self._clock = clock
        self._mapping_id = mapping_calibration_id.strip()
        self._ttl_s = float(plan_ttl_s)
        self._closed_center = closed
        self._ready_distance = float(ready_approach_distance_m)
        self._ready_lateral = float(ready_lateral_offset_m)
        self._retreat_distance = float(retreat_distance_m)

    def plan(
        self,
        target: TomatoTarget,
        arm_state: ArmState,
        hand_state: HandState,
    ) -> TomatoGraspPlan:
        if not isinstance(target, TomatoTarget):
            raise TypeError("target must be a TomatoTarget")
        if not isinstance(arm_state, ArmState) or not isinstance(hand_state, HandState):
            raise TypeError("arm_state and hand_state must use frozen DTOs")
        if arm_state.side is not hand_state.side:
            raise ValueError("arm_state and hand_state must use the same side")
        frame_q = target.center_pose.quaternion_wxyz
        frame_id = target.center_pose.frame_id
        planned_pull = rotate_vector(frame_q, (0.0, 0.0, 1.0))
        alignment = sum(a * b for a, b in zip(planned_pull, target.stem_direction_xyz))
        if alignment < 0.98:
            raise ValueError("target grasp frame z+ is not aligned away from the plant")
        ready_center_local = (
            self._closed_center[0] + self._ready_distance,
            self._closed_center[1] + self._ready_lateral,
            self._closed_center[2],
        )
        contact_palm = _sub(
            target.center_pose.position_m,
            rotate_vector(frame_q, self._closed_center),
        )
        ready_palm = _sub(
            target.center_pose.position_m,
            rotate_vector(frame_q, ready_center_local),
        )
        retreat_palm = _add(
            contact_palm,
            rotate_vector(frame_q, (0.0, 0.0, self._retreat_distance)),
        )
        now_ns = self._clock.monotonic_ns()
        return TomatoGraspPlan(
            grasp_plan_id=(
                f"plan-{target.target_id}-{arm_state.side.value}-{target.observation_sequence}"
            ),
            target_id=target.target_id,
            source_observation_sequence=target.observation_sequence,
            arm_side=arm_state.side,
            grasp_frame=Pose(target.center_pose.position_m, frame_q, frame_id),
            pregrasp_pose=Pose(ready_palm, frame_q, frame_id),
            contact_pose=Pose(contact_palm, frame_q, frame_id),
            retreat_pose=Pose(retreat_palm, frame_q, frame_id),
            max_residual_translation_m=(0.20, 0.08, 0.15),
            max_residual_rotation_rad=(0.60, 0.60, 1.20),
            mapping_calibration_id=self._mapping_id,
            planner_version="ppo_training_distribution_ready_v1",
            expires_monotonic_ns=max(1, now_ns + int(self._ttl_s * 1.0e9)),
        )


__all__ = [
    "TrainingDistributionTomatoGraspPlanner",
    "evaluate_ppo_ready",
]
