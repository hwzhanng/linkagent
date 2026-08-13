"""Phase-aware target-state estimation for the 20 Hz PPO observation loop."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from proto import ArmState, PickPhase, Pose, TomatoGraspPlan, TomatoTarget

from .geometry import quaternion_conjugate, rotate_vector


@runtime_checkable
class TargetStatePort(Protocol):
    """Produce a current metric target without changing track or plan identity."""

    def latest(
        self,
        *,
        initial_target: TomatoTarget,
        plan: TomatoGraspPlan,
        phase: PickPhase,
        arm: ArmState,
        now_ns: int,
    ) -> TomatoTarget: ...


class WorldThenHandTargetStatePort:
    """Use a fixed world target before detach, then a palm-attached estimate.

    This is an explicit kinematic estimator, not a camera measurement. It is
    valid only after calibrated RGB-D establishes a world-frame target and an
    independent verifier decides when detachment has occurred.
    """

    def __init__(self) -> None:
        self._attached_center_in_palm: tuple[float, float, float] | None = None
        self._attached_pull_in_palm: tuple[float, float, float] | None = None

    def latest(
        self,
        *,
        initial_target: TomatoTarget,
        plan: TomatoGraspPlan,
        phase: PickPhase,
        arm: ArmState,
        now_ns: int,
    ) -> TomatoTarget:
        if initial_target.target_id != plan.target_id:
            raise ValueError("target-state estimator target does not match the plan")
        if initial_target.observation_sequence != plan.source_observation_sequence:
            raise ValueError("target-state estimator target is not bound to the plan revision")
        if initial_target.center_pose.frame_id != "world" or arm.ee_pose.frame_id != "world":
            raise ValueError("world/hand target estimator requires world-frame poses")
        if phase is not PickPhase.HOLD:
            return self._copy(
                initial_target,
                center_pose=initial_target.center_pose,
                pull_axis=initial_target.stem_direction_xyz,
                now_ns=now_ns,
            )

        palm_inverse = quaternion_conjugate(arm.ee_pose.quaternion_wxyz)
        if self._attached_center_in_palm is None:
            delta_world = tuple(
                fruit - palm
                for fruit, palm in zip(
                    initial_target.center_pose.position_m,
                    arm.ee_pose.position_m,
                )
            )
            self._attached_center_in_palm = rotate_vector(palm_inverse, delta_world)
            self._attached_pull_in_palm = rotate_vector(
                palm_inverse, initial_target.stem_direction_xyz
            )
        assert self._attached_pull_in_palm is not None
        center_world_delta = rotate_vector(
            arm.ee_pose.quaternion_wxyz, self._attached_center_in_palm
        )
        center_world = tuple(
            palm + delta
            for palm, delta in zip(arm.ee_pose.position_m, center_world_delta)
        )
        pull_world = rotate_vector(
            arm.ee_pose.quaternion_wxyz, self._attached_pull_in_palm
        )
        return self._copy(
            initial_target,
            center_pose=Pose(
                center_world,
                arm.ee_pose.quaternion_wxyz,
                arm.ee_pose.frame_id,
            ),
            pull_axis=pull_world,
            now_ns=now_ns,
        )

    @staticmethod
    def _copy(
        target: TomatoTarget,
        *,
        center_pose: Pose,
        pull_axis: tuple[float, ...],
        now_ns: int,
    ) -> TomatoTarget:
        return TomatoTarget(
            target_id=target.target_id,
            center_pose=center_pose,
            semi_axes_m=target.semi_axes_m,
            ripeness_score=target.ripeness_score,
            pick_confidence=target.pick_confidence,
            occlusion=target.occlusion,
            stem_direction_xyz=pull_axis,
            observation_sequence=target.observation_sequence,
            sample_monotonic_ns=now_ns,
        )


__all__ = ["TargetStatePort", "WorldThenHandTargetStatePort"]
