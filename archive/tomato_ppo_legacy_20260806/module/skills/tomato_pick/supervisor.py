"""Continuous policy scaling and safety-bounded local Cartesian control."""

from __future__ import annotations

from dataclasses import dataclass

from proto import ArmState, HandState, PickPhase, Pose, TomatoGraspPlan

from .contracts import PolicyManifest, TomatoPickAction
from .geometry import apply_grasp_frame_delta, clamp, clamp_pose_to_plan_workspace
from .phase_contract import supervise_action_values


@dataclass(frozen=True, slots=True)
class SupervisedAction:
    normalized_action: TomatoPickAction
    target_pose: Pose
    target_hand_joint_positions_rad: tuple[float, ...]


class TomatoPickActionSupervisor:
    """Scale every policy dimension, then apply only controller safety bounds."""

    def __init__(self, manifest: PolicyManifest) -> None:
        if not isinstance(manifest, PolicyManifest):
            raise TypeError("manifest must be a PolicyManifest")
        self.manifest = manifest

    def supervise(
        self,
        action: TomatoPickAction,
        *,
        phase: PickPhase,
        arm: ArmState,
        hand: HandState,
        plan: TomatoGraspPlan,
    ) -> SupervisedAction:
        values = supervise_action_values(action.values, phase)
        normalized = TomatoPickAction(values)
        scales = self.manifest.action_scales
        delta_xyz = tuple(values[index] * scales[index] for index in range(3))
        delta_rotation_vector = tuple(
            values[index] * scales[index] for index in range(3, 6)
        )
        target_pose = apply_grasp_frame_delta(
            current=arm.ee_pose,
            grasp_frame=plan.grasp_frame,
            delta_xyz=delta_xyz,
            delta_rotation_vector=delta_rotation_vector,
        )
        target_pose = clamp_pose_to_plan_workspace(
            pose=target_pose,
            contact_pose=plan.contact_pose,
            grasp_frame=plan.grasp_frame,
            limits_m=plan.max_residual_translation_m,
            rotation_limits_rad=plan.max_residual_rotation_rad,
        )
        current_targets = hand.target_joint_positions_rad
        target_hand_joint_positions_rad = tuple(
            clamp(
                current_targets[index] + values[index + 6] * scales[index + 6],
                self.manifest.hand_position_lower_rad[index],
                self.manifest.hand_position_upper_rad[index],
            )
            for index in range(16)
        )
        return SupervisedAction(
            normalized,
            target_pose,
            target_hand_joint_positions_rad,
        )


__all__ = ["SupervisedAction", "TomatoPickActionSupervisor"]
