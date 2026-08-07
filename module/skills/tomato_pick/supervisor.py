"""Translate normalized policy output to the two validated controller ports."""

from __future__ import annotations

from dataclasses import dataclass

from proto import ArmState, HandState, Pose, TomatoGraspPlan

from .contracts import PolicyManifest, TomatoPickAction
from .geometry import apply_grasp_frame_delta, clamp_pose_to_plan_workspace
from .hand_synergy import g20_synergy_to_joint_targets


@dataclass(frozen=True, slots=True)
class SupervisedAction:
    normalized_action: TomatoPickAction
    target_pose: Pose
    target_hand_joint_positions_rad: tuple[float, ...]


class TomatoPickActionSupervisor:
    """Apply physical bounds without imposing phases or scripted movements."""

    def __init__(self, manifest: PolicyManifest) -> None:
        self.manifest = manifest

    def supervise(
        self,
        action: TomatoPickAction,
        *,
        phase: object | None = None,
        arm: ArmState,
        hand: HandState,
        plan: TomatoGraspPlan,
    ) -> SupervisedAction:
        del phase, hand
        if not isinstance(action, TomatoPickAction):
            raise TypeError("action must be a TomatoPickAction")
        scales = self.manifest.action_scales
        target_pose = apply_grasp_frame_delta(
            current=arm.ee_pose,
            grasp_frame=plan.grasp_frame,
            delta_xyz=tuple(value * scale for value, scale in zip(action.delta_xyz, scales[:3])),
            delta_rotation_vector=tuple(
                value * scale
                for value, scale in zip(action.delta_rotation_vector, scales[3:6])
            ),
        )
        target_pose = clamp_pose_to_plan_workspace(
            pose=target_pose,
            contact_pose=plan.contact_pose,
            grasp_frame=plan.grasp_frame,
            limits_m=plan.max_residual_translation_m,
            rotation_limits_rad=plan.max_residual_rotation_rad,
        )
        targets = g20_synergy_to_joint_targets(action.hand_synergy)
        return SupervisedAction(action, target_pose, targets)


__all__ = ["SupervisedAction", "TomatoPickActionSupervisor"]
