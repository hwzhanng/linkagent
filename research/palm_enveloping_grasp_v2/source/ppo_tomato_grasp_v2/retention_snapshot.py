"""State-consistent retention snapshots for v2 curriculum transitions.

The old seed bank stored only joint positions.  This module captures the
dynamic state required to resume a post-break episode without teleporting the
fruit or silently resetting tactile/actuator history.  It is intentionally
an adapter: no v1 source is edited.
"""

from __future__ import annotations

from typing import Iterable

import torch


_TENSOR_FIELDS: tuple[str, ...] = (
    "actions", "action_delta", "prev_targets", "post_break_targets",
    "post_break_targets_latched", "force_history", "sensor_delay",
    "sensor_dropout", "sensor_stale", "tactile_gain", "tactile_bias",
    "retention_force_ema", "actuator_response_scale", "actuator_bias",
    "actuator_deadband", "actuator_backlash", "last_target_direction",
    "contact_forces", "previous_contact_forces", "palm_force", "contact_counts",
    "peak_force", "force_exposure", "impact_exposure", "damage_score",
    "stable_steps", "post_break_steps", "prepick_stable_steps", "prepick_best_steps",
    "success", "secure_latched", "enclosure_latched", "grasp_center",
    "anchor_rest_pos", "anchor_pos", "anchor_rest_quat", "anchor_yaw",
    "stem_damage", "stem_attached", "stem_broke", "break_event", "last_break_event",
    "pick_started", "pick_done", "pick_progress_m", "break_opposition_ok",
    "secure_before_pick", "dropped", "slipped", "overloaded", "best_proximity",
    "best_fingertip_proximity", "best_contact_count", "metric_approach",
    "metric_multi_digit", "metric_thumb_opposition", "metric_palm_support",
    "metric_secure_enclosure", "last_termination_code", "last_drop_before_pick",
    "last_drop_after_pick", "last_drop_after_break", "strict_first_palm",
    "strict_first_three", "strict_palm_streak", "strict_hold_streak",
    "strict_hold_max", "strict_coverage_max", "strict_opposition_min",
    "strict_hemisphere_max", "strict_hemisphere_min", "strict_coverage_current",
    "strict_opposition_current", "strict_hemisphere_current",
    "strict_contact_geometry_measured", "strict_distal_participation",
    "strict_proximal_only", "strict_success_latched", "strict_initial_joint_pos",
)


def capture_retention_snapshot(env, env_ids: Iterable[int] | torch.Tensor | None = None) -> dict:
    """Capture all dynamic tensors and PhysX root states for selected envs."""

    if env_ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, torch.Tensor):
        ids = env_ids.to(device=env.device, dtype=torch.long)
    else:
        ids = torch.as_tensor(tuple(env_ids), device=env.device, dtype=torch.long)
    tensor_fields = {}
    for name in _TENSOR_FIELDS:
        if hasattr(env, name):
            tensor_fields[name] = getattr(env, name)[ids].detach().clone()
    robot_dof_pos = env.robot.root_physx_view.get_dof_positions()[ids.to(env.robot.root_physx_view.get_dof_positions().device)].detach().clone()
    robot_dof_vel = env.robot.root_physx_view.get_dof_velocities()[ids.to(env.robot.root_physx_view.get_dof_velocities().device)].detach().clone()
    tomato_transforms = env.tomato.root_physx_view.get_transforms()[ids.to(env.tomato.root_physx_view.get_transforms().device)].detach().clone()
    tomato_velocities = env.tomato.root_physx_view.get_velocities()[ids.to(env.tomato.root_physx_view.get_velocities().device)].detach().clone()
    return {
        "version": "v2_retention_snapshot_1",
        "env_ids": ids.detach().cpu(),
        "tensor_fields": tensor_fields,
        "robot_dof_pos": robot_dof_pos,
        "robot_dof_vel": robot_dof_vel,
        "tomato_transforms": tomato_transforms,
        "tomato_velocities": tomato_velocities,
    }


def restore_retention_snapshot(env, snapshot: dict, env_ids: Iterable[int] | torch.Tensor | None = None) -> None:
    """Restore a snapshot without invoking the episode reset path."""

    if snapshot.get("version") != "v2_retention_snapshot_1":
        raise ValueError("unsupported retention snapshot version")
    source_ids = snapshot["env_ids"]
    if env_ids is None:
        ids = source_ids.to(device=env.device, dtype=torch.long)
    elif isinstance(env_ids, torch.Tensor):
        ids = env_ids.to(device=env.device, dtype=torch.long)
    else:
        ids = torch.as_tensor(tuple(env_ids), device=env.device, dtype=torch.long)
    if ids.numel() != source_ids.numel():
        raise ValueError("restore env_ids must have the same count as the snapshot")
    # Isaac Lab reset hooks can leave bookkeeping buffers as inference
    # tensors.  Clone each destination before the indexed write so a restore
    # is valid both during PPO and in a standalone retention transition.
    with torch.inference_mode(False):
        for name, values in snapshot["tensor_fields"].items():
            target = getattr(env, name).clone()
            target[ids] = values.to(device=env.device, dtype=target.dtype)
            setattr(env, name, target)
    robot_view = env.robot.root_physx_view
    robot_ids = ids.to(device=robot_view.get_dof_positions().device)
    robot_pos = robot_view.get_dof_positions().clone()
    robot_vel = robot_view.get_dof_velocities().clone()
    robot_pos[robot_ids] = snapshot["robot_dof_pos"].to(device=robot_pos.device, dtype=robot_pos.dtype)
    robot_vel[robot_ids] = snapshot["robot_dof_vel"].to(device=robot_vel.device, dtype=robot_vel.dtype)
    robot_view.set_dof_positions(robot_pos, indices=robot_ids)
    robot_view.set_dof_velocities(robot_vel, indices=robot_ids)
    tomato_view = env.tomato.root_physx_view
    tomato_ids = ids.to(device=tomato_view.get_transforms().device)
    transforms = tomato_view.get_transforms().clone()
    velocities = tomato_view.get_velocities().clone()
    transforms[tomato_ids] = snapshot["tomato_transforms"].to(device=transforms.device, dtype=transforms.dtype)
    velocities[tomato_ids] = snapshot["tomato_velocities"].to(device=velocities.device, dtype=velocities.dtype)
    tomato_view.set_transforms(transforms, indices=tomato_ids)
    tomato_view.set_velocities(velocities, indices=tomato_ids)
    env.robot.update(dt=1.0e-6)
    env.tomato.update(dt=1.0e-6)


__all__ = ["capture_retention_snapshot", "restore_retention_snapshot"]
