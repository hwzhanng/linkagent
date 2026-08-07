"""Vectorized DirectRLEnv for local contact, grasp, detach, and hold.

This module belongs exclusively to the isolated Isaac runtime.  It deliberately
does not connect CAN, the vendor SDK, or any model API.
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.views import FrameView
from isaaclab.utils.math import (
    compute_pose_error,
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

from .env_cfg import (
    RIGHT_FINGERTIP_CONTACT_BODY_NAMES,
    RIGHT_FINGERTIP_FRAME_BODY_NAMES,
    RIGHT_HAND_BODY_NAMES,
    TomatoPickEnvCfg,
)
from .assets import (
    activate_nested_contact_reporters,
    validate_newton_mimic_constraints_disabled,
    validate_linkbot_mass_properties,
)
from .sim2real import Sim2RealRuntime
from simulation.tomato_pick.config import load_sim2real_profile
from simulation.tomato_pick.reward_potentials import positive_support_potential
from simulation.tomato_pick.stem_fracture import (
    directional_load_ratio,
    update_overload_streak,
)
from module.skills.tomato_pick.contracts import (
    ACTION_DIM,
    G20_JOINT_ORDER,
    OBSERVATION_DIM,
)
from module.skills.tomato_pick.phase_contract import (
    CONTACT_FORCE_THRESHOLD_N,
    CONTACT_STREAK_STEPS,
)


FINGERS = ("thumb", "index", "middle", "ring", "little")
RIGHT_HAND_CONTROL_RAD = {
    # (minimum, neutral/open, maximum, synergy source).  Opposition/spread
    # retain the validated neutral pose at synergy zero but can now explore a
    # broad, hard-limit-safe interval in both directions.  The former mapping
    # exposed only 0.02--0.05 rad and effectively locked the hand shape.
    "right_hand_thumb_joint0": (-0.35, 1.025843, 1.05, "opposition"),
    "right_hand_thumb_joint1": (-1.30, -0.093824, -0.02, "opposition"),
    "right_hand_thumb_joint2": (0.016471, 0.016471, 0.675294, "closure"),
    "right_hand_thumb_joint3": (0.025882, 0.025882, 0.931765, "closure"),
    "right_hand_index_joint0": (-0.24, 0.132549, 0.24, "spread"),
    "right_hand_index_joint1": (0.032549, 0.032549, 1.562353, "closure"),
    "right_hand_index_joint2": (0.026667, 0.026667, 1.333333, "closure"),
    "right_hand_middle_joint0": (-0.24, 0.041804, 0.24, "spread"),
    "right_hand_middle_joint1": (0.032549, 0.032549, 1.627451, "closure"),
    "right_hand_middle_joint2": (0.026667, 0.026667, 1.333333, "closure"),
    "right_hand_ring_joint0": (-0.24, -0.045883, 0.24, "spread"),
    "right_hand_ring_joint1": (0.032549, 0.032549, 1.627451, "closure"),
    "right_hand_ring_joint2": (0.026667, 0.026667, 1.306667, "closure"),
    "right_hand_little_joint0": (-0.24, -0.166196, 0.24, "spread"),
    "right_hand_little_joint1": (0.032549, 0.032549, 1.562353, "closure"),
    "right_hand_little_joint2": (0.026667, 0.026667, 1.333333, "closure"),
}
RIGHT_HAND_MIMIC_RAD = {
    "right_hand_thumb_joint4": ("right_hand_thumb_joint3", 1.11, -0.04545),
    "right_hand_index_joint3": ("right_hand_index_joint2", 0.94, -0.04545),
    "right_hand_middle_joint3": ("right_hand_middle_joint2", 0.94, -0.04545),
    "right_hand_ring_joint3": ("right_hand_ring_joint2", 0.94, -0.04545),
    "right_hand_little_joint3": ("right_hand_little_joint2", 0.94, -0.04545),
}
MIMIC_FOLLOWER_JOINT_NAMES = tuple(RIGHT_HAND_MIMIC_RAD) + tuple(
    name.replace("right_hand_", "left_hand_", 1) for name in RIGHT_HAND_MIMIC_RAD
)
RIGHT_HAND_CONTROL_JOINT_NAMES = tuple(
    f"right_hand_{name}" for name in G20_JOINT_ORDER
)


def phase_supervise_actions(actions: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    """Validate progress labels and leave every action under PPO control."""

    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape (N, {ACTION_DIM})")
    if phases.ndim != 1 or phases.shape[0] != actions.shape[0]:
        raise ValueError("phases must have shape (N,)")
    if bool(((phases < 0) | (phases > 3)).any()):
        raise ValueError("phase indices must be in 0..3")
    return actions.clone()


class TomatoPickEnv(DirectRLEnv):
    """Single-policy, one-active-arm tomato-pick physics environment."""

    cfg: TomatoPickEnvCfg

    def __init__(self, cfg: TomatoPickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if not hasattr(self, "_prestartup_fruit_axes_m"):
            raise RuntimeError("correlated fruit geometry prestartup event did not run")

        self._arm_joint_ids, _ = self._robot.find_joints(r"^R[1-7]_Joint$", preserve_order=True)
        self._left_arm_joint_ids, _ = self._robot.find_joints(
            r"^L[1-7]_Joint$", preserve_order=True
        )
        self._hand_joint_ids, self._hand_joint_names = self._robot.find_joints(
            list(RIGHT_HAND_CONTROL_JOINT_NAMES), preserve_order=True
        )
        self._hand_follower_ids, self._hand_follower_names = self._robot.find_joints(
            list(RIGHT_HAND_MIMIC_RAD), preserve_order=True
        )
        self._mimic_follower_ids, self._mimic_follower_names = self._robot.find_joints(
            list(MIMIC_FOLLOWER_JOINT_NAMES), preserve_order=True
        )
        self._actuated_joint_ids = list(range(self._robot.num_joints))
        if len(self._mimic_follower_ids) != 10 or len(self._actuated_joint_ids) != 56:
            raise RuntimeError(
                "expected 10 software-coupled G20 followers and 56 driven joints, "
                f"found {len(self._mimic_follower_ids)} and {len(self._actuated_joint_ids)}"
            )
        self._hand_joint_id_by_name = dict(zip(self._hand_joint_names, self._hand_joint_ids))
        self._ee_body_id = self._robot.find_bodies("right_hand_base_link")[0][0]
        self._ee_jacobian_id = self._ee_body_id - 1 if self._robot.is_fixed_base else self._ee_body_id
        self._fingertip_frame_body_ids = [
            self._robot.find_bodies(RIGHT_FINGERTIP_FRAME_BODY_NAMES[finger])[0][0]
            for finger in FINGERS
        ]
        self._fingertip_contact_body_ids = [
            self._robot.find_bodies(RIGHT_FINGERTIP_CONTACT_BODY_NAMES[finger])[0][0]
            for finger in FINGERS
        ]
        if self.cfg.enable_fingertip_contacts and tuple(self._contact_sensors) != FINGERS:
            raise RuntimeError(f"unexpected fingertip sensor order: {tuple(self._contact_sensors)}")
        self._self_collision_check = self._read_self_collision_state()
        if self._self_collision_check["enabled"] is not False:
            raise RuntimeError(f"robot self collisions are not disabled: {self._self_collision_check}")
        if not 0.0 <= float(self.cfg.tactile_policy_sample_retention) < 1.0:
            raise ValueError("tactile_policy_sample_retention must lie in [0, 1)")
        if (
            not isinstance(self.cfg.stem_break_persistence_substeps, int)
            or self.cfg.stem_break_persistence_substeps < 1
        ):
            raise ValueError("stem_break_persistence_substeps must be a positive integer")
        self._asset_initial_joint_pos = self._robot.data.joint_pos.torch.clone()
        self._nominal_joint_position_targets = self._robot.data.default_joint_pos.torch.clone()
        self._validate_nominal_arm_joint_margin()
        self._reset_arm_joint_ids = self._arm_joint_ids + self._left_arm_joint_ids
        self._reset_hand_joint_ids = self._hand_joint_ids + self._hand_follower_ids
        self._reset_joint_ids = self._reset_arm_joint_ids + self._reset_hand_joint_ids
        self._reset_previous_hand_joint_pos = self._robot.data.joint_pos.torch[
            :, self._reset_hand_joint_ids
        ].clone()
        self._tactile_force_filtered = torch.zeros((self.num_envs, 5), device=self.device)
        self._contact_substep_counts = torch.zeros(
            (self.num_envs, 5), dtype=torch.long, device=self.device
        )
        self._last_tactile_update_step = -1
        self._last_pressure_observation_step = -1
        self._previous_measured_pressure = torch.zeros(
            (self.num_envs, 5), device=self.device
        )
        self._pressure_rate_norm = torch.zeros((self.num_envs, 5), device=self.device)

        self._actions = torch.zeros((self.num_envs, gym.spaces.flatdim(self.single_action_space)), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._ee_command_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self._ee_command_pose_w[:, 6] = 1.0
        self._last_ik_predicted_delta_w = torch.zeros((self.num_envs, 6), device=self.device)
        self._last_ik_joint_delta = torch.zeros((self.num_envs, 7), device=self.device)
        self._synergy = torch.zeros((self.num_envs, 3), device=self.device)
        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._phase_at_action = torch.zeros_like(self._phase)
        self._reached_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reached_grasp = torch.zeros_like(self._reached_contact)
        self._reached_detach = torch.zeros_like(self._reached_contact)
        self._phase_steps = torch.zeros_like(self._phase)
        self._reset_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_fruit_placed = torch.zeros_like(self._reset_pending)
        self._reset_elapsed_steps = torch.zeros_like(self._phase)
        self._reset_robot_stable_steps = torch.zeros_like(self._phase)
        self._reset_validation_stable_steps = torch.zeros_like(self._phase)
        self._reset_collision_retries = torch.zeros_like(self._phase)
        self._reset_completed_count = 0
        self._reset_contact_rejection_count = 0
        self._contact_streak = torch.zeros_like(self._phase)
        self._hold_streak = torch.zeros_like(self._phase)
        self._stem_detached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stem_break_event = torch.zeros_like(self._stem_detached)
        self._stem_threshold_reached_this_step = torch.zeros_like(self._stem_detached)
        self._stem_overload_substeps = torch.zeros_like(self._phase)
        self._grasp_stable = torch.zeros_like(self._stem_detached)
        self._retained_without_slip = torch.zeros_like(self._stem_detached)
        self._success = torch.zeros_like(self._stem_detached)
        self._dropped = torch.zeros_like(self._stem_detached)
        self._excess_force = torch.zeros_like(self._stem_detached)
        self._task_timeout = torch.zeros_like(self._stem_detached)
        self._previous_approach_potential = torch.zeros(self.num_envs, device=self.device)
        self._previous_grasp_potential = torch.zeros(self.num_envs, device=self.device)
        self._previous_stem_load_potential = torch.zeros(self.num_envs, device=self.device)
        self._previous_retention_potential = torch.zeros(self.num_envs, device=self.device)
        self._ee_reference_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self._grasp_frame_basis_w = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        self._grasp_frame_origin_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._stem_anchor_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._stem_anchor_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._detach_start_ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._stem_anchor_quat_w[:, 3] = 1.0
        self._last_stem_force_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_stem_torque_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_stem_force_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_stem_torque_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._fruit_axes_m = self._prestartup_fruit_axes_m.to(self.device)
        self._fruit_density_kg_m3 = self._prestartup_fruit_density_kg_m3.to(self.device)
        self._fruit_mass_kg = (
            4.0 / 3.0 * math.pi * self._fruit_axes_m.prod(dim=1) * self._fruit_density_kg_m3
        )
        self._tomato.set_masses_index(masses=self._fruit_mass_kg.unsqueeze(1))
        self._sim2real = Sim2RealRuntime(
            profile=load_sim2real_profile(),
            num_envs=self.num_envs,
            action_dim=self.cfg.action_space,
            policy_rate_hz=round(1.0 / (self.cfg.sim.dt * self.cfg.decimation)),
            device=self.device,
            enabled=self.cfg.enable_sim2real_randomization,
        )
        self._break_force_n = self._sim2real.break_force_n
        self._break_torque_nm = self._sim2real.break_torque_nm
        self._joint_position_targets = self._nominal_joint_position_targets.clone()
        self._stem_visual_positions_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._stem_view = FrameView(self.cfg.stem.prim_path, device=self.device)
        if self._stem_view.count != self.num_envs:
            raise RuntimeError(
                f"expected {self.num_envs} stem visuals, found {self._stem_view.count}"
            )
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "contact",
                "approach_progress",
                "grasp_progress",
                "grasp",
                "stem_load_progress",
                "break_once",
                "retention",
                "success",
                "drop",
                "excess_force",
                "timeout",
                "soft_contact_force",
                "joint_limit",
                "action_magnitude",
                "action_rate",
            )
        }
        self._last_reward_terms = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self._episode_sums
        }

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self._software_coupled_mimic_paths = validate_newton_mimic_constraints_disabled(
            "/World/envs/env_0/Robot"
        )
        self._tomato = RigidObject(self.cfg.tomato)
        self._mass_profile_validation = validate_linkbot_mass_properties(
            "/World/envs/env_0/Robot",
            minimum_principal_inertia_kg_m2=self.cfg.minimum_hand_inertia_kg_m2,
        )
        self._contact_sensors = {}
        if self.cfg.enable_fingertip_contacts:
            activate_nested_contact_reporters(
                "/World/envs/env_0/Robot", RIGHT_HAND_BODY_NAMES
            )
            self._contact_sensors = {}
            for name, sensor_cfg in self.cfg.fingertip_contacts.items():
                # These sensors are registered manually rather than declared
                # on InteractiveSceneCfg, so apply its normal namespace
                # expansion before constructing the PhysX contact view.
                sensor_cfg.filter_prim_paths_expr = [
                    path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
                    for path in sensor_cfg.filter_prim_paths_expr
                ]
                self._contact_sensors[name] = ContactSensor(sensor_cfg)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["tomato"] = self._tomato
        for name, sensor in self._contact_sensors.items():
            self.scene.sensors[f"right_{name}_contact"] = sensor
        self.cfg.stem.spawn.func(
            self.cfg.stem.prim_path,
            self.cfg.stem.spawn,
            translation=self.cfg.stem.init_state.pos,
            orientation=self.cfg.stem.init_state.rot,
        )
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        sim_utils.DomeLightCfg(intensity=2200.0, color=(0.80, 0.82, 0.84)).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2200.0, color=(0.80, 0.82, 0.84))
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if tuple(actions.shape) != tuple(self._actions.shape):
            raise ValueError(
                f"action shape {tuple(actions.shape)} does not match {tuple(self._actions.shape)}"
            )
        if not torch.is_floating_point(actions):
            raise TypeError(f"actions must be floating point, got {actions.dtype}")
        if self.cfg.strict_runtime_checks:
            if actions.dtype != torch.float32:
                raise TypeError(f"actions must use float32, got {actions.dtype}")
            if not bool(torch.isfinite(actions).all()):
                raise ValueError("actions contain NaN or Inf")
            if bool(((actions < -1.0) | (actions > 1.0)).any()):
                raise ValueError("actions fall outside the declared [-1, 1] space")
        # `_apply_stem_wrench` runs once per physics substep.  Keep the
        # one-shot fracture visible for the whole policy step so reward,
        # metrics, and the phase machine cannot miss an event that occurred in
        # an early decimation substep.
        self._stem_break_event.zero_()
        self._stem_threshold_reached_this_step.zero_()
        self._contact_substep_counts.zero_()
        self._phase_at_action.copy_(self._phase)
        supervised_actions = phase_supervise_actions(actions, self._phase)
        self._previous_actions[:] = self._actions
        base_scales = torch.tensor(self.cfg.action_scales, device=self.device)
        delayed, scaled = self._sim2real.process_actions(
            supervised_actions,
            base_scales=base_scales,
            blocked=self._reset_pending,
        )
        self._actions[:] = delayed
        self._update_reset_state_machine()
        self._actions[self._reset_pending] = 0.0
        scaled[self._reset_pending] = 0.0
        ee_pose_w = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        ik_command = torch.zeros((self.num_envs, 6), device=self.device)
        translation_w = torch.bmm(
            self._grasp_frame_basis_w, scaled[:, :3].unsqueeze(2)
        ).squeeze(2)
        rotation_vector_w = torch.bmm(
            self._grasp_frame_basis_w, scaled[:, 3:6].unsqueeze(2)
        ).squeeze(2)
        ik_command[:, :3] = translation_w
        ik_command[:, 3:6] = rotation_vector_w
        self._last_ik_command = ik_command.clone()
        proposed_position_w = ee_pose_w[:, :3] + translation_w
        reference_delta_w = proposed_position_w - self._ee_reference_pose_w[:, :3]
        reference_delta_g = torch.bmm(
            self._grasp_frame_basis_w.transpose(1, 2),
            reference_delta_w.unsqueeze(2),
        ).squeeze(2)
        workspace_limits = torch.tensor(
            self.cfg.max_residual_translation_m,
            device=self.device,
        )
        reference_delta_g = torch.maximum(
            -workspace_limits,
            torch.minimum(workspace_limits, reference_delta_g),
        )
        self._ee_command_pose_w[:, :3] = self._ee_reference_pose_w[:, :3] + torch.bmm(
            self._grasp_frame_basis_w,
            reference_delta_g.unsqueeze(2),
        ).squeeze(2)
        rotation_angle = torch.linalg.vector_norm(rotation_vector_w, dim=1)
        rotation_axis_w = rotation_vector_w / rotation_angle.clamp_min(1.0e-8).unsqueeze(1)
        rotation_axis_w = torch.where(
            (rotation_angle > 1.0e-8).unsqueeze(1),
            rotation_axis_w,
            torch.tensor((0.0, 0.0, 1.0), device=self.device),
        )
        rotation_delta = quat_from_angle_axis(rotation_angle, rotation_axis_w)
        proposed_quat_w = quat_mul(rotation_delta, ee_pose_w[:, 3:7])
        proposed_rotvec_w = quat_box_minus(
            proposed_quat_w,
            self._ee_reference_pose_w[:, 3:7],
        )
        proposed_rotvec_g = torch.bmm(
            self._grasp_frame_basis_w.transpose(1, 2),
            proposed_rotvec_w.unsqueeze(2),
        ).squeeze(2)
        rotation_limits = torch.tensor(
            self.cfg.max_residual_rotation_rad, device=self.device
        )
        bounded_rotvec_g = torch.maximum(
            -rotation_limits,
            torch.minimum(rotation_limits, proposed_rotvec_g),
        )
        bounded_rotvec_w = torch.bmm(
            self._grasp_frame_basis_w, bounded_rotvec_g.unsqueeze(2)
        ).squeeze(2)
        bounded_angle = torch.linalg.vector_norm(bounded_rotvec_w, dim=1)
        bounded_axis_w = bounded_rotvec_w / bounded_angle.clamp_min(1.0e-8).unsqueeze(1)
        bounded_axis_w = torch.where(
            (bounded_angle > 1.0e-8).unsqueeze(1),
            bounded_axis_w,
            torch.tensor((0.0, 0.0, 1.0), device=self.device),
        )
        bounded_rotation = quat_from_angle_axis(bounded_angle, bounded_axis_w)
        self._ee_command_pose_w[:, 3:7] = quat_mul(
            bounded_rotation,
            self._ee_reference_pose_w[:, 3:7],
        )
        self._update_hand_targets_from_joint_action(scaled[:, 6:22])

    def _update_arm_target_from_cartesian_action(self) -> None:
        """Track the held policy-step Cartesian target with feedback IK.

        Translation is the primary task.  Rotation about the pull axis is
        solved only in the translational null space, so wrist twist cannot
        silently sacrifice approach motion near a singularity.
        """

        arm_joint_pos = self._robot.data.joint_pos.torch[:, self._arm_joint_ids]
        arm_target = self._joint_position_targets[:, self._arm_joint_ids].clone()
        active = torch.linalg.vector_norm(self._actions[:, :6], dim=1) > 1.0e-7
        if bool(active.any()):
            ee_pose_w = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
            position_command, rotation_command = compute_pose_error(
                ee_pose_w[:, :3],
                ee_pose_w[:, 3:7],
                self._ee_command_pose_w[:, :3],
                self._ee_command_pose_w[:, 3:7],
            )
            jacobian_joint_ids = [index + self._robot.num_base_dofs for index in self._arm_joint_ids]
            jacobian = self._robot.data.body_link_jacobian_w.torch[
                :, self._ee_jacobian_id, :, jacobian_joint_ids
            ]
            jacobian = torch.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0)
            position_jacobian = jacobian[:, :3]
            rotation_jacobian = jacobian[:, 3:]
            batch_size = self.num_envs
            task_identity = torch.eye(3, device=self.device).expand(batch_size, -1, -1)
            joint_identity = torch.eye(7, device=self.device).expand(batch_size, -1, -1)

            position_normal = torch.bmm(
                position_jacobian, position_jacobian.transpose(1, 2)
            ) + self.cfg.ik_position_damping**2 * task_identity
            position_inverse = torch.linalg.solve(position_normal, task_identity)
            position_pseudoinverse = torch.bmm(
                position_jacobian.transpose(1, 2), position_inverse
            )
            position_delta = torch.bmm(
                position_pseudoinverse, position_command.unsqueeze(2)
            ).squeeze(2)

            null_projection = joint_identity - torch.bmm(
                position_pseudoinverse, position_jacobian
            )
            null_rotation_jacobian = torch.bmm(rotation_jacobian, null_projection)
            rotation_residual = rotation_command - torch.bmm(
                rotation_jacobian, position_delta.unsqueeze(2)
            ).squeeze(2)
            rotation_normal = torch.bmm(
                null_rotation_jacobian, null_rotation_jacobian.transpose(1, 2)
            ) + self.cfg.ik_rotation_damping**2 * task_identity
            rotation_delta = torch.bmm(
                null_rotation_jacobian.transpose(1, 2),
                torch.linalg.solve(rotation_normal, rotation_residual.unsqueeze(2)),
            ).squeeze(2)
            nominal_arm_position = self._nominal_joint_position_targets[
                :, self._arm_joint_ids
            ]
            posture_error = nominal_arm_position - arm_joint_pos
            posture_delta = torch.bmm(
                null_projection,
                posture_error.unsqueeze(2),
            ).squeeze(2)
            secondary_delta = (
                rotation_delta
                + self.cfg.ik_nullspace_posture_gain * posture_delta
            )
            secondary_maximum = secondary_delta.abs().max(dim=1, keepdim=True).values
            secondary_scale = (
                self.cfg.maximum_arm_secondary_joint_delta_rad
                / secondary_maximum.clamp_min(
                    self.cfg.maximum_arm_secondary_joint_delta_rad
                )
            )
            secondary_delta = secondary_delta * secondary_scale
            joint_increment = position_delta + secondary_delta
            # Damped projectors are deliberately not exact near a
            # singularity.  Secondary rotation/posture terms can therefore
            # leak back into translation and, in the previous controller,
            # almost completely cancel the commanded approach.  Restore the
            # primary Cartesian translation after adding secondary terms.
            # This keeps the redundant elbow bias and wrist stabilization
            # subordinate to the motion requested by the policy.
            position_residual = position_command - torch.bmm(
                position_jacobian,
                joint_increment.unsqueeze(2),
            ).squeeze(2)
            joint_increment = joint_increment + torch.bmm(
                position_pseudoinverse,
                position_residual.unsqueeze(2),
            ).squeeze(2)
            joint_increment = torch.where(
                torch.isfinite(joint_increment),
                joint_increment,
                torch.zeros_like(joint_increment),
            )
            maximum_component = joint_increment.abs().max(dim=1, keepdim=True).values
            uniform_scale = (
                self.cfg.maximum_arm_joint_delta_rad
                / maximum_component.clamp_min(self.cfg.maximum_arm_joint_delta_rad)
            )
            # Preserve the IK vector direction.  Per-joint clipping changes
            # J*dq and can turn a straight approach into a lateral excursion.
            joint_increment = joint_increment * uniform_scale
            # Rebuild the next drive target from the measured joint state.
            # Accumulating from the previous target winds up the implicit
            # position servo: J*dq can point toward the Cartesian target while
            # the measured qdot still follows an older target in the opposite
            # direction.  The bounded one-substep increment keeps the executed
            # motion consistent with the action/log-probability pair.
            candidate = arm_joint_pos + joint_increment
            candidate = torch.maximum(
                arm_joint_pos - self.cfg.maximum_arm_target_lead_rad,
                torch.minimum(
                    arm_joint_pos + self.cfg.maximum_arm_target_lead_rad,
                    candidate,
                ),
            )
            arm_target[active] = candidate[active]
            self._last_ik_joint_delta[active] = joint_increment[active]
            predicted_delta = torch.bmm(
                jacobian, joint_increment.unsqueeze(2)
            ).squeeze(2)
            self._last_ik_predicted_delta_w[active] = predicted_delta[active]
        inactive = ~active
        self._last_ik_joint_delta[inactive] = 0.0
        self._last_ik_predicted_delta_w[inactive] = 0.0
        limits = self._robot.data.joint_pos_limits.torch[:, self._arm_joint_ids]
        arm_target = torch.maximum(limits[..., 0], torch.minimum(limits[..., 1], arm_target))
        self._joint_position_targets[:, self._arm_joint_ids] = arm_target

    def _apply_action(self) -> None:
        self._update_arm_target_from_cartesian_action()
        # Use the articulation position drives so PhysX sees the complete
        # arm/hand/contact dynamics.  Deployment maps the same bounded IK
        # target to the validated SDK followP boundary; PPO never controls raw
        # joint torque in either domain.
        self._robot.set_joint_position_target_index(
            target=self._joint_position_targets[:, self._actuated_joint_ids],
            joint_ids=self._actuated_joint_ids,
        )
        # The deployed moveL/followP boundary is gravity compensated.  Apply
        # PhysX inverse-dynamics g(q) as arm feed-forward so the simulated
        # position target represents that same controller instead of forcing
        # PPO to learn pose-dependent gravity sag.
        gravity_joint_ids = self._reset_arm_joint_ids
        gravity_columns = [
            index + self._robot.num_base_dofs for index in gravity_joint_ids
        ]
        gravity_effort = self._robot.data.gravity_compensation_forces.torch[
            :, gravity_columns
        ]
        self._robot.set_joint_effort_target_index(
            target=gravity_effort,
            joint_ids=gravity_joint_ids,
        )
        self._apply_stem_wrench()
        self._hold_pending_fruit_at_anchor()

    def _read_self_collision_state(self) -> dict[str, str | bool | None]:
        """Read back the PhysX articulation property actually authored on stage."""

        from pxr import Usd
        from isaaclab.sim.utils.stage import get_current_stage

        root = get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot")
        for prim in Usd.PrimRange(root):
            if "PhysxArticulationAPI" not in prim.GetAppliedSchemas():
                continue
            attribute = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            return {
                "prim_path": prim.GetPath().pathString,
                "enabled": attribute.Get() if attribute.IsValid() else None,
            }
        return {"prim_path": "", "enabled": None}

    def _validate_nominal_arm_joint_margin(self) -> None:
        arm_ids = self._arm_joint_ids + self._left_arm_joint_ids
        nominal = self._nominal_joint_position_targets[:, arm_ids]
        limits = self._robot.data.joint_pos_limits.torch[:, arm_ids]
        spans = (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6)
        margin = torch.minimum(nominal - limits[..., 0], limits[..., 1] - nominal) / spans
        minimum_margin = float(margin.min().item())
        if minimum_margin < self.cfg.minimum_arm_joint_limit_margin_fraction:
            flat_index = int(margin.reshape(-1).argmin().item())
            joint_index = flat_index % len(arm_ids)
            joint_name = self._robot.joint_names[arm_ids[joint_index]]
            raise RuntimeError(
                "nominal arm pose violates the joint-limit margin: "
                f"{joint_name} has {minimum_margin:.3f}, requires "
                f"{self.cfg.minimum_arm_joint_limit_margin_fraction:.3f}"
            )
        self._nominal_arm_minimum_margin_fraction = minimum_margin

    def _update_reset_state_machine(self) -> None:
        """Hold the fruit aside until the robot and the placed scene are stable."""

        pending = self._reset_pending.clone()
        if not bool(pending.any()):
            return
        self._reset_elapsed_steps[pending] += 1

        joint_pos = self._robot.data.joint_pos.torch[:, self._reset_joint_ids]
        joint_vel = self._robot.data.joint_vel.torch[:, self._reset_joint_ids]
        joint_target = self._joint_position_targets[:, self._reset_joint_ids]
        joint_error_values = (joint_target - joint_pos).abs()
        arm_count = len(self._reset_arm_joint_ids)
        arm_joint_error = joint_error_values[:, :arm_count].max(dim=1).values
        hand_joint_error = joint_error_values[:, arm_count:].max(dim=1).values
        joint_error = joint_error_values.max(dim=1).values
        maximum_joint_velocity = joint_vel.abs().max(dim=1).values
        maximum_arm_joint_velocity = joint_vel[:, :arm_count].abs().max(dim=1).values
        hand_joint_pos = joint_pos[:, arm_count:]
        maximum_hand_joint_step_delta = (
            hand_joint_pos - self._reset_previous_hand_joint_pos
        ).abs().max(dim=1).values
        self._reset_previous_hand_joint_pos[pending] = hand_joint_pos[pending]
        robot_stable = (
            (arm_joint_error <= self.cfg.reset_arm_joint_position_tolerance_rad)
            & (hand_joint_error <= self.cfg.reset_hand_joint_position_tolerance_rad)
            & (maximum_arm_joint_velocity <= self.cfg.reset_arm_joint_velocity_tolerance_rad_s)
            & (
                maximum_hand_joint_step_delta
                <= self.cfg.reset_hand_joint_step_delta_tolerance_rad
            )
        )
        self._reset_robot_stable_steps = torch.where(
            pending & robot_stable,
            self._reset_robot_stable_steps + 1,
            (self._reset_robot_stable_steps - 1).clamp_min(0),
        )

        validate_mask = pending & self._reset_fruit_placed
        if bool(validate_mask.any()):
            contact_force = self._instantaneous_finger_force_magnitudes().max(dim=1).values
            fruit_linear_speed = torch.linalg.vector_norm(
                self._tomato.data.root_lin_vel_w.torch, dim=1
            )
            fruit_angular_speed = torch.linalg.vector_norm(
                self._tomato.data.root_ang_vel_w.torch, dim=1
            )
            anchor_displacement = torch.linalg.vector_norm(
                self._tomato.data.root_pos_w.torch - self._stem_anchor_pos_w, dim=1
            )
            # The pose/velocity pin is authored before PhysX; the readback at
            # this point legitimately contains one dt of gravity.  Admit only
            # that known increment plus the frozen residual-speed tolerance.
            pinned_speed_tolerance = (
                self.cfg.reset_fruit_linear_velocity_tolerance_m_s
                + torch.linalg.vector_norm(
                    torch.tensor(self.cfg.sim.gravity, device=self.device)
                ).item()
                * self.cfg.sim.dt
                * 1.01
            )
            scene_valid = (
                robot_stable
                & (contact_force <= self.cfg.reset_maximum_initial_contact_force_n)
                & (fruit_linear_speed <= pinned_speed_tolerance)
                & (fruit_angular_speed <= self.cfg.reset_fruit_angular_velocity_tolerance_rad_s)
                & (anchor_displacement <= self.cfg.reset_maximum_anchor_displacement_m)
            )
            retry_contact = (
                validate_mask
                & (contact_force > self.cfg.reset_maximum_initial_contact_force_n)
                & (self._reset_collision_retries < self.cfg.maximum_collision_retries)
            )
            if bool(retry_contact.any()):
                retry_ids = retry_contact.nonzero(as_tuple=False).squeeze(1)
                self._reset_collision_retries[retry_ids] += 1
                self._reset_contact_rejection_count += int(len(retry_ids))
                self._place_fruit_at_precontact(retry_ids)
                scene_valid[retry_ids] = False
            self._reset_validation_stable_steps = torch.where(
                validate_mask & scene_valid,
                self._reset_validation_stable_steps + 1,
                torch.zeros_like(self._reset_validation_stable_steps),
            )
            ready = validate_mask & (
                self._reset_validation_stable_steps >= self.cfg.reset_required_stable_steps
            )
            if bool(ready.any()):
                ready_count = int(ready.sum().item())
                ready_ids = ready.nonzero(as_tuple=False).squeeze(1)
                settled_displacement = (
                    self._tomato.data.root_pos_w.torch[ready] - self._stem_anchor_pos_w[ready]
                )
                self._ee_reference_pose_w[ready, :3] += settled_displacement
                self._grasp_frame_origin_w[ready] = self._tomato.data.root_pos_w.torch[ready]
                self._tomato.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((ready_count, 6), device=self.device),
                    env_ids=ready_ids,
                )
                self._reset_pending[ready] = False
                self.episode_length_buf[ready] = 0
                self._reset_completed_count += ready_count

        place_mask = (
            pending
            & ~self._reset_fruit_placed
            & (self._reset_elapsed_steps >= self.cfg.reset_minimum_robot_settle_steps)
            & (self._reset_robot_stable_steps >= self.cfg.reset_required_stable_steps)
        )
        if bool(place_mask.any()):
            self._place_fruit_at_precontact(place_mask.nonzero(as_tuple=False).squeeze(1))

        timed_out = self._reset_pending & (
            self._reset_elapsed_steps >= self.cfg.reset_maximum_steps
        )
        if bool(timed_out.any()):
            count = int(timed_out.sum().item())
            first_env = int(timed_out.nonzero(as_tuple=False)[0, 0].item())
            first_error = (joint_target[first_env] - joint_pos[first_env]).abs()
            first_velocity = joint_vel[first_env].abs()
            error_local_id = int(first_error.argmax().item())
            velocity_local_id = int(first_velocity.argmax().item())
            first_fruit_linear_speed = float(
                torch.linalg.vector_norm(
                    self._tomato.data.root_lin_vel_w.torch[first_env]
                ).item()
            )
            first_fruit_angular_speed = float(
                torch.linalg.vector_norm(
                    self._tomato.data.root_ang_vel_w.torch[first_env]
                ).item()
            )
            first_anchor_displacement = float(
                torch.linalg.vector_norm(
                    self._tomato.data.root_pos_w.torch[first_env]
                    - self._stem_anchor_pos_w[first_env]
                ).item()
            )
            first_contact_force = float(self._finger_force_magnitudes()[first_env].max().item())
            first_tip_positions = torch.stack(
                tuple(sensor.data.pos_w.torch[first_env, 0] for sensor in self._contact_sensors.values())
            )
            first_tip_center_distances = torch.linalg.vector_norm(
                first_tip_positions - self._tomato.data.root_pos_w.torch[first_env], dim=1
            )
            raise RuntimeError(
                "pre-grasp reset validation timed out for "
                f"{count} environment(s); max_joint_error_rad="
                f"{float(joint_error[timed_out].max().item()):.5f}, "
                f"max_error_joint={self._robot.joint_names[self._reset_joint_ids[error_local_id]]}, "
                f"max_error_joint_position_rad={float(joint_pos[first_env, error_local_id].item()):.5f}, "
                f"max_error_joint_target_rad={float(joint_target[first_env, error_local_id].item()):.5f}, "
                f"arm_joint_error_rad={float(arm_joint_error[first_env].item()):.5f}, "
                f"arm_joint_velocity_rad_s={float(maximum_arm_joint_velocity[first_env].item()):.5f}, "
                f"hand_joint_step_delta_rad="
                f"{float(maximum_hand_joint_step_delta[first_env].item()):.5f}, "
                f"max_joint_velocity_rad_s="
                f"{float(maximum_joint_velocity[timed_out].max().item()):.5f}, "
                f"max_velocity_joint="
                f"{self._robot.joint_names[self._reset_joint_ids[velocity_local_id]]}, "
                f"fruit_linear_speed_m_s={first_fruit_linear_speed:.5f}, "
                f"fruit_angular_speed_rad_s={first_fruit_angular_speed:.5f}, "
                f"anchor_displacement_m={first_anchor_displacement:.5f}, "
                f"initial_contact_force_n={first_contact_force:.5f}"
                f", fruit_axes_m={self._fruit_axes_m[first_env].tolist()}"
                f", collision_retries={int(self._reset_collision_retries[first_env].item())}"
                f", fingertip_force_n={self._finger_force_magnitudes()[first_env].tolist()}"
                f", fingertip_center_distance_m={first_tip_center_distances.tolist()}"
            )

        self.extras["reset"] = {
            "pending_count": int(self._reset_pending.sum().item()),
            "fruit_placed_count": int((self._reset_pending & self._reset_fruit_placed).sum().item()),
            "completed_count_total": self._reset_completed_count,
            "contact_rejection_count_total": self._reset_contact_rejection_count,
            "maximum_collision_retry_count": int(self._reset_collision_retries[pending].max().item()),
            "maximum_joint_position_error_rad": float(joint_error[pending].max().item()),
            "maximum_arm_joint_position_error_rad": float(
                arm_joint_error[pending].max().item()
            ),
            "maximum_hand_joint_position_error_rad": float(
                hand_joint_error[pending].max().item()
            ),
            "maximum_joint_velocity_rad_s": float(maximum_joint_velocity[pending].max().item()),
            "maximum_hand_joint_step_delta_rad": float(
                maximum_hand_joint_step_delta[pending].max().item()
            ),
        }

    def _move_stem_visual(self, env_ids: torch.Tensor, positions_w: torch.Tensor) -> None:
        self._stem_visual_positions_w[env_ids] = positions_w
        self._stem_view.set_world_poses(
            positions=wp.from_torch(self._stem_visual_positions_w.contiguous(), dtype=wp.vec3)
        )

    def _apply_stem_wrench(self) -> None:
        if not self.cfg.enable_stem_wrench:
            self._last_stem_force_w.zero_()
            self._last_stem_torque_w.zero_()
            self._last_stem_force_b.zero_()
            self._last_stem_torque_b.zero_()
            self._tomato.permanent_wrench_composer.set_forces_and_torques_index(
                forces=self._last_stem_force_b.unsqueeze(1),
                torques=self._last_stem_torque_b.unsqueeze(1),
                is_global=False,
            )
            return
        fruit_pose = self._tomato.data.root_pose_w.torch
        fruit_lin_vel = self._tomato.data.root_lin_vel_w.torch
        fruit_ang_vel = self._tomato.data.root_ang_vel_w.torch
        position_error = self._stem_anchor_pos_w - fruit_pose[:, :3]
        rotation_error = quat_box_minus(self._stem_anchor_quat_w, fruit_pose[:, 3:7])
        linear_omega = self.cfg.stem_linear_natural_frequency_rad_s
        gravity_w = torch.tensor(self.cfg.sim.gravity, device=self.device)
        commanded_acceleration = (
            linear_omega * linear_omega * position_error
            - 2.0 * self.cfg.stem_linear_damping_ratio * linear_omega * fruit_lin_vel
            - gravity_w.unsqueeze(0)
        )
        force = self._fruit_mass_kg.unsqueeze(1) * commanded_acceleration
        angular_omega = self.cfg.stem_angular_natural_frequency_rad_s
        inertia_proxy = (
            0.4
            * self._fruit_mass_kg
            * torch.square(self._fruit_axes_m).mean(dim=1)
        )
        commanded_angular_acceleration = (
            angular_omega * angular_omega * rotation_error
            - 2.0 * self.cfg.stem_angular_damping_ratio * angular_omega * fruit_ang_vel
        )
        torque = inertia_proxy.unsqueeze(1) * commanded_angular_acceleration
        force_norm = torch.linalg.vector_norm(force, dim=1, keepdim=True).clamp_min(1.0e-6)
        torque_norm = torch.linalg.vector_norm(torque, dim=1, keepdim=True).clamp_min(1.0e-6)
        force = force * (self.cfg.maximum_stem_force_n / force_norm).clamp(max=1.0)
        torque = torque * (self.cfg.maximum_stem_torque_nm / torque_norm).clamp(max=1.0)
        # Reset constructs and validates the scene deterministically.  The
        # compliant stem is task physics and is enabled only for ready envs.
        attached = ~self._stem_detached & ~self._reset_pending
        force = force * attached.unsqueeze(1)
        torque = torque * attached.unsqueeze(1)
        instantaneous_force = self._instantaneous_finger_force_magnitudes()
        instantaneous_contact = (
            instantaneous_force >= self.cfg.minimum_contact_force_n
        )
        self._contact_substep_counts += instantaneous_contact.long()
        stem_load_ratio = self._directional_stem_load_ratio(
            force,
            torque,
        )
        self._stem_overload_substeps.copy_(
            update_overload_streak(
                self._stem_overload_substeps,
                stem_load_ratio >= 1.0,
                attached,
            )
        )
        self._stem_threshold_reached_this_step |= attached & (
            self._stem_overload_substeps
            >= self.cfg.stem_break_persistence_substeps
        )
        self._last_stem_force_w.copy_(force)
        self._last_stem_torque_w.copy_(torque)
        self._last_stem_force_b.copy_(
            quat_apply_inverse(fruit_pose[:, 3:7], self._last_stem_force_w)
        )
        self._last_stem_torque_b.copy_(
            quat_apply_inverse(fruit_pose[:, 3:7], self._last_stem_torque_w)
        )

        # Do not use RigidObject's permanent wrench composer for the stem.
        # In Isaac Lab 3.0.0-beta2.patch1 the composer buffer reports the
        # requested wrench but that wrench is not applied to this cloned,
        # non-uniformly-scaled rigid body.  Apply the mathematically equivalent
        # per-substep impulse to the *current* post-contact velocity instead.
        # This retains PhysX contact impulses while making the compliant
        # attachment deterministic across vectorized environments.
        zero_force_b = torch.zeros_like(self._last_stem_force_b)
        self._tomato.permanent_wrench_composer.set_forces_and_torques_index(
            forces=zero_force_b.unsqueeze(1),
            torques=zero_force_b.unsqueeze(1),
            is_global=False,
        )
        constrained = attached
        if bool(constrained.any()):
            env_ids = constrained.nonzero(as_tuple=False).squeeze(1)
            root_velocity = torch.cat((fruit_lin_vel, fruit_ang_vel), dim=1)[env_ids].clone()
            physics_dt = float(self.cfg.sim.dt)
            root_velocity[:, :3] += (
                force[env_ids] / self._fruit_mass_kg[env_ids].unsqueeze(1)
            ) * physics_dt
            root_velocity[:, 3:] += (
                torque[env_ids] / inertia_proxy[env_ids].clamp_min(1.0e-8).unsqueeze(1)
            ) * physics_dt
            self._tomato.write_root_velocity_to_sim_index(
                root_velocity=root_velocity,
                env_ids=env_ids,
            )

    def _hold_pending_fruit_at_anchor(self) -> None:
        """Pin reset fruit to its attachment frame until validation completes."""

        hold = self._reset_pending
        if not bool(hold.any()):
            return
        env_ids = hold.nonzero(as_tuple=False).squeeze(1)
        root_pose = torch.cat(
            (self._stem_anchor_pos_w[env_ids], self._stem_anchor_quat_w[env_ids]), dim=1
        )
        root_velocity = torch.zeros((len(env_ids), 6), device=self.device)
        self._tomato.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._tomato.write_root_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        physical_force = self._finger_force_magnitudes()
        contact_force, force_valid, force_age_norm = self._sim2real.measure_tactile(
            physical_force
        )
        contact_force = torch.where(force_valid, contact_force, 0.0)
        if self._last_pressure_observation_step != self.common_step_counter:
            self._pressure_rate_norm.copy_(
                ((contact_force - self._previous_measured_pressure) / 20.0).clamp(
                    -1.0, 1.0
                )
            )
            self._previous_measured_pressure.copy_(contact_force)
            self._last_pressure_observation_step = self.common_step_counter
        episode_progress = (
            self.episode_length_buf.float() / float(self.max_episode_length)
        ).clamp(0.0, 1.0)
        ee_pose_w = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        position_error, rotation_error = compute_pose_error(
            self._ee_reference_pose_w[:, :3],
            self._ee_reference_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )
        grasp_from_world = self._grasp_frame_basis_w.transpose(1, 2)
        position_error = torch.bmm(grasp_from_world, position_error.unsqueeze(2)).squeeze(2)
        rotation_error = torch.bmm(grasp_from_world, rotation_error.unsqueeze(2)).squeeze(2)
        ee_error = torch.cat((position_error, rotation_error), dim=1)
        ee_twist_w = self._robot.data.body_state_w.torch[:, self._ee_body_id, 7:13]
        ee_linear = torch.bmm(grasp_from_world, ee_twist_w[:, :3].unsqueeze(2)).squeeze(2)
        ee_angular = torch.bmm(grasp_from_world, ee_twist_w[:, 3:6].unsqueeze(2)).squeeze(2)
        ee_twist = torch.cat((ee_linear, ee_angular), dim=1)
        joint_pos = (
            self._robot.data.joint_pos.torch[:, self._arm_joint_ids]
            + self._sim2real.joint_encoder_offset
        )
        joint_vel = self._robot.data.joint_vel.torch[:, self._arm_joint_ids]
        limits = self._robot.data.joint_pos_limits.torch[:, self._arm_joint_ids]
        joint_pos_norm = (2.0 * (joint_pos - limits[..., 0]) / (limits[..., 1] - limits[..., 0]) - 1.0).clamp(-1.0, 1.0)
        velocity_limits = self._robot.data.joint_vel_limits.torch[:, self._arm_joint_ids]
        joint_vel_norm = (joint_vel / velocity_limits.clamp_min(1.0e-6)).clamp(-1.0, 1.0)
        hand_joint_pos = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
        hand_lower = torch.tensor(
            [RIGHT_HAND_CONTROL_RAD[name][0] for name in self._hand_joint_names],
            device=self.device,
        )
        hand_upper = torch.tensor(
            [RIGHT_HAND_CONTROL_RAD[name][2] for name in self._hand_joint_names],
            device=self.device,
        )
        hand_joint_pos_norm = (
            2.0 * (hand_joint_pos - hand_lower) / (hand_upper - hand_lower) - 1.0
        ).clamp(-1.0, 1.0)
        hand_joint_vel = self._robot.data.joint_vel.torch[:, self._hand_joint_ids]
        hand_velocity_limits = self._robot.data.joint_vel_limits.torch[
            :, self._hand_joint_ids
        ]
        hand_joint_vel_norm = (
            hand_joint_vel / hand_velocity_limits.clamp_min(1.0e-6)
        ).clamp(-1.0, 1.0)
        pressure_norm = (contact_force / 20.0).clamp(0.0, 1.0)
        gravity_w = torch.tensor(self.cfg.sim.gravity, device=self.device)
        gravity_w = gravity_w / torch.linalg.vector_norm(gravity_w).clamp_min(1.0e-8)
        gravity_g = torch.bmm(
            grasp_from_world,
            gravity_w.expand(self.num_envs, -1).unsqueeze(2),
        ).squeeze(2)
        axis_lower = torch.tensor(self.cfg.fruit_axis_lower_m, device=self.device)
        axis_upper = torch.tensor(self.cfg.fruit_axis_upper_m, device=self.device)
        axes_norm = (
            2.0 * (self._sim2real.measured_axes_m - axis_lower) / (axis_upper - axis_lower) - 1.0
        ).clamp(-1.0, 1.0)
        axes_norm = torch.where(
            self._sim2real.target_valid.unsqueeze(1), axes_norm, torch.zeros_like(axes_norm)
        )
        state_valid = torch.ones((self.num_envs, 2), device=self.device)
        target_state_valid = torch.cat(
            (self._sim2real.target_valid.float().unsqueeze(1), state_valid), dim=1
        )
        target_state_age = torch.cat(
            (
                (self._sim2real.target_age_s / self._sim2real.TARGET_MAX_AGE_S).unsqueeze(1),
                torch.zeros((self.num_envs, 2), device=self.device),
            ),
            dim=1,
        ).clamp(0.0, 1.0)
        observation = torch.cat(
            (
                ee_error,
                ee_twist,
                joint_pos_norm,
                joint_vel_norm,
                hand_joint_pos_norm,
                hand_joint_vel_norm,
                pressure_norm,
                self._pressure_rate_norm,
                gravity_g,
                self._previous_actions,
                target_state_valid,
                force_valid.float(),
                target_state_age,
                force_age_norm,
                axes_norm,
                episode_progress.unsqueeze(1),
            ),
            dim=1,
        )
        if observation.shape[1] != OBSERVATION_DIM:
            raise RuntimeError(f"actor observation drifted to {observation.shape[1]} values")
        self._synergy.copy_(self._estimated_hand_synergy())
        stem_load_ratio = self._directional_stem_load_ratio(
            self._last_stem_force_w,
            self._last_stem_torque_w,
        ).clamp(0.0, 1.0)
        critic = self._critic_state(observation, physical_force, stem_load_ratio)
        if self.cfg.strict_runtime_checks:
            for name, value in (("policy", observation), ("critic", critic)):
                if value.dtype != torch.float32:
                    raise TypeError(f"{name} observation must use float32, got {value.dtype}")
                if not bool(torch.isfinite(value).all()):
                    bad = (~torch.isfinite(value)).nonzero(as_tuple=False)[:16].tolist()
                    raise ValueError(
                        f"{name} observation contains NaN or Inf at indices {bad}"
                    )
        return {"policy": observation, "critic": critic}

    def _get_rewards(self) -> torch.Tensor:
        contact_force = self._finger_force_magnitudes()
        contact = contact_force >= self.cfg.minimum_contact_force_n
        active = (~self._reset_pending).float()
        ee_position = self._robot.data.body_pos_w.torch[:, self._ee_body_id]
        approach_distance = torch.linalg.vector_norm(
            ee_position - self._ee_reference_pose_w[:, :3], dim=1
        )
        approach_potential = torch.exp(
            -approach_distance / self.cfg.approach_potential_distance_scale_m
        )
        # Use the PPO discount for every dense potential.  The previous
        # clipped distance delta could be harvested by retreating in one large
        # step and returning in several smaller steps; at 22M transitions it
        # paid +7.1 per timeout episode.  Returning the potential to zero on
        # stage completion/failure makes such cycles non-positive.
        shaping_terminal = (
            self._stem_detached
            | self._dropped
            | self._excess_force
            | self._task_timeout
        )
        approach_potential = torch.where(
            shaping_terminal,
            torch.zeros_like(approach_potential),
            approach_potential,
        )
        approach_progress = (
            self.cfg.reward_potential_discount * approach_potential
            - self._previous_approach_potential
        )
        self._previous_approach_potential.copy_(approach_potential)
        # Reward the one grasp property that separates retained fruit from
        # drops in the physical audits: signed support against gravity.  Total
        # normal force alone cannot distinguish a load-bearing grasp from a
        # hard downward squeeze.  This simulation-only potential prescribes no
        # finger roles or hand shape and never gates actions or fracture.
        # A first fingertip contact commonly pushes down before the thumb and
        # opposing fingers arrive.  Negative shaping here created an audited
        # -0.64 to -0.99 exploration barrier that almost cancelled the +1
        # contact event.  Only upward load-bearing is progress; bad outcomes
        # are already handled by the terminal drop/excess/timeout penalties.
        grasp_potential = positive_support_potential(
            self._gravity_support_ratio()
        )
        # Detachment is not terminal: the same grasp must support the fruit
        # throughout HOLD.  Clearing this potential at fracture would refund a
        # bad downward grasp and charge a good upward grasp at exactly the
        # wrong transition.  Only true episode terminals return the potential
        # to zero, preserving policy-invariant shaping.
        support_terminal = (
            self._success
            | self._dropped
            | self._excess_force
            | self._task_timeout
        )
        grasp_potential = torch.where(
            support_terminal,
            torch.zeros_like(grasp_potential),
            grasp_potential,
        )
        grasp_progress = (
            self.cfg.reward_potential_discount * grasp_potential
            - self._previous_grasp_potential
        )
        self._previous_grasp_potential.copy_(grasp_potential)
        soft_force = (
            (contact_force - self.cfg.soft_contact_force_n).clamp_min(0.0)
            / (self.cfg.maximum_fingertip_force_n - self.cfg.soft_contact_force_n)
        ).clamp(0.0, 1.0).mean(dim=1)
        arm_position = self._robot.data.joint_pos.torch[:, self._arm_joint_ids]
        arm_limits = self._robot.data.joint_pos_limits.torch[:, self._arm_joint_ids]
        arm_position_unit = (
            2.0
            * (arm_position - arm_limits[..., 0])
            / (arm_limits[..., 1] - arm_limits[..., 0]).clamp_min(1.0e-6)
            - 1.0
        ).abs()
        joint_limit = ((arm_position_unit - 0.95) / 0.05).clamp(0.0, 1.0).mean(dim=1)
        action_magnitude = torch.square(self._actions).sum(dim=1)
        action_rate = torch.square(self._actions - self._previous_actions).sum(dim=1)
        # Use the exact same load semantics as physical fracture.  Lateral
        # bending and compression cannot detach the fruit, so neither the
        # critic nor the policy should be told that they are progress.
        stem_load_potential = self._directional_stem_load_ratio(
            self._last_stem_force_w,
            self._last_stem_torque_w,
        ).clamp(0.0, 1.0)
        stem_load_potential = torch.where(
            shaping_terminal,
            torch.zeros_like(stem_load_potential),
            stem_load_potential,
        )
        stem_load_progress = (
            self.cfg.reward_potential_discount * stem_load_potential
            - self._previous_stem_load_potential
        )
        self._previous_stem_load_potential.copy_(stem_load_potential)
        # HOLD shaping is a discounted potential, not an unconditional
        # per-frame payment.  A retained success keeps the terminal potential;
        # a later drop returns it to zero and therefore cancels the preceding
        # HOLD credit.  This removes the audited loophole where a trajectory
        # could hold briefly, drop the fruit, and still finish with positive
        # return, while preserving dense feedback throughout the 20-step hold.
        retention_potential = torch.where(
            self._retained_without_slip,
            self._hold_streak.float(),
            torch.zeros_like(self._previous_retention_potential),
        )
        retention_progress = (
            self.cfg.reward_potential_discount * retention_potential
            - self._previous_retention_potential
        )
        self._previous_retention_potential.copy_(retention_potential)
        reward_phase = self._phase_at_action
        # Progress labels generate one-shot telemetry/reward milestones but
        # never affect action authority, fracture, termination, or controller
        # state.  The approach potential remains available until detachment;
        # PPO is free to re-approach after losing contact.
        contact_completed = (
            (reward_phase < 1)
            & (self._contact_streak >= CONTACT_STREAK_STEPS)
        )
        grasp_completed = (reward_phase < 2) & self._grasp_stable
        terms = {
            "contact": active
            * contact_completed.float()
            * self.cfg.reward_contact_scale,
            "approach_progress": active
            * (~self._stem_detached).float()
            * approach_progress
            * self.cfg.reward_approach_progress_scale,
            "grasp_progress": active
            * (~self._stem_detached).float()
            * grasp_progress
            * self.cfg.reward_grasp_progress_scale,
            "grasp": active
            * grasp_completed.float()
            * self.cfg.reward_grasp_scale,
            "stem_load_progress": active
            * (~self._stem_detached).float()
            * stem_load_progress
            * self.cfg.reward_stem_load_progress_scale,
            "break_once": self._stem_break_event.float() * self.cfg.reward_break_once_scale,
            "retention": active
            * self._stem_detached.float()
            * retention_progress
            * self.cfg.reward_retention_scale,
            "success": self._success.float() * self.cfg.reward_success_scale,
            "drop": self._dropped.float() * self.cfg.reward_drop_scale,
            "excess_force": self._excess_force.float()
            * self.cfg.reward_excess_force_scale,
            "timeout": self._task_timeout.float() * self.cfg.reward_timeout_scale,
            "soft_contact_force": active
            * soft_force
            * self.cfg.reward_soft_contact_force_scale,
            "joint_limit": active
            * joint_limit
            * self.cfg.reward_joint_limit_scale,
            "action_magnitude": active
            * action_magnitude
            * self.cfg.reward_action_magnitude_scale,
            "action_rate": active
            * action_rate
            * self.cfg.reward_action_rate_scale,
        }
        self._last_reward_terms = {name: value.detach().clone() for name, value in terms.items()}
        for name, value in terms.items():
            self._episode_sums[name] += value
        reward = torch.stack(tuple(terms.values())).sum(dim=0)
        if self.cfg.strict_runtime_checks:
            if reward.dtype != torch.float32:
                raise TypeError(f"reward must use float32, got {reward.dtype}")
            if not bool(torch.isfinite(reward).all()):
                raise ValueError("reward contains NaN or Inf")
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        contact_force = self._finger_force_magnitudes()
        filtered_contact = contact_force >= self.cfg.minimum_contact_force_n
        contact = filtered_contact
        active = ~self._reset_pending
        # These are logging milestones, not gates.  They deliberately make no
        # assumption about which finger must oppose which other finger.
        multi_finger_contact = contact.sum(dim=1) >= 2
        load_bearing_grasp = (
            contact_force.sum(dim=1)
            >= self.cfg.minimum_grasp_total_normal_force_n
        )
        stable_contact = multi_finger_contact & active
        self._contact_streak = torch.where(
            stable_contact,
            self._contact_streak + 1,
            torch.zeros_like(self._contact_streak),
        )
        _, _, force_balance_quality = self._grasp_force_closure_quality()
        deployable_grasp_stable = (
            multi_finger_contact
            & load_bearing_grasp
            & active
            & (force_balance_quality >= 0.50)
        )
        self._grasp_stable = deployable_grasp_stable
        self._phase_steps += active.long()
        # Fracture is an outcome of the physical attachment load.  It is not
        # permissioned by a phase label, a hidden force-closure test, or a
        # prescribed axial displacement.  Bad impact-induced breaks naturally
        # receive the much larger drop penalty.
        break_now = (
            active
            & ~self._stem_detached
            & self._stem_threshold_reached_this_step
        )
        self._stem_break_event.copy_(break_now)
        self._stem_detached |= break_now
        if bool(break_now.any()):
            break_ids = break_now.nonzero(as_tuple=False).squeeze(1)
            fingertip_linear_velocity = self._robot.data.body_lin_vel_w.torch[
                :, self._fingertip_frame_body_ids
            ].mean(dim=1)
            hand_angular_velocity = self._robot.data.body_ang_vel_w.torch[
                :, self._ee_body_id
            ]
            break_velocity = torch.cat(
                (
                    fingertip_linear_velocity[break_ids],
                    hand_angular_velocity[break_ids],
                ),
                dim=1,
            )
            self._tomato.write_root_velocity_to_sim_index(
                root_velocity=break_velocity,
                env_ids=break_ids,
            )
        # Monotone deployable milestones are observations and metrics only.
        reached_contact = self._contact_streak >= CONTACT_STREAK_STEPS
        self._reached_contact |= reached_contact
        self._reached_grasp |= self._grasp_stable
        self._reached_detach |= self._stem_detached
        self._phase = torch.where(reached_contact, torch.maximum(self._phase, torch.ones_like(self._phase)), self._phase)
        self._phase = torch.where(self._grasp_stable, torch.maximum(self._phase, torch.full_like(self._phase, 2)), self._phase)
        self._phase = torch.where(self._stem_detached, torch.full_like(self._phase, 3), self._phase)
        fruit_pos = self._tomato.data.root_pos_w.torch
        fingertip_center = self._robot.data.body_pos_w.torch[
            :, self._fingertip_frame_body_ids
        ].mean(dim=1)
        # right_hand_base_link is the wrist/palm attachment frame.  On the
        # physical G20 it is already about 23--25 cm from a correctly enclosed
        # fruit, so the former 22 cm palm-distance test marked every fracture
        # as a drop in the same step.  Separation from the fingertip envelope
        # is the geometry that actually distinguishes retention from a fall.
        self._dropped = self._stem_detached & (
            torch.linalg.vector_norm(fruit_pos - fingertip_center, dim=1)
            > self.cfg.maximum_retained_fruit_to_fingertip_center_distance_m
        )
        self._excess_force = contact_force.max(dim=1).values > self.cfg.maximum_fingertip_force_n
        # The frozen skill contract defines successful retention as detached
        # and not slipped/dropped during the HOLD window.  Requiring the exact
        # three-finger GRASP threshold on every HOLD sample was stronger than
        # that contract and reset the streak on normal elastic contact dips.
        self._retained_without_slip = (
            self._stem_detached
            & ~self._dropped
            & ~self._excess_force
        )
        self._hold_streak = torch.where(
            self._retained_without_slip,
            self._hold_streak + 1,
            torch.zeros_like(self._hold_streak),
        )
        self._success = (
            self._stem_detached
            & self._retained_without_slip
            & (self._hold_streak >= self.cfg.success_hold_steps)
            & ~self._dropped
            & ~self._excess_force
        )
        task_timeout = (
            active
            & (self.episode_length_buf >= self.max_episode_length - 1)
            & ~self._success
            & ~self._dropped
            & ~self._excess_force
        )
        self._task_timeout.copy_(task_timeout)
        terminated = self._success | self._dropped | self._excess_force | task_timeout
        # The finite picking deadline is part of the task and therefore a true
        # terminal failure, not an external truncation that should bootstrap.
        time_out = torch.zeros_like(terminated)
        if self.cfg.strict_runtime_checks:
            if terminated.dtype != torch.bool or time_out.dtype != torch.bool:
                raise TypeError("terminated and truncated must be bool tensors")
            if bool((terminated & time_out).any()):
                raise RuntimeError("terminated and truncated must be disjoint")
        self.extras["step"] = {
            "success_count": int(self._success.sum().item()),
            "drop_count": int(self._dropped.sum().item()),
            "excess_force_count": int(self._excess_force.sum().item()),
            "task_timeout_count": int(task_timeout.sum().item()),
            "stem_detached_count": int(self._stem_detached.sum().item()),
            "maximum_tactile_proxy_n": float(contact_force.max().item()),
            "phase_histogram": [int((self._phase == index).sum().item()) for index in range(4)],
        }
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        super()._reset_idx(env_ids)
        self._log_reset(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        lateral_radius = self._fruit_axes_m[env_ids, :2].max(dim=1).values
        radius_unit = (
            (lateral_radius - self.cfg.pregrasp_small_fruit_lateral_radius_m)
            / (
                self.cfg.pregrasp_large_fruit_lateral_radius_m
                - self.cfg.pregrasp_small_fruit_lateral_radius_m
            )
        ).clamp(0.0, 1.0)
        self._synergy[env_ids, 0] = self.cfg.pregrasp_small_fruit_closure + radius_unit * (
            self.cfg.pregrasp_large_fruit_closure - self.cfg.pregrasp_small_fruit_closure
        )
        self._synergy[env_ids, 1] = self.cfg.pregrasp_thumb_opposition
        self._synergy[env_ids, 2] = self.cfg.pregrasp_finger_spread
        self._phase[env_ids] = 0
        self._phase_at_action[env_ids] = 0
        self._detach_start_ee_pos_w[env_ids] = self._robot.data.body_pos_w.torch[
            env_ids, self._ee_body_id
        ]
        self._reached_contact[env_ids] = False
        self._reached_grasp[env_ids] = False
        self._reached_detach[env_ids] = False
        self._phase_steps[env_ids] = 0
        self._reset_pending[env_ids] = False
        self._reset_fruit_placed[env_ids] = False
        self._reset_elapsed_steps[env_ids] = 0
        self._reset_robot_stable_steps[env_ids] = 0
        self._reset_validation_stable_steps[env_ids] = 0
        self._reset_collision_retries[env_ids] = 0
        self._reset_previous_hand_joint_pos[env_ids] = self._robot.data.joint_pos.torch[
            env_ids
        ][:, self._reset_hand_joint_ids]
        self._contact_streak[env_ids] = 0
        self._hold_streak[env_ids] = 0
        self._stem_detached[env_ids] = False
        self._stem_break_event[env_ids] = False
        self._stem_threshold_reached_this_step[env_ids] = False
        self._stem_overload_substeps[env_ids] = 0
        self._grasp_stable[env_ids] = False
        self._retained_without_slip[env_ids] = False
        self._success[env_ids] = False
        self._dropped[env_ids] = False
        self._excess_force[env_ids] = False
        self._task_timeout[env_ids] = False
        self._previous_approach_potential[env_ids] = 0.0
        self._previous_grasp_potential[env_ids] = 0.0
        self._previous_stem_load_potential[env_ids] = 0.0
        self._previous_retention_potential[env_ids] = 0.0

        joint_pos = self._nominal_joint_position_targets[env_ids].clone()
        self._joint_position_targets[env_ids] = joint_pos
        self._set_hand_targets(env_ids)
        reset_position = self._joint_position_targets[env_ids][
            :, self._actuated_joint_ids
        ]
        reset_velocity = torch.zeros_like(reset_position)
        self._robot.write_joint_position_to_sim_index(
            position=reset_position,
            joint_ids=self._actuated_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=reset_velocity,
            joint_ids=self._actuated_joint_ids,
            env_ids=env_ids,
        )

        fruit_pose = self._tomato.data.default_root_pose.torch[env_ids].clone()
        staging_offset = torch.tensor(self.cfg.reset_staging_offset_m, device=self.device)
        fruit_pose[:, :3] = self.scene.env_origins[env_ids] + staging_offset
        fruit_vel = torch.zeros((len(env_ids), 6), device=self.device)
        self._tomato.write_root_pose_to_sim_index(root_pose=fruit_pose, env_ids=env_ids)
        self._tomato.write_root_velocity_to_sim_index(root_velocity=fruit_vel, env_ids=env_ids)
        self._stem_anchor_pos_w[env_ids] = fruit_pose[:, :3]
        self._stem_anchor_quat_w[env_ids] = fruit_pose[:, 3:7]
        staging_stem_pos = fruit_pose[:, :3].clone()
        staging_stem_pos[:, 2] += self.cfg.stem_visual_height_m * 0.5
        self._move_stem_visual(env_ids, staging_stem_pos)
        # Refresh kinematics without advancing simulation time.  Reset states
        # are complete before they enter an on-policy rollout.
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        gravity_force = self._fruit_mass_kg[env_ids] * abs(float(self.cfg.sim.gravity[2]))
        self._sim2real.reset(
            env_ids,
            true_axes_m=self._fruit_axes_m[env_ids],
            gravity_force_n=gravity_force,
        )
        self._sim2real.apply_fruit_material(self._tomato, env_ids)
        ee_pose = self._robot.data.body_pose_w.torch[env_ids, self._ee_body_id].clone()
        self._ee_command_pose_w[env_ids] = ee_pose
        self._ee_reference_pose_w[env_ids] = ee_pose
        self._place_fruit_at_precontact(env_ids)
        self._reset_previous_hand_joint_pos[env_ids] = self._robot.data.joint_pos.torch[
            env_ids
        ][:, self._reset_hand_joint_ids]
        self._reset_completed_count += len(env_ids)
        self._tactile_force_filtered[env_ids] = 0.0
        self._previous_measured_pressure[env_ids] = 0.0
        self._pressure_rate_norm[env_ids] = 0.0
        self._contact_substep_counts[env_ids] = 0
        self._last_stem_force_w[env_ids] = 0.0
        self._last_stem_torque_w[env_ids] = 0.0
        self._last_stem_force_b[env_ids] = 0.0
        self._last_stem_torque_b[env_ids] = 0.0

    def _place_fruit_at_precontact(self, env_ids: torch.Tensor) -> None:
        # Geometry uses the tiny fixed terminal frames (link4/link5) while
        # tactile force comes from the actual distal pad bodies (link3/link4).
        # Conflating them was the reason valid pad contacts were invisible.
        fingertip_pos = self._robot.data.body_pos_w.torch[
            env_ids[:, None], self._fingertip_frame_body_ids
        ]
        support_fingers = fingertip_pos[:, 1:5]
        support_center = support_fingers.mean(dim=1)
        thumb_tip = fingertip_pos[:, 0]
        ee_pose = self._robot.data.body_pose_w.torch[env_ids, self._ee_body_id]
        approach_axis = torch.nn.functional.normalize(support_center - ee_pose[:, :3], dim=1)
        thumb_delta = thumb_tip - support_center
        thumb_in_support_plane = thumb_tip - (
            (thumb_delta * approach_axis).sum(dim=1, keepdim=True) * approach_axis
        )
        # Put the fruit inside the open hand aperture, between the thumb and
        # support fingers.  The former placement projected a whole fruit
        # radius beyond the support fingertips; flexion then moved every
        # finger away from the fruit and made thumb opposition unreachable.
        aperture_center = 0.5 * (support_center + thumb_tip)

        pull_axis = -approach_axis
        opposition_axis = support_center - thumb_in_support_plane
        opposition_axis = opposition_axis - (
            (opposition_axis * pull_axis).sum(dim=1, keepdim=True) * pull_axis
        )
        opposition_axis = torch.nn.functional.normalize(opposition_axis, dim=1)
        lateral_axis = torch.nn.functional.normalize(
            torch.linalg.cross(pull_axis, opposition_axis, dim=1), dim=1
        )
        opposition_axis = torch.nn.functional.normalize(
            torch.linalg.cross(lateral_axis, pull_axis, dim=1), dim=1
        )
        grasp_basis = torch.stack((opposition_axis, lateral_axis, pull_axis), dim=2)

        fruit_extent_along_approach = torch.sqrt(
            torch.square(self._fruit_axes_m[env_ids] * approach_axis).sum(dim=1)
        )
        fruit_pose = self._tomato.data.root_pose_w.torch[env_ids].clone()
        clearance = (
            self.cfg.initial_fruit_clearance_m
            + self._reset_collision_retries[env_ids].float()
            * self.cfg.collision_retry_clearance_increment_m
        )
        precontact_distance = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.fruit_precontact_distance_range_m
        )
        lateral_offsets = torch.empty((len(env_ids), 2), device=self.device).uniform_(
            -self.cfg.fruit_precontact_lateral_offset_m,
            self.cfg.fruit_precontact_lateral_offset_m,
        )
        precontact_delta_w = (
            approach_axis * precontact_distance.unsqueeze(1)
            + opposition_axis * lateral_offsets[:, 0:1]
            + lateral_axis * lateral_offsets[:, 1:2]
        )
        fruit_pose[:, :3] = aperture_center + precontact_delta_w
        fruit_pose[:, 3:7] = 0.0
        fruit_pose[:, 6] = 1.0
        contact_pose = ee_pose.clone()
        contact_pose[:, :3] += precontact_delta_w + approach_axis * clearance.unsqueeze(1)
        # YOLO/RGB-D and hand-eye uncertainty is sampled once per plan.  It
        # changes the frozen contact reference, not the live proprioceptive
        # measurement every 50 ms.
        plan_position_error_w = torch.bmm(
            grasp_basis,
            self._sim2real.target_position_error_m[env_ids].unsqueeze(2),
        ).squeeze(2)
        contact_pose[:, :3] += plan_position_error_w
        plan_rotvec_g = self._sim2real.target_rotation_error_rad[env_ids]
        plan_angle = torch.linalg.vector_norm(plan_rotvec_g, dim=1)
        plan_axis_g = plan_rotvec_g / plan_angle.clamp_min(1.0e-8).unsqueeze(1)
        plan_axis_w = torch.bmm(grasp_basis, plan_axis_g.unsqueeze(2)).squeeze(2)
        plan_axis_w = torch.where(
            (plan_angle > 1.0e-8).unsqueeze(1),
            plan_axis_w,
            torch.tensor((0.0, 0.0, 1.0), device=self.device),
        )
        plan_rotation_w = quat_from_angle_axis(plan_angle, plan_axis_w)
        contact_pose[:, 3:7] = quat_mul(plan_rotation_w, contact_pose[:, 3:7])
        self._ee_reference_pose_w[env_ids] = contact_pose
        reset_approach_distance = torch.linalg.vector_norm(
            ee_pose[:, :3] - contact_pose[:, :3], dim=1
        )
        self._previous_approach_potential[env_ids] = torch.exp(
            -reset_approach_distance
            / self.cfg.approach_potential_distance_scale_m
        )
        self._grasp_frame_basis_w[env_ids] = grasp_basis
        self._grasp_frame_origin_w[env_ids] = fruit_pose[:, :3]
        stem_visual_pos = fruit_pose[:, :3] + pull_axis * (
            fruit_extent_along_approach + self.cfg.stem_visual_height_m * 0.5
        ).unsqueeze(1)
        fruit_velocity = torch.zeros((len(env_ids), 6), device=self.device)
        self._tomato.write_root_pose_to_sim_index(root_pose=fruit_pose, env_ids=env_ids)
        self._tomato.write_root_velocity_to_sim_index(root_velocity=fruit_velocity, env_ids=env_ids)
        self._stem_anchor_pos_w[env_ids] = fruit_pose[:, :3]
        self._stem_anchor_quat_w[env_ids] = fruit_pose[:, 3:7]
        self._move_stem_visual(env_ids, stem_visual_pos)
        self._reset_fruit_placed[env_ids] = True
        self._reset_validation_stable_steps[env_ids] = 0
        self._tactile_force_filtered[env_ids] = 0.0

    def _set_hand_targets(self, env_ids: torch.Tensor | None = None) -> None:
        """Set only the reset pre-shape from three compact geometry parameters."""

        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids.long()
        synergy = self._synergy[ids]
        for joint_id, name in zip(self._hand_joint_ids, self._hand_joint_names):
            lower, neutral, upper, source = RIGHT_HAND_CONTROL_RAD[name]
            if source == "closure":
                unit = synergy[:, 0]
                target = lower + unit * (upper - lower)
            elif source == "opposition":
                value = synergy[:, 1]
                target = torch.where(
                    value >= 0.0,
                    neutral + value * (upper - neutral),
                    neutral + (-value) * (lower - neutral),
                )
            else:
                value = synergy[:, 2]
                target = torch.where(
                    value >= 0.0,
                    neutral + value * (upper - neutral),
                    neutral + (-value) * (lower - neutral),
                )
            self._joint_position_targets[ids, joint_id] = target
        self._update_mimic_targets(ids)

    def _update_hand_targets_from_joint_action(self, joint_delta: torch.Tensor) -> None:
        """Integrate sixteen independent G20 commands and enforce hard limits."""

        if joint_delta.shape != (self.num_envs, 16):
            raise ValueError(
                f"hand action must have shape ({self.num_envs}, 16), "
                f"got {tuple(joint_delta.shape)}"
            )
        current = self._joint_position_targets[:, self._hand_joint_ids]
        lower = torch.tensor(
            [RIGHT_HAND_CONTROL_RAD[name][0] for name in self._hand_joint_names],
            device=self.device,
        )
        upper = torch.tensor(
            [RIGHT_HAND_CONTROL_RAD[name][2] for name in self._hand_joint_names],
            device=self.device,
        )
        target = torch.maximum(lower, torch.minimum(upper, current + joint_delta))
        target[self._reset_pending] = current[self._reset_pending]
        self._joint_position_targets[:, self._hand_joint_ids] = target
        self._update_mimic_targets(torch.arange(self.num_envs, device=self.device))

    def _update_mimic_targets(self, ids: torch.Tensor) -> None:
        """Derive the five distal followers per hand from mechanical coupling."""

        limits = self._robot.data.joint_pos_limits.torch[ids]
        for follower_name in MIMIC_FOLLOWER_JOINT_NAMES:
            side_prefix = (
                "right_hand_" if follower_name.startswith("right_hand_") else "left_hand_"
            )
            right_name = follower_name.replace("left_hand_", "right_hand_", 1)
            driver_right_name, multiplier, offset = RIGHT_HAND_MIMIC_RAD[right_name]
            driver_name = driver_right_name.replace("right_hand_", side_prefix, 1)
            follower_id = self._robot.joint_names.index(follower_name)
            driver_id = self._robot.joint_names.index(driver_name)
            target = self._joint_position_targets[ids, driver_id] * multiplier + offset
            self._joint_position_targets[ids, follower_id] = target.clamp(
                limits[:, follower_id, 0], limits[:, follower_id, 1]
            )

    def _estimated_hand_synergy(self) -> torch.Tensor:
        """Estimate closure/opposition/spread from deployable joint encoders."""

        position = self._robot.data.joint_pos.torch
        by_source: dict[str, list[torch.Tensor]] = {
            "closure": [],
            "opposition": [],
            "spread": [],
        }
        for name, joint_id in zip(self._hand_joint_names, self._hand_joint_ids):
            lower, neutral, upper, source = RIGHT_HAND_CONTROL_RAD[name]
            joint_position = position[:, joint_id]
            if source == "closure":
                value = ((joint_position - lower) / (upper - lower)).clamp(0.0, 1.0)
            else:
                value = torch.where(
                    joint_position >= neutral,
                    (joint_position - neutral) / max(upper - neutral, 1.0e-6),
                    (joint_position - neutral) / max(neutral - lower, 1.0e-6),
                ).clamp(-1.0, 1.0)
            by_source[source].append(value)
        closure = torch.stack(by_source["closure"], dim=1).mean(dim=1)
        opposition = torch.stack(by_source["opposition"], dim=1).mean(dim=1)
        spread = torch.stack(by_source["spread"], dim=1).mean(dim=1)
        return torch.stack((closure, opposition, spread), dim=1)

    def _ee_pose_in_root(self) -> torch.Tensor:
        ee_pose_w = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        root_pose_w = self._robot.data.root_pose_w.torch
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )
        return torch.cat((ee_pos_b, ee_quat_b), dim=1)

    def _finger_force_magnitudes(self) -> torch.Tensor:
        if not self._contact_sensors:
            return self._tactile_force_filtered
        if self._last_tactile_update_step != self.common_step_counter:
            instantaneous = self._instantaneous_finger_force_magnitudes()
            instantaneous = torch.where(
                self._reset_pending.unsqueeze(1), 0.0, instantaneous
            )
            retention = float(self.cfg.tactile_policy_sample_retention)
            self._tactile_force_filtered.mul_(retention).add_(
                instantaneous, alpha=1.0 - retention
            )
            self._last_tactile_update_step = self.common_step_counter
        return self._tactile_force_filtered

    def _instantaneous_finger_force_magnitudes(self) -> torch.Tensor:
        """Read exact Tomato-filtered forces without temporal filtering."""

        per_finger = []
        for sensor in self._contact_sensors.values():
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                raise RuntimeError("fingertip contact filter did not initialize")
            values = force_matrix.torch[:, 0, 0]
            per_finger.append(torch.linalg.vector_norm(values, dim=-1))
        return torch.stack(per_finger, dim=1).clamp(0.0, 25.0)

    def _grasp_force_closure_quality(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return generic opposition, resultant, and smooth balance quality.

        No finger is assigned a privileged role.  Opposition is the strongest
        opposing contacted pair and balance is the residual normal-force
        resultant.  This is a soft training/diagnostic signal only: it never
        gates fracture, action authority, termination, or success.
        """

        force_on_fruit_w = -torch.stack(
            [
                sensor.data.force_matrix_w.torch[:, 0, 0]
                for sensor in self._contact_sensors.values()
            ],
            dim=1,
        )
        force_n = torch.linalg.vector_norm(force_on_fruit_w, dim=2)
        contacted = force_n >= self.cfg.minimum_contact_force_n
        directions_w = force_on_fruit_w / force_n.clamp_min(1.0e-6).unsqueeze(2)
        pair_opposition = -torch.bmm(directions_w, directions_w.transpose(1, 2))
        pair_valid = contacted.unsqueeze(2) & contacted.unsqueeze(1)
        diagonal = torch.eye(len(FINGERS), dtype=torch.bool, device=self.device).unsqueeze(0)
        pair_valid &= ~diagonal
        opposition_cosine = torch.where(
            pair_valid,
            pair_opposition,
            torch.full_like(pair_opposition, -1.0),
        ).amax(dim=(1, 2)).clamp(-1.0, 1.0)
        total_force_n = force_n.sum(dim=1)
        resultant_ratio = torch.linalg.vector_norm(
            force_on_fruit_w.sum(dim=1), dim=1
        ) / total_force_n.clamp_min(1.0e-6)
        valid_pair = contacted.sum(dim=1) >= 2
        opposition_score = ((opposition_cosine + 1.0) * 0.5).clamp(0.0, 1.0)
        balance_score = (
            1.0 - resultant_ratio / self.cfg.grasp_force_resultant_score_ceiling
        ).clamp(0.0, 1.0)
        quality = torch.sqrt(opposition_score * balance_score) * valid_pair.float()
        return opposition_cosine, resultant_ratio, quality

    def _gravity_support_ratio(self) -> torch.Tensor:
        """Return fingertip support along world-up divided by fruit weight."""

        force_on_fruit_w = -torch.stack(
            [
                sensor.data.force_matrix_w.torch[:, 0, 0]
                for sensor in self._contact_sensors.values()
            ],
            dim=1,
        ).sum(dim=1)
        gravity_w = torch.tensor(
            self.cfg.sim.gravity, dtype=torch.float32, device=self.device
        )
        gravity_magnitude = torch.linalg.vector_norm(gravity_w).clamp_min(1.0e-6)
        world_up = -gravity_w / gravity_magnitude
        fruit_weight_n = (self._fruit_mass_kg * gravity_magnitude).clamp_min(1.0e-6)
        return torch.sum(force_on_fruit_w * world_up.unsqueeze(0), dim=1) / fruit_weight_n

    def _directional_stem_load_ratio(
        self,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return the shared physical, critic, and reward detachment load."""

        local_stem_axis = torch.zeros_like(force_w)
        local_stem_axis[:, 2] = 1.0
        stem_axis_w = quat_apply(self._stem_anchor_quat_w, local_stem_axis)
        return directional_load_ratio(
            force_w,
            torque_w,
            stem_axis_w,
            self._break_force_n,
            self._break_torque_nm,
        )

    @staticmethod
    def _tactile_features(
        contact_force: torch.Tensor,
        contact: torch.Tensor,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(contact_force)
        features = torch.stack(
            (
                (contact_force / 20.0).clamp(0.0, 1.0),
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                contact.float(),
                zeros,
            ),
            dim=2,
        )
        return features.reshape(features.shape[0], 40)

    def _critic_state(
        self,
        policy_observation: torch.Tensor,
        contact_force: torch.Tensor,
        stem_load_progress: torch.Tensor,
    ) -> torch.Tensor:
        """Return a Markov, asymmetric value-function observation.

        PPO's actor remains restricted to the frozen deployable 113-D input.
        The critic additionally receives exact simulator state and the hidden
        state that determines force-closure rewards and phase transitions.
        This is training-only information and is never exported with the
        deployment actor.
        """

        fruit_pose = self._tomato.data.root_pose_w.torch.clone()
        fruit_pose[:, :3] -= self.scene.env_origins
        fruit_velocity = torch.cat(
            (self._tomato.data.root_lin_vel_w.torch, self._tomato.data.root_ang_vel_w.torch), dim=1
        )
        phase = torch.nn.functional.one_hot(self._phase, 4).float()
        privileged_state = torch.cat(
            (
                fruit_pose,
                fruit_velocity,
                self._fruit_axes_m,
                self._fruit_mass_kg.unsqueeze(1),
                contact_force,
                self._stem_detached.float().unsqueeze(1),
                self._break_force_n.unsqueeze(1),
                self._break_torque_nm.unsqueeze(1),
                stem_load_progress.unsqueeze(1),
                phase,
                self._synergy,
            ),
            dim=1,
        )
        opposition_cosine, resultant_ratio, force_closure_quality = (
            self._grasp_force_closure_quality()
        )
        contact_streak_progress = (
            self._contact_streak.float() / float(CONTACT_STREAK_STEPS)
        ).clamp(0.0, 1.0)
        hold_streak_progress = (
            self._hold_streak.float() / float(self.cfg.success_hold_steps)
        ).clamp(0.0, 1.0)
        critic_internal_state = torch.stack(
            (
                opposition_cosine,
                resultant_ratio,
                force_closure_quality,
                contact_streak_progress,
                hold_streak_progress,
                self._previous_grasp_potential,
                (
                    self._previous_retention_potential
                    / float(self.cfg.success_hold_steps)
                ).clamp(0.0, 1.0),
            ),
            dim=1,
        )
        state = torch.cat(
            (policy_observation, privileged_state, critic_internal_state), dim=1
        )
        if state.shape[1] != self.cfg.state_space:
            raise RuntimeError(
                f"critic state drifted to {state.shape[1]} values; "
                f"expected {self.cfg.state_space}"
            )
        return state

    def _log_reset(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self, "_episode_sums"):
            return
        self.extras["log"] = {
            f"Episode_Reward/{name}": values[env_ids].mean().item()
            for name, values in self._episode_sums.items()
        }
        self.extras["log"]["Metrics/success_rate"] = self._success[env_ids].float().mean().item()
        self.extras["log"]["Metrics/drop_rate"] = self._dropped[env_ids].float().mean().item()
        self.extras["log"]["Metrics/reached_contact_rate"] = (
            self._reached_contact[env_ids].float().mean().item()
        )
        self.extras["log"]["Metrics/reached_grasp_rate"] = (
            self._reached_grasp[env_ids].float().mean().item()
        )
        self.extras["log"]["Metrics/reached_detach_rate"] = (
            self._reached_detach[env_ids].float().mean().item()
        )
        self.extras["log"]["Metrics/reached_hold_rate"] = (
            self._hold_streak[env_ids] > 0
        ).float().mean().item()
        for values in self._episode_sums.values():
            values[env_ids] = 0.0


__all__ = ["TomatoPickEnv"]
