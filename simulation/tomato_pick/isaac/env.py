"""Minimal continuous-control tomato-picking environment.

The environment contains no phase machine and no scripted grasp.  PPO owns a
6D local Cartesian command and a 5D absolute G20 grasp manifold for the whole
episode.  Simulator-only state is restricted to the critic, reward, physics,
and final verification.
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
)

from module.skills.tomato_pick import (
    ACTION_DIM,
    CRITIC_STATE_DIM,
    G20_JOINT_ORDER,
    OBSERVATION_DIM,
)
from module.skills.tomato_pick.hand_synergy import (
    DISTAL_CURL_SHAPE_GAIN,
    FINGER_SPREAD_PATTERN,
    HAND_JOINT_LIMITS,
    PREGRASP_FLEXION_FRACTION,
    PREGRASP_SYNERGY,
    PROXIMAL_CURL_SHAPE_GAIN,
    RADIAL_ULNAR_BALANCE_GAIN,
    RADIAL_ULNAR_BALANCE_PATTERN,
    THUMB_ROLL_FIXED_RAD,
    THUMB_ROLL_JOINT_INDEX,
    THUMB_ROOT_JOINT_INDEX,
    THUMB_ROOT_MAX_FRACTION,
    THUMB_SIDE_JOINT_INDEX,
    THUMB_SIDE_SAFE_RAD,
    THUMB_TIP_JOINT_INDEX,
    g20_synergy_to_joint_targets,
)
from simulation.tomato_pick.reward import (
    alignment_alpha_for_iteration,
    break_grasp_quality,
    grasp_contact_score,
    opposed_contact_score,
    tomato_pick_reward,
)
from simulation.tomato_pick.stem_fracture import (
    attachment_broken,
    attachment_load_ratio,
    update_overload_streak,
)

from .assets import (
    activate_nested_contact_reporters,
    extract_collision_aabbs_by_body,
    validate_linkbot_mass_properties,
    validate_newton_mimic_constraints_disabled,
)
from .env_cfg import (
    RIGHT_FINGERTIP_CONTACT_BODY_NAMES,
    RIGHT_FINGERTIP_FRAME_BODY_NAMES,
    RIGHT_HAND_BODY_NAMES,
    TomatoPickEnvCfg,
)


FINGERS = ("thumb", "index", "middle", "ring", "little")
RIGHT_HAND_CONTROL_JOINT_NAMES = tuple(
    f"right_hand_{name}" for name in G20_JOINT_ORDER
)
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


class TomatoPickEnv(DirectRLEnv):
    cfg: TomatoPickEnvCfg

    def __init__(self, cfg: TomatoPickEnvCfg, render_mode: str | None = None, **kwargs):
        if (
            not isinstance(cfg.stem_break_persistence_substeps, int)
            or cfg.stem_break_persistence_substeps < 1
        ):
            raise ValueError("stem_break_persistence_substeps must be an integer >= 1")
        if not 1 <= cfg.stable_detach_credit_steps < cfg.success_hold_steps:
            raise ValueError(
                "stable_detach_credit_steps must be in [1, success_hold_steps)"
            )
        if cfg.reward_rollout_steps < 1 or cfg.reward_curriculum_iteration_offset < 0:
            raise ValueError("reward curriculum counters must be non-negative")
        super().__init__(cfg, render_mode, **kwargs)
        if not hasattr(self, "_prestartup_fruit_axes_m"):
            raise RuntimeError("fruit geometry prestartup event did not run")

        self._arm_joint_ids, _ = self._robot.find_joints(
            r"^R[1-7]_Joint$", preserve_order=True
        )
        self._left_arm_joint_ids, _ = self._robot.find_joints(
            r"^L[1-7]_Joint$", preserve_order=True
        )
        self._hand_joint_ids, self._hand_joint_names = self._robot.find_joints(
            list(RIGHT_HAND_CONTROL_JOINT_NAMES), preserve_order=True
        )
        self._mimic_follower_ids, _ = self._robot.find_joints(
            list(MIMIC_FOLLOWER_JOINT_NAMES), preserve_order=True
        )
        if len(self._arm_joint_ids) != 7 or len(self._hand_joint_ids) != 16:
            raise RuntimeError("expected one 7DoF arm and sixteen active G20 joints")
        if len(self._mimic_follower_ids) != 10:
            raise RuntimeError("expected five mimic followers on each G20 hand")
        self._actuated_joint_ids = list(range(self._robot.num_joints))
        self._ee_body_id = self._robot.find_bodies("right_hand_base_link")[0][0]
        self._ee_jacobian_id = (
            self._ee_body_id - 1 if self._robot.is_fixed_base else self._ee_body_id
        )
        self._fingertip_frame_body_ids = [
            self._robot.find_bodies(RIGHT_FINGERTIP_FRAME_BODY_NAMES[name])[0][0]
            for name in FINGERS
        ]
        self._fingertip_contact_body_ids = [
            self._robot.find_bodies(RIGHT_FINGERTIP_CONTACT_BODY_NAMES[name])[0][0]
            for name in FINGERS
        ]
        collision_specs = extract_collision_aabbs_by_body(
            "/World/envs/env_0/Robot", RIGHT_HAND_BODY_NAMES
        )
        collision_names = [spec[0] for spec in collision_specs]
        self._hand_collision_body_ids, found_collision_names = self._robot.find_bodies(
            collision_names, preserve_order=True
        )
        if found_collision_names != collision_names:
            raise RuntimeError("G20 collision-body order did not resolve exactly")
        self._hand_collision_aabb_centers = torch.tensor(
            [spec[1] for spec in collision_specs], device=self.device
        )
        self._hand_collision_aabb_half_extents = torch.tensor(
            [spec[2] for spec in collision_specs], device=self.device
        )

        self._fruit_axes_m = self._prestartup_fruit_axes_m.to(self.device)
        self._fruit_density_kg_m3 = self._prestartup_fruit_density_kg_m3.to(self.device)
        self._fruit_mass_kg = (
            4.0
            / 3.0
            * math.pi
            * self._fruit_axes_m.prod(dim=1)
            * self._fruit_density_kg_m3
        )
        self._tomato.set_masses_index(masses=self._fruit_mass_kg.unsqueeze(1))

        self._nominal_joint_targets = self._robot.data.default_joint_pos.torch.clone()
        self._joint_targets = self._nominal_joint_targets.clone()
        self._actions = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._action_history = torch.zeros((2, self.num_envs, ACTION_DIM), device=self.device)
        self._action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ee_reference_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self._ee_reference_pose_w[:, 6] = 1.0
        self._ee_command_pose_w = self._ee_reference_pose_w.clone()
        self._grasp_basis_w = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        self._perceived_fruit_position_error_w = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._measured_fruit_axes_m = self._fruit_axes_m.clone()
        self._tactile_gain = torch.ones((self.num_envs, 5), device=self.device)
        self._tactile_bias_n = torch.zeros((self.num_envs, 5), device=self.device)

        # One stem geometry convention is used by physics and rendering:
        # the fruit-side attachment is the material point at local +Z on the
        # ellipsoid, and the plant anchor is one visual stem length above it.
        self._stem_rest_attachment_pos_w = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._stem_plant_anchor_pos_w = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._stem_anchor_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._stem_anchor_quat_w[:, 3] = 1.0
        self._stem_detached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stem_break_event = torch.zeros_like(self._stem_detached)
        self._stem_overload_substeps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._last_stem_force_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_stem_torque_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._break_force_n = torch.full((self.num_envs,), 5.0, device=self.device)

        self._best_reach_score = torch.zeros(self.num_envs, device=self.device)
        self._best_grasp_score = torch.zeros(self.num_envs, device=self.device)
        self._best_opposed_score = torch.zeros(self.num_envs, device=self.device)
        self._best_stem_load_score = torch.zeros(self.num_envs, device=self.device)
        self._best_stable_detach_score = torch.zeros(
            self.num_envs, device=self.device
        )
        self._best_retention_score = torch.zeros(self.num_envs, device=self.device)
        self._hold_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success = torch.zeros_like(self._stem_detached)
        self._dropped = torch.zeros_like(self._stem_detached)
        self._excess_force = torch.zeros_like(self._stem_detached)
        self._out_of_bounds = torch.zeros_like(self._stem_detached)
        self._retained = torch.zeros_like(self._stem_detached)
        self._terminal_failure = torch.zeros_like(self._stem_detached)
        self._timed_out = torch.zeros_like(self._stem_detached)
        self._last_physical_force_n = torch.zeros((self.num_envs, 5), device=self.device)
        self._last_reward_terms = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "reach_progress",
                "grasp_progress",
                "stem_load_progress",
                "break_grasped",
                "opposed_progress",
                "stable_detach",
                "retention_progress",
                "time_cost",
                "success",
                "drop_failure",
                "excess_force_failure",
                "out_of_bounds_failure",
                "timeout_failure",
            )
        }
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self._last_reward_terms
        }
        self._episode_sums.update(
            {
                "detached": torch.zeros(self.num_envs, device=self.device),
                "contact": torch.zeros(self.num_envs, device=self.device),
            }
        )

        self._stem_visual_positions_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._stem_visual_orientations_w = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self._stem_visual_orientations_w[:, 0] = 1.0
        self._stem_view = FrameView(self.cfg.stem.prim_path, device=self.device)
        if self._stem_view.count != self.num_envs:
            raise RuntimeError("stem visual count does not match environment count")
        self._validate_nominal_joint_margin()
        if render_mode is not None:
            # Recordings are an engineering diagnostic: center the first
            # hand--fruit pair in world coordinates after the scene has
            # resolved its true kinematic pose.  Configuration-only camera
            # origins are evaluated too early for this nested articulation.
            ee = self._robot.data.body_pos_w.torch[0, self._ee_body_id]
            fruit = self._tomato.data.root_pos_w.torch[0]
            target = 0.5 * (ee + fruit)
            eye = target + torch.tensor((0.36, -0.46, 0.22), device=self.device)
            self.sim.set_camera_view(
                eye=tuple(float(value) for value in eye),
                target=tuple(float(value) for value in target),
            )

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        validate_newton_mimic_constraints_disabled("/World/envs/env_0/Robot")
        validate_linkbot_mass_properties(
            "/World/envs/env_0/Robot",
            minimum_principal_inertia_kg_m2=2.5e-7,
        )
        self._tomato = RigidObject(self.cfg.tomato)
        activate_nested_contact_reporters("/World/envs/env_0/Robot", RIGHT_HAND_BODY_NAMES)
        self._contact_sensors: dict[str, ContactSensor] = {}
        for name, sensor_cfg in self.cfg.fingertip_contacts.items():
            sensor_cfg.filter_prim_paths_expr = [
                value.format(ENV_REGEX_NS=self.scene.env_regex_ns)
                for value in sensor_cfg.filter_prim_paths_expr
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
        light = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.80, 0.82, 0.84))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != self._actions.shape:
            raise ValueError(
                f"actions must have shape {tuple(self._actions.shape)}, got {tuple(actions.shape)}"
            )
        if self.cfg.strict_runtime_checks and not bool(torch.isfinite(actions).all()):
            raise ValueError("actions contain NaN or Inf")
        bounded = actions.clamp(-1.0, 1.0)
        self._previous_actions.copy_(self._actions)
        self._action_history[1].copy_(self._action_history[0])
        self._action_history[0].copy_(bounded)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._actions.copy_(self._action_history[self._action_delay, env_ids])
        self._stem_break_event.zero_()

        ee_pose = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        scales = torch.tensor(self.cfg.action_scales, device=self.device)
        local_delta = self._actions[:, :6] * scales
        translation_w = torch.bmm(
            self._grasp_basis_w, local_delta[:, :3].unsqueeze(2)
        ).squeeze(2)
        rotation_w = torch.bmm(
            self._grasp_basis_w, local_delta[:, 3:6].unsqueeze(2)
        ).squeeze(2)
        proposed_pos = ee_pose[:, :3] + translation_w
        reference_delta = torch.bmm(
            self._grasp_basis_w.transpose(1, 2),
            (proposed_pos - self._ee_reference_pose_w[:, :3]).unsqueeze(2),
        ).squeeze(2)
        translation_limit = torch.tensor(
            self.cfg.max_residual_translation_m, device=self.device
        )
        bounded_delta = torch.maximum(
            -translation_limit, torch.minimum(translation_limit, reference_delta)
        )
        self._ee_command_pose_w[:, :3] = self._ee_reference_pose_w[:, :3] + torch.bmm(
            self._grasp_basis_w, bounded_delta.unsqueeze(2)
        ).squeeze(2)

        angle = torch.linalg.vector_norm(rotation_w, dim=1)
        axis = rotation_w / angle.clamp_min(1.0e-8).unsqueeze(1)
        axis = torch.where(
            (angle > 1.0e-8).unsqueeze(1),
            axis,
            torch.tensor((0.0, 0.0, 1.0), device=self.device),
        )
        proposed_quat = quat_mul(quat_from_angle_axis(angle, axis), ee_pose[:, 3:7])
        reference_rotvec = quat_box_minus(
            proposed_quat, self._ee_reference_pose_w[:, 3:7]
        )
        reference_rotvec_g = torch.bmm(
            self._grasp_basis_w.transpose(1, 2), reference_rotvec.unsqueeze(2)
        ).squeeze(2)
        rotation_limit = torch.tensor(self.cfg.max_residual_rotation_rad, device=self.device)
        reference_rotvec_g = torch.maximum(
            -rotation_limit, torch.minimum(rotation_limit, reference_rotvec_g)
        )
        bounded_rotvec_w = torch.bmm(
            self._grasp_basis_w, reference_rotvec_g.unsqueeze(2)
        ).squeeze(2)
        bounded_angle = torch.linalg.vector_norm(bounded_rotvec_w, dim=1)
        bounded_axis = bounded_rotvec_w / bounded_angle.clamp_min(1.0e-8).unsqueeze(1)
        bounded_axis = torch.where(
            (bounded_angle > 1.0e-8).unsqueeze(1),
            bounded_axis,
            torch.tensor((0.0, 0.0, 1.0), device=self.device),
        )
        self._ee_command_pose_w[:, 3:7] = quat_mul(
            quat_from_angle_axis(bounded_angle, bounded_axis),
            self._ee_reference_pose_w[:, 3:7],
        )
        self._set_hand_targets_from_synergy(self._actions[:, 6:11])

    def _apply_action(self) -> None:
        self._update_arm_target()
        self._robot.set_joint_position_target_index(
            target=self._joint_targets[:, self._actuated_joint_ids],
            joint_ids=self._actuated_joint_ids,
        )
        arm_ids = self._arm_joint_ids + self._left_arm_joint_ids
        gravity_columns = [index + self._robot.num_base_dofs for index in arm_ids]
        gravity_effort = self._robot.data.gravity_compensation_forces.torch[
            :, gravity_columns
        ]
        self._robot.set_joint_effort_target_index(
            target=gravity_effort, joint_ids=arm_ids
        )
        self._apply_attachment_wrench()

    def _update_arm_target(self) -> None:
        joint_pos = self._robot.data.joint_pos.torch[:, self._arm_joint_ids]
        ee_pose = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        pos_error, rot_error = compute_pose_error(
            ee_pose[:, :3],
            ee_pose[:, 3:7],
            self._ee_command_pose_w[:, :3],
            self._ee_command_pose_w[:, 3:7],
        )
        error = torch.cat((pos_error, rot_error), dim=1)
        jacobian_ids = [index + self._robot.num_base_dofs for index in self._arm_joint_ids]
        jacobian = self._robot.data.body_link_jacobian_w.torch[
            :, self._ee_jacobian_id, :, jacobian_ids
        ]
        jacobian = torch.nan_to_num(jacobian)
        identity6 = torch.eye(6, device=self.device).expand(self.num_envs, -1, -1)
        dls = torch.linalg.solve(
            torch.bmm(jacobian, jacobian.transpose(1, 2))
            + self.cfg.ik_damping**2 * identity6,
            error.unsqueeze(2),
        )
        pseudo = torch.bmm(jacobian.transpose(1, 2), dls).squeeze(2)
        identity7 = torch.eye(7, device=self.device).expand(self.num_envs, -1, -1)
        j_pinv = torch.bmm(
            jacobian.transpose(1, 2),
            torch.linalg.solve(
                torch.bmm(jacobian, jacobian.transpose(1, 2))
                + self.cfg.ik_damping**2 * identity6,
                identity6,
            ),
        )
        nullspace = identity7 - torch.bmm(j_pinv, jacobian)
        posture_error = (
            self._nominal_joint_targets[:, self._arm_joint_ids] - joint_pos
        )
        dq = pseudo + self.cfg.ik_nullspace_gain * torch.bmm(
            nullspace, posture_error.unsqueeze(2)
        ).squeeze(2)
        dq = torch.nan_to_num(dq)
        largest = dq.abs().amax(dim=1, keepdim=True)
        dq *= (
            self.cfg.maximum_arm_joint_delta_rad
            / largest.clamp_min(self.cfg.maximum_arm_joint_delta_rad)
        )
        target = joint_pos + dq
        target = torch.maximum(
            joint_pos - self.cfg.maximum_arm_target_lead_rad,
            torch.minimum(joint_pos + self.cfg.maximum_arm_target_lead_rad, target),
        )
        limits = self._robot.data.joint_pos_limits.torch[:, self._arm_joint_ids]
        self._joint_targets[:, self._arm_joint_ids] = torch.maximum(
            limits[..., 0], torch.minimum(limits[..., 1], target)
        )

    def _set_hand_targets_from_synergy(self, synergy: torch.Tensor) -> None:
        if synergy.shape != (self.num_envs, 5):
            raise ValueError("hand synergy tensor must have shape (num_envs, 5)")
        global_flexion_action = synergy[:, 0].clamp(-1.0, 1.0)
        curl_shape = synergy[:, 1].clamp(-1.0, 1.0)
        radial_ulnar_balance = synergy[:, 2].clamp(-1.0, 1.0)
        spread = synergy[:, 3].clamp(-1.0, 1.0)
        thumb_opposition_action = synergy[:, 4].clamp(-1.0, 1.0)

        def centered_unit(action: torch.Tensor) -> torch.Tensor:
            return torch.where(
                action >= 0.0,
                PREGRASP_FLEXION_FRACTION
                + action * (1.0 - PREGRASP_FLEXION_FRACTION),
                PREGRASP_FLEXION_FRACTION
                + (-action) * (0.0 - PREGRASP_FLEXION_FRACTION),
            )

        global_flexion = centered_unit(global_flexion_action)
        balance_pattern = torch.tensor(
            RADIAL_ULNAR_BALANCE_PATTERN, device=self.device
        )
        balance_offset = (
            RADIAL_ULNAR_BALANCE_GAIN
            * radial_ulnar_balance[:, None]
            * balance_pattern[None, :]
        )
        finger_proximal = (
            global_flexion[:, None]
            - PROXIMAL_CURL_SHAPE_GAIN * curl_shape[:, None]
            + balance_offset
        ).clamp(0.0, 1.0)
        finger_distal = (
            global_flexion[:, None]
            + DISTAL_CURL_SHAPE_GAIN * curl_shape[:, None]
            + balance_offset
        ).clamp(0.0, 1.0)
        thumb_opposition = centered_unit(thumb_opposition_action)
        thumb_root = THUMB_ROOT_MAX_FRACTION * thumb_opposition
        thumb_tip = thumb_opposition
        limits = torch.tensor(HAND_JOINT_LIMITS, device=self.device)
        targets = torch.zeros((self.num_envs, 16), device=self.device)
        targets[:, THUMB_ROOT_JOINT_INDEX] = limits[THUMB_ROOT_JOINT_INDEX, 0] + (
            thumb_root
            * (limits[THUMB_ROOT_JOINT_INDEX, 2] - limits[THUMB_ROOT_JOINT_INDEX, 0])
        )
        targets[:, 1:5] = limits[1:5, 0] + finger_proximal * (
            limits[1:5, 2] - limits[1:5, 0]
        )
        targets[:, THUMB_SIDE_JOINT_INDEX] = THUMB_SIDE_SAFE_RAD
        targets[:, THUMB_ROLL_JOINT_INDEX] = THUMB_ROLL_FIXED_RAD
        spread_pattern = torch.tensor(FINGER_SPREAD_PATTERN, device=self.device)
        for offset, index in enumerate(range(6, 10)):
            value = (spread * spread_pattern[offset]).clamp(-1.0, 1.0)
            targets[:, index] = torch.where(
                value >= 0.0,
                limits[index, 1] + value * (limits[index, 2] - limits[index, 1]),
                limits[index, 1] + (-value) * (limits[index, 0] - limits[index, 1]),
            )
        targets[:, THUMB_TIP_JOINT_INDEX] = limits[THUMB_TIP_JOINT_INDEX, 0] + (
            thumb_tip
            * (limits[THUMB_TIP_JOINT_INDEX, 2] - limits[THUMB_TIP_JOINT_INDEX, 0])
        )
        targets[:, 12:16] = limits[12:16, 0] + finger_distal * (
            limits[12:16, 2] - limits[12:16, 0]
        )
        self._joint_targets[:, self._hand_joint_ids] = targets
        self._update_mimic_targets(torch.arange(self.num_envs, device=self.device))

    def _update_mimic_targets(self, env_ids: torch.Tensor) -> None:
        limits = self._robot.data.joint_pos_limits.torch[env_ids]
        for follower_name in MIMIC_FOLLOWER_JOINT_NAMES:
            side = "right_hand_" if follower_name.startswith("right_hand_") else "left_hand_"
            right_name = follower_name.replace("left_hand_", "right_hand_", 1)
            driver_right, multiplier, offset = RIGHT_HAND_MIMIC_RAD[right_name]
            driver_name = driver_right.replace("right_hand_", side, 1)
            follower_id = self._robot.joint_names.index(follower_name)
            driver_id = self._robot.joint_names.index(driver_name)
            target = self._joint_targets[env_ids, driver_id] * multiplier + offset
            self._joint_targets[env_ids, follower_id] = target.clamp(
                limits[:, follower_id, 0], limits[:, follower_id, 1]
            )

    def _apply_attachment_wrench(self) -> None:
        fruit_pose = self._tomato.data.root_pose_w.torch
        fruit_lin_vel = self._tomato.data.root_lin_vel_w.torch
        fruit_ang_vel = self._tomato.data.root_ang_vel_w.torch
        local_attachment = torch.zeros((self.num_envs, 3), device=self.device)
        local_attachment[:, 2] = self._fruit_axes_m[:, 2]
        attachment_offset_w = quat_apply(fruit_pose[:, 3:7], local_attachment)
        attachment_pos_w = fruit_pose[:, :3] + attachment_offset_w
        attachment_velocity_w = fruit_lin_vel + torch.linalg.cross(
            fruit_ang_vel, attachment_offset_w, dim=1
        )
        position_error = self._stem_rest_attachment_pos_w - attachment_pos_w
        rotation_error = quat_box_minus(self._stem_anchor_quat_w, fruit_pose[:, 3:7])
        gravity = torch.tensor(self.cfg.sim.gravity, device=self.device)
        omega = self.cfg.stem_linear_natural_frequency_rad_s
        acceleration = (
            omega * omega * position_error
            - 2.0
            * self.cfg.stem_linear_damping_ratio
            * omega
            * attachment_velocity_w
            - gravity.unsqueeze(0)
        )
        force = self._fruit_mass_kg.unsqueeze(1) * acceleration
        angular_omega = self.cfg.stem_angular_natural_frequency_rad_s
        inertia = (
            0.4
            * self._fruit_mass_kg
            * torch.square(self._fruit_axes_m).mean(dim=1)
        ).clamp_min(1.0e-8)
        angular_acceleration = (
            angular_omega * angular_omega * rotation_error
            - 2.0
            * self.cfg.stem_angular_damping_ratio
            * angular_omega
            * fruit_ang_vel
        )
        # Orientation is restored separately from translation.  Injecting a
        # point-force lever torque through the explicit velocity integrator
        # double-counts this constraint and is numerically unstable at 120 Hz.
        torque = inertia.unsqueeze(1) * angular_acceleration
        force *= (
            self.cfg.maximum_stem_force_n
            / torch.linalg.vector_norm(force, dim=1, keepdim=True).clamp_min(1.0e-6)
        ).clamp(max=1.0)
        torque *= (
            self.cfg.maximum_stem_torque_nm
            / torch.linalg.vector_norm(torque, dim=1, keepdim=True).clamp_min(1.0e-6)
        ).clamp(max=1.0)
        attached = ~self._stem_detached
        pull_axis = self._grasp_basis_w[:, :, 2]
        instantaneous_overload = attachment_broken(
            force,
            pull_axis,
            self._break_force_n,
        )
        self._stem_overload_substeps.copy_(
            update_overload_streak(
                self._stem_overload_substeps,
                instantaneous_overload,
                attached,
            )
        )
        break_now = attached & (
            self._stem_overload_substeps >= self.cfg.stem_break_persistence_substeps
        )
        self._stem_break_event |= break_now
        self._stem_detached |= break_now
        attached_after = attached & ~break_now
        force *= attached_after.unsqueeze(1)
        torque *= attached_after.unsqueeze(1)
        self._last_stem_force_w.copy_(force)
        self._last_stem_torque_w.copy_(torque)
        if bool(attached_after.any()):
            env_ids = attached_after.nonzero(as_tuple=False).squeeze(1)
            self._move_stem_visual(
                env_ids,
                attachment_pos_w[env_ids],
                self._stem_plant_anchor_pos_w[env_ids],
            )

        zeros = torch.zeros_like(force)
        self._tomato.permanent_wrench_composer.set_forces_and_torques_index(
            forces=zeros.unsqueeze(1), torques=zeros.unsqueeze(1), is_global=False
        )
        if bool(attached_after.any()):
            env_ids = attached_after.nonzero(as_tuple=False).squeeze(1)
            velocity = torch.cat((fruit_lin_vel, fruit_ang_vel), dim=1)[env_ids].clone()
            velocity[:, :3] += (
                force[env_ids] / self._fruit_mass_kg[env_ids].unsqueeze(1)
            ) * float(self.cfg.sim.dt)
            velocity[:, 3:] += (
                torque[env_ids] / inertia[env_ids].unsqueeze(1)
            ) * float(self.cfg.sim.dt)
            self._tomato.write_root_velocity_to_sim_index(
                root_velocity=velocity, env_ids=env_ids
            )

    def _physical_finger_force(self) -> torch.Tensor:
        per_finger = []
        for sensor in self._contact_sensors.values():
            matrix = sensor.data.force_matrix_w
            if matrix is None:
                raise RuntimeError("tomato-filtered contact sensor did not initialize")
            per_finger.append(torch.linalg.vector_norm(matrix.torch[:, 0, 0], dim=-1))
        return torch.stack(per_finger, dim=1).clamp(0.0, 25.0)

    def _measured_finger_pressure(self) -> torch.Tensor:
        physical = self._physical_finger_force()
        self._last_physical_force_n.copy_(physical)
        return (physical * self._tactile_gain + self._tactile_bias_n).clamp(0.0, 25.0)

    def _reach_score(self) -> torch.Tensor:
        fruit_pose = self._tomato.data.root_pose_w.torch
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        palm = self._robot.data.body_pos_w.torch[:, self._ee_body_id]
        return self._reach_score_for(fruit_pose[:, :3], tips, palm, self._fruit_axes_m)

    def _enclosure_score(self) -> torch.Tensor:
        """Bounded geometric diagnostic retained for visualization audits."""

        fruit_pose = self._tomato.data.root_pose_w.torch
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        delta = tips - fruit_pose[:, None, :3]
        fruit_quat = fruit_pose[:, None, 3:7].expand(-1, len(FINGERS), -1)
        local = quat_apply_inverse(
            fruit_quat.reshape(-1, 4), delta.reshape(-1, 3)
        ).reshape(self.num_envs, len(FINGERS), 3)
        radial = torch.sqrt(
            torch.square(local / self._fruit_axes_m[:, None, :].clamp_min(1.0e-4)).sum(
                dim=2
            )
        )
        surface_distance = torch.abs(radial - 1.0) * self._fruit_axes_m.mean(
            dim=1, keepdim=True
        )
        tip_score = torch.exp(-surface_distance / 0.018)
        closest_three = torch.topk(tip_score, 3, dim=1).values.mean(dim=1)
        center_error = torch.linalg.vector_norm(
            tips.mean(dim=1) - fruit_pose[:, :3], dim=1
        )
        center_score = torch.exp(-center_error / 0.055)
        return (0.65 * closest_three + 0.35 * center_score).clamp(0.0, 1.0)

    def _hand_collision_clearance_for(
        self,
        fruit_pos: torch.Tensor,
        fruit_axes_m: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative fruit clearance from every right-hand collider.

        The ellipsoid is enclosed by its maximum-radius sphere and each mesh
        collider by an AABB in its owning rigid-body frame.  Positive values
        are separation; negative values are penetration.  This is intentionally
        conservative and is used only for reset validation.
        """

        body_pos = self._robot.data.body_pos_w.torch[env_ids][
            :, self._hand_collision_body_ids
        ]
        body_quat = self._robot.data.body_quat_w.torch[env_ids][
            :, self._hand_collision_body_ids
        ]
        relative = fruit_pos.unsqueeze(1) - body_pos
        local = quat_apply_inverse(
            body_quat.reshape(-1, 4), relative.reshape(-1, 3)
        ).reshape(len(env_ids), len(self._hand_collision_body_ids), 3)
        outside = (
            torch.abs(local - self._hand_collision_aabb_centers)
            - self._hand_collision_aabb_half_extents
        ).clamp_min(0.0)
        distance_to_aabb = torch.linalg.vector_norm(outside, dim=2)
        fruit_radius = fruit_axes_m.amax(dim=1, keepdim=True)
        return (distance_to_aabb - fruit_radius).amin(dim=1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        fruit_pose = self._tomato.data.root_pose_w.torch
        ee_pose = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        perceived_fruit = fruit_pose[:, :3] + self._perceived_fruit_position_error_w
        relative_position = quat_apply_inverse(
            ee_pose[:, 3:7], perceived_fruit - ee_pose[:, :3]
        ) / 0.15
        axes = (
            2.0 * (self._measured_fruit_axes_m - 0.025) / (0.035 - 0.025) - 1.0
        ).clamp(-1.0, 1.0)
        pull_axis_palm = quat_apply_inverse(
            ee_pose[:, 3:7], self._grasp_basis_w[:, :, 2]
        )
        ee_linear = self._robot.data.body_lin_vel_w.torch[:, self._ee_body_id]
        ee_angular = self._robot.data.body_ang_vel_w.torch[:, self._ee_body_id]
        twist = torch.cat(
            (
                quat_apply_inverse(ee_pose[:, 3:7], ee_linear) / 0.5,
                quat_apply_inverse(ee_pose[:, 3:7], ee_angular) / 2.0,
            ),
            dim=1,
        ).clamp(-1.0, 1.0)
        hand_pos = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
        hand_vel = self._robot.data.joint_vel.torch[:, self._hand_joint_ids]
        limits = torch.tensor(HAND_JOINT_LIMITS, device=self.device)
        hand_pos_norm = (
            2.0 * (hand_pos - limits[:, 0]) / (limits[:, 2] - limits[:, 0]) - 1.0
        ).clamp(-1.0, 1.0)
        hand_vel_norm = (hand_vel / 2.0).clamp(-1.0, 1.0)
        pressure = (self._measured_finger_pressure() / 20.0).clamp(0.0, 1.0)
        gravity = torch.tensor((0.0, 0.0, -1.0), device=self.device).expand(
            self.num_envs, -1
        )
        gravity_palm = quat_apply_inverse(ee_pose[:, 3:7], gravity)
        policy = torch.cat(
            (
                relative_position.clamp(-1.0, 1.0),
                axes,
                pull_axis_palm,
                twist,
                hand_pos_norm,
                hand_vel_norm,
                pressure,
                gravity_palm,
                self._previous_actions,
            ),
            dim=1,
        )
        if policy.shape[1] != OBSERVATION_DIM:
            raise RuntimeError(f"Actor observation drifted to {policy.shape[1]}D")

        exact_relative = quat_apply_inverse(
            ee_pose[:, 3:7], fruit_pose[:, :3] - ee_pose[:, :3]
        ).clamp(-0.30, 0.30)
        fruit_linear_velocity = quat_apply_inverse(
            ee_pose[:, 3:7], self._tomato.data.root_lin_vel_w.torch
        ).clamp(-2.0, 2.0)
        fruit_angular_velocity = quat_apply_inverse(
            ee_pose[:, 3:7], self._tomato.data.root_ang_vel_w.torch
        ).clamp(-10.0, 10.0)
        fruit_velocity = torch.cat(
            (
                fruit_linear_velocity,
                fruit_angular_velocity,
            ),
            dim=1,
        )
        stem_load = attachment_load_ratio(
            self._last_stem_force_w,
            self._grasp_basis_w[:, :, 2],
            self._break_force_n,
        ).unsqueeze(1)
        privileged = torch.cat(
            (
                exact_relative,
                fruit_velocity,
                self._last_physical_force_n.clamp(
                    0.0, self.cfg.maximum_fingertip_force_n
                ),
                stem_load,
                self._stem_detached.float().unsqueeze(1),
                self._fruit_mass_kg.clamp(0.03, 0.50).unsqueeze(1),
            ),
            dim=1,
        )
        critic = torch.cat((policy, privileged), dim=1)
        if critic.shape[1] != CRITIC_STATE_DIM:
            raise RuntimeError(f"Critic observation drifted to {critic.shape[1]}D")
        if self.cfg.strict_runtime_checks:
            if not bool(torch.isfinite(policy).all() and torch.isfinite(critic).all()):
                raise RuntimeError("observation contains NaN or Inf")
        return {"policy": policy, "critic": critic}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # DirectRLEnv asks for dones before observations.  Refresh contact force
        # here so terminal decisions are based on the current physics step.
        self._last_physical_force_n.copy_(self._physical_finger_force())
        fruit_pos = self._tomato.data.root_pos_w.torch
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        tip_center = tips.mean(dim=1)
        tip_velocity = self._robot.data.body_lin_vel_w.torch[
            :, self._fingertip_contact_body_ids
        ].mean(dim=1)
        distance = torch.linalg.vector_norm(fruit_pos - tip_center, dim=1)
        relative_speed = torch.linalg.vector_norm(
            self._tomato.data.root_lin_vel_w.torch - tip_velocity, dim=1
        )
        self._excess_force = (
            self._last_physical_force_n.amax(dim=1) > self.cfg.maximum_fingertip_force_n
        )
        self._dropped = self._stem_detached & (
            distance > self.cfg.maximum_retained_distance_m
        )
        # Retention is the physical outcome: after detachment the fruit stays
        # with the hand, moves with it, and is neither dropped nor overloaded.
        # Do not require two fingertip sensors to remain above threshold on
        # every 50 ms policy step: a valid grasp may transfer load to the palm
        # or between fingers.  Two-finger/thumb opposition still gates detach
        # credit, and aligned stable/hold shaping still multiplies opposition.
        self._retained = (
            self._stem_detached
            & (distance <= self.cfg.maximum_retained_distance_m)
            & (relative_speed <= self.cfg.maximum_retained_relative_speed_m_s)
            & ~self._dropped
            & ~self._excess_force
        )
        self._hold_streak = torch.where(
            self._retained, self._hold_streak + 1, torch.zeros_like(self._hold_streak)
        )
        self._success = self._retained & (
            self._hold_streak >= self.cfg.success_hold_steps
        )

        ee_pos = self._robot.data.body_pos_w.torch[:, self._ee_body_id]
        local_offset = torch.bmm(
            self._grasp_basis_w.transpose(1, 2),
            (ee_pos - self._ee_reference_pose_w[:, :3]).unsqueeze(2),
        ).squeeze(2)
        workspace_limit = torch.tensor(
            self.cfg.max_residual_translation_m, device=self.device
        ) + 0.02
        workspace_violation = (local_offset.abs() > workspace_limit).any(dim=1)
        arm_pos = self._robot.data.joint_pos.torch[:, self._arm_joint_ids]
        arm_limits = self._robot.data.joint_pos_limits.torch[:, self._arm_joint_ids]
        span = (arm_limits[..., 1] - arm_limits[..., 0]).clamp_min(1.0e-6)
        margin = torch.minimum(
            arm_pos - arm_limits[..., 0], arm_limits[..., 1] - arm_pos
        ) / span
        joint_violation = margin.amin(dim=1) < 0.02
        self._out_of_bounds = workspace_violation | joint_violation
        physical_failure = self._dropped | self._excess_force | self._out_of_bounds
        self._timed_out = (
            (self.episode_length_buf >= self.max_episode_length - 1)
            & ~self._success
            & ~physical_failure
        )
        # Preserve the released run's exploration semantics: a time limit is a
        # value-bootstrap boundary, while only true task outcomes terminate.
        self._terminal_failure = physical_failure
        terminated = self._success | physical_failure
        truncated = self._timed_out
        self.extras["step"] = {
            "success_count": int(self._success.sum().item()),
            "detached_count": int(self._stem_detached.sum().item()),
            "retained_count": int(self._retained.sum().item()),
            "drop_count": int(self._dropped.sum().item()),
            "excess_force_count": int(self._excess_force.sum().item()),
            "out_of_bounds_count": int(self._out_of_bounds.sum().item()),
            "timeout_count": int(self._timed_out.sum().item()),
        }
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        reach = self._reach_score()
        grasp = grasp_contact_score(
            self._last_physical_force_n,
            full_contact_force_n=self.cfg.grasp_full_contact_force_n,
        )
        opposed = opposed_contact_score(
            self._last_physical_force_n,
            full_contact_force_n=self.cfg.grasp_full_contact_force_n,
        )
        break_quality = break_grasp_quality(
            self._last_physical_force_n,
            contact_force_n=self.cfg.grasp_full_contact_force_n,
        )
        fruit_pos = self._tomato.data.root_pos_w.torch
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        fruit_in_hand = (
            torch.linalg.vector_norm(fruit_pos - tips.mean(dim=1), dim=1)
            <= self.cfg.maximum_retained_distance_m
        )
        stem_load = self._stem_progress_reward_score(opposed)
        stable_detach = (
            self._retained
            & (self._hold_streak >= self.cfg.stable_detach_credit_steps)
        ).float() * opposed
        # Physical hold duration is already downstream of a qualified detach.
        # Reward its new-best duration even when load moves temporarily from a
        # fingertip sensor to the palm; opposition remains a separate signal.
        retention = (
            self._hold_streak.float() / float(self.cfg.success_hold_steps)
        ).clamp(0.0, 1.0)
        learning_iteration = (
            float(self.cfg.reward_curriculum_iteration_offset)
            + float(self.common_step_counter) / float(self.cfg.reward_rollout_steps)
        )
        alignment_alpha = alignment_alpha_for_iteration(learning_iteration)
        (
            reward,
            next_reach,
            next_grasp,
            next_opposed,
            next_stem_load,
            next_stable_detach,
            next_retention,
            terms,
        ) = tomato_pick_reward(
            reach_score=reach,
            previous_best_reach=self._best_reach_score,
            grasp_score=grasp,
            previous_best_grasp=self._best_grasp_score,
            opposed_score=opposed,
            previous_best_opposed=self._best_opposed_score,
            stem_load_score=stem_load,
            previous_best_stem_load=self._best_stem_load_score,
            stable_detach_score=stable_detach,
            previous_best_stable_detach=self._best_stable_detach_score,
            retention_score=retention,
            previous_best_retention=self._best_retention_score,
            break_grasped_score=self._break_grasped_reward_score(
                fruit_in_hand, break_quality
            ),
            success=self._success,
            dropped=self._dropped,
            excess_force=self._excess_force,
            out_of_bounds=self._out_of_bounds,
            timed_out=self._timed_out,
            stem_detached=self._stem_detached,
            alignment_alpha=alignment_alpha,
        )
        self._best_reach_score.copy_(next_reach)
        self._best_grasp_score.copy_(next_grasp)
        self._best_opposed_score.copy_(next_opposed)
        self._best_stem_load_score.copy_(next_stem_load)
        self._best_stable_detach_score.copy_(next_stable_detach)
        self._best_retention_score.copy_(next_retention)
        for name, value in terms.items():
            self._last_reward_terms[name].copy_(value)
            self._episode_sums[name] += value
        self._episode_sums["detached"] += self._stem_detached.float()
        self._episode_sums["contact"] += (
            self._last_physical_force_n.amax(dim=1) >= 0.6
        ).float()
        self.extras.setdefault("step", {})["reward_alignment_alpha"] = alignment_alpha
        self.extras["step"]["reward_learning_iteration"] = learning_iteration
        if self.cfg.strict_runtime_checks and not bool(torch.isfinite(reward).all()):
            raise RuntimeError("reward contains NaN or Inf")
        return reward

    def _stem_progress_reward_score(self, opposed: torch.Tensor) -> torch.Tensor:
        """Return the task-specific bounded stem progress score.

        Clean-v5 keeps its released instantaneous axial-load semantics.
        Independent physics overlays may override this hook when their actual
        fracture state uses a different load model.
        """

        return (
            attachment_load_ratio(
                self._last_stem_force_w,
                self._grasp_basis_w[:, :, 2],
                self._break_force_n,
            )
            * opposed
        )

    def _break_grasped_reward_score(
        self, fruit_in_hand: torch.Tensor, break_quality: torch.Tensor
    ) -> torch.Tensor:
        """Return task-specific one-shot credit for a grasped fracture."""

        return (
            self._stem_break_event.float()
            * fruit_in_hand.float()
            * break_quality
        )

    def _prepare_reset_common(self, env_ids: torch.Tensor) -> None:
        """Reset manager state and buffers shared by every tomato task."""

        super()._reset_idx(env_ids)
        self._log_episodes(env_ids)
        count = len(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._action_history[:, env_ids] = 0.0
        self._action_delay[env_ids] = torch.randint(
            self.cfg.action_delay_steps[0],
            self.cfg.action_delay_steps[1] + 1,
            (count,),
            device=self.device,
        )

    def _reset_measurement_randomization(self, env_ids: torch.Tensor) -> None:
        """Sample observation-side reset randomization shared by task variants."""

        count = len(env_ids)
        self._perceived_fruit_position_error_w[env_ids] = torch.randn(
            (count, 3), device=self.device
        ) * self.cfg.target_position_noise_m
        measured_axes = self._fruit_axes_m[env_ids] * (
            1.0
            + torch.randn((count, 3), device=self.device)
            * self.cfg.fruit_axis_relative_noise
        )
        self._measured_fruit_axes_m[env_ids] = measured_axes.clamp(0.025, 0.035)
        self._tactile_gain[env_ids] = torch.empty(
            (count, 5), device=self.device
        ).uniform_(*self.cfg.tactile_force_gain)
        self._tactile_bias_n[env_ids] = torch.empty(
            (count, 5), device=self.device
        ).uniform_(*self.cfg.tactile_force_bias_n)

    def _clear_stem_episode_state(self, env_ids: torch.Tensor) -> None:
        self._stem_detached[env_ids] = False
        self._stem_break_event[env_ids] = False
        self._stem_overload_substeps[env_ids] = 0
        self._last_stem_force_w[env_ids] = 0.0
        self._last_stem_torque_w[env_ids] = 0.0

    def _clear_task_episode_state(self, env_ids: torch.Tensor) -> None:
        self._best_grasp_score[env_ids] = 0.0
        self._best_opposed_score[env_ids] = 0.0
        self._best_stem_load_score[env_ids] = 0.0
        self._best_stable_detach_score[env_ids] = 0.0
        self._best_retention_score[env_ids] = 0.0
        self._hold_streak[env_ids] = 0
        self._success[env_ids] = False
        self._dropped[env_ids] = False
        self._excess_force[env_ids] = False
        self._out_of_bounds[env_ids] = False
        self._retained[env_ids] = False
        self._terminal_failure[env_ids] = False
        self._timed_out[env_ids] = False
        self._last_physical_force_n[env_ids] = 0.0

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        self._prepare_reset_common(env_ids)
        count = len(env_ids)

        joint_pos = self._nominal_joint_targets[env_ids].clone()
        noise_rad = math.radians(self.cfg.pregrasp_arm_joint_noise_deg)
        right_noise = torch.empty((count, 7), device=self.device).uniform_(
            -noise_rad, noise_rad
        )
        right_limits = self._robot.data.joint_pos_limits.torch[
            env_ids[:, None], self._arm_joint_ids
        ]
        joint_pos[:, self._arm_joint_ids] = torch.maximum(
            right_limits[..., 0],
            torch.minimum(
                right_limits[..., 1],
                joint_pos[:, self._arm_joint_ids] + right_noise,
            ),
        )
        pregrasp = torch.tensor(
            g20_synergy_to_joint_targets(PREGRASP_SYNERGY), device=self.device
        )
        joint_pos[:, self._hand_joint_ids] = pregrasp
        self._joint_targets[env_ids] = joint_pos
        self._update_mimic_targets(env_ids)
        joint_pos = self._joint_targets[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_position_to_sim_index(
            position=joint_pos,
            joint_ids=self._actuated_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel,
            joint_ids=self._actuated_joint_ids,
            env_ids=env_ids,
        )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

        ee_pose = self._robot.data.body_pose_w.torch[env_ids, self._ee_body_id].clone()
        tips = self._robot.data.body_pos_w.torch[
            env_ids[:, None], self._fingertip_frame_body_ids
        ]
        support_center = tips[:, 1:5].mean(dim=1)
        thumb = tips[:, 0]
        # Tomato fruit hangs below its stem.  Keep the attachment direction a
        # property of the plant, not of the current hand pose: z+ in the grasp
        # frame points down, away from the plant.  The palm approach is a
        # separate horizontal axis, so curling fruit into the palm cannot be
        # mistaken for axial stem tension.
        pull_axis = torch.tensor(
            (0.0, 0.0, -1.0), device=self.device
        ).expand(count, -1)
        approach = support_center - ee_pose[:, :3]
        approach -= (approach * pull_axis).sum(dim=1, keepdim=True) * pull_axis
        approach = torch.nn.functional.normalize(approach, dim=1)
        lateral = torch.nn.functional.normalize(
            torch.linalg.cross(pull_axis, approach, dim=1), dim=1
        )
        basis = torch.stack((approach, lateral, pull_axis), dim=2)
        self._grasp_basis_w[env_ids] = basis
        self._ee_reference_pose_w[env_ids] = ee_pose
        self._ee_command_pose_w[env_ids] = ee_pose

        distance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.pregrasp_distance_range_m
        )
        lateral_radius = torch.empty(count, device=self.device).uniform_(
            *self.cfg.pregrasp_lateral_offset_range_m
        )
        lateral_angle = torch.empty(count, device=self.device).uniform_(
            -math.pi, math.pi
        )
        lateral_offset = (
            lateral * (lateral_radius * torch.cos(lateral_angle)).unsqueeze(1)
            + pull_axis * (lateral_radius * torch.sin(lateral_angle)).unsqueeze(1)
        )
        axial_noise = torch.empty(count, device=self.device).uniform_(
            -self.cfg.pregrasp_axial_noise_m,
            self.cfg.pregrasp_axial_noise_m,
        )
        calibrated_center_g = torch.tensor(
            self.cfg.closed_grasp_fruit_center_m, device=self.device
        ).expand(count, -1)
        calibrated_center_w = ee_pose[:, :3] + torch.bmm(
            basis, calibrated_center_g.unsqueeze(2)
        ).squeeze(2)
        fruit_pos = calibrated_center_w + approach * (
            distance + axial_noise
        ).unsqueeze(1) + lateral_offset
        # Validate against all 27 G20 collision bodies, not fingertip points.
        # Move only along the sampled approach so the accepted reset retains
        # the same grasp frame and lateral distribution.
        clearance_target = self.cfg.minimum_initial_clearance_m + 3.0e-4
        clearance = self._hand_collision_clearance_for(
            fruit_pos, self._fruit_axes_m[env_ids], env_ids
        )
        invalid = clearance < clearance_target
        if bool(invalid.any()):
            # The previous shortfall iteration could stall for a ray almost
            # tangent to a collider.  Resolve the same geometric constraint
            # exactly: enclose the fruit by a sphere, expand each body-local
            # AABB by that radius plus the reset margin, and advance to the
            # first exit face of every box currently containing the centre.
            # Progress is strictly positive along one ray; an exited convex
            # box cannot be re-entered, so at most one pass per body is needed.
            body_pos = self._robot.data.body_pos_w.torch[env_ids][
                :, self._hand_collision_body_ids
            ]
            body_quat = self._robot.data.body_quat_w.torch[env_ids][
                :, self._hand_collision_body_ids
            ]
            local_direction = quat_apply_inverse(
                body_quat.reshape(-1, 4),
                approach.unsqueeze(1)
                .expand(-1, len(self._hand_collision_body_ids), -1)
                .reshape(-1, 3),
            ).reshape(count, len(self._hand_collision_body_ids), 3)
            expanded_half_extents = self._hand_collision_aabb_half_extents.unsqueeze(0) + (
                self._fruit_axes_m[env_ids].amax(dim=1) + clearance_target
            ).view(count, 1, 1)
            epsilon_m = 1.0e-4
            for _ in range(len(self._hand_collision_body_ids) + 1):
                relative = fruit_pos.unsqueeze(1) - body_pos
                local = quat_apply_inverse(
                    body_quat.reshape(-1, 4), relative.reshape(-1, 3)
                ).reshape(count, len(self._hand_collision_body_ids), 3)
                local -= self._hand_collision_aabb_centers.unsqueeze(0)
                contains = (local.abs() < expanded_half_extents).all(dim=2) & invalid.unsqueeze(1)
                if not bool(contains.any()):
                    break
                positive_face = torch.where(
                    local_direction >= 0.0, expanded_half_extents, -expanded_half_extents
                )
                exit_by_axis = torch.where(
                    local_direction.abs() > 1.0e-8,
                    (positive_face - local) / local_direction,
                    torch.full_like(local_direction, float("inf")),
                )
                exit_by_body = exit_by_axis.amin(dim=2)
                shift = torch.where(contains, exit_by_body, torch.zeros_like(exit_by_body)).amax(dim=1)
                fruit_pos += approach * (shift + epsilon_m).unsqueeze(1) * invalid.unsqueeze(1)
                # The expanded AABB is only a convenient way to compute a
                # guaranteed exit distance.  Re-evaluate the actual spherical
                # clearance after every exit so corner-safe samples are not
                # moved farther than the original reset constraint requires.
                clearance = self._hand_collision_clearance_for(
                    fruit_pos, self._fruit_axes_m[env_ids], env_ids
                )
                invalid = clearance < clearance_target
        clearance = self._hand_collision_clearance_for(
            fruit_pos, self._fruit_axes_m[env_ids], env_ids
        )
        if bool((clearance < self.cfg.minimum_initial_clearance_m).any()):
            raise RuntimeError(
                "could not produce a collision-free tomato reset from the pre-grasp pose"
            )

        fruit_pose = self._tomato.data.default_root_pose.torch[env_ids].clone()
        fruit_pose[:, :3] = fruit_pos
        fruit_pose[:, 3:7] = 0.0
        fruit_pose[:, 6] = 1.0
        fruit_vel = torch.zeros((count, 6), device=self.device)
        self._tomato.write_root_pose_to_sim_index(
            root_pose=fruit_pose, env_ids=env_ids
        )
        self._tomato.write_root_velocity_to_sim_index(
            root_velocity=fruit_vel, env_ids=env_ids
        )
        fruit_side = fruit_pos - pull_axis * self._fruit_axes_m[env_ids, 2].unsqueeze(1)
        plant_side = fruit_side - pull_axis * self.cfg.stem_visual_height_m
        self._stem_rest_attachment_pos_w[env_ids] = fruit_side
        self._stem_plant_anchor_pos_w[env_ids] = plant_side
        self._stem_anchor_quat_w[env_ids] = fruit_pose[:, 3:7]
        self._clear_stem_episode_state(env_ids)
        self._break_force_n[env_ids] = torch.empty(count, device=self.device).uniform_(
            *self.cfg.break_force_range_n
        )

        self._reset_measurement_randomization(env_ids)

        self._move_stem_visual(env_ids, fruit_side, plant_side)
        self._best_reach_score[env_ids] = self._reach_score_for(
            fruit_pos,
            tips,
            ee_pose[:, :3],
            self._fruit_axes_m[env_ids],
        )
        self._clear_task_episode_state(env_ids)

    def _reach_score_for(
        self,
        fruit_pos: torch.Tensor,
        tips: torch.Tensor,
        palm: torch.Tensor,
        fruit_axes_m: torch.Tensor,
    ) -> torch.Tensor:
        """Score approach of the palm and every fingertip to the fruit."""

        hand_points = torch.cat((palm.unsqueeze(1), tips), dim=1)
        center_distance = torch.linalg.vector_norm(
            hand_points - fruit_pos.unsqueeze(1), dim=2
        )
        surface_distance = (
            center_distance - fruit_axes_m.mean(dim=1, keepdim=True)
        ).clamp_min(0.0)
        furthest_point = surface_distance.amax(dim=1)
        return 1.0 - torch.tanh(
            furthest_point / self.cfg.reach_distance_scale_m
        )

    @staticmethod
    def _enclosure_score_for(
        fruit_pos: torch.Tensor,
        tips: torch.Tensor,
        fruit_axes_m: torch.Tensor,
    ) -> torch.Tensor:
        """Reset-time enclosure approximation used by the released run."""

        distance = torch.linalg.vector_norm(tips - fruit_pos.unsqueeze(1), dim=2)
        surface = torch.abs(distance - fruit_axes_m.mean(dim=1, keepdim=True))
        tip_score = torch.exp(-surface / 0.018)
        closest = torch.topk(tip_score, 3, dim=1).values.mean(dim=1)
        center = torch.exp(
            -torch.linalg.vector_norm(tips.mean(dim=1) - fruit_pos, dim=1) / 0.055
        )
        return (0.65 * closest + 0.35 * center).clamp(0.0, 1.0)

    def _move_stem_visual(
        self,
        env_ids: torch.Tensor,
        fruit_side_w: torch.Tensor,
        plant_side_w: torch.Tensor,
    ) -> None:
        """Place the visible stem exactly between its two physical endpoints."""

        segment = plant_side_w - fruit_side_w
        direction = torch.nn.functional.normalize(segment, dim=1)
        self._stem_visual_positions_w[env_ids] = 0.5 * (
            fruit_side_w + plant_side_w
        )

        # Shortest-arc quaternion rotating the cylinder's local +Z axis onto
        # the fruit-to-plant segment.  Quaternions are stored as (w, x, y, z).
        dot = direction[:, 2].clamp(-1.0, 1.0)
        quaternion = torch.stack(
            (
                1.0 + dot,
                -direction[:, 1],
                direction[:, 0],
                torch.zeros_like(dot),
            ),
            dim=1,
        )
        opposite = dot < -0.999999
        quaternion[opposite] = torch.tensor(
            (0.0, 1.0, 0.0, 0.0), device=self.device
        )
        quaternion = torch.nn.functional.normalize(quaternion, dim=1)
        self._stem_visual_orientations_w[env_ids] = quaternion
        self._stem_view.set_world_poses(
            positions=wp.from_torch(self._stem_visual_positions_w.contiguous(), dtype=wp.vec3),
            orientations=wp.from_torch(
                self._stem_visual_orientations_w.contiguous(), dtype=wp.quat
            ),
        )

    def _validate_nominal_joint_margin(self) -> None:
        all_ids = self._arm_joint_ids + self._left_arm_joint_ids
        all_nominal = self._nominal_joint_targets[:, all_ids]
        all_limits = self._robot.data.joint_pos_limits.torch[:, all_ids]
        if bool(
            (
                (all_nominal < all_limits[..., 0])
                | (all_nominal > all_limits[..., 1])
            ).any()
        ):
            raise RuntimeError("nominal arm pose exceeds a hard joint limit")

        # The inactive left arm intentionally stays at the SDK zero pose.  Its
        # asymmetric L2 software range places zero close to one edge, so the
        # training margin applies only to the active, IK-controlled right arm.
        nominal = self._nominal_joint_targets[:, self._arm_joint_ids]
        limits = self._robot.data.joint_pos_limits.torch[:, self._arm_joint_ids]
        span = (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6)
        margin = torch.minimum(
            nominal - limits[..., 0], limits[..., 1] - nominal
        ) / span
        if float(margin.amin().item()) < self.cfg.minimum_joint_limit_margin_fraction:
            raise RuntimeError("active arm nominal pose is too close to a hard joint limit")

    def _log_episodes(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self, "_episode_sums"):
            return
        count = max(1, len(env_ids))
        self.extras["log"] = {
            f"Episode_Reward/{name}": value[env_ids].mean().item()
            for name, value in self._episode_sums.items()
        }
        self.extras["log"].update(
            {
                "Metrics/success_rate": self._success[env_ids].float().mean().item(),
                "Metrics/drop_rate": self._dropped[env_ids].float().mean().item(),
                "Metrics/detach_rate": self._stem_detached[env_ids].float().mean().item(),
                "Metrics/retained_rate": self._retained[env_ids].float().mean().item(),
                "Metrics/mean_hold_steps": self._hold_streak[env_ids].float().mean().item(),
                "Metrics/timeout_rate": self._timed_out[env_ids].float().mean().item(),
                "Metrics/reset_count": float(count),
            }
        )
        for value in self._episode_sums.values():
            value[env_ids] = 0.0


__all__ = ["TomatoPickEnv"]
