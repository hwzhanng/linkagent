"""Physics-v1 environment layered over the frozen Clean-v5 task interface."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_box_minus

from module.skills.tomato_pick.hand_synergy_calibrated_v2 import (
    CENTER_RAD as CALIBRATED_HAND_CENTER_RAD,
    FINGER_ACTIVE_JOINT_INDICES as CALIBRATED_FINGER_ACTIVE_JOINT_INDICES,
    NEGATIVE_ENDPOINTS_RAD as CALIBRATED_HAND_NEGATIVE_ENDPOINTS_RAD,
    POSITIVE_ENDPOINTS_RAD as CALIBRATED_HAND_POSITIVE_ENDPOINTS_RAD,
    THUMB_ACTIVE_JOINT_INDICES as CALIBRATED_THUMB_ACTIVE_JOINT_INDICES,
)

from simulation.tomato_pick.isaac.env import (
    MIMIC_FOLLOWER_JOINT_NAMES,
    RIGHT_HAND_MIMIC_RAD,
    TomatoPickEnv,
)
from simulation.tomato_pick.physics_v1 import (
    combined_stem_load_ratio,
    secure_kinematic_break_credit_gate,
    secure_support_credit_gate,
    stem_load_damage_progress_score,
    update_stem_damage,
)
from simulation.tomato_pick.reset_v1 import (
    sample_bounded_joint_ellipsoid_offsets,
    sample_bounded_joint_workspace_offsets,
    sample_hanging_fruit_quaternions,
    sample_pregrasp_fruit_offsets,
)

from .env_cfg import TomatoPickPhysicsV1EnvCfg


class TomatoPickPhysicsV1Env(TomatoPickEnv):
    """Clean-v5 behavior with independent actuator, contact, and stem physics."""

    cfg: TomatoPickPhysicsV1EnvCfg

    def __init__(self, cfg: TomatoPickPhysicsV1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        profiles = self._prestartup_physics_v1_profiles
        self._stem_tensile_limit_n = profiles["stem_tensile_force_n"].to(self.device)
        self._stem_shear_limit_n = profiles["stem_shear_force_n"].to(self.device)
        self._stem_bending_limit_nm = profiles["stem_bending_moment_nm"].to(self.device)
        self._stem_critical_duration_s = profiles["stem_critical_duration_s"].to(self.device)
        self._stem_damage = torch.zeros(self.num_envs, device=self.device)
        self._stem_combined_load_ratio = torch.zeros(self.num_envs, device=self.device)
        self._shadow_secure_support = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Diagnostic-only latches capture the exact 120 Hz substep at which a
        # stem fractures.  Policy-step values can otherwise be overwritten by
        # later substeps before DirectRLEnv asks for dones.
        self._break_substep_supported_fingertips = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._break_substep_thumb_support = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._break_substep_finger_support = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._break_substep_secure_support = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._break_substep_kinematic_retained = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._break_substep_distance_m = torch.zeros(self.num_envs, device=self.device)
        self._break_substep_relative_speed_m_s = torch.zeros(
            self.num_envs, device=self.device
        )
        self._break_substep_damage = torch.zeros(self.num_envs, device=self.device)
        self._break_substep_load_ratio = torch.zeros(self.num_envs, device=self.device)
        self._invalid_contact_force_n = torch.zeros(
            (self.num_envs, 5), device=self.device
        )
        self._reset_workspace_arm_joint_offset_rad = torch.zeros(
            (self.num_envs, 7), device=self.device
        )
        self._reset_workspace_base_arm_joint_offset_rad = torch.zeros(
            (self.num_envs, 7), device=self.device
        )
        self._reset_workspace_mode_coefficients = torch.zeros(
            (self.num_envs, len(self.cfg.reset_arm_correlated_modes_rad)),
            device=self.device,
        )
        self._reset_workspace_fruit_local_m = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._reset_workspace_fruit_tilt_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._reset_workspace_clearance_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._reset_workspace_translation_headroom_m = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._break_force_n.copy_(self._stem_tensile_limit_n)
        configured_velocity = torch.tensor(
            self.cfg.hand_joint_max_velocity_rad_s, device=self.device
        )
        asset_velocity = self._robot.data.joint_vel_limits.torch[
            0, self._hand_joint_ids
        ]
        self._effective_hand_velocity_rad_s = torch.minimum(
            configured_velocity, asset_velocity
        )
        self._linkage_driver_velocity_caps_rad_s: dict[str, float] = {}
        # A driver cannot physically outrun its passive follower.  The source
        # L20 URDF limits every distal follower to 1 rad/s; applying only the
        # driver's imported limit allowed 2.3--3.1 rad/s commands and produced
        # the measured 0.4537 rad tracking error.  Referring the follower limit
        # through the mimic ratio closes that modeling error without powering
        # the follower or weakening the 0.35 rad acceptance threshold.  A
        # small configured safety factor leaves finite phase-lag margin for
        # the compliant transmission at the 20 Hz policy boundary.
        for follower_name, (driver_name, multiplier, _offset) in (
            RIGHT_HAND_MIMIC_RAD.items()
        ):
            driver_id = self._robot.joint_names.index(driver_name)
            follower_id = self._robot.joint_names.index(follower_name)
            active_index = self._hand_joint_ids.index(driver_id)
            follower_limit = self._robot.data.joint_vel_limits.torch[
                0, follower_id
            ]
            referred_limit = (
                self.cfg.passive_mimic_driver_velocity_safety_factor
                * follower_limit
                / abs(float(multiplier))
            )
            self._effective_hand_velocity_rad_s[active_index] = torch.minimum(
                self._effective_hand_velocity_rad_s[active_index], referred_limit
            )
            self._linkage_driver_velocity_caps_rad_s[driver_name] = float(
                referred_limit
            )

    def _stem_progress_reward_score(self, opposed: torch.Tensor) -> torch.Tensor:
        """Credit load-duration progress only during real secure support."""

        return secure_support_credit_gate(
            stem_load_damage_progress_score(
                self._stem_combined_load_ratio,
                self._stem_damage,
                opposed,
            ),
            self._shadow_secure_support,
        )

    def _break_grasped_reward_score(
        self, fruit_in_hand: torch.Tensor, break_quality: torch.Tensor
    ) -> torch.Tensor:
        """Credit only a supported, kinematically retained 120 Hz fracture."""

        raw_credit = (
            self._stem_break_event.float()
            * fruit_in_hand.float()
            * break_quality
        )
        return secure_kinematic_break_credit_gate(
            raw_credit,
            self._break_substep_secure_support,
            self._break_substep_kinematic_retained,
        )

    def _set_hand_targets_from_synergy(self, synergy: torch.Tensor) -> None:
        """Apply the independent user-calibrated 5D manifold in Physics-v1."""

        if synergy.shape != (self.num_envs, 5):
            raise ValueError("hand synergy tensor must have shape (num_envs, 5)")
        action = synergy.clamp(-1.0, 1.0)
        center = torch.tensor(
            CALIBRATED_HAND_CENTER_RAD,
            device=self.device,
            dtype=action.dtype,
        )
        negative = torch.tensor(
            CALIBRATED_HAND_NEGATIVE_ENDPOINTS_RAD,
            device=self.device,
            dtype=action.dtype,
        )
        positive = torch.tensor(
            CALIBRATED_HAND_POSITIVE_ENDPOINTS_RAD,
            device=self.device,
            dtype=action.dtype,
        )
        endpoints = torch.where(
            (action >= 0.0).unsqueeze(2),
            positive.unsqueeze(0),
            negative.unsqueeze(0),
        )
        coordinate_mask = torch.zeros((5, 16), device=self.device, dtype=action.dtype)
        coordinate_mask[:4, list(CALIBRATED_FINGER_ACTIVE_JOINT_INDICES)] = 1.0
        coordinate_mask[4, list(CALIBRATED_THUMB_ACTIVE_JOINT_INDICES)] = 1.0
        targets = center.unsqueeze(0) + (
            action.abs().unsqueeze(2)
            * (endpoints - center.view(1, 1, -1))
            * coordinate_mask.unsqueeze(0)
        ).sum(dim=1)
        limits = self._robot.data.joint_pos_limits.torch[
            :, self._hand_joint_ids
        ]
        self._joint_targets[:, self._hand_joint_ids] = torch.maximum(
            limits[..., 0], torch.minimum(limits[..., 1], targets)
        )
        self._update_mimic_targets(torch.arange(self.num_envs, device=self.device))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        previous = self._joint_targets[:, self._hand_joint_ids].clone()
        super()._pre_physics_step(actions)
        desired = self._joint_targets[:, self._hand_joint_ids]
        maximum_delta = self._effective_hand_velocity_rad_s * (
            float(self.cfg.sim.dt) * float(self.cfg.decimation)
        )
        rate_limited = previous + (desired - previous).clamp(
            min=-maximum_delta, max=maximum_delta
        )
        self._joint_targets[:, self._hand_joint_ids] = rate_limited
        self._update_mimic_targets(torch.arange(self.num_envs, device=self.device))

    def _physical_finger_force(self) -> torch.Tensor:
        force = super()._physical_finger_force()
        if hasattr(self, "_invalid_contact_force_n"):
            valid_by_finger = []
            maximum_radius = self._fruit_axes_m.amax(dim=1)
            for sensor in self._contact_sensors.values():
                contact_position = sensor.data.contact_pos_w
                if contact_position is None:
                    valid_by_finger.append(
                        torch.ones(
                            self.num_envs, dtype=torch.bool, device=self.device
                        )
                    )
                    continue
                radius = torch.linalg.vector_norm(
                    contact_position.torch[:, 0, 0]
                    - self._tomato.data.root_pos_w.torch,
                    dim=1,
                )
                valid_by_finger.append(
                    radius
                    <= maximum_radius
                    + self.cfg.contact_point_maximum_surface_gap_m
                )
            valid = torch.stack(valid_by_finger, dim=1)
            self._invalid_contact_force_n.copy_(
                torch.where(valid, torch.zeros_like(force), force)
            )
            force = torch.where(valid, force, torch.zeros_like(force))
        return force

    def _update_mimic_targets(self, env_ids: torch.Tensor) -> None:
        """Keep reset/telemetry targets consistent with native mimic physics."""

        limits = self._robot.data.joint_pos_limits.torch[env_ids]
        for follower_name in MIMIC_FOLLOWER_JOINT_NAMES:
            side = (
                "right_hand_"
                if follower_name.startswith("right_hand_")
                else "left_hand_"
            )
            right_name = follower_name.replace("left_hand_", "right_hand_", 1)
            driver_right, multiplier, _legacy_offset = RIGHT_HAND_MIMIC_RAD[
                right_name
            ]
            driver_name = driver_right.replace("right_hand_", side, 1)
            follower_id = self._robot.joint_names.index(follower_name)
            driver_id = self._robot.joint_names.index(driver_name)
            target = (
                self._joint_targets[env_ids, driver_id] * multiplier
                + self.cfg.passive_mimic_offset_rad
            )
            self._joint_targets[env_ids, follower_id] = target.clamp(
                limits[:, follower_id, 0], limits[:, follower_id, 1]
            )

    def _project_native_mimic_state(self, env_ids: torch.Tensor) -> None:
        """Place follower state exactly on the transmission manifold at reset."""

        positions = self._robot.data.joint_pos.torch[env_ids].clone()
        # The first constraint solve may move the lightweight finger chain
        # slightly. Make the settled active pose the new drive equilibrium;
        # otherwise the old pre-solve target immediately pulls against the
        # mimic constraint on the first policy step.
        self._joint_targets[env_ids] = positions
        limits = self._robot.data.joint_pos_limits.torch[env_ids]
        follower_ids = []
        follower_positions = []
        for follower_name in MIMIC_FOLLOWER_JOINT_NAMES:
            side = (
                "right_hand_"
                if follower_name.startswith("right_hand_")
                else "left_hand_"
            )
            right_name = follower_name.replace("left_hand_", "right_hand_", 1)
            driver_right, multiplier, _legacy_offset = RIGHT_HAND_MIMIC_RAD[
                right_name
            ]
            driver_name = driver_right.replace("right_hand_", side, 1)
            follower_id = self._robot.joint_names.index(follower_name)
            driver_id = self._robot.joint_names.index(driver_name)
            target = (
                positions[:, driver_id] * multiplier
                + self.cfg.passive_mimic_offset_rad
            ).clamp(
                limits[:, follower_id, 0], limits[:, follower_id, 1]
            )
            follower_ids.append(follower_id)
            follower_positions.append(target)
            self._joint_targets[env_ids, follower_id] = target
        self._robot.write_joint_position_to_sim_index(
            position=torch.stack(follower_positions, dim=1),
            joint_ids=follower_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(positions),
            joint_ids=self._actuated_joint_ids,
            env_ids=env_ids,
        )
        self.scene.write_data_to_sim()

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
        load_ratio = combined_stem_load_ratio(
            force,
            torque,
            pull_axis,
            self._stem_tensile_limit_n,
            self._stem_shear_limit_n,
            self._stem_bending_limit_nm,
        )
        self._stem_combined_load_ratio.copy_(load_ratio)
        damage, break_now = update_stem_damage(
            self._stem_damage,
            load_ratio,
            dt_s=float(self.cfg.sim.dt),
            critical_duration_s=self._stem_critical_duration_s,
            onset_ratio=self.cfg.stem_damage_onset_ratio,
            recovery_time_s=self.cfg.stem_damage_recovery_time_s,
            attached=attached,
        )
        self._stem_damage.copy_(damage)
        if bool(break_now.any()):
            physical_force = self._physical_finger_force()
            supported_mask = physical_force >= self.cfg.minimum_support_force_n
            supported = supported_mask.sum(dim=1)
            tips = self._robot.data.body_pos_w.torch[
                :, self._fingertip_contact_body_ids
            ]
            tip_center = tips.mean(dim=1)
            tip_velocity = self._robot.data.body_lin_vel_w.torch[
                :, self._fingertip_contact_body_ids
            ].mean(dim=1)
            distance = torch.linalg.vector_norm(fruit_pose[:, :3] - tip_center, dim=1)
            relative_speed = torch.linalg.vector_norm(
                fruit_lin_vel - tip_velocity, dim=1
            )
            self._break_substep_supported_fingertips[break_now] = supported[break_now]
            self._break_substep_thumb_support[break_now] = supported_mask[break_now, 0]
            self._break_substep_finger_support[break_now] = supported_mask[
                break_now, 1:
            ].any(dim=1)
            substep_secure = supported >= self.cfg.minimum_supported_fingertips
            if self.cfg.thumb_opposition_required:
                substep_secure &= supported_mask[:, 0] & supported_mask[:, 1:].any(
                    dim=1
                )
            self._break_substep_secure_support[break_now] = substep_secure[break_now]
            self._break_substep_kinematic_retained[break_now] = (
                (distance[break_now] <= self.cfg.maximum_retained_distance_m)
                & (
                    relative_speed[break_now]
                    <= self.cfg.maximum_retained_relative_speed_m_s
                )
            )
            self._break_substep_distance_m[break_now] = distance[break_now]
            self._break_substep_relative_speed_m_s[break_now] = relative_speed[break_now]
            self._break_substep_damage[break_now] = damage[break_now]
            self._break_substep_load_ratio[break_now] = load_ratio[break_now]
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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        supported_mask = (
            self._last_physical_force_n >= self.cfg.minimum_support_force_n
        )
        supported = supported_mask.sum(dim=1)
        secure_support = supported >= self.cfg.minimum_supported_fingertips
        thumb_support = supported_mask[:, 0]
        finger_support = supported_mask[:, 1:].any(dim=1)
        if self.cfg.thumb_opposition_required:
            secure_support &= thumb_support & finger_support
        self._shadow_secure_support.copy_(secure_support)
        # Physics-v1 promotes secure load-bearing contact into the actual task
        # outcome.  Proximity and low relative speed alone can no longer count
        # a fruit that is merely flying alongside the hand as retained.
        self._retained &= self._shadow_secure_support
        self._hold_streak = torch.where(
            self._retained,
            self._hold_streak,
            torch.zeros_like(self._hold_streak),
        )
        self._success = self._retained & (
            self._hold_streak >= self.cfg.success_hold_steps
        )
        self._timed_out = (
            (self.episode_length_buf >= self.max_episode_length - 1)
            & ~self._success
            & ~self._terminal_failure
        )
        terminated = self._success | self._terminal_failure
        truncated = self._timed_out
        self.extras["step"]["success_count"] = int(self._success.sum().item())
        self.extras["step"]["retained_count"] = int(self._retained.sum().item())
        self.extras["step"]["timeout_count"] = int(self._timed_out.sum().item())
        self.extras["step"]["shadow_secure_support_count"] = int(
            self._shadow_secure_support.sum().item()
        )
        self.extras["step"]["maximum_fingertip_force_n"] = float(
            self._last_physical_force_n.max().item()
        )
        self.extras["step"]["maximum_stem_load_ratio"] = float(
            self._stem_combined_load_ratio.max().item()
        )
        self.extras["step"]["maximum_invalid_contact_force_n"] = float(
            self._invalid_contact_force_n.max().item()
        )
        # Read-only event diagnostics.  These aggregates are captured before
        # DirectRLEnv resets terminal environments, so a fracture immediately
        # followed by drop cannot disappear from an offline audit.
        break_event = self._stem_break_event
        break_count = int(break_event.sum().item())
        self.extras["step"]["physics_v1_break_count"] = break_count
        self.extras["step"]["physics_v1_break_secure_support_count"] = int(
            (break_event & self._break_substep_secure_support).sum().item()
        )
        self.extras["step"]["physics_v1_break_kinematic_retained_count"] = int(
            (break_event & self._break_substep_kinematic_retained).sum().item()
        )
        distance_ok = (
            self._break_substep_distance_m <= self.cfg.maximum_retained_distance_m
        )
        speed_ok = (
            self._break_substep_relative_speed_m_s
            <= self.cfg.maximum_retained_relative_speed_m_s
        )
        self.extras["step"]["physics_v1_break_distance_ok_count"] = int(
            (break_event & distance_ok).sum().item()
        )
        self.extras["step"]["physics_v1_break_speed_ok_count"] = int(
            (break_event & speed_ok).sum().item()
        )
        self.extras["step"]["physics_v1_break_secure_and_kinematic_count"] = int(
            (
                break_event
                & self._break_substep_secure_support
                & self._break_substep_kinematic_retained
            ).sum().item()
        )
        self.extras["step"]["physics_v1_break_secure_and_distance_ok_count"] = int(
            (break_event & self._break_substep_secure_support & distance_ok)
            .sum()
            .item()
        )
        self.extras["step"]["physics_v1_break_secure_and_speed_ok_count"] = int(
            (break_event & self._break_substep_secure_support & speed_ok)
            .sum()
            .item()
        )
        self.extras["step"]["physics_v1_break_thumb_support_count"] = int(
            (break_event & self._break_substep_thumb_support).sum().item()
        )
        self.extras["step"]["physics_v1_break_finger_support_count"] = int(
            (break_event & self._break_substep_finger_support).sum().item()
        )
        self.extras["step"]["physics_v1_break_supported_fingertip_sum"] = int(
            self._break_substep_supported_fingertips[break_event].sum().item()
        )
        self.extras["step"]["physics_v1_break_drop_count"] = int(
            (break_event & self._dropped).sum().item()
        )
        self.extras["step"]["physics_v1_break_excess_force_count"] = int(
            (break_event & self._excess_force).sum().item()
        )
        if break_count:
            self.extras["step"]["physics_v1_break_distance_sum_m"] = float(
                self._break_substep_distance_m[break_event].sum().item()
            )
            self.extras["step"]["physics_v1_break_relative_speed_sum_m_s"] = float(
                self._break_substep_relative_speed_m_s[break_event].sum().item()
            )
            self.extras["step"]["physics_v1_break_damage_sum"] = float(
                self._break_substep_damage[break_event].sum().item()
            )
            self.extras["step"]["physics_v1_break_load_ratio_sum"] = float(
                self._break_substep_load_ratio[break_event].sum().item()
            )
        contact_radii = []
        fruit_position = self._tomato.data.root_pos_w.torch
        for sensor in self._contact_sensors.values():
            contact_position = sensor.data.contact_pos_w
            force_matrix = sensor.data.force_matrix_w
            if contact_position is None or force_matrix is None:
                continue
            position = contact_position.torch[:, 0, 0]
            force_norm = torch.linalg.vector_norm(
                force_matrix.torch[:, 0, 0], dim=1
            )
            radius = torch.linalg.vector_norm(position - fruit_position, dim=1)
            active = (force_norm > 1.0e-4) & (
                radius
                <= self._fruit_axes_m.amax(dim=1)
                + self.cfg.contact_point_maximum_surface_gap_m
            )
            if bool(active.any()):
                contact_radii.append(
                    torch.linalg.vector_norm(
                        position[active] - fruit_position[active], dim=1
                    )
                )
        if contact_radii:
            radii = torch.cat(contact_radii)
            self.extras["step"]["minimum_contact_radius_m"] = float(radii.min())
            self.extras["step"]["maximum_contact_radius_m"] = float(radii.max())
        return terminated, truncated

    def _push_fruit_to_reset_clearance(
        self,
        fruit_position_w: torch.Tensor,
        escape_direction_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move an intersecting fruit outward on its sampled approach ray."""

        target = self.cfg.minimum_initial_clearance_m + 3.0e-4
        position = fruit_position_w.clone()
        clearance = self._hand_collision_clearance_for(
            position, self._fruit_axes_m[env_ids], env_ids
        )
        invalid = clearance < target
        if not bool(invalid.any()):
            return position, clearance
        body_position = self._robot.data.body_pos_w.torch[env_ids][
            :, self._hand_collision_body_ids
        ]
        body_quaternion = self._robot.data.body_quat_w.torch[env_ids][
            :, self._hand_collision_body_ids
        ]
        body_count = len(self._hand_collision_body_ids)
        local_direction = quat_apply_inverse(
            body_quaternion.reshape(-1, 4),
            escape_direction_w.unsqueeze(1)
            .expand(-1, body_count, -1)
            .reshape(-1, 3),
        ).reshape(len(env_ids), body_count, 3)
        expanded_half_extent = self._hand_collision_aabb_half_extents.unsqueeze(0) + (
            self._fruit_axes_m[env_ids].amax(dim=1) + target
        ).view(len(env_ids), 1, 1)
        for _ in range(body_count + 1):
            relative = position.unsqueeze(1) - body_position
            local = quat_apply_inverse(
                body_quaternion.reshape(-1, 4), relative.reshape(-1, 3)
            ).reshape(len(env_ids), body_count, 3)
            local -= self._hand_collision_aabb_centers.unsqueeze(0)
            contains = (
                (local.abs() < expanded_half_extent).all(dim=2)
                & invalid.unsqueeze(1)
            )
            if not bool(contains.any()):
                break
            exit_face = torch.where(
                local_direction >= 0.0,
                expanded_half_extent,
                -expanded_half_extent,
            )
            exit_distance_by_axis = torch.where(
                local_direction.abs() > 1.0e-8,
                (exit_face - local) / local_direction,
                torch.full_like(local_direction, float("inf")),
            )
            exit_distance_by_body = exit_distance_by_axis.amin(dim=2)
            shift = torch.where(
                contains,
                exit_distance_by_body,
                torch.zeros_like(exit_distance_by_body),
            ).amax(dim=1)
            position += (
                escape_direction_w
                * (shift + 1.0e-4).unsqueeze(1)
                * invalid.unsqueeze(1)
            )
            clearance = self._hand_collision_clearance_for(
                position, self._fruit_axes_m[env_ids], env_ids
            )
            invalid = clearance < target
        clearance = self._hand_collision_clearance_for(
            position, self._fruit_axes_m[env_ids], env_ids
        )
        if bool((clearance < self.cfg.minimum_initial_clearance_m).any()):
            raise RuntimeError(
                "Physics-v1 reset could not produce a collision-free fruit pose"
            )
        return position, clearance

    def _apply_workspace_reset(self, env_ids: torch.Tensor) -> None:
        """Sample A7 first, then place one reachable fruit around the palm."""

        count = len(env_ids)
        # Move the fruit away before changing arm configuration.  A reset must
        # never create a transient fruit-hand impulse at the previous pose.
        parking_pose = self._tomato.data.root_pose_w.torch[env_ids].clone()
        parking_pose[:, :3] = self.scene.env_origins[env_ids]
        parking_pose[:, 2] += 2.5
        parking_pose[:, 3:7] = 0.0
        parking_pose[:, 3] = 1.0
        self._tomato.write_root_pose_to_sim_index(
            root_pose=parking_pose, env_ids=env_ids
        )
        self._tomato.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((count, 6), device=self.device),
            env_ids=env_ids,
        )

        joint_radius = torch.tensor(
            self.cfg.reset_arm_joint_ellipsoid_radius_rad,
            device=self.device,
        )
        nominal = self._nominal_joint_targets[env_ids][:, self._arm_joint_ids]
        limits = self._robot.data.joint_pos_limits.torch[
            env_ids[:, None], self._arm_joint_ids
        ]
        margin = (
            self.cfg.reset_joint_limit_margin_fraction
            * (limits[..., 1] - limits[..., 0])
        )
        lower = limits[..., 0] + margin
        upper = limits[..., 1] - margin
        correlated_modes = torch.tensor(
            self.cfg.reset_arm_correlated_modes_rad, device=self.device
        )
        if correlated_modes.numel() == 0:
            offset = sample_bounded_joint_ellipsoid_offsets(
                count,
                joint_radius,
                lower_offset_rad=lower - nominal,
                upper_offset_rad=upper - nominal,
            )
            base_offset = offset
            mode_coefficients = torch.empty((count, 0), device=self.device)
        else:
            sampled_workspace = sample_bounded_joint_workspace_offsets(
                count,
                joint_radius,
                correlated_modes_rad=correlated_modes,
                mode_coefficient_ranges=torch.tensor(
                    self.cfg.reset_arm_correlated_mode_coefficient_ranges,
                    device=self.device,
                ),
                lower_offset_rad=lower - nominal,
                upper_offset_rad=upper - nominal,
            )
            offset = sampled_workspace["offset_rad"]
            base_offset = sampled_workspace["base_ellipsoid_offset_rad"]
            mode_coefficients = sampled_workspace["mode_coefficients"]
        arm_position = nominal + offset
        self._joint_targets[env_ids[:, None], self._arm_joint_ids] = arm_position
        self._update_mimic_targets(env_ids)
        joint_position = self._joint_targets[env_ids]
        self._robot.write_joint_position_to_sim_index(
            position=joint_position,
            joint_ids=self._actuated_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(joint_position),
            joint_ids=self._actuated_joint_ids,
            env_ids=env_ids,
        )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        self._project_native_mimic_state(env_ids)
        self.sim.forward()
        self.scene.update(dt=0.0)

        ee_pose = self._robot.data.body_pose_w.torch[env_ids, self._ee_body_id].clone()
        tips = self._robot.data.body_pos_w.torch[
            env_ids[:, None], self._fingertip_frame_body_ids
        ]
        translation_limit = torch.tensor(
            self.cfg.max_residual_translation_m, device=self.device
        )
        required_headroom = (
            self.cfg.minimum_closure_translation_headroom_fraction
            * translation_limit
        )
        closed_center = torch.tensor(
            self.cfg.closed_grasp_fruit_center_m, device=self.device
        )
        fruit_quaternion = torch.empty((count, 4), device=self.device)
        fruit_tilt = torch.empty(count, device=self.device)
        pull_axis = torch.empty((count, 3), device=self.device)
        basis = torch.empty((count, 3, 3), device=self.device)
        fruit_position = torch.empty((count, 3), device=self.device)
        actual_local_offset = torch.empty((count, 3), device=self.device)
        clearance = torch.empty(count, device=self.device)
        translation_headroom = torch.empty((count, 3), device=self.device)
        pending = torch.arange(count, device=self.device)
        for _ in range(self.cfg.reset_fruit_sampling_max_attempts):
            if pending.numel() == 0:
                break
            pending_count = int(pending.numel())
            candidate_orientation = sample_hanging_fruit_quaternions(
                pending_count,
                maximum_tilt_rad=self.cfg.reset_fruit_stem_tilt_rad_max,
                yaw_range_rad=self.cfg.reset_fruit_yaw_rad,
                device=self.device,
            )
            candidate_quaternion = candidate_orientation["quaternion_wxyz"]
            candidate_up = quat_apply(
                candidate_quaternion,
                torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(
                    pending_count, -1
                ),
            )
            candidate_pull = -candidate_up
            pending_ee = ee_pose[pending]
            support_center = tips[pending, 1:5].mean(dim=1)
            candidate_approach = support_center - pending_ee[:, :3]
            candidate_approach -= (
                candidate_approach * candidate_pull
            ).sum(dim=1, keepdim=True) * candidate_pull
            approach_norm = torch.linalg.vector_norm(
                candidate_approach, dim=1, keepdim=True
            )
            fallback = quat_apply(
                pending_ee[:, 3:7],
                torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(
                    pending_count, -1
                ),
            )
            fallback -= (fallback * candidate_pull).sum(
                dim=1, keepdim=True
            ) * candidate_pull
            candidate_approach = torch.where(
                approach_norm > 1.0e-5, candidate_approach, fallback
            )
            candidate_approach = torch.nn.functional.normalize(
                candidate_approach, dim=1
            )
            candidate_lateral = torch.nn.functional.normalize(
                torch.linalg.cross(
                    candidate_pull, candidate_approach, dim=1
                ),
                dim=1,
            )
            candidate_basis = torch.stack(
                (candidate_approach, candidate_lateral, candidate_pull), dim=2
            )
            candidate_local = sample_pregrasp_fruit_offsets(
                pending_count,
                closed_grasp_center_m=closed_center,
                distance_range_m=self.cfg.reset_fruit_approach_distance_m,
                yaw_range_rad=self.cfg.reset_fruit_approach_yaw_rad,
                pitch_range_rad=self.cfg.reset_fruit_approach_pitch_rad,
            )
            candidate_position = pending_ee[:, :3] + torch.bmm(
                candidate_basis,
                candidate_local["local_position_m"].unsqueeze(2),
            ).squeeze(2)
            escape_direction = torch.bmm(
                candidate_basis,
                candidate_local["direction_local"].unsqueeze(2),
            ).squeeze(2)
            candidate_position, candidate_clearance = (
                self._push_fruit_to_reset_clearance(
                    candidate_position, escape_direction, env_ids[pending]
                )
            )
            candidate_actual_local = torch.bmm(
                candidate_basis.transpose(1, 2),
                (candidate_position - pending_ee[:, :3]).unsqueeze(2),
            ).squeeze(2)
            candidate_closure_translation = candidate_actual_local - closed_center
            candidate_headroom = (
                translation_limit - candidate_closure_translation.abs()
            )
            candidate_closure_distance = torch.linalg.vector_norm(
                candidate_closure_translation, dim=1
            )
            valid = (
                (candidate_headroom >= required_headroom).all(dim=1)
                & (
                    candidate_closure_distance
                    <= self.cfg.maximum_closure_translation_norm_m
                )
            )
            if bool(valid.any()):
                selected = pending[valid]
                fruit_quaternion[selected] = candidate_quaternion[valid]
                fruit_tilt[selected] = candidate_orientation["tilt_rad"][valid]
                pull_axis[selected] = candidate_pull[valid]
                basis[selected] = candidate_basis[valid]
                fruit_position[selected] = candidate_position[valid]
                actual_local_offset[selected] = candidate_actual_local[valid]
                clearance[selected] = candidate_clearance[valid]
                translation_headroom[selected] = candidate_headroom[valid]
            pending = pending[~valid]
        if pending.numel() != 0:
            print(
                "PHYSICS_V1_FRUIT_RESET_SAMPLING_FAILED="
                f"pending={int(pending.numel())},"
                f"attempts={self.cfg.reset_fruit_sampling_max_attempts},"
                f"closure_limit_m={self.cfg.maximum_closure_translation_norm_m}"
            )
            raise RuntimeError(
                "Physics-v1 reset could not sample a fruit with closure workspace margin"
            )

        fruit_pose = self._tomato.data.default_root_pose.torch[env_ids].clone()
        fruit_pose[:, :3] = fruit_position
        fruit_pose[:, 3:7] = fruit_quaternion
        self._tomato.write_root_pose_to_sim_index(
            root_pose=fruit_pose, env_ids=env_ids
        )
        self._tomato.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((count, 6), device=self.device),
            env_ids=env_ids,
        )

        local_attachment = torch.zeros((count, 3), device=self.device)
        local_attachment[:, 2] = self._fruit_axes_m[env_ids, 2]
        attachment = fruit_position + quat_apply(fruit_quaternion, local_attachment)
        plant_anchor = attachment - pull_axis * self.cfg.stem_visual_height_m
        self._grasp_basis_w[env_ids] = basis
        self._ee_reference_pose_w[env_ids] = ee_pose
        self._ee_command_pose_w[env_ids] = ee_pose
        self._stem_rest_attachment_pos_w[env_ids] = attachment
        self._stem_plant_anchor_pos_w[env_ids] = plant_anchor
        self._stem_anchor_quat_w[env_ids] = fruit_quaternion
        self._move_stem_visual(env_ids, attachment, plant_anchor)
        self._best_reach_score[env_ids] = self._reach_score_for(
            fruit_position,
            tips,
            ee_pose[:, :3],
            self._fruit_axes_m[env_ids],
        )
        if hasattr(self, "_reset_workspace_arm_joint_offset_rad"):
            self._reset_workspace_arm_joint_offset_rad[env_ids] = arm_position - nominal
            self._reset_workspace_base_arm_joint_offset_rad[env_ids] = base_offset
            self._reset_workspace_mode_coefficients[env_ids] = mode_coefficients
            self._reset_workspace_fruit_local_m[env_ids] = actual_local_offset
            self._reset_workspace_fruit_tilt_rad[env_ids] = fruit_tilt
            self._reset_workspace_clearance_m[env_ids] = clearance
            self._reset_workspace_translation_headroom_m[env_ids] = translation_headroom
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        center = torch.tensor(
            CALIBRATED_HAND_CENTER_RAD,
            device=self.device,
            dtype=self._joint_targets.dtype,
        )
        limits = self._robot.data.joint_pos_limits.torch[
            env_ids[:, None], self._hand_joint_ids
        ]
        self._joint_targets[env_ids[:, None], self._hand_joint_ids] = torch.maximum(
            limits[..., 0], torch.minimum(limits[..., 1], center)
        )
        self._update_mimic_targets(env_ids)
        self._apply_workspace_reset(env_ids)
        # Clear contact history only after the final sampled arm and fruit
        # poses have been forwarded. Intermediate parking/reset poses must not
        # leak a force into the first policy observation.
        for sensor in self._contact_sensors.values():
            sensor.reset(env_ids)
        if not hasattr(self, "_stem_damage"):
            return
        self._invalid_contact_force_n[env_ids] = 0.0
        self._stem_damage[env_ids] = 0.0
        self._stem_combined_load_ratio[env_ids] = 0.0
        self._shadow_secure_support[env_ids] = False
        self._break_substep_supported_fingertips[env_ids] = 0
        self._break_substep_thumb_support[env_ids] = False
        self._break_substep_finger_support[env_ids] = False
        self._break_substep_secure_support[env_ids] = False
        self._break_substep_kinematic_retained[env_ids] = False
        self._break_substep_distance_m[env_ids] = 0.0
        self._break_substep_relative_speed_m_s[env_ids] = 0.0
        self._break_substep_damage[env_ids] = 0.0
        self._break_substep_load_ratio[env_ids] = 0.0
        self._break_force_n[env_ids] = self._stem_tensile_limit_n[env_ids]


__all__ = ["TomatoPickPhysicsV1Env"]
