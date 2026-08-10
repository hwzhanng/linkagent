"""Step-1 Physics-v2 environment with no behavior override."""

from __future__ import annotations

import importlib

import torch
import warp as wp

from isaaclab.utils.math import quat_apply, quat_box_minus

from module.skills.tomato_pick.hand_synergy_calibrated_v2 import (
    CENTER_RAD as CALIBRATED_HAND_CENTER_RAD,
    FINGER_ACTIVE_JOINT_INDICES as CALIBRATED_FINGER_ACTIVE_JOINT_INDICES,
    NEGATIVE_ENDPOINTS_RAD as CALIBRATED_HAND_NEGATIVE_ENDPOINTS_RAD,
    THUMB_ACTIVE_JOINT_INDICES as CALIBRATED_THUMB_ACTIVE_JOINT_INDICES,
)

from simulation.tomato_pick.isaac.physics_v1.env import TomatoPickPhysicsV1Env
from simulation.tomato_pick.physics_v1 import update_stem_damage
from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_models import (
    apply_joint_target_uncertainty_v2,
    initial_fruit_damage_state,
    initial_tactile_uncertainty_state,
    sample_uncertainty_profiles_v2,
    stem_load_components_v2,
    update_fruit_damage_diagnostics,
    update_tactile_uncertainty_v2,
)

from .env_cfg import TomatoPickPhysicsV2EnvCfg


class TomatoPickPhysicsV2Env(TomatoPickPhysicsV1Env):
    """Physics-v1 task with v2 fruit/contact priors and damage diagnostics."""

    cfg: TomatoPickPhysicsV2EnvCfg

    def __init__(self, cfg: TomatoPickPhysicsV2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Physics-v1 inherited the parent-of-tip contact bodies even though
        # the formal USD exposes one additional distal body per finger.  Keep
        # reward geometry, physical support, and ContactSensor prims on the
        # same true fingertip bodies in Physics-v2.
        self._fingertip_contact_body_ids = list(self._fingertip_frame_body_ids)
        if not hasattr(self, "_prestartup_physics_v2_profiles"):
            raise RuntimeError("Physics-v2 fruit profile event did not run")
        self._fruit_damage_cfg = load_physics_v2_profile()["fruit_damage_proxy_v2"]
        reference = torch.zeros((self.num_envs, 5), device=self.device)
        self._fruit_damage_state = initial_fruit_damage_state(reference)
        self._fruit_effective_stiffness_n_m = self._prestartup_physics_v2_profiles[
            "contact_stiffness_n_m"
        ].to(self.device)
        self._fruit_estimated_indentation_m = torch.zeros_like(reference)
        self._stem_torsional_limit_nm = self._prestartup_physics_v2_profiles[
            "stem_torsional_moment_nm"
        ].to(self.device)
        self._stem_load_components_v2 = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("tension", "shear", "bending", "torsion", "combined")
        }
        self._uncertainty_cfg = load_physics_v2_profile()["uncertainty_v2"]
        self._uncertainty_profiles = sample_uncertainty_profiles_v2(
            self.num_envs, device=self.device
        )
        maximum_tactile_delay = int(
            self._uncertainty_cfg["episode_fixed"]["tactile_delay_policy_steps"][1]
        )
        self._tactile_uncertainty_state = initial_tactile_uncertainty_state(
            reference, maximum_delay_steps=maximum_tactile_delay
        )
        if "palm" not in self._aux_contact_sensors:
            raise RuntimeError("Physics-v2 requires an explicit palm contact sensor")
        self._palm_contact_force_n = torch.zeros(self.num_envs, device=self.device)
        self._palm_support = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._invalid_palm_contact_force_n = torch.zeros(
            self.num_envs, device=self.device
        )

    def _apply_workspace_reset(self, env_ids: torch.Tensor) -> None:
        """Open the hand before placing a fruit around the closure envelope.

        Physics-v1 places the fruit while the calibrated hand is at its center
        pose.  For centers that are actually inside the closing envelope, that
        pose can already overlap the fruit and force the reset sampler to push
        it beyond the local skill workspace.  Physics-v2 uses the frozen
        negative flexion and thumb endpoints only for the pre-contact reset;
        the policy still owns the full five-dimensional hand command.
        """

        def apply_parent_reset_with_xyzw_quaternions() -> None:
            # The frozen Physics-v1 sampler intentionally returns wxyz, while
            # Isaac Lab v3/Sim 6 math and tensor APIs use xyzw.  Adapt only the
            # Physics-v2 call boundary so the parent evidence stays untouched.
            parent_module = importlib.import_module(
                "simulation.tomato_pick.isaac.physics_v1.env"
            )
            original_sampler = parent_module.sample_hanging_fruit_quaternions

            def sample_xyzw(*args, **kwargs):
                sampled = original_sampler(*args, **kwargs)
                sampled["quaternion_wxyz"] = torch.roll(
                    sampled["quaternion_wxyz"], shifts=-1, dims=1
                )
                return sampled

            parent_module.sample_hanging_fruit_quaternions = sample_xyzw
            try:
                super(TomatoPickPhysicsV2Env, self)._apply_workspace_reset(env_ids)
            finally:
                parent_module.sample_hanging_fruit_quaternions = original_sampler

        if not self.cfg.open_hand_before_workspace_reset:
            apply_parent_reset_with_xyzw_quaternions()
            return

        pregrasp = torch.tensor(
            CALIBRATED_HAND_CENTER_RAD,
            device=self.device,
            dtype=self._joint_targets.dtype,
        )
        finger_open = CALIBRATED_HAND_NEGATIVE_ENDPOINTS_RAD[0]
        thumb_open = CALIBRATED_HAND_NEGATIVE_ENDPOINTS_RAD[4]
        pregrasp[list(CALIBRATED_FINGER_ACTIVE_JOINT_INDICES)] = torch.tensor(
            [finger_open[index] for index in CALIBRATED_FINGER_ACTIVE_JOINT_INDICES],
            device=self.device,
            dtype=pregrasp.dtype,
        )
        pregrasp[list(CALIBRATED_THUMB_ACTIVE_JOINT_INDICES)] = torch.tensor(
            [thumb_open[index] for index in CALIBRATED_THUMB_ACTIVE_JOINT_INDICES],
            device=self.device,
            dtype=pregrasp.dtype,
        )
        limits = self._robot.data.joint_pos_limits.torch[
            env_ids[:, None], self._hand_joint_ids
        ]
        target = pregrasp.unsqueeze(0).expand(len(env_ids), -1)
        self._joint_targets[env_ids[:, None], self._hand_joint_ids] = torch.maximum(
            limits[..., 0], torch.minimum(limits[..., 1], target)
        )
        self._update_mimic_targets(env_ids)
        apply_parent_reset_with_xyzw_quaternions()

    def _move_stem_visual(
        self,
        env_ids: torch.Tensor,
        fruit_side_w: torch.Tensor,
        plant_side_w: torch.Tensor,
    ) -> None:
        """Place the stem using the Isaac Lab v3 xyzw quaternion convention."""

        segment = plant_side_w - fruit_side_w
        direction = torch.nn.functional.normalize(segment, dim=1)
        self._stem_visual_positions_w[env_ids] = 0.5 * (
            fruit_side_w + plant_side_w
        )
        dot = direction[:, 2].clamp(-1.0, 1.0)
        quaternion_xyzw = torch.stack(
            (
                -direction[:, 1],
                direction[:, 0],
                torch.zeros_like(dot),
                1.0 + dot,
            ),
            dim=1,
        )
        opposite = dot < -0.999999
        quaternion_xyzw[opposite] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), device=self.device
        )
        quaternion_xyzw = torch.nn.functional.normalize(quaternion_xyzw, dim=1)
        self._stem_visual_orientations_w[env_ids] = quaternion_xyzw
        self._stem_view.set_world_poses(
            positions=wp.from_torch(
                self._stem_visual_positions_w.contiguous(), dtype=wp.vec3
            ),
            orientations=wp.from_torch(
                self._stem_visual_orientations_w.contiguous(), dtype=wp.quat
            ),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        if not hasattr(self, "_uncertainty_profiles"):
            return
        hand_current = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
        hand_desired = self._joint_targets[:, self._hand_joint_ids]
        hand_limits = self._robot.data.joint_pos_limits.torch[:, self._hand_joint_ids]
        hand_maximum_delta = self._effective_hand_velocity_rad_s.unsqueeze(0).expand_as(
            hand_current
        ) * (float(self.cfg.sim.dt) * float(self.cfg.decimation))
        self._joint_targets[:, self._hand_joint_ids] = apply_joint_target_uncertainty_v2(
            hand_desired,
            hand_current,
            hand_limits[..., 0],
            hand_limits[..., 1],
            hand_maximum_delta,
            self._uncertainty_profiles["hand_response_scale"],
            self._uncertainty_profiles["hand_target_bias_rad"],
            self._uncertainty_profiles["hand_deadband_rad"],
            self._uncertainty_profiles["hand_backlash_rad"],
        )
        arm_current = self._robot.data.joint_pos.torch[:, self._arm_joint_ids]
        arm_desired = self._joint_targets[:, self._arm_joint_ids]
        arm_limits = self._robot.data.joint_pos_limits.torch[:, self._arm_joint_ids]
        arm_shape = arm_current.shape
        self._joint_targets[:, self._arm_joint_ids] = apply_joint_target_uncertainty_v2(
            arm_desired,
            arm_current,
            arm_limits[..., 0],
            arm_limits[..., 1],
            torch.full_like(arm_current, self.cfg.maximum_arm_joint_delta_rad),
            self._uncertainty_profiles["arm_response_scale"].expand(arm_shape),
            torch.zeros_like(arm_current),
            torch.zeros_like(arm_current),
            torch.zeros_like(arm_current),
        )
        self._update_mimic_targets(torch.arange(self.num_envs, device=self.device))

    def _measured_finger_pressure(self) -> torch.Tensor:
        if not hasattr(self, "_uncertainty_profiles"):
            return super()._measured_finger_pressure()
        physical = self._physical_finger_force()
        self._last_physical_force_n.copy_(physical)
        self._tactile_uncertainty_state = update_tactile_uncertainty_v2(
            self._tactile_uncertainty_state,
            physical,
            self._uncertainty_profiles,
            dt_s=float(self.cfg.sim.dt) * float(self.cfg.decimation),
        )
        return self._tactile_uncertainty_state["measured_n"]

    def _physical_palm_force(self) -> torch.Tensor:
        """Return tomato-filtered physical force on the palm body."""

        sensor = self._aux_contact_sensors["palm"]
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            raise RuntimeError("tomato-filtered palm contact sensor did not initialize")
        force = torch.linalg.vector_norm(matrix.torch[:, 0, 0], dim=-1).clamp(0.0, 25.0)
        contact_position = sensor.data.contact_pos_w
        if contact_position is None:
            valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            radius = torch.linalg.vector_norm(
                contact_position.torch[:, 0, 0] - self._tomato.data.root_pos_w.torch,
                dim=1,
            )
            valid = radius <= (
                self._fruit_axes_m.amax(dim=1)
                + self.cfg.contact_point_maximum_surface_gap_m
            )
        self._invalid_palm_contact_force_n.copy_(
            torch.where(valid, torch.zeros_like(force), force)
        )
        return torch.where(valid, force, torch.zeros_like(force))

    def _retention_reference(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the palm as the retained-distance frame when it bears the fruit."""

        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        tip_center = tips.mean(dim=1)
        tip_velocity = self._robot.data.body_lin_vel_w.torch[
            :, self._fingertip_contact_body_ids
        ].mean(dim=1)
        palm_position = self._robot.data.body_pos_w.torch[:, self._ee_body_id]
        palm_velocity = self._robot.data.body_lin_vel_w.torch[:, self._ee_body_id]
        palm_force = self._physical_palm_force()
        palm_support = palm_force >= self.cfg.minimum_palm_support_force_n
        self._palm_contact_force_n.copy_(palm_force)
        self._palm_support.copy_(palm_support)
        use_palm = self._stem_detached & palm_support
        return (
            torch.where(use_palm.unsqueeze(1), palm_position, tip_center),
            torch.where(use_palm.unsqueeze(1), palm_velocity, tip_velocity),
        )

    def _retention_support_mask(self) -> torch.Tensor:
        """Require finger opposition for detachment, then allow palm support."""

        supported_mask = (
            self._last_physical_force_n >= self.cfg.minimum_support_force_n
        )
        supported = supported_mask.sum(dim=1)
        fingertip_support = supported >= self.cfg.minimum_supported_fingertips
        if self.cfg.thumb_opposition_required:
            fingertip_support &= supported_mask[:, 0] & supported_mask[:, 1:].any(dim=1)
        palm_force = self._physical_palm_force()
        palm_support = palm_force >= self.cfg.minimum_palm_support_force_n
        self._palm_contact_force_n.copy_(palm_force)
        self._palm_support.copy_(palm_support)
        if self.cfg.palm_support_requires_detach:
            palm_support = self._stem_detached & palm_support
        return fingertip_support | palm_support

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-credit post-detachment palm support without changing Physics-v1."""

        previous_hold_streak = self._hold_streak.clone()
        super()._get_dones()
        additional_excess = self._additional_excess_force_mask()
        if additional_excess.shape != self._excess_force.shape:
            raise RuntimeError("additional excess-force mask has the wrong shape")
        self._excess_force |= additional_excess
        self._terminal_failure |= additional_excess

        secure_support = self._retention_support_mask()
        self._shadow_secure_support.copy_(secure_support)
        tip_center, tip_velocity = self._retention_reference()
        fruit_position = self._tomato.data.root_pos_w.torch
        fruit_velocity = self._tomato.data.root_lin_vel_w.torch
        distance = torch.linalg.vector_norm(fruit_position - tip_center, dim=1)
        relative_speed = torch.linalg.vector_norm(fruit_velocity - tip_velocity, dim=1)
        kinematic_retained = (
            self._stem_detached
            & (distance <= self.cfg.maximum_retained_distance_m)
            & (relative_speed <= self.cfg.maximum_retained_relative_speed_m_s)
            & ~self._dropped
            & ~self._excess_force
        )
        self._retained = kinematic_retained & secure_support
        self._hold_streak = torch.where(
            self._retained,
            previous_hold_streak + 1,
            torch.zeros_like(previous_hold_streak),
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
        self.extras["step"].update(self._retention_support_diagnostics())
        self.extras["step"]["success_count"] = int(self._success.sum().item())
        self.extras["step"]["retained_count"] = int(self._retained.sum().item())
        self.extras["step"]["timeout_count"] = int(self._timed_out.sum().item())
        self.extras["step"]["shadow_secure_support_count"] = int(
            self._shadow_secure_support.sum().item()
        )
        return terminated, truncated

    def _additional_excess_force_mask(self) -> torch.Tensor:
        return self._physical_palm_force() > self.cfg.palm_support_force_ceiling_n

    def _retention_support_diagnostics(self) -> dict[str, int | float | bool]:
        return {
            "physics_v2_palm_support_enabled": True,
            "physics_v2_palm_support_count": int(self._palm_support.sum().item()),
            "physics_v2_max_palm_force_n": float(self._palm_contact_force_n.max().item()),
            "physics_v2_invalid_palm_contact_force_n": float(
                self._invalid_palm_contact_force_n.max().item()
            ),
        }

    def _apply_attachment_wrench(self) -> None:
        """Apply the v1 attachment dynamics with four-mode v2 failure load."""

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
        components = stem_load_components_v2(
            force,
            torque,
            pull_axis,
            self._stem_tensile_limit_n,
            self._stem_shear_limit_n,
            self._stem_bending_limit_nm,
            self._stem_torsional_limit_nm,
        )
        for name, values in components.items():
            self._stem_load_components_v2[name].copy_(values)
        load_ratio = components["combined"]
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
            tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
            tip_center = tips.mean(dim=1)
            tip_velocity = self._robot.data.body_lin_vel_w.torch[
                :, self._fingertip_contact_body_ids
            ].mean(dim=1)
            distance = torch.linalg.vector_norm(fruit_pose[:, :3] - tip_center, dim=1)
            relative_speed = torch.linalg.vector_norm(fruit_lin_vel - tip_velocity, dim=1)
            self._break_substep_supported_fingertips[break_now] = supported[break_now]
            self._break_substep_thumb_support[break_now] = supported_mask[break_now, 0]
            self._break_substep_finger_support[break_now] = supported_mask[
                break_now, 1:
            ].any(dim=1)
            substep_secure = supported >= self.cfg.minimum_supported_fingertips
            if self.cfg.thumb_opposition_required:
                substep_secure &= supported_mask[:, 0] & supported_mask[:, 1:].any(dim=1)
            self._break_substep_secure_support[break_now] = substep_secure[break_now]
            self._break_substep_kinematic_retained[break_now] = (
                (distance[break_now] <= self.cfg.maximum_retained_distance_m)
                & (relative_speed[break_now] <= self.cfg.maximum_retained_relative_speed_m_s)
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
                env_ids, attachment_pos_w[env_ids], self._stem_plant_anchor_pos_w[env_ids]
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

    def _apply_action(self) -> None:
        super()._apply_action()
        force = self._physical_finger_force()
        stiffness = self._fruit_effective_stiffness_n_m.unsqueeze(1).clamp_min(1.0)
        indentation = force / stiffness
        self._fruit_estimated_indentation_m.copy_(indentation)
        self._fruit_damage_state = update_fruit_damage_diagnostics(
            self._fruit_damage_state,
            force,
            indentation,
            dt_s=float(self.cfg.sim.dt),
            config=self._fruit_damage_cfg,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        score = self._fruit_damage_state["damage_score"]
        self.extras["step"]["physics_v2_max_damage_risk"] = float(score.max())
        self.extras["step"]["physics_v2_mean_damage_risk"] = float(score.mean())
        self.extras["step"]["physics_v2_max_estimated_indentation_m"] = float(
            self._fruit_estimated_indentation_m.max()
        )
        self.extras["step"]["physics_v2_max_force_exposure_n_s"] = float(
            self._fruit_damage_state["force_exposure_n_s"].max()
        )
        for name, values in self._stem_load_components_v2.items():
            self.extras["step"][f"physics_v2_max_stem_{name}_ratio"] = float(
                values.max()
            )
        if "dropout_mask" in self._tactile_uncertainty_state:
            self.extras["step"]["physics_v2_tactile_dropout_count"] = int(
                self._tactile_uncertainty_state["dropout_mask"].sum()
            )
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "_fruit_damage_state"):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._fruit_estimated_indentation_m[env_ids] = 0.0
        self._palm_contact_force_n[env_ids] = 0.0
        self._palm_support[env_ids] = False
        self._invalid_palm_contact_force_n[env_ids] = 0.0
        for values in self._fruit_damage_state.values():
            values[env_ids] = 0.0
        for values in self._stem_load_components_v2.values():
            values[env_ids] = 0.0
        sampled = sample_uncertainty_profiles_v2(len(env_ids), device=self.device)
        for name, values in self._uncertainty_profiles.items():
            values[env_ids] = sampled[name]
        for name, values in self._tactile_uncertainty_state.items():
            if name == "history_n":
                values[:, env_ids] = 0.0
            else:
                values[env_ids] = 0.0


__all__ = ["TomatoPickPhysicsV2Env"]
