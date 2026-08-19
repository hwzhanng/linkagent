"""Physics-v2 environment with multi-mode public-prior abscission."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import (
    compute_pose_error,
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
)

from simulation.tomato_pick.reward import break_grasp_quality
from simulation.tomato_pick.isaac.physics_v1.env import TomatoPickPhysicsV1Env
from simulation.tomato_pick.physics_v1 import (
    secure_support_credit_gate,
    stem_load_damage_progress_score,
)
from simulation.tomato_pick.physics_v2 import (
    combined_abscission_load_ratio,
    discovery_capacity_scale,
    issued_command_history_observation,
    kinematic_retention_margin_score,
    load_bearing_opposition_score,
    load_bearing_opposition_target_n,
    normalized_controller_tracking_error,
    retention_ready_conditioned_capacity_scale,
    relative_position_history_observation,
    update_multimode_abscission,
    v2_outcome_alignment_alpha_for_iteration,
)

from .env_cfg import (
    PHYSICS_V2_CRITIC_STATE_DIM,
    PHYSICS_V2_OBSERVATION_DIM,
    TomatoPickPhysicsV2EnvCfg,
)


def _quat_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=1)
    rw, rx, ry, rz = right.unbind(dim=1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=1,
    )


class TomatoPickPhysicsV2Env(TomatoPickPhysicsV1Env):
    """V1 robot/control baseline with isolated V2 crop-failure semantics."""

    cfg: TomatoPickPhysicsV2EnvCfg

    def __init__(
        self,
        cfg: TomatoPickPhysicsV2EnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        if not (
            0.0 < cfg.discovery_tensile_shear_initial_scale <= 1.0
            and 0.0 < cfg.discovery_bending_torsion_initial_scale <= 1.0
        ):
            raise ValueError("discovery capacity scales must lie in (0, 1]")
        if not (
            0.0
            <= cfg.discovery_capacity_hold_initial_scale_iterations
            < cfg.discovery_capacity_full_iteration
        ):
            raise ValueError(
                "discovery capacity hold must precede full capacity"
            )
        if cfg.discovery_capacity_full_iteration <= 0.0:
            raise ValueError("discovery full-capacity iteration must be positive")
        if (
            cfg.discovery_bending_torsion_requires_secure_support
            is not True
        ):
            raise ValueError(
                "bending/torsion discovery assistance must require secure support"
            )
        if (
            cfg.discovery_bending_torsion_requires_kinematic_retention
            is not True
        ):
            raise ValueError(
                "bending/torsion discovery assistance must require current "
                "kinematic retention"
            )
        super().__init__(cfg, render_mode, **kwargs)
        # Episode-level audit latches.  Terminal-only retained logging can miss
        # a valid transient retention chain that is later lost; these metrics
        # expose that distinction without changing reward or termination.
        self._ever_retained = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._maximum_hold_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        required_action_history = int(max(cfg.action_delay_steps)) + 1
        if self._action_history.shape[0] < required_action_history:
            self._action_history = torch.zeros(
                (
                    required_action_history,
                    self.num_envs,
                    self._actions.shape[1],
                ),
                device=self.device,
                dtype=self._actions.dtype,
            )
        profiles = self._prestartup_physics_v2_profiles
        actual_pair_static_friction = self._prestartup_contact_material_profiles[
            "effective_pair_static_friction"
        ].to(self.device)
        self._load_bearing_opposition_target_n = (
            load_bearing_opposition_target_n(
                self._fruit_mass_kg,
                actual_pair_static_friction,
                full_contact_force_n=self.cfg.grasp_full_contact_force_n,
                safety_factor=self.cfg.load_bearing_opposition_safety_factor,
            )
        )
        self._stem_torsion_limit_nm = profiles["stem_torsion_moment_nm"].to(
            self.device
        )
        self._stem_subcritical_onset_ratio = profiles[
            "stem_subcritical_onset_ratio"
        ].to(self.device)
        self._stem_subcritical_damage_exponent = profiles[
            "stem_subcritical_damage_exponent"
        ].to(self.device)
        self._stem_critical_duration_s = profiles["stem_critical_duration_s"].to(
            self.device
        )
        self._stem_integrity = torch.ones(self.num_envs, device=self.device)
        self._stem_cumulative_twist_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._stem_effective_twist_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._stem_duration_load_ratio = torch.zeros(
            self.num_envs, device=self.device
        )
        self._stem_torsion_load_ratio = torch.zeros(
            self.num_envs, device=self.device
        )
        # Training assistance may lower the fracture threshold, but the
        # best-ever load-progress reward must retain one fixed normalization.
        # Otherwise a 10%-capacity fracture exhausts the unit load score and
        # leaves no incremental signal while capacity returns to the public
        # prior.  This buffer is reward-only; fracture uses the scheduled
        # capacities in _simulate_stem_substep.
        self._stem_public_combined_load_ratio = torch.zeros(
            self.num_envs, device=self.device
        )
        self._stem_load_progress_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stem_load_progress_retention_margin = torch.zeros(
            self.num_envs, device=self.device
        )
        self._task_orientation_category = profiles[
            "world_orientation_category"
        ].to(device=self.device, dtype=torch.long)
        self._task_orientation_azimuth_rad = profiles[
            "orientation_azimuth_rad"
        ].to(self.device)
        self._task_orientation_polar_tilt_rad = profiles[
            "orientation_polar_tilt_rad"
        ].to(self.device)
        self._break_tension_n = torch.zeros(self.num_envs, device=self.device)
        self._break_shear_n = torch.zeros(self.num_envs, device=self.device)
        self._break_bending_ratio = torch.zeros(
            self.num_envs, device=self.device
        )
        self._break_torsion_ratio = torch.zeros(
            self.num_envs, device=self.device
        )
        self._break_integrity = torch.ones(self.num_envs, device=self.device)
        self._break_twist_rad = torch.zeros(self.num_envs, device=self.device)
        self._break_effective_twist_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._break_thumb_force_n = torch.zeros(
            self.num_envs, device=self.device
        )
        self._break_other_force_n = torch.zeros(
            self.num_envs, device=self.device
        )
        self._break_load_bearing_opposition_score = torch.zeros(
            self.num_envs, device=self.device
        )
        # Fracture happens inside a 120 Hz substep, but reward is evaluated at
        # the later 20 Hz policy boundary. Preserve the exact-substep grasp
        # quality so the remainder of an already-issued action cannot erase a
        # valid controlled-break event before it receives credit.
        self._break_substep_grasp_quality = torch.zeros(
            self.num_envs, device=self.device
        )
        self._break_twist_assisted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._break_bend_assisted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._break_moderate_pull = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._break_paper_strategy_candidate = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._break_bending_torsion_curriculum_assisted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._current_discovery_capacity_scale = 1.0
        self._current_tensile_shear_capacity_scale = 1.0
        self._current_bending_torsion_capacity_scale = 1.0
        self._previous_perceived_relative_position = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._previous_perceived_relative_position_reset_pending = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_perceived_tip_center_relative_position = torch.zeros(
            (self.num_envs, 3), device=self.device
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Preserve every V2 action-delay slot before the inherited shift.

        The V1 baseline has a two-slot history because its delay support is
        limited to zero or one policy step.  V2 deliberately samples zero to
        two steps, so shift the additional tail first and let the inherited
        implementation retain its validated slots 0 and 1 behavior.
        """

        if self._action_history.shape[0] > 2:
            self._action_history[2:].copy_(
                self._action_history[1:-1].clone()
            )
        super()._pre_physics_step(actions)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Expose deployable motion and delayed-command history to V2 only.

        The inherited observation contains the current noisy fruit position
        relative to the palm but no fruit-motion signal. The explicit
        current/previous deployable pairs were retained because completed gates
        improved support and speed before recurrent phase memory was introduced.
        Current command-vs-actual end-effector tracking error provides
        deployable proprioceptive load information that cannot be reconstructed
        from the finite action queue alone. V2 also samples a two-step actuator delay,
        so append the two newest issued commands that may still be queued
        beyond the inherited previously-applied command. Exact simulator fruit
        velocity and sampled delay remain hidden, and the frozen 66D Clean-v5/
        V1 interface is unchanged.
        """

        observations = super()._get_observations()
        base_policy = observations["policy"]
        current_relative_position = base_policy[:, :3]
        previous_relative_position = torch.where(
            self._previous_perceived_relative_position_reset_pending.unsqueeze(1),
            current_relative_position,
            self._previous_perceived_relative_position,
        )
        fruit_pose = self._tomato.data.root_pose_w.torch
        ee_pose = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        fingertip_center = self._robot.data.body_pos_w.torch[
            :, self._fingertip_contact_body_ids
        ].mean(dim=1)
        perceived_fruit = (
            fruit_pose[:, :3] + self._perceived_fruit_position_error_w
        )
        tip_center_relative_position = (
            quat_apply_inverse(
                ee_pose[:, 3:7], perceived_fruit - fingertip_center
            )
            / 0.15
        ).clamp(-1.0, 1.0)
        tip_center_position_history = relative_position_history_observation(
            tip_center_relative_position,
            self._previous_perceived_tip_center_relative_position,
            self._previous_perceived_relative_position_reset_pending,
        )
        position_error_w, rotation_error_w = compute_pose_error(
            ee_pose[:, :3],
            ee_pose[:, 3:7],
            self._ee_command_pose_w[:, :3],
            self._ee_command_pose_w[:, 3:7],
        )
        controller_tracking_error = normalized_controller_tracking_error(
            position_error_w,
            rotation_error_w,
            self._grasp_basis_w,
            torch.tensor(
                self.cfg.action_scales,
                device=self.device,
                dtype=position_error_w.dtype,
            ),
        )
        issued_command_history = issued_command_history_observation(
            self._action_history,
            history_steps=2,
        )
        policy = torch.cat(
            (
                base_policy,
                previous_relative_position,
                tip_center_position_history,
                controller_tracking_error,
                issued_command_history,
            ),
            dim=1,
        )
        privileged = observations["critic"][:, base_policy.shape[1] :]
        critic = torch.cat((policy, privileged), dim=1)
        self._previous_perceived_relative_position.copy_(
            current_relative_position
        )
        self._previous_perceived_tip_center_relative_position.copy_(
            tip_center_relative_position
        )
        self._previous_perceived_relative_position_reset_pending.fill_(False)
        if policy.shape[1] != PHYSICS_V2_OBSERVATION_DIM:
            raise RuntimeError(
                f"Physics-v2 actor observation drifted to {policy.shape[1]}D"
            )
        if critic.shape[1] != PHYSICS_V2_CRITIC_STATE_DIM:
            raise RuntimeError(
                f"Physics-v2 critic observation drifted to {critic.shape[1]}D"
            )
        return {"policy": policy, "critic": critic}

    def _stem_progress_reward_score(self, opposed: torch.Tensor) -> torch.Tensor:
        """Reward physical load plus either twist or load-duration progress."""

        public_progress = torch.maximum(
            self._stem_damage.clamp(0.0, 1.0),
            (1.0 - self._stem_integrity).clamp(0.0, 1.0),
        )
        load_progress = stem_load_damage_progress_score(
            self._stem_public_combined_load_ratio,
            public_progress,
            opposed,
        )
        # Repairs 30-31 proved that a strict boolean envelope alone can align
        # low-capacity fractures yet still leave support, distance, and speed
        # margins temporally disjoint as public capacity returns.  Modulate the
        # same load-progress credit by the centered-and-slow margin *inside*
        # that unchanged strict envelope.  This is zero at either boundary and
        # cannot create weak credit outside it.  Evaluation/non-curriculum V2
        # behavior keeps the released secure-support gate and unit margin.
        if self.cfg.discovery_capacity_curriculum_enabled:
            load_progress = (
                load_progress * self._stem_load_progress_retention_margin
            )
            credit_ready = self._stem_load_progress_ready
        else:
            credit_ready = self._shadow_secure_support
        return secure_support_credit_gate(load_progress, credit_ready)

    def _opposed_reward_score(self) -> torch.Tensor:
        """Normalize opposition by the fruit's actual static load target.

        The reward weight and best-ever topology remain inherited.  Only the
        saturation point is corrected from contact presence to the grip force
        required by this environment's mass and quantized pair friction.
        """

        return load_bearing_opposition_score(
            self._last_physical_force_n,
            self._load_bearing_opposition_target_n,
            full_contact_force_n=self.cfg.grasp_full_contact_force_n,
        )

    def _break_grasped_reward_score(
        self, fruit_in_hand: torch.Tensor, break_quality: torch.Tensor
    ) -> torch.Tensor:
        """Credit the exact 120 Hz controlled-break state once.

        The inherited arguments are sampled at the later 20 Hz reward
        boundary. Using them discarded a valid break whenever the fruit moved
        during the remaining substeps of the already-issued action. The
        latched replacement is stricter: exact-substep grasp quality,
        thumb-plus-other secure support, and the unchanged distance and speed
        predicates must all hold. Only temporal credit changes; mechanics,
        reward weight, termination, retention, and success are unchanged.
        """

        del fruit_in_hand, break_quality
        return (
            self._stem_break_event.float()
            * self._break_substep_grasp_quality
            * self._break_substep_secure_support.float()
            * self._break_substep_kinematic_retained.float()
        )

    def _reward_alignment_alpha(self, learning_iteration: float) -> float:
        """Use V2's bounded gate-scale objective schedule."""

        return v2_outcome_alignment_alpha_for_iteration(learning_iteration)

    def _break_grasped_reward_scale(self, alignment_alpha: float) -> float:
        """Keep the strict one-shot joint-ready fracture bridge available.

        Clean-v5 fades its coarse fruit-in-hand break credit because that event
        can be exploited without a controlled release. V2's replacement is
        already stricter than retained-at-break: it requires exact-substep
        grasp quality, thumb-plus-other secure support, and the unchanged
        distance and speed predicates. Repair45 showed that fading this signal
        to zero also drove exact joint-ready fractures to zero while failure
        penalties successfully reduced drop. Preserve the same 40-point
        one-shot term while keeping the aligned failure and stable-detach
        channels unchanged.
        """

        del alignment_alpha
        return 1.0

    def _time_cost_alignment_scale(self, alignment_alpha: float) -> float:
        """Keep gate-scale V2 returns conditioned on physical outcomes.

        Repair45's terminal failure terms reduced public-tail drop, but the
        simultaneously strengthened per-step cost contributed about -14.7 per
        episode, raised tail value loss from 4.67 to 25.82, and erased the
        transient-retention chain. Repair46's strict break bridge recovered
        only one joint-ready tail iteration while value loss stayed at 26.49.
        Preserve the discovery endpoint's small anti-stall cost and keep the
        proven terminal failure, stable-detach, success, and strict-break
        channels unchanged. This is a component schedule correction, not a
        new reward or a relaxed outcome predicate.
        """

        del alignment_alpha
        return 0.0

    def _apply_public_orientation(self, env_ids: torch.Tensor) -> None:
        """Map S09 world categories to bounded stem-tilt quadrants.

        The category changes world geometry only.  The inherited policy input
        remains the local grasp basis/pull axis, so greenhouse-side prevalence
        cannot become an actor shortcut.
        """

        if not hasattr(self, "_task_orientation_category"):
            return
        count = len(env_ids)
        tilt = self._task_orientation_polar_tilt_rad[env_ids]
        azimuth = self._task_orientation_azimuth_rad[env_ids]
        yaw = -math.pi + 2.0 * math.pi * torch.rand(count, device=self.device)
        half_tilt = 0.5 * tilt
        tilt_quaternion = torch.stack(
            (
                torch.cos(half_tilt),
                torch.sin(half_tilt) * torch.cos(azimuth),
                torch.sin(half_tilt) * torch.sin(azimuth),
                torch.zeros_like(tilt),
            ),
            dim=1,
        )
        half_yaw = 0.5 * yaw
        yaw_quaternion = torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(half_yaw),
            ),
            dim=1,
        )
        quaternion = torch.nn.functional.normalize(
            _quat_multiply_wxyz(tilt_quaternion, yaw_quaternion), dim=1
        )
        fruit_pose = self._tomato.data.root_pose_w.torch[env_ids].clone()
        fruit_pose[:, 3:7] = quaternion
        self._tomato.write_root_pose_to_sim_index(
            root_pose=fruit_pose, env_ids=env_ids
        )

        up = quat_apply(
            quaternion,
            torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(count, -1),
        )
        pull_axis = -up
        ee_pose = self._robot.data.body_pose_w.torch[
            env_ids, self._ee_body_id
        ].clone()
        tips = self._robot.data.body_pos_w.torch[
            env_ids[:, None], self._fingertip_frame_body_ids
        ]
        approach = tips[:, 1:5].mean(dim=1) - ee_pose[:, :3]
        approach -= (approach * pull_axis).sum(dim=1, keepdim=True) * pull_axis
        fallback = quat_apply(
            ee_pose[:, 3:7],
            torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(count, -1),
        )
        fallback -= (fallback * pull_axis).sum(dim=1, keepdim=True) * pull_axis
        approach = torch.where(
            torch.linalg.vector_norm(approach, dim=1, keepdim=True) > 1.0e-5,
            approach,
            fallback,
        )
        approach = torch.nn.functional.normalize(approach, dim=1)
        lateral = torch.nn.functional.normalize(
            torch.linalg.cross(pull_axis, approach, dim=1), dim=1
        )
        self._grasp_basis_w[env_ids] = torch.stack(
            (approach, lateral, pull_axis), dim=2
        )
        local_attachment = torch.zeros((count, 3), device=self.device)
        local_attachment[:, 2] = self._fruit_axes_m[env_ids, 2]
        attachment = fruit_pose[:, :3] + quat_apply(quaternion, local_attachment)
        plant_anchor = attachment - pull_axis * self.cfg.stem_visual_height_m
        self._stem_rest_attachment_pos_w[env_ids] = attachment
        self._stem_plant_anchor_pos_w[env_ids] = plant_anchor
        self._stem_anchor_quat_w[env_ids] = quaternion
        self._move_stem_visual(env_ids, attachment, plant_anchor)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

    def _apply_workspace_reset(self, env_ids: torch.Tensor) -> None:
        super()._apply_workspace_reset(env_ids)
        self._apply_public_orientation(env_ids)

    def _apply_attachment_wrench(self) -> None:
        fruit_pose = self._tomato.data.root_pose_w.torch
        fruit_lin_vel = self._tomato.data.root_lin_vel_w.torch
        fruit_ang_vel = self._tomato.data.root_ang_vel_w.torch
        attachment_offset_w = quat_apply(
            fruit_pose[:, 3:7], self._stem_local_attachment
        )
        attachment_pos_w = fruit_pose[:, :3] + attachment_offset_w
        attachment_velocity_w = fruit_lin_vel + torch.linalg.cross(
            fruit_ang_vel, attachment_offset_w, dim=1
        )
        position_error = self._stem_rest_attachment_pos_w - attachment_pos_w
        rotation_error = quat_box_minus(
            self._stem_anchor_quat_w, fruit_pose[:, 3:7]
        )
        omega = self.cfg.stem_linear_natural_frequency_rad_s
        acceleration = (
            omega * omega * position_error
            - 2.0
            * self.cfg.stem_linear_damping_ratio
            * omega
            * attachment_velocity_w
            - self._gravity_w.unsqueeze(0)
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
        pull_axis = torch.nn.functional.normalize(
            self._grasp_basis_w[:, :, 2], dim=1
        )
        learning_iteration = (
            float(self.cfg.reward_curriculum_iteration_offset)
            + float(self.common_step_counter) / float(self.cfg.reward_rollout_steps)
        )
        physical_force: torch.Tensor | None = None
        supported_mask: torch.Tensor | None = None
        substep_distance: torch.Tensor | None = None
        substep_relative_speed: torch.Tensor | None = None
        if self.cfg.discovery_capacity_curriculum_enabled:
            tensile_shear_scale = discovery_capacity_scale(
                learning_iteration,
                initial_scale=(
                    self.cfg.discovery_tensile_shear_initial_scale
                ),
                hold_initial_scale_iterations=(
                    self.cfg.discovery_capacity_hold_initial_scale_iterations
                ),
                full_capacity_iteration=(
                    self.cfg.discovery_capacity_full_iteration
                ),
            )
            bending_torsion_schedule_scale = discovery_capacity_scale(
                learning_iteration,
                initial_scale=(
                    self.cfg.discovery_bending_torsion_initial_scale
                ),
                hold_initial_scale_iterations=(
                    self.cfg.discovery_capacity_hold_initial_scale_iterations
                ),
                full_capacity_iteration=(
                    self.cfg.discovery_capacity_full_iteration
                ),
            )
            # Use the current 120 Hz contact state, not the previous 50 ms
            # policy boundary, so assistance cannot survive loss of support.
            physical_force = self._physical_finger_force()
            supported_mask = physical_force >= self.cfg.minimum_support_force_n
            curriculum_secure_support = (
                supported_mask.sum(dim=1)
                >= self.cfg.minimum_supported_fingertips
            )
            if self.cfg.thumb_opposition_required:
                curriculum_secure_support &= (
                    supported_mask[:, 0]
                    & supported_mask[:, 1:].any(dim=1)
                )
            tips = self._robot.data.body_pos_w.torch[
                :, self._fingertip_contact_body_ids
            ]
            tip_center = tips.mean(dim=1)
            tip_velocity = self._robot.data.body_lin_vel_w.torch[
                :, self._fingertip_contact_body_ids
            ].mean(dim=1)
            substep_distance = torch.linalg.vector_norm(
                fruit_pose[:, :3] - tip_center, dim=1
            )
            substep_relative_speed = torch.linalg.vector_norm(
                fruit_lin_vel - tip_velocity, dim=1
            )
            curriculum_kinematic_retention = (
                substep_distance <= self.cfg.maximum_retained_distance_m
            ) & (
                substep_relative_speed
                <= self.cfg.maximum_retained_relative_speed_m_s
            )
            curriculum_retention_ready = (
                curriculum_secure_support & curriculum_kinematic_retention
            )
            self._stem_load_progress_ready.copy_(
                curriculum_retention_ready
            )
            self._stem_load_progress_retention_margin.copy_(
                kinematic_retention_margin_score(
                    substep_distance,
                    substep_relative_speed,
                    maximum_distance_m=self.cfg.maximum_retained_distance_m,
                    maximum_relative_speed_m_s=(
                        self.cfg.maximum_retained_relative_speed_m_s
                    ),
                )
            )
            bending_torsion_scale = retention_ready_conditioned_capacity_scale(
                bending_torsion_schedule_scale, curriculum_retention_ready
            )
            bending_torsion_assistance_active = (
                curriculum_retention_ready
                & (bending_torsion_schedule_scale < 1.0)
            )
        else:
            tensile_shear_scale = 1.0
            bending_torsion_schedule_scale = 1.0
            bending_torsion_scale = torch.ones(
                self.num_envs, device=self.device
            )
            bending_torsion_assistance_active = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._stem_load_progress_ready.copy_(
                self._shadow_secure_support
            )
            self._stem_load_progress_retention_margin.fill_(1.0)
        self._current_tensile_shear_capacity_scale = tensile_shear_scale
        self._current_bending_torsion_capacity_scale = (
            bending_torsion_schedule_scale
        )
        self._current_discovery_capacity_scale = min(
            tensile_shear_scale,
            bending_torsion_schedule_scale,
        )
        tensile_limit_n = self._stem_tensile_limit_n * tensile_shear_scale
        shear_limit_n = self._stem_shear_limit_n * tensile_shear_scale
        bending_limit_nm = self._stem_bending_limit_nm * bending_torsion_scale
        torsion_limit_nm = self._stem_torsion_limit_nm * bending_torsion_scale
        # Elastoplastic torsional yielding: cap the restoring torque at the
        # current integrity-scaled torsional capacity.  Rotation beyond this
        # point drives integrity loss instead of an unbounded spring impulse.
        signed_torsion = (torque * pull_axis).sum(dim=1)
        current_torsion_capacity = torsion_limit_nm * (
            self._stem_integrity.clamp_min(self.cfg.stem_capacity_floor_fraction)
        )
        capped_torsion = torch.maximum(
            -current_torsion_capacity,
            torch.minimum(current_torsion_capacity, signed_torsion),
        )
        torque += (capped_torsion - signed_torsion).unsqueeze(1) * pull_axis
        self._stem_public_combined_load_ratio.copy_(
            combined_abscission_load_ratio(
                force,
                torque,
                pull_axis,
                self._stem_tensile_limit_n,
                self._stem_shear_limit_n,
                self._stem_bending_limit_nm,
                self._stem_torsion_limit_nm,
                self._stem_integrity,
                capacity_floor_fraction=self.cfg.stem_capacity_floor_fraction,
            )
        )
        twist_increment = (
            (fruit_ang_vel * pull_axis).sum(dim=1).abs() * float(self.cfg.sim.dt)
        )
        torsion_utilization = (
            capped_torsion.abs() / current_torsion_capacity.clamp_min(1.0e-8)
        ).clamp(0.0, 1.0)
        effective_twist_increment = twist_increment * torch.pow(
            torsion_utilization, self.cfg.stem_twist_integrity_q
        )
        state = update_multimode_abscission(
            self._stem_integrity,
            self._stem_damage,
            force,
            torque,
            pull_axis,
            tensile_limit_n,
            shear_limit_n,
            bending_limit_nm,
            torsion_limit_nm,
            twist_increment,
            dt_s=float(self.cfg.sim.dt),
            attached=attached,
            k_theta_per_rad=self.cfg.stem_twist_integrity_k_per_rad,
            twist_exponent_q=self.cfg.stem_twist_integrity_q,
            subcritical_onset_ratio=self._stem_subcritical_onset_ratio,
            subcritical_exponent=self._stem_subcritical_damage_exponent,
            subcritical_time_constant_s=self._stem_critical_duration_s,
            instantaneous_failure_ratio=(
                self.cfg.stem_instantaneous_failure_ratio
            ),
            torsion_detach_integrity_threshold=(
                self.cfg.stem_torsion_detach_integrity_threshold
            ),
            capacity_floor_fraction=self.cfg.stem_capacity_floor_fraction,
        )
        self._stem_integrity.copy_(state["integrity"])
        self._stem_damage.copy_(state["load_damage"])
        self._stem_combined_load_ratio.copy_(state["combined_load_ratio"])
        self._stem_duration_load_ratio.copy_(state["duration_load_ratio"])
        self._stem_torsion_load_ratio.copy_(state["torsion_ratio"])
        self._stem_cumulative_twist_rad += twist_increment * attached
        self._stem_effective_twist_rad += effective_twist_increment * attached
        break_now = state["broken"]

        if bool(break_now.any()):
            capacity_scale = self._stem_integrity.clamp_min(
                self.cfg.stem_capacity_floor_fraction
            )
            self._break_tension_n[break_now] = (
                state["tension_ratio"]
                * tensile_limit_n
                * capacity_scale
            )[break_now]
            self._break_shear_n[break_now] = (
                state["shear_ratio"]
                * shear_limit_n
                * capacity_scale
            )[break_now]
            self._break_bending_ratio[break_now] = state["bending_ratio"][
                break_now
            ]
            self._break_torsion_ratio[break_now] = state["torsion_ratio"][
                break_now
            ]
            self._break_integrity[break_now] = self._stem_integrity[break_now]
            self._break_twist_rad[break_now] = self._stem_cumulative_twist_rad[
                break_now
            ]
            self._break_effective_twist_rad[break_now] = (
                self._stem_effective_twist_rad[break_now]
            )
            twist_assisted = (
                self._stem_effective_twist_rad
                >= self.cfg.paper_strategy_effective_twist_minimum_rad
            )
            bend_assisted = (
                state["bending_ratio"]
                >= self.cfg.paper_strategy_bending_utilization_minimum
            )
            pull_minimum, pull_maximum = (
                self.cfg.paper_strategy_final_axial_pull_n
            )
            moderate_pull = (
                (self._break_tension_n >= pull_minimum)
                & (self._break_tension_n <= pull_maximum)
            )
            self._break_twist_assisted[break_now] = twist_assisted[break_now]
            self._break_bend_assisted[break_now] = bend_assisted[break_now]
            self._break_moderate_pull[break_now] = moderate_pull[break_now]
            self._break_bending_torsion_curriculum_assisted[break_now] = (
                bending_torsion_assistance_active[break_now]
            )
            if physical_force is None:
                physical_force = self._physical_finger_force()
            self._break_thumb_force_n[break_now] = physical_force[break_now, 0]
            self._break_other_force_n[break_now] = physical_force[
                break_now, 1:
            ].amax(dim=1)
            self._break_load_bearing_opposition_score[break_now] = (
                load_bearing_opposition_score(
                    physical_force,
                    self._load_bearing_opposition_target_n,
                    full_contact_force_n=self.cfg.grasp_full_contact_force_n,
                )[break_now]
            )
            self._break_substep_grasp_quality[break_now] = break_grasp_quality(
                physical_force,
                contact_force_n=self.cfg.grasp_full_contact_force_n,
            )[break_now]
            if supported_mask is None:
                supported_mask = (
                    physical_force >= self.cfg.minimum_support_force_n
                )
            supported = supported_mask.sum(dim=1)
            if substep_distance is None or substep_relative_speed is None:
                tips = self._robot.data.body_pos_w.torch[
                    :, self._fingertip_contact_body_ids
                ]
                tip_center = tips.mean(dim=1)
                tip_velocity = self._robot.data.body_lin_vel_w.torch[
                    :, self._fingertip_contact_body_ids
                ].mean(dim=1)
                substep_distance = torch.linalg.vector_norm(
                    fruit_pose[:, :3] - tip_center, dim=1
                )
                substep_relative_speed = torch.linalg.vector_norm(
                    fruit_lin_vel - tip_velocity, dim=1
                )
            self._break_substep_supported_fingertips[break_now] = supported[break_now]
            self._break_substep_thumb_support[break_now] = supported_mask[
                break_now, 0
            ]
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
                (
                    substep_distance[break_now]
                    <= self.cfg.maximum_retained_distance_m
                )
                & (
                    substep_relative_speed[break_now]
                    <= self.cfg.maximum_retained_relative_speed_m_s
                )
            )
            self._break_substep_distance_m[break_now] = substep_distance[break_now]
            self._break_substep_relative_speed_m_s[break_now] = (
                substep_relative_speed[break_now]
            )
            self._break_substep_damage[break_now] = state["load_damage"][break_now]
            self._break_substep_load_ratio[break_now] = state[
                "combined_load_ratio"
            ][break_now]
            paper_candidate = (
                (twist_assisted | bend_assisted)
                & moderate_pull
                & substep_secure
                & self._break_substep_kinematic_retained
            )
            self._break_paper_strategy_candidate[break_now] = paper_candidate[
                break_now
            ]
        self._stem_break_event |= break_now
        self._stem_detached |= break_now
        attached_after = attached & ~break_now
        force *= attached_after.unsqueeze(1)
        torque *= attached_after.unsqueeze(1)
        self._last_stem_force_w.copy_(force)
        self._last_stem_torque_w.copy_(torque)
        if bool(attached_after.any()):
            active_ids = attached_after.nonzero(as_tuple=False).squeeze(1)
            self._move_stem_visual(
                active_ids,
                attachment_pos_w[active_ids],
                self._stem_plant_anchor_pos_w[active_ids],
            )

        zeros = torch.zeros_like(force)
        self._tomato.permanent_wrench_composer.set_forces_and_torques_index(
            forces=zeros.unsqueeze(1), torques=zeros.unsqueeze(1), is_global=False
        )
        if bool(attached_after.any()):
            active_ids = attached_after.nonzero(as_tuple=False).squeeze(1)
            velocity = torch.cat((fruit_lin_vel, fruit_ang_vel), dim=1)[
                active_ids
            ].clone()
            velocity[:, :3] += (
                force[active_ids] / self._fruit_mass_kg[active_ids].unsqueeze(1)
            ) * float(self.cfg.sim.dt)
            velocity[:, 3:] += (
                torque[active_ids] / inertia[active_ids].unsqueeze(1)
            ) * float(self.cfg.sim.dt)
            self._tomato.write_root_velocity_to_sim_index(
                root_velocity=velocity, env_ids=active_ids
            )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        self._ever_retained |= self._retained
        self._maximum_hold_streak = torch.maximum(
            self._maximum_hold_streak, self._hold_streak
        )
        if self.cfg.detailed_step_diagnostics:
            self.extras["step"]["physics_v2_minimum_integrity"] = float(
                self._stem_integrity.min().item()
            )
            self.extras["step"]["physics_v2_maximum_twist_rad"] = float(
                self._stem_cumulative_twist_rad.max().item()
            )
            self.extras["step"]["physics_v2_maximum_effective_twist_rad"] = (
                float(self._stem_effective_twist_rad.max().item())
            )
            self.extras["step"]["physics_v2_maximum_torsion_ratio"] = float(
                self._stem_torsion_load_ratio.max().item()
            )
            category_count = torch.bincount(
                self._task_orientation_category, minlength=4
            )
            for index, name in enumerate(("front", "back", "left", "right")):
                self.extras["step"][
                    f"physics_v2_world_orientation_{name}_count"
                ] = int(category_count[index].item())
            break_event = self._stem_break_event
            self.extras["step"]["physics_v2_twist_assisted_break_count"] = int(
                (break_event & self._break_twist_assisted).sum().item()
            )
            self.extras["step"]["physics_v2_bend_assisted_break_count"] = int(
                (break_event & self._break_bend_assisted).sum().item()
            )
            self.extras["step"]["physics_v2_moderate_pull_break_count"] = int(
                (break_event & self._break_moderate_pull).sum().item()
            )
            self.extras["step"][
                "physics_v2_paper_strategy_candidate_count"
            ] = int(
                (break_event & self._break_paper_strategy_candidate)
                .sum()
                .item()
            )
        return terminated, truncated

    def _log_episodes(self, env_ids: torch.Tensor) -> None:
        """Add strategy evidence needed to audit the learned behavior."""

        detached = self._stem_detached[env_ids]
        detached_count = int(detached.sum().item())

        def detached_mean(values: torch.Tensor) -> float:
            if detached_count == 0:
                return 0.0
            return float(values[env_ids][detached].float().mean().item())

        super()._log_episodes(env_ids)
        if hasattr(self, "_ever_retained"):
            self.extras.setdefault("log", {}).update(
                {
                    "Metrics/ever_retained_rate": self._ever_retained[env_ids]
                    .float()
                    .mean()
                    .item(),
                    "Metrics/mean_max_hold_steps": self._maximum_hold_streak[
                        env_ids
                    ]
                    .float()
                    .mean()
                    .item(),
                }
            )
        self.extras.setdefault("log", {}).update(
            {
                "PhysicsV2/mean_break_tension_n": detached_mean(
                    self._break_tension_n
                ),
                "PhysicsV2/mean_break_shear_n": detached_mean(
                    self._break_shear_n
                ),
                "PhysicsV2/mean_break_bending_ratio": detached_mean(
                    self._break_bending_ratio
                ),
                "PhysicsV2/mean_break_torsion_ratio": detached_mean(
                    self._break_torsion_ratio
                ),
                "PhysicsV2/mean_break_integrity": detached_mean(
                    self._break_integrity
                ),
                "PhysicsV2/mean_break_twist_deg": math.degrees(
                    detached_mean(self._break_twist_rad)
                ),
                "PhysicsV2/mean_break_effective_twist_deg": math.degrees(
                    detached_mean(self._break_effective_twist_rad)
                ),
                "PhysicsV2/mean_break_thumb_force_n": detached_mean(
                    self._break_thumb_force_n
                ),
                "PhysicsV2/mean_break_other_force_n": detached_mean(
                    self._break_other_force_n
                ),
                "PhysicsV2/mean_break_opposition_target_n": detached_mean(
                    self._load_bearing_opposition_target_n
                ),
                "PhysicsV2/mean_break_load_bearing_opposition_score": (
                    detached_mean(self._break_load_bearing_opposition_score)
                ),
                "PhysicsV2/twist_assisted_break_rate": self._break_twist_assisted[
                    env_ids
                ]
                .float()
                .mean()
                .item(),
                "PhysicsV2/bend_assisted_break_rate": self._break_bend_assisted[
                    env_ids
                ]
                .float()
                .mean()
                .item(),
                "PhysicsV2/moderate_pull_break_rate": self._break_moderate_pull[
                    env_ids
                ]
                .float()
                .mean()
                .item(),
                "PhysicsV2/paper_strategy_candidate_rate": (
                    self._break_paper_strategy_candidate[env_ids]
                    .float()
                    .mean()
                    .item()
                ),
                "PhysicsV2/paper_strategy_success_rate": (
                    (
                        self._break_paper_strategy_candidate[env_ids]
                        & self._success[env_ids]
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "PhysicsV2/bending_torsion_curriculum_assisted_break_rate": (
                    detached_mean(
                        self._break_bending_torsion_curriculum_assisted
                    )
                ),
                "PhysicsV2/mean_break_distance_m": detached_mean(
                    self._break_substep_distance_m
                ),
                "PhysicsV2/mean_break_relative_speed_m_s": detached_mean(
                    self._break_substep_relative_speed_m_s
                ),
                "PhysicsV2/break_secure_support_rate": detached_mean(
                    self._break_substep_secure_support
                ),
                "PhysicsV2/break_distance_ok_rate": detached_mean(
                    self._break_substep_distance_m
                    <= self.cfg.maximum_retained_distance_m
                ),
                "PhysicsV2/break_speed_ok_rate": detached_mean(
                    self._break_substep_relative_speed_m_s
                    <= self.cfg.maximum_retained_relative_speed_m_s
                ),
                "PhysicsV2/break_kinematic_retained_rate": detached_mean(
                    self._break_substep_kinematic_retained
                ),
                "PhysicsV2/break_joint_ready_rate": detached_mean(
                    self._break_substep_secure_support
                    & self._break_substep_kinematic_retained
                ),
                "PhysicsV2/discovery_capacity_scale": (
                    self._current_discovery_capacity_scale
                ),
                "PhysicsV2/tensile_shear_capacity_scale": (
                    self._current_tensile_shear_capacity_scale
                ),
                "PhysicsV2/bending_torsion_capacity_scale": (
                    self._current_bending_torsion_capacity_scale
                ),
            }
        )

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        if hasattr(self, "_ever_retained"):
            self._ever_retained[env_ids] = False
            self._maximum_hold_streak[env_ids] = 0
        if hasattr(
            self, "_previous_perceived_relative_position_reset_pending"
        ):
            self._previous_perceived_relative_position_reset_pending[
                env_ids
            ] = True
        if not hasattr(self, "_stem_integrity"):
            return
        self._stem_integrity[env_ids] = 1.0
        self._stem_cumulative_twist_rad[env_ids] = 0.0
        self._stem_effective_twist_rad[env_ids] = 0.0
        self._stem_duration_load_ratio[env_ids] = 0.0
        self._stem_torsion_load_ratio[env_ids] = 0.0
        self._stem_public_combined_load_ratio[env_ids] = 0.0
        self._stem_load_progress_ready[env_ids] = False
        self._stem_load_progress_retention_margin[env_ids] = 0.0
        self._break_tension_n[env_ids] = 0.0
        self._break_shear_n[env_ids] = 0.0
        self._break_bending_ratio[env_ids] = 0.0
        self._break_torsion_ratio[env_ids] = 0.0
        self._break_integrity[env_ids] = 1.0
        self._break_twist_rad[env_ids] = 0.0
        self._break_effective_twist_rad[env_ids] = 0.0
        self._break_thumb_force_n[env_ids] = 0.0
        self._break_other_force_n[env_ids] = 0.0
        self._break_load_bearing_opposition_score[env_ids] = 0.0
        self._break_substep_grasp_quality[env_ids] = 0.0
        self._break_twist_assisted[env_ids] = False
        self._break_bend_assisted[env_ids] = False
        self._break_moderate_pull[env_ids] = False
        self._break_paper_strategy_candidate[env_ids] = False
        self._break_bending_torsion_curriculum_assisted[env_ids] = False


__all__ = ["TomatoPickPhysicsV2Env"]
