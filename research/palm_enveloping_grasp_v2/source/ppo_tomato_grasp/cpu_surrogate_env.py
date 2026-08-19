"""Vectorized CPU surrogate for interface and learning-pipeline validation.

This is not a replacement for Isaac/PhysX and its metrics are not physical
evidence.  It provides a deterministic, batched path through the frozen action
manifold, actuator uncertainty, contact proxy, tactile matrices, observations,
critic state, reward, termination, and auto-reset logic without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from simulation.ppo_tomato_grasp.domain_randomization import sample_hand_grasp_profiles
from simulation.ppo_tomato_grasp.hand_grasp_mdp import (
    FINGER_FORCE_CEILING_N,
    HandGraspState,
    hand_grasp_transition,
    initial_hand_grasp_state,
)
from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    LEFT_SAFE_VELOCITY_RAD_S,
    hybrid_action_to_joint_targets,
    joint_targets_to_digit_closure,
    synergy_to_joint_targets,
)
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
    pack_actor_observation,
    pack_critic_state,
)
from simulation.ppo_tomato_grasp.tactile_features import extract_tactile_features


SURROGATE_ENV_VERSION = "cpu_hand_grasp_surrogate_v3_urdf_velocity_capped"


@dataclass(frozen=True)
class CpuSurrogateStep:
    observation: torch.Tensor
    critic_state: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    success: torch.Tensor
    metrics: dict[str, torch.Tensor]


class CpuHandGraspEnv:
    """Batched hand-only surrogate that never imports Isaac or hardware code."""

    def __init__(
        self,
        num_envs: int,
        *,
        seed: int = 42,
        curriculum_fraction: float = 0.35,
        tail_focus_fraction: float = 0.0,
        episode_steps: int = 120,
        policy_dt_s: float = 0.05,
        auto_reset: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be an integer >= 1")
        if not 0.0 <= curriculum_fraction <= 1.0:
            raise ValueError("curriculum_fraction must lie in [0, 1]")
        if not 0.0 <= tail_focus_fraction <= 1.0:
            raise ValueError("tail_focus_fraction must lie in [0, 1]")
        if episode_steps < 2 or policy_dt_s <= 0.0:
            raise ValueError("episode_steps and policy_dt_s must be positive")
        self.num_envs = num_envs
        self.curriculum_fraction = float(curriculum_fraction)
        self.tail_focus_fraction = float(tail_focus_fraction)
        self.episode_steps = int(episode_steps)
        self.dt = float(policy_dt_s)
        self.auto_reset = bool(auto_reset)
        self.dtype = dtype
        self.device = torch.device(device)
        # Keep one CPU generator for deterministic cross-device episode resets;
        # randomized tensors are copied to the requested execution device.
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

        self.lower = torch.tensor(LEFT_SAFE_LOWER_RAD, dtype=dtype, device=self.device)
        self.upper = torch.tensor(LEFT_SAFE_UPPER_RAD, dtype=dtype, device=self.device)
        self.velocity = torch.tensor(LEFT_SAFE_VELOCITY_RAD_S, dtype=dtype, device=self.device)
        self.force_ceiling = torch.tensor(FINGER_FORCE_CEILING_N, dtype=dtype, device=self.device)
        self.finger_angles = torch.tensor(
            (-1.15, -0.52, -0.14, 0.27, 0.67), dtype=dtype, device=self.device
        )
        self.finger_direction = torch.stack(
            (torch.sin(self.finger_angles), torch.cos(self.finger_angles)), dim=1
        )
        self.maximum_tactile_delay = 3
        self.reset()

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.profiles = sample_hand_grasp_profiles(
            self.num_envs,
            curriculum_fraction=self.curriculum_fraction,
            tail_focus_fraction=self.tail_focus_fraction,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        self.object_offset = self.profiles["object_position_offset_m"].clone()
        self.previous_object_offset = self.object_offset.clone()
        self.object_velocity = torch.zeros_like(self.object_offset)
        self.action = torch.zeros((self.num_envs, ACTION_DIM), dtype=self.dtype, device=self.device)
        initial_action = torch.zeros((self.num_envs, 5), dtype=self.dtype, device=self.device)
        self.joint_position = synergy_to_joint_targets(initial_action)
        self.joint_velocity = torch.zeros_like(self.joint_position)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.mdp_state = initial_hand_grasp_state(torch.zeros(self.num_envs, dtype=self.dtype, device=self.device))
        self.physical_force = torch.zeros((self.num_envs, 5), dtype=self.dtype, device=self.device)
        self.palm_force = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        self.surface_gap = torch.full((self.num_envs, 5), 0.02, dtype=self.dtype, device=self.device)
        self.indentation = torch.zeros((self.num_envs, 5), dtype=self.dtype, device=self.device)
        self.contact_points = torch.zeros((self.num_envs, 5, 3), dtype=self.dtype, device=self.device)
        self.tactile_history = torch.zeros(
            (self.maximum_tactile_delay + 1, self.num_envs, 5, 12, 6), dtype=self.dtype, device=self.device
        )
        self.tactile_release = torch.zeros((self.num_envs, 5, 12, 6), dtype=self.dtype, device=self.device)
        self.tactile_measured = torch.zeros_like(self.tactile_release)
        self.tactile_drift = torch.zeros((self.num_envs, 5), dtype=self.dtype, device=self.device)
        self.tactile_age = torch.zeros((self.num_envs, 5), dtype=self.dtype, device=self.device)
        self.previous_tactile_total = torch.zeros((self.num_envs, 5), dtype=self.dtype, device=self.device)
        self.peak_indentation = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        return self._build_observations()

    def _replace_profiles(self, env_ids: torch.Tensor) -> None:
        count = int(env_ids.numel())
        sampled = sample_hand_grasp_profiles(
            count,
            curriculum_fraction=self.curriculum_fraction,
            tail_focus_fraction=self.tail_focus_fraction,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        for name, value in self.profiles.items():
            value[env_ids] = sampled[name]

    @staticmethod
    def _replace_mdp_state(
        state: HandGraspState,
        initial: HandGraspState,
        env_ids: torch.Tensor,
    ) -> HandGraspState:
        values = {}
        for item in fields(HandGraspState):
            tensor = getattr(state, item.name).clone()
            tensor[env_ids] = getattr(initial, item.name)
            values[item.name] = tensor
        return HandGraspState(**values)

    def _reset_indices(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._replace_profiles(env_ids)
        count = int(env_ids.numel())
        self.object_offset[env_ids] = self.profiles["object_position_offset_m"][env_ids]
        self.previous_object_offset[env_ids] = self.object_offset[env_ids]
        self.object_velocity[env_ids] = 0.0
        self.action[env_ids] = 0.0
        initial_action = torch.zeros((count, 5), dtype=self.dtype, device=self.device)
        self.joint_position[env_ids] = synergy_to_joint_targets(initial_action)
        self.joint_velocity[env_ids] = 0.0
        self.step_count[env_ids] = 0
        initial = initial_hand_grasp_state(torch.zeros(count, dtype=self.dtype, device=self.device))
        self.mdp_state = self._replace_mdp_state(self.mdp_state, initial, env_ids)
        self.physical_force[env_ids] = 0.0
        self.palm_force[env_ids] = 0.0
        self.surface_gap[env_ids] = 0.02
        self.indentation[env_ids] = 0.0
        self.contact_points[env_ids] = 0.0
        self.tactile_history[:, env_ids] = 0.0
        self.tactile_release[env_ids] = 0.0
        self.tactile_measured[env_ids] = 0.0
        self.tactile_drift[env_ids] = 0.0
        self.tactile_age[env_ids] = 0.0
        self.previous_tactile_total[env_ids] = 0.0
        self.peak_indentation[env_ids] = 0.0

    def _apply_actuator_model(self, action: torch.Tensor) -> None:
        desired = hybrid_action_to_joint_targets(action)
        error = desired - self.joint_position
        deadband = self.profiles["actuator_deadband_rad"]
        backlash = self.profiles["actuator_backlash_rad"]
        active = error.abs() > deadband
        transmitted = torch.sign(error) * (error.abs() - backlash).clamp_min(0.0)
        delta = (
            self.profiles["actuator_response_scale"] * transmitted
            + self.profiles["actuator_bias_rad"]
        )
        # Match the derived left-hand URDF's 1 rad/s per-joint velocity limit,
        # even when measured no-load motion is faster.
        maximum_delta = torch.minimum(0.5 * self.velocity, torch.ones_like(self.velocity)) * self.dt
        delta = delta.clamp(-maximum_delta, maximum_delta)
        next_position = torch.where(active, self.joint_position + delta, self.joint_position)
        next_position = torch.maximum(self.lower, torch.minimum(self.upper, next_position))
        self.joint_velocity = (next_position - self.joint_position) / self.dt
        self.joint_position = next_position

    def _contact_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        closure = joint_targets_to_digit_closure(self.joint_position)
        axes = self.profiles["axes_m"]
        mean_radius = axes[:, :2].mean(dim=1)
        projection = torch.einsum("nd,fd->nf", self.object_offset[:, :2], self.finger_direction)
        size_shift = (mean_radius - 0.0325).unsqueeze(1) / 0.025
        spread_pattern = torch.tensor(
            (-1.0, -0.45, 0.0, 0.45, 1.0), dtype=self.dtype, device=self.device
        )
        spread_shift = 0.025 * self.action[:, 3:4] * spread_pattern
        onset = (0.57 - size_shift - projection / 0.025 + spread_shift).clamp(0.20, 0.90)
        self.surface_gap = (onset - closure) * 0.007
        self.indentation = (-self.surface_gap).clamp(0.0, 0.006)
        closing_speed = self.joint_velocity.abs().mean(dim=1, keepdim=True) * 0.001
        effective_stiffness = self.profiles["contact_stiffness_n_m"].unsqueeze(1) * 0.10
        effective_damping = self.profiles["contact_damping_n_s_m"].unsqueeze(1) * 0.10
        force = effective_stiffness * self.indentation + effective_damping * closing_speed
        force = torch.where(self.indentation > 0.0, force, torch.zeros_like(force))
        self.physical_force = force.clamp(0.0, 12.0)

        palm_penetration = (
            (closure.mean(dim=1) - 0.80) * 0.004
            - self.object_offset[:, 1].abs()
            + (mean_radius - 0.0325).clamp_min(0.0) * 0.15
        ).clamp(0.0, 0.004)
        self.palm_force = (
            self.profiles["contact_stiffness_n_m"] * 0.06 * palm_penetration
        ).clamp(0.0, 10.0)
        self.peak_indentation = torch.maximum(
            self.peak_indentation, self.indentation.amax(dim=1)
        )
        return closure, mean_radius

    def _update_object(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_normal = self.physical_force.sum(dim=1) + self.palm_force
        capacity = self.profiles["static_friction"] * total_normal
        gravity_load = self.profiles["mass_kg"] * 9.81
        has_contact = self.physical_force.amax(dim=1) >= 0.25
        slip_ratio = ((gravity_load - capacity) / gravity_load.clamp_min(1.0e-4)).clamp(0.0, 1.0)
        slip = has_contact & (slip_ratio > 0.05)
        self.previous_object_offset.copy_(self.object_offset)
        centering = (capacity / gravity_load.clamp_min(1.0e-4)).clamp(0.0, 1.0)
        self.object_offset[:, :2] *= (1.0 - 0.08 * centering.unsqueeze(1))
        self.object_offset[:, 2] -= self.dt * (
            0.030 * slip_ratio + 0.004 * (~has_contact).float()
        )
        self.object_velocity = (self.object_offset - self.previous_object_offset) / self.dt
        distance = torch.linalg.vector_norm(self.object_offset, dim=1)
        speed = torch.linalg.vector_norm(self.object_velocity, dim=1)
        dropped = (self.object_offset[:, 2] < -0.065) | (distance > 0.095)
        return slip, dropped, distance

    def _physical_tactile_matrices(self) -> torch.Tensor:
        row = torch.linspace(-1.0, 1.0, 12, dtype=self.dtype, device=self.device).view(1, 1, 12, 1)
        column = torch.linspace(-1.0, 1.0, 6, dtype=self.dtype, device=self.device).view(1, 1, 1, 6)
        center_row = (
            0.20 * torch.sin(self.finger_angles).unsqueeze(0)
            + self.object_offset[:, 2:3] / 0.040
        ).clamp(-0.8, 0.8)
        lateral_projection = torch.einsum(
            "nd,fd->nf", self.object_offset[:, :2], self.finger_direction
        )
        center_column = (lateral_projection / 0.025).clamp(-0.8, 0.8)
        sigma_row = 0.24 + 0.10 * (self.physical_force / self.force_ceiling).clamp(0.0, 1.0)
        sigma_col = 0.30 + 0.08 * (self.physical_force / self.force_ceiling).clamp(0.0, 1.0)
        gaussian = torch.exp(
            -0.5
            * (
                ((row - center_row.unsqueeze(-1).unsqueeze(-1)) / sigma_row.unsqueeze(-1).unsqueeze(-1)).square()
                + ((column - center_column.unsqueeze(-1).unsqueeze(-1)) / sigma_col.unsqueeze(-1).unsqueeze(-1)).square()
            )
        )
        gaussian /= gaussian.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
        return gaussian * (100.0 * self.physical_force).unsqueeze(-1).unsqueeze(-1)

    def _update_tactile(self) -> torch.Tensor:
        physical = self._physical_tactile_matrices()
        self.tactile_history[1:] = self.tactile_history[:-1].clone()
        self.tactile_history[0] = physical
        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed = self.tactile_history[self.profiles["tactile_delay_steps"], env_ids]
        tau = self.profiles["tactile_release_tau_s"].clamp_min(self.dt)
        alpha = (1.0 - torch.exp(-self.dt / tau)).unsqueeze(-1).unsqueeze(-1)
        self.tactile_release += alpha * (delayed - self.tactile_release)
        self.tactile_drift += self.profiles["tactile_drift_normalized_s"] * self.dt
        bias = self.profiles["tactile_bias_normalized"] + self.tactile_drift
        candidate = (
            self.tactile_release * self.profiles["tactile_gain"].unsqueeze(-1).unsqueeze(-1)
            + (1000.0 / 72.0) * bias.unsqueeze(-1).unsqueeze(-1)
        ).clamp_min(0.0)
        dropout = torch.rand(
            (self.num_envs, 5), generator=self.generator, dtype=self.dtype, device="cpu"
        ).to(self.device) < self.profiles["tactile_dropout_probability"]
        dropout |= self.profiles["tactile_failure_mask"]
        self.tactile_measured = torch.where(
            dropout.unsqueeze(-1).unsqueeze(-1), self.tactile_measured, candidate
        )
        self.tactile_age = torch.where(
            dropout, self.tactile_age + self.dt, torch.zeros_like(self.tactile_age)
        )
        extracted = extract_tactile_features(
            self.tactile_measured,
            previous_total_load=self.previous_tactile_total,
            dt_s=self.dt,
            sample_age_s=self.tactile_age,
            maximum_age_s=0.15,
            contact_taxel_threshold=0.5,
            total_load_full_scale=1000.0,
            peak_taxel_full_scale=250.0,
            load_rate_full_scale_per_s=5000.0,
        )
        self.previous_tactile_total = extracted.total_load
        return extracted.features

    def _build_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        tactile_features = self._update_tactile()
        axes = self.profiles["axes_m"]
        axes_norm = (2.0 * (axes - 0.021) / (0.044 - 0.021) - 1.0).clamp(-1.0, 1.0)
        yaw = self.profiles["object_yaw_rad"]
        orientation = torch.stack(
            (
                torch.cos(yaw),
                torch.sin(yaw),
                torch.zeros_like(yaw),
                -torch.sin(yaw),
                torch.cos(yaw),
                torch.zeros_like(yaw),
            ),
            dim=1,
        )
        q_norm = (2.0 * (self.joint_position - self.lower) / (self.upper - self.lower) - 1.0).clamp(-1.0, 1.0)
        qd_norm = (self.joint_velocity / self.velocity).clamp(-1.0, 1.0)
        actor = pack_actor_observation(
            fruit_center_in_palm=(self.object_offset / 0.08).clamp(-1.0, 1.0),
            fruit_semi_axes=axes_norm,
            fruit_orientation_6d=orientation,
            fruit_twist_in_palm=torch.cat(
                ((self.object_velocity / 0.20).clamp(-1.0, 1.0), torch.zeros_like(self.object_velocity)),
                dim=1,
            ),
            hand_joint_position=q_norm,
            hand_joint_velocity=qd_norm,
            tactile_matrix_features=tactile_features,
            previous_action=self.action,
        )
        contact_points = self.object_offset.unsqueeze(1).repeat(1, 5, 1)
        contact_points[:, :, :2] += self.finger_direction.unsqueeze(0) * axes[:, :2].mean(dim=1).view(-1, 1, 1)
        self.contact_points = contact_points
        critic = pack_critic_state(
            actor,
            physical_fingertip_force=(self.physical_force / self.force_ceiling).clamp(0.0, 2.0),
            physical_palm_force=(self.palm_force / 8.0).clamp(0.0, 2.0).unsqueeze(1),
            contact_points_in_palm=(self.contact_points / 0.08).reshape(self.num_envs, 15).clamp(-2.0, 2.0),
            object_mass=(self.profiles["mass_kg"] / 0.30).unsqueeze(1),
            static_friction=self.profiles["static_friction"].unsqueeze(1),
            effective_stiffness=(self.profiles["contact_stiffness_n_m"] / 9000.0).unsqueeze(1),
            peak_indentation=(self.peak_indentation / 0.010).unsqueeze(1),
            force_exposure=(self.mdp_state.force_exposure_n_s.mean(dim=1) / 3.0).unsqueeze(1),
            impact_exposure=(self.mdp_state.impact_exposure_n.mean(dim=1) / 10.0).unsqueeze(1),
            damage_score=self.mdp_state.damage_score.unsqueeze(1),
            tactile_latent_gain=((self.profiles["tactile_gain"] - 1.0) / 0.18).clamp(-1.0, 1.0),
            tactile_latent_bias=(self.profiles["tactile_bias_normalized"] / 0.03).clamp(-1.0, 1.0),
        )
        if actor.shape != (self.num_envs, ACTOR_OBSERVATION_DIM):
            raise AssertionError("surrogate actor observation drifted")
        if critic.shape != (self.num_envs, CRITIC_STATE_DIM):
            raise AssertionError("surrogate critic state drifted")
        return actor, critic

    def step(self, action: torch.Tensor) -> CpuSurrogateStep:
        if action.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"action must have shape {(self.num_envs, ACTION_DIM)}")
        if not torch.isfinite(action).all():
            raise ValueError("action contains NaN or Inf")
        bounded = action.clamp(-1.0, 1.0)
        action_delta = bounded - self.action
        self.action = bounded
        self._apply_actuator_model(bounded)
        self._contact_model()
        slip, dropped, center_distance = self._update_object()
        relative_speed = torch.linalg.vector_norm(self.object_velocity, dim=1)
        # The CPU surrogate does not make collision geometry claims.
        self_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        normalized_joint_position = (
            (self.joint_position - self.lower) / (self.upper - self.lower)
        ).clamp(0.0, 1.0)
        joint_margin = torch.minimum(
            normalized_joint_position, 1.0 - normalized_joint_position
        )
        normalized_joint_limit_proximity = (
            (0.05 - joint_margin).clamp_min(0.0) / 0.05
        ).mean(dim=1)
        transition = hand_grasp_transition(
            self.mdp_state,
            surface_gap_m=self.surface_gap,
            physical_finger_force_n=self.physical_force,
            physical_palm_force_n=self.palm_force,
            center_distance_m=center_distance,
            relative_speed_m_s=relative_speed,
            slip_detected=slip,
            dropped=dropped,
            self_collision=self_collision,
            action=bounded,
            action_delta=action_delta,
            normalized_joint_limit_proximity=normalized_joint_limit_proximity,
            dt_s=self.dt,
        )
        self.mdp_state = transition.state
        self.step_count += 1
        truncated = self.step_count >= self.episode_steps
        terminated = transition.terminated
        done = terminated | truncated
        reward = transition.reward.clone()
        success = transition.success.clone()
        metrics = {name: value.clone() for name, value in transition.metrics.items()}
        metrics.update(
            {
                "drop": dropped.clone(),
                "slip": slip.clone(),
                "mean_force_n": self.physical_force.mean(dim=1).clone(),
                "max_force_n": self.physical_force.amax(dim=1).clone(),
            }
        )
        if self.auto_reset and done.any():
            self._reset_indices(done.nonzero().flatten())
        observation, critic = self._build_observations()
        return CpuSurrogateStep(
            observation=observation,
            critic_state=critic,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=success,
            metrics=metrics,
        )


__all__ = ["CpuHandGraspEnv", "CpuSurrogateStep", "SURROGATE_ENV_VERSION"]
