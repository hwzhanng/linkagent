"""Pure-CPU contract layer for the future Isaac/PhysX v1 hand environment.

This module intentionally imports neither Isaac Lab nor hardware code.  It
defines the exact name-based joint extraction, SI-unit normalization, and
rate-limited action command that an Isaac adapter must call.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_ACTIVE_JOINT_ORDER,
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    LEFT_SAFE_VELOCITY_RAD_S,
    hybrid_action_to_joint_targets,
    rate_limit_joint_targets,
    reorder_joint_targets,
)
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
    POLICY_RATE_HZ,
    TACTILE_FEATURE_DIM,
    pack_actor_observation,
    pack_critic_state,
)
from simulation.ppo_tomato_grasp.hand_grasp_mdp import FINGER_FORCE_CEILING_N


ISAAC_V1_ADAPTER_ID = "left_g20_isaac_v1_contract_dry_cpu_v1"


@dataclass(frozen=True)
class ActiveJointCommand:
    articulation_indices: torch.Tensor
    targets_rad: torch.Tensor
    canonical_targets_rad: torch.Tensor


def _validate_named_width(names: Sequence[str], values: torch.Tensor, label: str) -> tuple[str, ...]:
    order = tuple(names)
    if len(order) != len(set(order)):
        raise ValueError(f"{label} names contain duplicates")
    if values.ndim < 1 or values.shape[-1] != len(order):
        raise ValueError(f"{label} values do not match named width")
    if not torch.isfinite(values).all():
        raise ValueError(f"{label} values contain NaN or Inf")
    return order


def canonical_joint_state(
    articulation_joint_names: Sequence[str],
    articulation_values: torch.Tensor,
) -> torch.Tensor:
    """Extract the 16 active joints from any articulation order by exact name."""

    names = _validate_named_width(
        articulation_joint_names, articulation_values, "articulation joint"
    )
    missing = [name for name in LEFT_ACTIVE_JOINT_ORDER if name not in names]
    if missing:
        raise ValueError(f"articulation is missing active joints: {missing}")
    indices = [names.index(name) for name in LEFT_ACTIVE_JOINT_ORDER]
    return articulation_values[..., indices]


def pack_deployable_actor_si(
    *,
    fruit_center_in_palm_m: torch.Tensor,
    fruit_semi_axes_m: torch.Tensor,
    fruit_orientation_6d: torch.Tensor,
    fruit_twist_in_palm_si: torch.Tensor,
    canonical_joint_position_rad: torch.Tensor,
    canonical_joint_velocity_rad_s: torch.Tensor,
    tactile_matrix_features: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Normalize deployable SI signals and pack the frozen 110D Actor input."""

    if canonical_joint_position_rad.shape[-1:] != (16,) or canonical_joint_velocity_rad_s.shape[-1:] != (16,):
        raise ValueError("canonical joint position/velocity must end in 16 values")
    if tactile_matrix_features.shape[-1:] != (TACTILE_FEATURE_DIM,):
        raise ValueError(f"tactile features must end in {TACTILE_FEATURE_DIM} values")
    if previous_action.shape[-1:] != (ACTION_DIM,):
        raise ValueError(f"previous action must end in {ACTION_DIM} values")
    lower = canonical_joint_position_rad.new_tensor(LEFT_SAFE_LOWER_RAD)
    upper = canonical_joint_position_rad.new_tensor(LEFT_SAFE_UPPER_RAD)
    velocity = canonical_joint_velocity_rad_s.new_tensor(LEFT_SAFE_VELOCITY_RAD_S)
    axes_normalized = (
        2.0 * (fruit_semi_axes_m - 0.021) / (0.044 - 0.021) - 1.0
    ).clamp(-1.0, 1.0)
    q_normalized = (
        2.0 * (canonical_joint_position_rad - lower) / (upper - lower) - 1.0
    ).clamp(-1.0, 1.0)
    qd_normalized = (canonical_joint_velocity_rad_s / velocity).clamp(-1.0, 1.0)
    actor = pack_actor_observation(
        fruit_center_in_palm=(fruit_center_in_palm_m / 0.08).clamp(-1.0, 1.0),
        fruit_semi_axes=axes_normalized,
        fruit_orientation_6d=fruit_orientation_6d,
        fruit_twist_in_palm=torch.cat(
            (
                (fruit_twist_in_palm_si[..., :3] / 0.20).clamp(-1.0, 1.0),
                (fruit_twist_in_palm_si[..., 3:] / 4.0).clamp(-1.0, 1.0),
            ),
            dim=-1,
        ),
        hand_joint_position=q_normalized,
        hand_joint_velocity=qd_normalized,
        tactile_matrix_features=tactile_matrix_features,
        previous_action=previous_action,
    )
    if actor.shape[-1] != ACTOR_OBSERVATION_DIM:
        raise AssertionError("Isaac adapter Actor contract drifted")
    return actor


def action_to_active_joint_command(
    action: torch.Tensor,
    current_canonical_position_rad: torch.Tensor,
    articulation_joint_names: Sequence[str],
    *,
    policy_dt_s: float = 1.0 / POLICY_RATE_HZ,
    velocity_scale: float = 0.50,
) -> ActiveJointCommand:
    """Map 10D actions to rate-limited targets and explicit articulation indices."""

    if abs(policy_dt_s - 1.0 / POLICY_RATE_HZ) > 1.0e-9:
        raise ValueError(f"policy_dt_s must remain exactly {1.0 / POLICY_RATE_HZ} s")
    names = tuple(articulation_joint_names)
    if len(names) != len(set(names)):
        raise ValueError("articulation joint names contain duplicates")
    missing = [name for name in LEFT_ACTIVE_JOINT_ORDER if name not in names]
    if missing:
        raise ValueError(f"articulation is missing active joints: {missing}")
    desired = hybrid_action_to_joint_targets(action)
    canonical = rate_limit_joint_targets(
        current_canonical_position_rad,
        desired,
        policy_dt_s=policy_dt_s,
        velocity_scale=velocity_scale,
        absolute_velocity_cap_rad_s=1.0,
    )
    active_in_articulation = tuple(name for name in names if name in LEFT_ACTIVE_JOINT_ORDER)
    targets = reorder_joint_targets(canonical, active_in_articulation)
    indices = torch.tensor(
        [names.index(name) for name in active_in_articulation], dtype=torch.long
    )
    return ActiveJointCommand(indices, targets, canonical)


def pack_physx_critic_si(
    actor_observation: torch.Tensor,
    *,
    physical_fingertip_force_n: torch.Tensor,
    physical_palm_force_n: torch.Tensor,
    contact_points_in_palm_m: torch.Tensor,
    object_mass_kg: torch.Tensor,
    static_friction: torch.Tensor,
    effective_stiffness_n_m: torch.Tensor,
    peak_indentation_m: torch.Tensor,
    mean_force_exposure_n_s: torch.Tensor,
    mean_impact_exposure_n: torch.Tensor,
    damage_score: torch.Tensor,
    tactile_latent_gain: torch.Tensor,
    tactile_latent_bias_normalized: torch.Tensor,
) -> torch.Tensor:
    """Normalize simulator-only SI truth into the frozen 148D Critic state."""

    if contact_points_in_palm_m.shape[-2:] != (5, 3):
        raise ValueError("contact points must end in shape (5, 3)")
    if physical_palm_force_n.shape == actor_observation.shape[:-1]:
        physical_palm_force_n = physical_palm_force_n.unsqueeze(-1)
    scalar_inputs = (
        object_mass_kg,
        static_friction,
        effective_stiffness_n_m,
        peak_indentation_m,
        mean_force_exposure_n_s,
        mean_impact_exposure_n,
        damage_score,
    )
    scalar_inputs = tuple(
        value.unsqueeze(-1) if value.shape == actor_observation.shape[:-1] else value
        for value in scalar_inputs
    )
    (
        object_mass_kg,
        static_friction,
        effective_stiffness_n_m,
        peak_indentation_m,
        mean_force_exposure_n_s,
        mean_impact_exposure_n,
        damage_score,
    ) = scalar_inputs
    state = pack_critic_state(
        actor_observation,
        physical_fingertip_force=(
            physical_fingertip_force_n
            / physical_fingertip_force_n.new_tensor(FINGER_FORCE_CEILING_N)
        ).clamp(0.0, 2.0),
        physical_palm_force=(physical_palm_force_n / 8.0).clamp(0.0, 2.0),
        contact_points_in_palm=(contact_points_in_palm_m / 0.08).flatten(-2).clamp(-2.0, 2.0),
        object_mass=(object_mass_kg / 0.30).clamp(0.0, 2.0),
        static_friction=static_friction.clamp(0.0, 2.0),
        effective_stiffness=(effective_stiffness_n_m / 9000.0).clamp(0.0, 2.0),
        peak_indentation=(peak_indentation_m / 0.010).clamp(0.0, 2.0),
        force_exposure=(mean_force_exposure_n_s / 3.0).clamp(0.0, 2.0),
        impact_exposure=(mean_impact_exposure_n / 10.0).clamp(0.0, 2.0),
        damage_score=damage_score.clamp(0.0, 1.0),
        tactile_latent_gain=((tactile_latent_gain - 1.0) / 0.18).clamp(-1.0, 1.0),
        tactile_latent_bias=(tactile_latent_bias_normalized / 0.03).clamp(-1.0, 1.0),
    )
    if state.shape[-1] != CRITIC_STATE_DIM:
        raise AssertionError("Isaac adapter Critic contract drifted")
    return state


__all__ = [
    "ActiveJointCommand",
    "ISAAC_V1_ADAPTER_ID",
    "action_to_active_joint_command",
    "canonical_joint_state",
    "pack_deployable_actor_si",
    "pack_physx_critic_si",
]
