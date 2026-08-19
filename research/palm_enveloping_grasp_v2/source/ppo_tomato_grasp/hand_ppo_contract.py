"""Frozen tensor contract for the hand-only tomato grasp policy.

This module contains no Isaac Lab or hardware imports.  Actor fields are
restricted to signals that can eventually be produced by vision, hand state,
and the five 12x6 fingertip matrices.  Simulator-only truth is appended only
to the critic state.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

from simulation.ppo_tomato_grasp.hand_manifold import (
    DIGIT_RESIDUAL_ORDER,
    LEFT_ACTIVE_JOINT_ORDER,
    SYNERGY_ORDER,
)


INTERFACE_VERSION = "1.0.0-cpu-gated"
ACTION_LAYOUT_ID = "left_g20_manifold5_digit_residual5_v1"
ACTOR_OBSERVATION_LAYOUT_ID = "hand_tomato_actor_110d_v1"
CRITIC_STATE_LAYOUT_ID = "hand_tomato_critic_148d_v1"
TACTILE_FEATURE_LAYOUT_ID = "g20_matrix_5x10_v1_uncalibrated"
POLICY_RATE_HZ = 20
ACTION_DIM = 10
TACTILE_FEATURES_PER_FINGER = 10
TACTILE_FEATURE_DIM = 5 * TACTILE_FEATURES_PER_FINGER

ACTION_ORDER = (*SYNERGY_ORDER, *DIGIT_RESIDUAL_ORDER)

ACTOR_OBSERVATION_BLOCKS: Mapping[str, slice] = MappingProxyType(
    {
        "fruit_center_in_palm": slice(0, 3),
        "fruit_semi_axes": slice(3, 6),
        "fruit_orientation_6d": slice(6, 12),
        "fruit_twist_in_palm": slice(12, 18),
        "hand_joint_position": slice(18, 34),
        "hand_joint_velocity": slice(34, 50),
        "tactile_matrix_features": slice(50, 100),
        "previous_action": slice(100, 110),
    }
)
ACTOR_OBSERVATION_DIM = 110

CRITIC_PRIVILEGED_BLOCKS: Mapping[str, slice] = MappingProxyType(
    {
        "physical_fingertip_force": slice(110, 115),
        "physical_palm_force": slice(115, 116),
        "contact_points_in_palm": slice(116, 131),
        "object_mass": slice(131, 132),
        "static_friction": slice(132, 133),
        "effective_stiffness": slice(133, 134),
        "peak_indentation": slice(134, 135),
        "force_exposure": slice(135, 136),
        "impact_exposure": slice(136, 137),
        "damage_score": slice(137, 138),
        "tactile_latent_gain": slice(138, 143),
        "tactile_latent_bias": slice(143, 148),
    }
)
CRITIC_STATE_DIM = 148

# Tomato MDP critic layout (palm-up stem task).  The plan §4.3 adds the stem
# load components, stem damage, slip speed, palm contact point, and per-digit
# contact counts to the privileged state.  The 148D layout above stays frozen
# for the sphere MDP and the CPU surrogate; the tomato environment packs with
# this extended layout instead.
TOMATO_CRITIC_STATE_LAYOUT_ID = "hand_tomato_critic_163d_v1"
TOMATO_CRITIC_PRIVILEGED_BLOCKS: Mapping[str, slice] = MappingProxyType(
    {
        "physical_fingertip_force": slice(110, 115),
        "physical_palm_force": slice(115, 116),
        "contact_points_in_palm": slice(116, 131),
        "palm_contact_point_in_palm": slice(131, 134),
        "contact_counts": slice(134, 140),
        "object_mass": slice(140, 141),
        "static_friction": slice(141, 142),
        "effective_stiffness": slice(142, 143),
        "peak_indentation": slice(143, 144),
        "force_exposure": slice(144, 145),
        "impact_exposure": slice(145, 146),
        "damage_score": slice(146, 147),
        "stem_loads": slice(147, 151),
        "stem_damage": slice(151, 152),
        "slip_speed": slice(152, 153),
        "tactile_latent_gain": slice(153, 158),
        "tactile_latent_bias": slice(158, 163),
    }
)
TOMATO_CRITIC_STATE_DIM = 163

TACTILE_FEATURE_ORDER = (
    "total_load",
    "peak_taxel",
    "contact_area",
    "centroid_row",
    "centroid_col",
    "spread_row",
    "spread_col",
    "load_rate",
    "valid",
    "stale_or_dropout",
)


def _validate_blocks(blocks: Mapping[str, slice], expected_dim: int, start: int = 0) -> None:
    cursor = start
    for name, block in blocks.items():
        if block.step not in (None, 1) or block.start != cursor or block.stop is None:
            raise AssertionError(f"non-contiguous contract block: {name}")
        cursor = block.stop
    if cursor != expected_dim:
        raise AssertionError(f"contract ends at {cursor}, expected {expected_dim}")


_validate_blocks(ACTOR_OBSERVATION_BLOCKS, ACTOR_OBSERVATION_DIM)
_validate_blocks(CRITIC_PRIVILEGED_BLOCKS, CRITIC_STATE_DIM, ACTOR_OBSERVATION_DIM)
_validate_blocks(TOMATO_CRITIC_PRIVILEGED_BLOCKS, TOMATO_CRITIC_STATE_DIM, ACTOR_OBSERVATION_DIM)
if len(ACTION_ORDER) != ACTION_DIM or len(LEFT_ACTIVE_JOINT_ORDER) != 16:
    raise AssertionError("action or joint contract arithmetic changed")


def _check_tensor(
    name: str,
    value: torch.Tensor,
    batch_shape: tuple[int, ...],
    width: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (*batch_shape, width):
        raise ValueError(
            f"{name} must have shape {(*batch_shape, width)}, got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def pack_actor_observation(**blocks: torch.Tensor) -> torch.Tensor:
    """Pack named deployable blocks in the frozen 110D order."""

    expected = set(ACTOR_OBSERVATION_BLOCKS)
    if set(blocks) != expected:
        raise ValueError(
            f"actor blocks mismatch: missing={sorted(expected - set(blocks))}, "
            f"extra={sorted(set(blocks) - expected)}"
        )
    first = next(iter(blocks.values()))
    if first.ndim < 1:
        raise ValueError("actor blocks must include a feature dimension")
    batch_shape = tuple(first.shape[:-1])
    ordered: list[torch.Tensor] = []
    for name, block in ACTOR_OBSERVATION_BLOCKS.items():
        width = int(block.stop - block.start)
        _check_tensor(name, blocks[name], batch_shape, width)
        ordered.append(blocks[name])
    packed = torch.cat(ordered, dim=-1)
    if packed.shape[-1] != ACTOR_OBSERVATION_DIM:
        raise AssertionError("actor observation arithmetic drifted")
    return packed.clamp(-1.0, 1.0)


def pack_critic_state(
    actor_observation: torch.Tensor,
    **privileged_blocks: torch.Tensor,
) -> torch.Tensor:
    """Append simulator truth to the unchanged deployable observation."""

    if actor_observation.ndim < 1 or actor_observation.shape[-1] != ACTOR_OBSERVATION_DIM:
        raise ValueError("actor_observation must end in 110 values")
    if not torch.isfinite(actor_observation).all():
        raise ValueError("actor_observation contains NaN or Inf")
    expected = set(CRITIC_PRIVILEGED_BLOCKS)
    if set(privileged_blocks) != expected:
        raise ValueError(
            f"critic blocks mismatch: missing={sorted(expected - set(privileged_blocks))}, "
            f"extra={sorted(set(privileged_blocks) - expected)}"
        )
    batch_shape = tuple(actor_observation.shape[:-1])
    ordered = [actor_observation]
    for name, block in CRITIC_PRIVILEGED_BLOCKS.items():
        width = int(block.stop - block.start)
        _check_tensor(name, privileged_blocks[name], batch_shape, width)
        ordered.append(privileged_blocks[name])
    state = torch.cat(ordered, dim=-1)
    if state.shape[-1] != CRITIC_STATE_DIM:
        raise AssertionError("critic state arithmetic drifted")
    return state


def pack_tomato_critic_state(
    actor_observation: torch.Tensor,
    **privileged_blocks: torch.Tensor,
) -> torch.Tensor:
    """Append simulator truth in the frozen 163D tomato critic order."""

    if actor_observation.ndim < 1 or actor_observation.shape[-1] != ACTOR_OBSERVATION_DIM:
        raise ValueError("actor_observation must end in 110 values")
    if not torch.isfinite(actor_observation).all():
        raise ValueError("actor_observation contains NaN or Inf")
    expected = set(TOMATO_CRITIC_PRIVILEGED_BLOCKS)
    if set(privileged_blocks) != expected:
        raise ValueError(
            f"tomato critic blocks mismatch: missing={sorted(expected - set(privileged_blocks))}, "
            f"extra={sorted(set(privileged_blocks) - expected)}"
        )
    batch_shape = tuple(actor_observation.shape[:-1])
    ordered = [actor_observation]
    for name, block in TOMATO_CRITIC_PRIVILEGED_BLOCKS.items():
        width = int(block.stop - block.start)
        _check_tensor(name, privileged_blocks[name], batch_shape, width)
        ordered.append(privileged_blocks[name])
    state = torch.cat(ordered, dim=-1)
    if state.shape[-1] != TOMATO_CRITIC_STATE_DIM:
        raise AssertionError("tomato critic state arithmetic drifted")
    return state


def block(vector: torch.Tensor, name: str) -> torch.Tensor:
    """Return one named Actor or Critic slice with strict width validation."""

    if name in ACTOR_OBSERVATION_BLOCKS:
        if vector.shape[-1] not in (ACTOR_OBSERVATION_DIM, CRITIC_STATE_DIM, TOMATO_CRITIC_STATE_DIM):
            raise ValueError("vector is not an Actor observation or Critic state")
        return vector[..., ACTOR_OBSERVATION_BLOCKS[name]]
    if name in TOMATO_CRITIC_PRIVILEGED_BLOCKS:
        if vector.shape[-1] != TOMATO_CRITIC_STATE_DIM:
            raise ValueError("privileged blocks require a tomato Critic state")
        return vector[..., TOMATO_CRITIC_PRIVILEGED_BLOCKS[name]]
    if name in CRITIC_PRIVILEGED_BLOCKS:
        if vector.shape[-1] != CRITIC_STATE_DIM:
            raise ValueError("privileged blocks require a Critic state")
        return vector[..., CRITIC_PRIVILEGED_BLOCKS[name]]
    raise KeyError(f"unknown observation block {name!r}")


__all__ = [
    "ACTION_DIM",
    "ACTION_LAYOUT_ID",
    "ACTION_ORDER",
    "ACTOR_OBSERVATION_BLOCKS",
    "ACTOR_OBSERVATION_DIM",
    "ACTOR_OBSERVATION_LAYOUT_ID",
    "CRITIC_PRIVILEGED_BLOCKS",
    "CRITIC_STATE_DIM",
    "CRITIC_STATE_LAYOUT_ID",
    "INTERFACE_VERSION",
    "TOMATO_CRITIC_PRIVILEGED_BLOCKS",
    "TOMATO_CRITIC_STATE_DIM",
    "TOMATO_CRITIC_STATE_LAYOUT_ID",
    "POLICY_RATE_HZ",
    "TACTILE_FEATURE_DIM",
    "TACTILE_FEATURE_LAYOUT_ID",
    "TACTILE_FEATURE_ORDER",
    "TACTILE_FEATURES_PER_FINGER",
    "block",
    "pack_actor_observation",
    "pack_critic_state",
    "pack_tomato_critic_state",
]
