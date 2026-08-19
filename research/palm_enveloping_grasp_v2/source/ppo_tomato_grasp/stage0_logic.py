"""Tensor-only logic for the Stage-0 rigid-sphere enclosure task.

This module deliberately has no Isaac Lab imports.  It is used by the real
environment and by fast unit tests, so reward changes can be checked without
starting Isaac Sim.  The contact loads are normalized simulation proxies, not
calibrated Newton values and not hardware tactile readings.
"""

from __future__ import annotations

import torch


def proxy_tip_positions(
    object_pos: torch.Tensor,
    actions: torch.Tensor,
    tomato_radius: float,
    open_radius: float = 0.090,
    closed_surface_offset: float = -0.002,
) -> torch.Tensor:
    """Create a differentiable five-tip enclosure proxy around a sphere.

    Action order is thumb opposition, thumb flex, index, middle, ring and
    little flex.  The thumb radius weakly depends on opposition so all six
    actions affect the proxy while opposition also changes its approach angle.
    """

    if object_pos.ndim != 2 or object_pos.shape[-1] != 3:
        raise ValueError(f"object_pos must have shape (N, 3), got {tuple(object_pos.shape)}")
    if actions.ndim != 2 or actions.shape[-1] != 6:
        raise ValueError(f"actions must have shape (N, 6), got {tuple(actions.shape)}")
    if object_pos.shape[0] != actions.shape[0]:
        raise ValueError("object_pos and actions must have the same batch size")

    bounded = actions.clamp(-1.0, 1.0)
    thumb_closure = 0.25 * bounded[:, 0] + 0.75 * bounded[:, 1]
    finger_closure = torch.stack(
        (thumb_closure, bounded[:, 2], bounded[:, 3], bounded[:, 4], bounded[:, 5]), dim=-1
    )
    closure_01 = 0.5 * (finger_closure + 1.0)
    closed_radius = tomato_radius + closed_surface_offset
    radial_distance = open_radius + closure_01 * (closed_radius - open_radius)

    base_angles = torch.tensor((-1.18, -0.50, -0.12, 0.28, 0.68), device=actions.device, dtype=actions.dtype)
    angles = base_angles.unsqueeze(0).repeat(actions.shape[0], 1)
    angles[:, 0] += 0.30 * (0.5 * (bounded[:, 0] + 1.0))

    tips = object_pos.unsqueeze(1).repeat(1, 5, 1)
    tips[:, :, 0] += radial_distance * torch.sin(angles)
    tips[:, :, 1] += radial_distance * torch.cos(angles)
    tips[:, :, 2] += 0.006
    return tips


def normalized_contact_loads(
    tip_pos: torch.Tensor,
    object_pos: torch.Tensor,
    tomato_radius: float,
    contact_band: float = 0.008,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized contact-load proxies and fingertip surface gaps."""

    distances = torch.linalg.vector_norm(tip_pos - object_pos.unsqueeze(1), dim=-1)
    surface_gap = distances - tomato_radius
    loads = ((contact_band - surface_gap) / contact_band).clamp(0.0, 1.5)
    return loads, surface_gap


def enclosure_reward(
    contact_loads: torch.Tensor,
    surface_gap: torch.Tensor,
    action_delta: torch.Tensor,
    *,
    target_load: float = 0.35,
    contact_threshold: float = 0.10,
    safe_load: float = 0.90,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reward gentle five-finger enclosure without requiring arm motion."""

    contacts = contact_loads >= contact_threshold
    coverage = contacts.float().mean(dim=-1)
    enclosure = contacts[:, 0] & (contacts[:, 1:].sum(dim=-1) >= 3)
    proximity = torch.exp(-surface_gap.clamp(min=0.0) / 0.025).mean(dim=-1)

    load_error = (contact_loads - target_load).square().mean(dim=-1)
    load_balance = contact_loads.var(dim=-1, unbiased=False)
    overload = (contact_loads - safe_load).clamp(min=0.0).mean(dim=-1)
    action_rate = action_delta.square().mean(dim=-1)
    gentle_contact = (
        torch.exp(-((contact_loads - target_load) / 0.22).square()) * contacts.float()
    ).mean(dim=-1)

    reward = proximity + 1.5 * coverage
    reward += 1.5 * enclosure.float()
    reward += 2.5 * gentle_contact
    reward -= 2.0 * load_error
    reward -= 0.5 * load_balance * enclosure.float()
    reward -= 20.0 * overload
    reward -= 0.02 * action_rate

    stable = enclosure & (contact_loads.max(dim=-1).values <= safe_load) & (
        contact_loads.mean(dim=-1) >= 0.15
    )
    metrics = {
        "proximity": proximity,
        "coverage": coverage,
        "enclosure": enclosure,
        "stable": stable,
        "mean_load": contact_loads.mean(dim=-1),
        "max_load": contact_loads.max(dim=-1).values,
        "overload": overload,
        "gentle_contact": gentle_contact,
        "action_rate": action_rate,
    }
    return reward, metrics
