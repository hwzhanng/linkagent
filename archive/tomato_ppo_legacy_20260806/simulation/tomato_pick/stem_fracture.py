"""Directional tomato-stem fracture helpers.

The attachment axis points from the fruit toward the peduncle.  Detachment is
caused by tensile load along that axis or twist about it; compression and
lateral bending are not interchangeable with a pull/twist picking action.
"""

from __future__ import annotations

import torch


def directional_load_ratio(
    force_w: torch.Tensor,
    torque_w: torch.Tensor,
    stem_axis_w: torch.Tensor,
    break_force_n: torch.Tensor,
    break_torque_nm: torch.Tensor,
) -> torch.Tensor:
    """Return the larger normalized axial pull or twist load."""

    axis = torch.nn.functional.normalize(stem_axis_w, dim=1)
    axial_tension_n = torch.sum(force_w * axis, dim=1).clamp_min(0.0)
    axial_twist_nm = torch.abs(torch.sum(torque_w * axis, dim=1))
    return torch.maximum(
        axial_tension_n / break_force_n.clamp_min(1.0e-8),
        axial_twist_nm / break_torque_nm.clamp_min(1.0e-8),
    )


def directional_overload(
    force_w: torch.Tensor,
    torque_w: torch.Tensor,
    stem_axis_w: torch.Tensor,
    break_force_n: torch.Tensor,
    break_torque_nm: torch.Tensor,
) -> torch.Tensor:
    """Return whether each stem exceeds its axial pull or twist threshold."""

    return directional_load_ratio(
        force_w,
        torque_w,
        stem_axis_w,
        break_force_n,
        break_torque_nm,
    ) >= 1.0


def update_overload_streak(
    previous: torch.Tensor,
    overloaded: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Count consecutive overloaded physics substeps for active attachments."""

    return torch.where(
        active & overloaded,
        previous + 1,
        torch.zeros_like(previous),
    )


__all__ = [
    "directional_load_ratio",
    "directional_overload",
    "update_overload_streak",
]
