"""Single coordinate convention for the virtual tomato attachment."""

from __future__ import annotations

import torch


def attachment_load_ratio(
    force_on_fruit_w: torch.Tensor,
    pull_axis_w: torch.Tensor,
    break_force_n: torch.Tensor,
) -> torch.Tensor:
    """Return axial tensile load for ``z+ away from plant``.

    Pulling the fruit in ``pull_axis_w`` makes the attachment exert a restoring
    force in the opposite direction. Compression, transverse bending, and
    uncalibrated whole-fruit rotation do not count as tensile picking load.
    """

    axis = torch.nn.functional.normalize(pull_axis_w, dim=1)
    axial_tension_n = (-(force_on_fruit_w * axis).sum(dim=1)).clamp_min(0.0)
    return axial_tension_n / break_force_n.clamp_min(1.0e-8)


def attachment_broken(
    force_on_fruit_w: torch.Tensor,
    pull_axis_w: torch.Tensor,
    break_force_n: torch.Tensor,
) -> torch.Tensor:
    return attachment_load_ratio(
        force_on_fruit_w,
        pull_axis_w,
        break_force_n,
    ) >= 1.0


def update_overload_streak(
    previous: torch.Tensor,
    overloaded: torch.Tensor,
    attached: torch.Tensor,
) -> torch.Tensor:
    """Count consecutive overloaded physics substeps for attached fruit."""

    return torch.where(
        attached & overloaded,
        previous + 1,
        torch.zeros_like(previous),
    )


__all__ = ["attachment_broken", "attachment_load_ratio", "update_overload_streak"]
