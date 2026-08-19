"""Numerically explicit v2 observation normalizers."""

from __future__ import annotations

import torch


def normalize_semi_axes(axes_m: torch.Tensor) -> torch.Tensor:
    """Map 30--50 mm semi-axes to [-1, 1] without the v1 early saturation.

    Values outside the designed domain are clipped only after the full
    20-mm interval has been represented.  This is an actor-observation prior,
    not a claim about real tomato dimensions.
    """

    if axes_m.shape[-1] != 3:
        raise ValueError("axes_m must end in three semi-axes")
    return ((axes_m - 0.040) / 0.010).clamp(-1.0, 1.0)
