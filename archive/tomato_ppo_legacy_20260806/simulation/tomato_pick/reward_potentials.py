"""Small, policy-agnostic reward potentials for tomato manipulation."""

from __future__ import annotations

import torch


def positive_support_potential(gravity_support_ratio: torch.Tensor) -> torch.Tensor:
    """Reward upward load support without penalizing necessary first contact."""

    return torch.tanh(gravity_support_ratio.clamp_min(0.0))


__all__ = ["positive_support_potential"]
