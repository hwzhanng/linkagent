"""Pure action-channel filters used by the v2 controller and unit tests."""

from __future__ import annotations

import torch


def filter_distal_residual(
    previous: torch.Tensor | None,
    action: torch.Tensor,
    *,
    alpha: float | torch.Tensor = 0.70,
) -> torch.Tensor:
    """Low-pass only the four distal residual channels.

    The first ten manifold channels remain untouched by this helper.  Keeping
    it pure makes the different root/distal rates auditable without starting
    Isaac Sim.
    """
    if action.shape[-1] != 14:
        raise ValueError("v2 action must have 14 channels")
    bounded = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    if torch.is_tensor(alpha):
        if alpha.ndim == 1 and alpha.shape != (4,):
            raise ValueError("per-digit alpha must have shape (4,)")
        if alpha.ndim not in (0, 1):
            raise ValueError("alpha tensor must be scalar or shape (4,)")
        if not torch.isfinite(alpha).all() or bool((alpha < 0.0).any()) or bool((alpha >= 1.0).any()):
            raise ValueError("alpha must lie in [0, 1)")
        alpha_value = alpha.to(device=bounded.device, dtype=bounded.dtype)
        if alpha_value.ndim == 1:
            alpha_value = alpha_value.view(*([1] * (bounded.ndim - 1)), 4)
    else:
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must lie in [0, 1)")
        alpha_value = float(alpha)
    if previous is None:
        distal = bounded[..., 10:]
    else:
        if previous.shape != bounded[..., 10:].shape:
            raise ValueError("previous distal action shape mismatch")
        distal = alpha_value * previous + (1.0 - alpha_value) * bounded[..., 10:]
    return torch.cat((bounded[..., :10], distal), dim=-1)


__all__ = ["filter_distal_residual"]
