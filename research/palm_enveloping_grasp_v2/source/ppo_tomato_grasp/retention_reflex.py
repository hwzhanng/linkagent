"""Bounded post-detachment retention correction for the enveloping task.

The helper is deliberately stateless and Torch-only so it can be unit tested
without Isaac Sim.  It is not a pose replay or a kinematic capture: each
finger receives a small signed target correction from its own filtered force,
and a separate slip term is enabled only while the fruit is still inside the
palm workspace.  The PhysX environment owns the low-pass state and applies
the correction through its normal rate-limited joint-target path.
"""

from __future__ import annotations

import torch


def adaptive_retention_correction(
    *,
    filtered_force_n: torch.Tensor,
    target_force_n: torch.Tensor,
    high_force_n: torch.Tensor,
    relative_speed_m_s: torch.Tensor,
    in_hand: torch.Tensor,
    palm_support: torch.Tensor,
    contact_threshold_n: float,
    force_gain_rad_per_n: float,
    slip_gain_rad_per_m_s: float,
    slip_threshold_m_s: float,
    max_correction_rad: float,
) -> torch.Tensor:
    """Return one bounded correction per digit.

    Positive correction means curl further under the hand's calibrated joint
    convention.  The correction is zero when the object has left the usable
    palm workspace and is attenuated when the palm already carries support.
    ``filtered_force_n`` must have shape ``(N, 5)``; all other per-digit
    vectors have shape ``(5,)`` and the phase masks have shape ``(N,)``.
    """

    if filtered_force_n.ndim != 2 or filtered_force_n.shape[-1] != 5:
        raise ValueError("filtered_force_n must have shape (N, 5)")
    if target_force_n.shape != (5,) or high_force_n.shape != (5,):
        raise ValueError("target_force_n and high_force_n must have shape (5,)")
    if relative_speed_m_s.shape != (filtered_force_n.shape[0],):
        raise ValueError("relative_speed_m_s must have shape (N,)")
    if in_hand.shape != relative_speed_m_s.shape or palm_support.shape != relative_speed_m_s.shape:
        raise ValueError("in_hand and palm_support must have shape (N,)")
    if force_gain_rad_per_n < 0.0 or slip_gain_rad_per_m_s < 0.0:
        raise ValueError("reflex gains must be non-negative")
    if max_correction_rad < 0.0 or slip_threshold_m_s < 0.0:
        raise ValueError("reflex limits must be non-negative")

    force_error = (target_force_n.unsqueeze(0) - filtered_force_n).clamp(-2.0, 2.0)
    correction = force_gain_rad_per_n * force_error
    high = filtered_force_n > high_force_n.unsqueeze(0)
    correction = torch.where(
        high,
        -force_gain_rad_per_n * (filtered_force_n - high_force_n.unsqueeze(0)).clamp(0.0, 2.0),
        correction,
    )
    active = filtered_force_n >= float(contact_threshold_n)
    slip = (relative_speed_m_s - float(slip_threshold_m_s)).clamp_min(0.0)
    # A slipping fruit needs a gentle all-digit preload, but only while it is
    # still near the palm or the palm is carrying it.  This prevents an empty
    # hand after a real drop from continuing to close indefinitely.
    correction = correction + slip_gain_rad_per_m_s * slip.unsqueeze(-1) * (
        active | (in_hand | palm_support).unsqueeze(-1)
    ).float()
    # Palm support is useful evidence, not a reason to squeeze harder.  Keep
    # enough fingertip correction for lateral retention but reduce its force
    # component when the palm is already carrying the load.
    correction = torch.where(
        palm_support.unsqueeze(-1),
        correction * 0.65,
        correction,
    )
    correction = torch.where(
        (in_hand | palm_support).unsqueeze(-1),
        correction,
        torch.zeros_like(correction),
    )
    return correction.clamp(-float(max_correction_rad), float(max_correction_rad))


__all__ = ["adaptive_retention_correction"]
