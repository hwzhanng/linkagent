"""A 100-line, model-free object-pressure-to-action gradient track.

Online damped Broyden update estimates sparse pressure response J=dp/da
from actual joint action *changes* and measured next-step pressure changes.
J is a local action sensitivity; it is NOT a neural-network gradient.
The derived -J^T dD/dp correction is task-tangent projected and accepted
only if its own one-step model predicts pressure relief without losing spin.

This is the ONLY proposed new control algorithm on top of 2026 RoTO2 PPO.
Every parameter is a testable starting prior, not a proven safety guarantee.
"""
from __future__ import annotations

import torch

CHANNELS = 24
ACTIONS = 16
HOTSPOT_THRESHOLD = 0.70  # pressure divided by 5kPa; not calibrated fruit damage


def hotspot_cost(p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Risk must be insensitive to unobserved patches (NOT assume them safe)."""
    valid = mask.clamp(0, 1)
    exceed = (p.clamp_min(0) - HOTSPOT_THRESHOLD).clamp_min(0) * valid
    return exceed.square().sum(-1) / valid.sum(-1).clamp_min(1)


class PressureJacobianTrack:
    """Bathed per-environment rank-one secant gradient, entirely on GPU."""

    def __init__(self, num_envs: int, device: str, *, enabled: bool = True,
                 beta: float = 0.12, step: float = 0.12,
                 min_excitations: int = 8):
        self.num_envs, self.enabled = num_envs, enabled
        self.beta, self.step = beta, step
        self.min_excitations = min_excitations
        self.J = torch.zeros(num_envs, CHANNELS, ACTIONS, device=device)
        self.spin_J = torch.zeros(num_envs, ACTIONS, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_a = torch.zeros(num_envs, ACTIONS, device=device)
        self.prev_p = torch.zeros(num_envs, CHANNELS, device=device)
        self.prev_spin = torch.zeros(num_envs, device=device)
        self.last_correction = torch.zeros(num_envs, ACTIONS, device=device)
        self.last_accepted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @torch.no_grad()
    def reset(self, ids: torch.Tensor) -> None:
        self.J[ids] = 0
        self.spin_J[ids] = 0
        self.count[ids] = 0
        self.prev_a[ids] = 0
        self.prev_p[ids] = 0
        self.prev_spin[ids] = 0
        self.last_correction[ids] = 0
        self.last_accepted[ids] = False
        self.ready[ids] = False
        self.initialized[ids] = False

    @torch.no_grad()
    def observe(self, applied: torch.Tensor, next_pressure: torch.Tensor,
                panel_mask: torch.Tensor, spin: torch.Tensor) -> None:
        """Call EXACTLY once per completed physics/control transition.

        Uses action innovation and response innovation. State drift and
        non-smooth contact can bias this local secant estimate, so it is
        strongly damped, bounded, and shielded by acceptance gates.
        """
        if applied.shape != (self.num_envs, ACTIONS):
            raise ValueError("bad applied action shape")
        if next_pressure.shape != (self.num_envs, CHANNELS):
            raise ValueError("bad sparse pressure shape")
        if panel_mask.shape != next_pressure.shape or spin.shape != (self.num_envs,):
            raise ValueError("bad sparse pressure validity/spin shape")

        da = applied - self.prev_a
        dp = (next_pressure - self.prev_p) * panel_mask
        dw = spin - self.prev_spin
        energy = (da * da).sum(-1)
        learn = self.initialized & (energy > 0.01) & (panel_mask.sum(-1) >= 8)
        j_da = torch.bmm(self.J, da.unsqueeze(-1)).squeeze(-1)
        prediction_error = (dp - j_da) * panel_mask
        correction = self.beta * prediction_error.unsqueeze(-1) * da[:, None, :] / (
            energy[:, None, None] + 0.08
        )
        self.J = torch.where(learn[:, None, None],
                             (self.J + correction).clamp(-4., 4.), self.J)
        dw_pred = (self.spin_J * da).sum(-1)
        j_spin_delta = self.beta * (dw - dw_pred)[:, None] * da / (
            energy[:, None] + 0.08
        )
        self.spin_J = torch.where(
            learn[:, None], (self.spin_J + j_spin_delta).clamp(-2., 2.),
            self.spin_J
        )
        self.count += learn.long()
        self.ready = self.count >= self.min_excitations
        self.prev_a.copy_(applied)
        self.prev_p.copy_(next_pressure * panel_mask)
        self.prev_spin.copy_(spin)
        self.initialized[:] = True

    @torch.no_grad()
    def correct(self, proposed: torch.Tensor, pressure: torch.Tensor,
                mask: torch.Tensor, grip_ok: torch.Tensor) -> torch.Tensor:
        """Apply only a bounded, predicted beneficial *tangent* action change."""
        self.last_correction.zero_()
        self.last_accepted.zero_()
        if not self.enabled:
            return proposed.clamp(-1., 1.)
        if proposed.shape != (self.num_envs, ACTIONS):
            raise ValueError("bad proposed action")
        # Explicit derivative of local pressure hotspot potential.
        positive = (pressure - HOTSPOT_THRESHOLD).clamp_min(0) * mask
        dp_cost = 2.0 * positive / mask.sum(-1, keepdim=True).clamp_min(1)
        g_action = torch.bmm(
            self.J.transpose(1, 2), dp_cost.unsqueeze(-1)
        ).squeeze(-1)
        # Prevent pressure relief from canceling in-hand rotation to first order.
        spin = self.spin_J
        projected = g_action - spin * (g_action * spin).sum(-1, keepdim=True) / (
            spin.square().sum(-1, keepdim=True) + 0.03
        )
        norm = projected.norm(dim=-1, keepdim=True)
        adjustment = -self.step * projected / norm.clamp_min(0.20)
        candidate = (proposed + adjustment).clamp(-1., 1.)
        applied_delta = candidate - proposed
        # Counterfactual check with local secant estimate only.
        predicted = (pressure + torch.bmm(
            self.J, applied_delta.unsqueeze(-1)
        ).squeeze(-1)).clamp_min(0)
        before = hotspot_cost(pressure, mask)
        after = hotspot_cost(predicted, mask)
        accepted = (
            self.ready & grip_ok & (mask.sum(-1) >= 8)
            & (before > 1e-4) & (after < before - 1e-5)
            & ((self.spin_J * applied_delta).sum(-1).abs() < 0.03)
            & torch.isfinite(predicted).all(-1)
        )
        self.last_accepted.copy_(accepted)
        self.last_correction[:] = torch.where(
            accepted[:, None], applied_delta, 0.
        )
        return torch.where(accepted[:, None], candidate, proposed).clamp(-1, 1)
