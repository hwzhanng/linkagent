"""Task-tangent stress descent (TTSD): physically informed actor guidance.

The gradients originate from an *ensemble of learned one-step models* and
the sensor-derived pressure field, not from differentiating PhysX. Predictions
are cross-checked for agreement, lower risk, preserved rotation and contact.
Uncertainty or weakly identifiable gradients disable this auxiliary path,
leaving the constrained on-policy PPO update intact.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .core import ACTION_DIM, POLICY_DIM, POLICY_SLICES, pressure_risk


def _mlp(input_dim: int, sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for size in sizes:
        layers.extend((nn.Linear(input_dim, size), nn.ELU()))
        input_dim = size
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ContactDynamicsEnsemble(nn.Module):
    """Bootstrap heads predict action-dependent 96-site pressure, spin, support, slip.

    Every head is separately initialized and trained on a random subset of each
    on-policy batch. Only the sparse actor observation is an input. Dense field,
    object spin, support and slip are simulated privileged *training targets*.
    """
    def __init__(self, members: int = 3):
        super().__init__()
        if members < 2:
            raise ValueError("At least two independent models are required")
        self.networks = nn.ModuleList(
            _mlp(POLICY_DIM + ACTION_DIM, (256, 192), 99)
            for _ in range(members)
        )

    def forward_members(self, obs: torch.Tensor, action: torch.Tensor):
        inp = torch.cat((obs, action), dim=-1)
        out = torch.stack(tuple(network(inp) for network in self.networks), 0)
        return (
            F.softplus(out[..., :96]).clamp(0.0, 8.0),
            out[..., 96].tanh(),
            out[..., 97].sigmoid(),
            F.softplus(out[..., 98]),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        pressures, omega, support, slip = self.forward_members(obs, action)
        return pressures.mean(0), omega.mean(0), support.mean(0), slip.mean(0)


def project_task_tangent(
    gradient: torch.Tensor,
    constraints: torch.Tensor,
    damping: float = 0.02,
) -> torch.Tensor:
    """Project pressure gradient outside task-sensitive row space.

    gradient: [N,A], constraints: [N,K,A] derivatives of spin,
    contact support and sliding risk wrt action. Damped batched least squares.
    """
    if gradient.ndim != 2 or constraints.ndim != 3:
        raise ValueError("Bad task/pressure Jacobian ranks")
    if gradient.shape[0] != constraints.shape[0] or gradient.shape[1] != constraints.shape[2]:
        raise ValueError("Task/pressure Jacobian dimensions mismatch")
    norm = torch.linalg.vector_norm(constraints, dim=-1, keepdim=True)
    normalized = torch.where(norm > 1e-6, constraints / norm.clamp_min(1e-6), 0)
    gram = normalized @ normalized.transpose(-1, -2)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    coords = torch.linalg.solve(
        gram + damping * eye, (normalized @ gradient.unsqueeze(-1))
    )
    return gradient - (normalized.transpose(-1, -2) @ coords).squeeze(-1)


@dataclass
class GuidanceConfig:
    candidate_step: float = 0.09
    model_risk_activation: float = 0.008
    risk_disagreement_limit: float = 0.30
    angular_velocity_disagreement_limit: float = 0.22
    gradient_cosine_min: float = 0.30
    predicted_support_min: float = 0.45
    allowed_spin_change: float = 0.035
    allowed_support_drop: float = 0.025
    allowed_slip_increase: float = 0.04
    min_predicted_risk_improvement: float = 1.0e-5
    minimum_valid_sparse_taxels: int = 8
    maximum_examples: int = 192


def _effective_gradient(risk: torch.Tensor, actions: torch.Tensor, retain_graph: bool):
    return torch.autograd.grad(
        risk.sum(), actions, retain_graph=retain_graph,
        create_graph=False, allow_unused=False,
    )[0].detach()


def stress_guidance_loss(
    action_mean: torch.Tensor,
    observation: torch.Tensor,
    ensemble: ContactDynamicsEnsemble,
    config: GuidanceConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Produce a verified model-based correction target and guide the Actor.

    Uses gradients through learned pressure *with respect to robot actions*.
    The target is detached so PPO's stochastic policy learns to mimic the
    local risk-reducing correction while PPO's reward/cost advantages remain
    independent. There is no gradient through true simulator pressure.
    """
    n = min(config.maximum_examples, action_mean.shape[0])
    mean_action = action_mean[:n]
    obs = observation[:n].detach()
    probe = mean_action.detach().clone().requires_grad_(True)
    fields, omegas, supports, slips = ensemble.forward_members(obs, probe)
    members = fields.shape[0]
    all_risks = torch.stack(tuple(pressure_risk(fields[m]) for m in range(members)))
    mean_risk = all_risks.mean(dim=0)
    grad_stress = _effective_gradient(mean_risk, probe, retain_graph=True)
    grad_omega = _effective_gradient(omegas.mean(0), probe, retain_graph=True)
    grad_support = _effective_gradient(supports.mean(0), probe, retain_graph=True)
    grad_slip = _effective_gradient(slips.mean(0), probe, retain_graph=True)
    members_grad = []
    for m in range(members):
        members_grad.append(
            _effective_gradient(all_risks[m], probe, retain_graph=m < members - 1)
        )
    each_grad = torch.stack(members_grad, 0)
    grad_norm = torch.linalg.vector_norm(grad_stress, dim=-1)
    cosine = F.cosine_similarity(each_grad, grad_stress.unsqueeze(0), dim=-1)
    consistency = cosine.min(0).values
    risk_disagreement = all_risks.std(0, unbiased=False)
    omega_disagreement = omegas.std(0, unbiased=False)

    task_jac = torch.stack((grad_omega, grad_support, grad_slip), dim=1)
    projected = project_task_tangent(grad_stress, task_jac)
    projected_norm = projected.norm(dim=-1, keepdim=True)
    correction = projected / projected_norm.clamp_min(0.08)
    candidate = (probe.detach() - config.candidate_step * correction).clamp(-0.95, 0.95)
    with torch.no_grad():
        post_fields, post_omega, post_support, post_slip = ensemble.forward_members(
            obs, candidate
        )
        after_risk = torch.stack(tuple(
            pressure_risk(post_fields[m]) for m in range(members)
        )).mean(0)
        delta_risk = mean_risk.detach() - after_risk
        delta_omega = (
            post_omega.mean(0) - omegas.detach().mean(0)
        ).abs()
        delta_support = supports.detach().mean(0) - post_support.mean(0)
        delta_slip = post_slip.mean(0) - slips.detach().mean(0)
        valid_count = obs[:, POLICY_SLICES["sparse_valid"]].sum(-1)
        gate = (
            (valid_count >= config.minimum_valid_sparse_taxels)
            & (mean_risk.detach() >= config.model_risk_activation)
            & (risk_disagreement.detach() < config.risk_disagreement_limit)
            & (omega_disagreement.detach() < config.angular_velocity_disagreement_limit)
            & (consistency >= config.gradient_cosine_min)
            & (supports.detach().mean(0) >= config.predicted_support_min)
            & (grad_norm >= 1.0e-5)
            & (projected_norm[:, 0] >= 1.0e-6)
            & (delta_risk >= config.min_predicted_risk_improvement)
            & (delta_omega <= config.allowed_spin_change)
            & (delta_support <= config.allowed_support_drop)
            & (delta_slip <= config.allowed_slip_increase)
        )
        # Counterfactual acceptance is checked on EVERY learned model. Mean
        # improvement alone may hide harmful outlier models.
        per_member_improvement = all_risks.detach() - torch.stack(tuple(
            pressure_risk(post_fields[m]) for m in range(members)
        ))
        gate = gate & (per_member_improvement.min(0).values >= 0)
        # Conservative direct noncontact action gate: no gradient is trusted
        # unless the tactile maps indicate at least two active fingertips.
        fingertip_peaks = obs[:, :360].reshape(n, 5, 72).amax(-1)
        tactile_supported = (fingertip_peaks > 0.10).sum(-1) >= 2
        gate = gate & tactile_supported

    target = candidate.detach()
    # 16 dimensions retained, no high-pressure shortcut from letting go.
    per_sample = (mean_action - target).square().mean(-1)
    loss = (per_sample * gate.float()).mean()
    stats = {
        "guidance_active_fraction": float(gate.float().mean()),
        "guidance_accepted_improvement": float(
            (delta_risk * gate).sum() / gate.sum().clamp_min(1)
        ),
        "guidance_gradient_agreement": float(consistency.mean()),
        "guidance_predicted_mean_risk": float(mean_risk.detach().mean()),
        "guidance_model_disagreement": float(risk_disagreement.mean()),
    }
    return loss, stats


def ensemble_update(
    ensemble: ContactDynamicsEnsemble,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_pressure: torch.Tensor,
    next_omega: torch.Tensor,
    next_support: torch.Tensor,
    next_slip: torch.Tensor,
):
    """Bootstrap each head on independent 80% subsets; preserve head diversity."""
    model_losses = []
    n = observations.shape[0]
    if n == 0:
        return torch.zeros((), device=observations.device)
    for net in ensemble.networks:
        mask = torch.rand(n, device=observations.device) < 0.8
        if not bool(mask.any()):
            mask[0] = True
        raw = net(torch.cat((observations[mask], actions[mask]), dim=-1))
        pred_p = F.softplus(raw[..., :96])
        target_p = next_pressure[mask]
        hotspots = (target_p > 0.5).float()
        weighted_pressure = (
            F.smooth_l1_loss(pred_p, target_p, reduction="none")
            * (1.0 + 3.0 * hotspots)
        ).mean()
        omega = F.smooth_l1_loss(raw[..., 96].tanh(), next_omega[mask])
        support = F.binary_cross_entropy_with_logits(raw[..., 97], next_support[mask])
        slip = F.smooth_l1_loss(
            F.softplus(raw[..., 98]), next_slip[mask].clamp(0, 4)
        )
        model_losses.append(weighted_pressure + 0.5 * omega + 0.35 * support + 0.15 * slip)
    loss = torch.stack(model_losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(ensemble.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def validate_ensemble(
    ensemble: ContactDynamicsEnsemble,
    observations: torch.Tensor,
    actions: torch.Tensor,
    pressure: torch.Tensor,
    omega: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, float]:
    if not observations.shape[0]:
        return {"pressure_mae": float("inf"), "omega_mae": float("inf"),
                "support_mae": float("inf"), "risk_mae": float("inf")}
    p, w, s, _ = ensemble(observations, actions)
    has_pressure = pressure.amax(-1) > 0.3
    if not bool(has_pressure.any()):
        pressure_mae = float("inf")
    else:
        pressure_mae = (p[has_pressure] - pressure[has_pressure]).abs().mean().item()
    return {
        "pressure_mae": pressure_mae,
        "omega_mae": (w - omega).abs().mean().item(),
        "support_mae": (s - support).abs().mean().item(),
        "risk_mae": (pressure_risk(p) - pressure_risk(pressure)).abs().mean().item(),
    }
