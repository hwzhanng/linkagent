"""Evidence that pressure-derived action gradients update the PPO actor.

Tests an analytic, physically plausible action->pressure map. Random dynamics
models alone cannot demonstrate that stress guidance is actually effective.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from simulation.fruit_rotation.core import POLICY_DIM
from simulation.fruit_rotation.ppo import ActorCritic, DualState, PPOSettings
from simulation.fruit_rotation.pressure_guidance import (
    ContactDynamicsEnsemble, GuidanceConfig, project_task_tangent,
    stress_guidance_loss, ensemble_update, validate_ensemble,
)


class AnalyticContactDynamics(nn.Module):
    """Three agreeing pressure maps with orthogonal spin/support/slip gradients."""
    def __init__(self, opposite: bool = False):
        super().__init__()
        self.opposite = opposite

    def forward_members(self, observation, action):
        n = action.shape[0]
        p, w, s, slip = [], [], [], []
        for m in range(3):
            sign = -1 if self.opposite and m == 2 else 1
            load = (
                .94 + .20 * sign * action[:, 0] + .10 * action[:, 1]
                + (m - 1) * .004
            )
            p.append(load[:, None].expand(n, 96))
            w.append(.5 + .30 * action[:, 2])
            s.append((.85 + .02 * action[:, 3]).clamp(.01, .99))
            slip.append((.05 + .02 * action[:, 4]).clamp_min(0.0))
        return torch.stack(p), torch.stack(w), torch.stack(s), torch.stack(slip)


def realistic_sparse_obs(n=4):
    obs = torch.zeros((n, POLICY_DIM))
    obs[:, :72] = .45
    obs[:, 72:144] = .55
    obs[:, 144:216] = .40
    obs[:, 408:432] = 1.0
    return obs


def test_multiconstraint_tangent_projection_preserves_three_task_directions():
    gradients = torch.tensor([[3., 2., 5., -2.]])
    j = torch.tensor([[
        [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
    ]])
    projected = project_task_tangent(gradients, j, damping=1e-5)
    assert projected.shape == gradients.shape
    assert abs(projected[0, 0]) < .001
    assert abs(projected[0, 1]) < .001
    assert abs(projected[0, 2]) < .001
    assert abs(projected[0, 3] + 2) < .001


def test_counterfactual_pressure_descent_actually_updates_actor():
    torch.manual_seed(21)
    actor = ActorCritic()
    observations = realistic_sparse_obs()
    action_mean = actor.mean(observations).tanh()
    loss, stats = stress_guidance_loss(
        action_mean, observations, AnalyticContactDynamics(), GuidanceConfig()
    )
    assert stats["guidance_active_fraction"] == 1.0
    assert stats["guidance_accepted_improvement"] > 0.001
    assert stats["guidance_gradient_agreement"] > 0.99
    assert loss.item() > 0
    actor.zero_grad(set_to_none=True)
    loss.backward()
    grad = actor.actor_mean.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0, "Stress signal failed to reach Actor"


def test_no_tactile_contact_cannot_farm_pressure_gradient_reward():
    action = torch.zeros(4, 16, requires_grad=True)
    observation = realistic_sparse_obs()
    observation[:, :360] = 0
    loss, stats = stress_guidance_loss(
        action, observation, AnalyticContactDynamics(), GuidanceConfig()
    )
    assert stats["guidance_active_fraction"] == 0
    assert loss.item() == 0


def test_opposed_models_disable_unsafe_gradient():
    action = torch.zeros(4, 16, requires_grad=True)
    _, stats = stress_guidance_loss(
        action, realistic_sparse_obs(),
        AnalyticContactDynamics(opposite=True), GuidanceConfig()
    )
    assert stats["guidance_active_fraction"] == 0


def test_ensemble_predictor_trains_on_measured_transition_targets():
    torch.manual_seed(42)
    ensemble = ContactDynamicsEnsemble(members=3)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
    observations = realistic_sparse_obs(20)
    action = torch.randn(20, 16) * .1
    pressure = torch.ones(20, 96) * 0.92
    omega = torch.ones(20) * 0.12
    support = torch.ones(20)
    slip = torch.zeros(20)
    pre = validate_ensemble(ensemble, observations, action, pressure, omega, support)
    loss = ensemble_update(
        ensemble, optimizer, observations, action, pressure,
        omega, support, slip
    )
    post = validate_ensemble(ensemble, observations, action, pressure, omega, support)
    assert math.isfinite(float(loss))
    assert all(math.isfinite(value) for value in post.values())
    assert sum(math.isfinite(value) for value in pre.values()) == 4


def test_primal_dual_budget_does_not_claim_hard_safety_guarantee():
    settings = PPOSettings(dual_initial=.1, dual_max=.5, dual_lr=.5)
    dual = DualState(settings.dual_initial)
    dual.update(settings.pressure_cost_budget + 2., settings)
    assert dual.multiplier == settings.dual_max
    dual.update(0., settings)
    assert 0 <= dual.multiplier < settings.dual_max
    settings.disable_pressure_constraint = True
    dual.update(99., settings)
    assert dual.multiplier == 0
