"""CPU checks for PPO/pressure math. Run: python -m pytest tests/test_fruit_rotation_core.py -q"""
import math

import torch

from simulation.fruit_rotation.core import (
    ACTION_DIM, ACTIVE_JOINTS, CRITIC_DIM, POLICY_DIM, POLICY_SLICES,
    pressure_from_contacts, pressure_risk, rotation_reward, surface_sites,
    tangent_project, fingertip_taxels,
)
from simulation.fruit_rotation.ppo import (
    ActorCritic, ContactDynamics, PPOSettings, auxiliary_loss, compute_gae,
    ppo_update,
)


def test_layouts_and_full_tip_maps():
    assert POLICY_DIM == 502 and CRITIC_DIM == 622
    assert ACTION_DIM == len(ACTIVE_JOINTS) == 16
    assert POLICY_SLICES["tactile"] == slice(0, 360)
    assert POLICY_SLICES["sparse_valid"] == slice(408, 432)
    force = torch.tensor([[1., 0.5, 0, 0, 0]])
    taxels = fingertip_taxels(force, torch.zeros(1, 5, 2))
    assert taxels.shape == (1, 5, 12, 6)
    assert taxels[0, 0].sum() > taxels[0, 1].sum()


def test_six_sensor_panels_and_dense_surface():
    dense, sparse = surface_sites("cpu")
    assert dense.shape == (96, 3) and sparse.shape == (24, 3)
    assert torch.allclose(dense.norm(dim=-1), torch.ones(96), atol=1e-6)
    assert torch.allclose(sparse.norm(dim=-1), torch.ones(24), atol=1e-6)


def test_contact_pressure_spatial_response():
    dense, _ = surface_sites("cpu")
    p = torch.zeros(2, 3)
    q = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
    axes = torch.ones(2, 3) * .032
    force = torch.tensor([[1., 0, 0, 0, 0, 0], [2., 0, 0, 0, 0, 0]])
    location = torch.zeros(2, 6, 3)
    location[:, 0, 0] = .032
    output = pressure_from_contacts(force, location, p, q, axes, dense)
    assert output.shape == (2, 96)
    assert torch.isfinite(output).all() and (output >= 0).all()
    assert torch.allclose(output[1], 2.0 * output[0], atol=.001)
    assert float(output[0].max()) < 1.0 / (2.0 * math.pi * .01 ** 2)


def test_risk_and_projection():
    low = torch.full((2, 96), .25)
    high = low.clone()
    high[:, :12] = 1.5
    assert pressure_risk(low).sum() == 0
    assert pressure_risk(high).min() > 0
    gd = torch.tensor([[3., 2., 1.], [-1., 1., 2.]])
    gw = torch.tensor([[1., 0., 0.], [0., 2., 0.]])
    safe = tangent_project(gd, gw, damping=0)
    assert torch.allclose((safe * gw).sum(-1), torch.zeros(2), atol=1e-6)


def test_reward_rotation_exceeds_stillness():
    args = dict(
        desired_speed=torch.ones(2) * .6,
        force_n=torch.ones((2, 6)),
        dense_pressure=torch.zeros((2, 96)),
        translation_error=torch.zeros(2),
        tilt_error=torch.zeros(2),
        slip_speed=torch.zeros(2),
        actions=torch.zeros(2, 16),
        previous_actions=torch.zeros(2, 16), dt=.05,
    )
    steady, _ = rotation_reward(torch.zeros(2), **args)
    spinning, _ = rotation_reward(torch.ones(2) * .6, **args)
    assert torch.all(spinning > steady)


def test_actor_predictor_and_aux_gradients():
    torch.manual_seed(7)
    actor = ActorCritic()
    dynamics = ContactDynamics()
    obs = torch.randn(4, POLICY_DIM) * .2
    critic = torch.randn(4, CRITIC_DIM) * .2
    action, latent, logp, value = actor.act(obs, critic)
    assert action.shape == (4, 16) and latent.shape == (4, 16)
    assert logp.shape == (4,) and value.shape == (4,)
    pred, spin = dynamics(obs, action)
    assert pred.shape == (4, 96) and spin.shape == (4,)
    loss, _ = auxiliary_loss(actor, dynamics, obs, PPOSettings())
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(actor.actor_mean.weight.grad).all()
    assert dynamics.network[0].weight.grad is None


def test_gae_terminal_no_bootstrap():
    reward = torch.tensor([[1., 2.], [3., 4.]])
    done = torch.tensor([[0, 0], [1, 0]], dtype=torch.bool)
    value = torch.zeros_like(reward)
    final = torch.ones(2) * 10
    advantages, returns = compute_gae(reward, done, value, final, gamma=0.9, lam=1.0)
    assert torch.allclose(returns[:, 0], torch.tensor([3.7, 3.0]))
    assert torch.allclose(returns[:, 1], torch.tensor([14.54, 13.0]))


def test_predictor_and_ppo_update_cpu():
    torch.manual_seed(3)
    settings = PPOSettings(epochs=1, minibatches=2, minimum_samples_for_aux=999999)
    actor = ActorCritic()
    predictor = ContactDynamics()
    optim = torch.optim.Adam(actor.parameters(), lr=settings.learning_rate)
    model_optim = torch.optim.Adam(predictor.parameters(), lr=settings.predictor_learning_rate)
    batch = 8
    policy = torch.randn(2, batch, POLICY_DIM) * 0.1
    critic = torch.randn(2, batch, CRITIC_DIM) * 0.1
    with torch.no_grad():
        action, latent, lp, value = actor.act(policy.reshape(-1, POLICY_DIM), critic.reshape(-1, CRITIC_DIM))
    data = {
        "policy": policy, "critic": critic,
        "latent": latent.reshape(2, batch, ACTION_DIM),
        "logp": lp.reshape(2, batch),
        "value": value.reshape(2, batch),
        "reward": torch.ones(2, batch) * .1,
        "done": torch.zeros(2, batch, dtype=torch.bool),
        "next_pressure": torch.ones(2, batch, 96) * .3,
        "next_omega": torch.ones(2, batch) * .1,
        "predictor_mask": torch.ones(2, batch, dtype=torch.bool),
        "advantages": torch.randn(2, batch),
        "returns": torch.randn(2, batch),
    }
    stats = ppo_update(actor, predictor, optim, model_optim, data, settings, prior_samples=0)
    assert stats["ppo_minibatch_updates"] >= 1
    assert math.isfinite(stats["predictor_loss"])
    assert not stats["aux_enabled"]
