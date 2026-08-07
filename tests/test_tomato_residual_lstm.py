from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from simulation.tomato_pick.isaac.agents.residual_lstm_model import ResidualLSTMModel


def _obs(batch: int) -> TensorDict:
    return TensorDict({"policy": torch.randn(batch, 66)}, batch_size=[batch])


def _distribution() -> dict[str, object]:
    return {"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "log"}


def _models(batch: int = 8) -> tuple[MLPModel, ResidualLSTMModel]:
    obs = _obs(batch)
    groups = {"actor": ["policy"]}
    baseline = MLPModel(
        obs,
        groups,
        "actor",
        11,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution(),
    )
    residual = ResidualLSTMModel(
        obs,
        groups,
        "actor",
        11,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution(),
    )
    source = baseline.state_dict()
    target = residual.state_dict()
    for name, value in source.items():
        target[name] = value.clone()
    residual.load_state_dict(target)
    return baseline, residual


def test_zero_residual_exactly_preserves_mlp_across_time() -> None:
    baseline, residual = _models()
    for _ in range(5):
        obs = _obs(8)
        expected = baseline(obs)
        actual = residual(obs)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    hidden, cell = residual.get_hidden_state()
    assert hidden.abs().max() > 0.0
    assert cell.abs().max() > 0.0
    assert not residual.mlp[0].weight.requires_grad


def test_done_mask_only_resets_finished_hidden_states() -> None:
    _, residual = _models(batch=4)
    residual(_obs(4))
    dones = torch.tensor([False, True, False, True])
    residual.reset(dones)
    hidden, cell = residual.get_hidden_state()
    assert torch.count_nonzero(hidden[:, dones, :]) == 0
    assert torch.count_nonzero(cell[:, dones, :]) == 0
    assert torch.count_nonzero(hidden[:, ~dones, :]) > 0
    assert torch.count_nonzero(cell[:, ~dones, :]) > 0


def test_zero_head_opens_gradient_path_without_moving_base() -> None:
    _, residual = _models()
    optimizer = torch.optim.Adam(
        [parameter for parameter in residual.parameters() if parameter.requires_grad], lr=1.0e-3
    )
    target = torch.randn(8, 11)

    loss = (residual(_obs(8)) - target).square().mean()
    loss.backward()
    assert residual.residual_head.weight.grad is not None
    assert residual.residual_head.weight.grad.abs().max() > 0.0
    assert residual.rnn.rnn.weight_ih_l0.grad is not None
    assert residual.rnn.rnn.weight_ih_l0.grad.abs().max() == 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    residual.reset()

    loss = (residual(_obs(8)) - target).square().mean()
    loss.backward()
    assert residual.rnn.rnn.weight_ih_l0.grad is not None
    assert residual.rnn.rnn.weight_ih_l0.grad.abs().max() > 0.0
