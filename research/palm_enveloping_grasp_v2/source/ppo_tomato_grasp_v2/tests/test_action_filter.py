import torch

from simulation.ppo_tomato_grasp_v2.action_filter import filter_distal_residual
from simulation.ppo_tomato_grasp.enveloping_action_manifold import distal_residual_jacobian


def test_filter_preserves_root_channels_and_smooths_only_distal():
    previous = torch.zeros(2, 4)
    action = torch.ones(2, 14)
    filtered = filter_distal_residual(previous, action, alpha=0.75)
    assert torch.allclose(filtered[:, :10], torch.ones(2, 10))
    assert torch.allclose(filtered[:, 10:], torch.full((2, 4), 0.25))


def test_filter_accepts_per_digit_alpha():
    previous = torch.zeros(1, 4)
    action = torch.ones(1, 14)
    filtered = filter_distal_residual(previous, action, alpha=torch.tensor([0.0, 0.25, 0.5, 0.75]))
    assert torch.allclose(filtered[0, 10:], torch.tensor([1.0, 0.75, 0.5, 0.25]))


def test_distal_jacobian_has_no_root_cross_motion():
    action = torch.zeros(3, 14)
    jac = distal_residual_jacobian(action)
    assert torch.count_nonzero(jac[..., :10]).item() == 0
    assert torch.count_nonzero(jac[..., 10:]).item() == 12
