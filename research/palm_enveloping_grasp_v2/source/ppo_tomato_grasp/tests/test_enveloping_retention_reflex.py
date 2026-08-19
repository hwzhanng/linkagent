import torch

from simulation.ppo_tomato_grasp.retention_reflex import adaptive_retention_correction


def _kwargs(force: torch.Tensor, speed: torch.Tensor, in_hand: torch.Tensor, palm: torch.Tensor):
    return dict(
        filtered_force_n=force,
        target_force_n=torch.tensor([0.55, 0.42, 0.42, 0.36, 0.36]),
        high_force_n=torch.tensor([2.8, 2.2, 2.2, 2.0, 2.0]),
        relative_speed_m_s=speed,
        in_hand=in_hand,
        palm_support=palm,
        contact_threshold_n=0.04,
        force_gain_rad_per_n=0.012,
        slip_gain_rad_per_m_s=0.018,
        slip_threshold_m_s=0.04,
        max_correction_rad=0.018,
    )


def test_reflex_is_zero_after_object_leaves_workspace():
    force = torch.full((2, 5), 0.10)
    result = adaptive_retention_correction(
        **_kwargs(force, torch.tensor([0.2, 0.2]), torch.tensor([False, False]), torch.tensor([False, False]))
    )
    assert torch.equal(result, torch.zeros_like(result))


def test_reflex_curls_missing_digits_and_is_bounded():
    force = torch.zeros((1, 5))
    result = adaptive_retention_correction(
        **_kwargs(force, torch.tensor([0.20]), torch.tensor([True]), torch.tensor([False]))
    )
    assert torch.all(result > 0.0)
    assert float(result.abs().max()) <= 0.018 + 1.0e-7


def test_reflex_relaxes_only_high_force_digit():
    force = torch.tensor([[3.5, 0.0, 0.0, 0.0, 0.0]])
    result = adaptive_retention_correction(
        **_kwargs(force, torch.tensor([0.0]), torch.tensor([True]), torch.tensor([False]))
    )
    assert result[0, 0] < 0.0
    assert torch.all(result[0, 1:] > 0.0)
