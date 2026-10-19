"""CPU semantics for the conservative third-load margin bonus."""

from __future__ import annotations

import pytest
import torch

from research.firm_grasp_v1.reward_margin import third_load_margin_reward


@pytest.mark.parametrize(
    ("third_n", "expected"),
    [(0.14, 0.0), (0.15, 0.0), (0.225, 1.0), (0.30, 2.0), (0.60, 2.0)],
)
def test_third_load_margin_ramp_and_cap(third_n, expected):
    pad = torch.tensor([[0.4, third_n, third_n, 0.0, 0.0]], dtype=torch.float32)
    reward = third_load_margin_reward(
        pad,
        torch.tensor([True]),
        torch.zeros((1, 5), dtype=torch.bool),
        torch.zeros((1, 5)),
        torch.tensor([False]),
        torch.tensor([False]),
    )

    assert reward.device.type == "cpu"
    assert reward.tolist() == pytest.approx([expected])


def test_margin_requires_clean_real_contact_and_no_hard_or_drop():
    pad = torch.tensor([[0.4, 0.30, 0.30, 0.0, 0.0]] * 6, dtype=torch.float32)
    thumb_ok = torch.tensor([True, False, True, True, True, True])
    unknown = torch.zeros((6, 5), dtype=torch.bool)
    unknown[2, 1] = True
    nonpad = torch.zeros((6, 5))
    nonpad[3, 2] = 1e-6
    hard = torch.tensor([False, False, False, False, True, False])
    dropped = torch.tensor([False, False, False, False, False, True])

    reward = third_load_margin_reward(pad, thumb_ok, unknown, nonpad, hard, dropped)

    assert reward.tolist() == pytest.approx([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
