from __future__ import annotations

import torch

from research.small_fruit_pinch_v1.source.verification_geometry import contact_topology


def test_opposed_spread_contacts_pass_geometry_gate() -> None:
    points = torch.tensor([[[-0.034, 0.0, 0.0], [0.030, 0.016, 0.0], [0.030, -0.016, 0.0]]])
    result = contact_topology(points, torch.zeros((1, 3)), torch.tensor([0.034]))
    assert result.valid.tolist() == [True]
    assert result.opposition_angle_deg.item() == 0.0


def test_three_contacts_on_same_side_do_not_prove_opposition() -> None:
    points = torch.tensor([[[0.034, 0.0, 0.0], [0.030, 0.016, 0.0], [0.030, -0.016, 0.0]]])
    result = contact_topology(points, torch.zeros((1, 3)), torch.tensor([0.034]))
    assert result.valid.tolist() == [False]


def test_missing_or_coincident_contact_points_fail_closed() -> None:
    points = torch.tensor([
        [[float("nan"), 0.0, 0.0], [0.030, 0.016, 0.0], [0.030, -0.016, 0.0]],
        [[-0.034, 0.0, 0.0], [0.034, 0.0, 0.0], [0.034, 0.0, 0.0]],
    ])
    result = contact_topology(points, torch.zeros((2, 3)), torch.tensor([0.034, 0.034]))
    assert result.valid.tolist() == [False, False]
