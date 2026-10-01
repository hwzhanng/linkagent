from __future__ import annotations

import torch

from simulation.ppo_tomato_grasp_v3.mimic import advance_mimic_positions, apply_urdf_mimic_targets


def test_urdf_mimic_equations_and_limits() -> None:
    names = [
        "thumb_joint3", "thumb_joint4", "index_joint2", "index_joint3",
        "middle_joint2", "middle_joint3", "ring_joint2", "ring_joint3",
        "little_joint2", "little_joint3",
    ]
    targets = torch.tensor([[0.50, 0.0, 0.60, 0.0, 0.70, 0.0, 0.80, 0.0, 0.01, 0.0]])
    lower = torch.zeros_like(targets)
    upper = torch.ones_like(targets)
    result = apply_urdf_mimic_targets(targets, names, lower, upper)
    assert torch.isclose(result[0, 1], torch.tensor(1.11 * 0.50 - 0.04545))
    assert torch.isclose(result[0, 3], torch.tensor(0.94 * 0.60 - 0.04545))
    assert torch.isclose(result[0, 5], torch.tensor(0.94 * 0.70 - 0.04545))
    assert torch.isclose(result[0, 7], torch.tensor(0.94 * 0.80 - 0.04545))
    assert result[0, 9] == 0.0


def test_urdf_mimic_rejects_missing_joint() -> None:
    targets = torch.zeros((1, 2))
    limits = torch.ones((1, 2))
    try:
        apply_urdf_mimic_targets(targets, ["thumb_joint3", "thumb_joint4"], -limits, limits)
    except ValueError as error:
        assert "missing URDF mimic joints" in str(error)
    else:
        raise AssertionError("missing mimic joint was accepted")


def test_mimic_position_servo_only_moves_selected_joints() -> None:
    current = torch.tensor([[0.2, 0.0, 0.4]])
    desired = torch.tensor([[0.7, 0.8, -0.4]])
    result, delta = advance_mimic_positions(current, desired, [1], 0.1)
    assert torch.allclose(result, torch.tensor([[0.2, 0.1, 0.4]]))
    assert torch.allclose(delta, torch.tensor([[0.1]]))


def test_mimic_position_servo_rejects_nonpositive_step() -> None:
    values = torch.zeros((1, 2))
    try:
        advance_mimic_positions(values, values, [1], 0.0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive mimic servo step was accepted")
