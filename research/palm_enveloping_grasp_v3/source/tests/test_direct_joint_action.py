from __future__ import annotations

import torch

from simulation.ppo_tomato_grasp_v3.direct_joint_action import (
    joint_targets_to_v3_direct_action,
    v3_direct_action_to_joint_targets,
)


def test_direct_contract_controls_twelve_flexion_axes_independently() -> None:
    action = torch.zeros((1, 14))
    base = v3_direct_action_to_joint_targets(action)
    for channel in range(12):
        changed = action.clone()
        changed[0, channel] = 0.4
        delta = (v3_direct_action_to_joint_targets(changed) - base).abs() > 1.0e-7
        assert int(delta.sum()) == 1


def test_direct_contract_inverse_is_exact_for_flexion_axes() -> None:
    generator = torch.Generator().manual_seed(7)
    action = 1.6 * torch.rand((8, 14), generator=generator) - 0.8
    target = v3_direct_action_to_joint_targets(action)
    recovered = joint_targets_to_v3_direct_action(target)
    remapped = v3_direct_action_to_joint_targets(recovered)
    direct_joint_ids = [0, 5, 10, 11, 1, 2, 3, 4, 12, 13, 14, 15]
    assert torch.allclose(remapped[:, direct_joint_ids], target[:, direct_joint_ids], atol=1.0e-6)


def test_direct_contract_clamps_and_rejects_bad_shape() -> None:
    target = v3_direct_action_to_joint_targets(torch.full((2, 14), 8.0))
    assert torch.isfinite(target).all()
    try:
        v3_direct_action_to_joint_targets(torch.zeros((1, 13)))
    except ValueError as error:
        assert "14" in str(error)
    else:
        raise AssertionError("invalid direct action shape was accepted")
