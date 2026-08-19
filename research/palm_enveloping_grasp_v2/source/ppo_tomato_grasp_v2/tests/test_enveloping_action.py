import torch

from simulation.ppo_tomato_grasp.enveloping_action_manifold import (
    enveloping_action_to_joint_targets,
)


def test_v2_root_span_none_preserves_default_mapper():
    action = torch.linspace(-0.8, 0.8, 14).reshape(1, 14)
    baseline = enveloping_action_to_joint_targets(
        action, residual_span_fraction=0.12, distal_residual_span_fraction=0.50
    )
    explicit_default = enveloping_action_to_joint_targets(
        action,
        residual_span_fraction=0.12,
        distal_residual_span_fraction=0.50,
        per_digit_root_residual_span_fraction=None,
    )
    assert torch.equal(baseline, explicit_default)


def test_v2_root_span_only_widens_straight_finger_residuals():
    action = torch.zeros((1, 14))
    action[:, 6:10] = 1.0
    narrow = enveloping_action_to_joint_targets(
        action, residual_span_fraction=0.12, distal_residual_span_fraction=0.50
    )
    wide = enveloping_action_to_joint_targets(
        action,
        residual_span_fraction=0.12,
        distal_residual_span_fraction=0.50,
        per_digit_root_residual_span_fraction=0.40,
    )
    # Thumb and straight-finger distal residual joints are unchanged; only
    # each straight finger's root/proximal axis receives the wider span.
    assert torch.equal(narrow[:, (0, 11)], wide[:, (0, 11)])
    assert torch.equal(narrow[:, 12:16], wide[:, 12:16])
    assert torch.all(wide[:, 1:5] >= narrow[:, 1:5])
    assert torch.any(wide[:, 1:5] > narrow[:, 1:5])
