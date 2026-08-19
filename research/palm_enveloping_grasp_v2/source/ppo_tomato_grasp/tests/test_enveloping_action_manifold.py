import torch

from simulation.ppo_tomato_grasp.enveloping_action_manifold import (
    ENVELOPING_ACTION_DIM,
    STRAIGHT_FINGER_DISTAL_JOINT_INDICES,
    distal_residual_jacobian,
    enveloping_action_to_joint_targets,
)


def test_distal_residuals_move_only_distal_joints() -> None:
    action = torch.zeros((1, ENVELOPING_ACTION_DIM), dtype=torch.float32)
    action[0, 10:] = torch.tensor((1.0, -1.0, 0.5, -0.5))
    baseline = enveloping_action_to_joint_targets(action * 0.0)
    targets = enveloping_action_to_joint_targets(action)
    delta = targets - baseline
    for joint in range(16):
        if joint not in STRAIGHT_FINGER_DISTAL_JOINT_INDICES:
            assert torch.allclose(delta[..., joint], torch.zeros(1), atol=1e-7)
    assert (delta[..., list(STRAIGHT_FINGER_DISTAL_JOINT_INDICES)].abs() > 1e-4).all()


def test_distal_jacobian_is_diagonal() -> None:
    action = torch.zeros((2, ENVELOPING_ACTION_DIM), dtype=torch.float32)
    jac = distal_residual_jacobian(action)
    assert jac.shape == (2, 4, 16)
    for row, joint in enumerate(STRAIGHT_FINGER_DISTAL_JOINT_INDICES):
        nonzero = (jac[:, row].abs() > 0).nonzero(as_tuple=False)
        assert set(nonzero[:, 1].tolist()) == {joint}


def test_bounds_are_respected() -> None:
    action = torch.full((4, ENVELOPING_ACTION_DIM), 10.0)
    targets = enveloping_action_to_joint_targets(action)
    from simulation.ppo_tomato_grasp.hand_manifold import LEFT_SAFE_LOWER_RAD, LEFT_SAFE_UPPER_RAD

    lower = torch.tensor(LEFT_SAFE_LOWER_RAD)
    upper = torch.tensor(LEFT_SAFE_UPPER_RAD)
    assert bool((targets >= lower).all())
    assert bool((targets <= upper).all())
