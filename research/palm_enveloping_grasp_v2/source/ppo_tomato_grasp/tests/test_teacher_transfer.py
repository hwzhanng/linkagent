import pytest
import torch

from simulation.ppo_tomato_grasp.teacher_transfer import (
    hand_actor_to_teacher_observation,
    teacher_action_to_hand_action,
)


def test_common_actor_fields_map_to_clean_v5_contract():
    actor = torch.zeros((2, 110))
    actor[:, 0:3] = torch.tensor((0.5, -0.5, 0.25))
    actor[:, 3:6] = 0.4
    actor[:, 18:34] = 0.2
    actor[:, 34:50] = -0.3
    actor[:, 50:100:10] = torch.tensor((0.1, 0.2, 0.3, 0.4, 0.5))
    actor[:, 100:105] = torch.tensor((1.0, 0.5, 0.0, -0.5, -1.0))
    teacher = hand_actor_to_teacher_observation(actor)
    assert teacher.shape == (2, 66)
    assert torch.allclose(teacher[0, 0:3], actor[0, 0:3] * (0.08 / 0.15))
    assert torch.allclose(teacher[:, 15:31], actor[:, 18:34])
    assert torch.allclose(teacher[:, 47:52], actor[:, 50:100:10])
    assert torch.allclose(teacher[:, 61:66], actor[:, 100:105])


def test_teacher_arm_action_is_discarded_and_residuals_start_zero():
    source = torch.linspace(-2.0, 2.0, 11).reshape(1, 11)
    mapped = teacher_action_to_hand_action(source)
    assert mapped.shape == (1, 10)
    assert torch.allclose(mapped[:, :5], source[:, 6:11].clamp(-1.0, 1.0))
    assert torch.count_nonzero(mapped[:, 5:]) == 0


def test_transfer_rejects_nonfinite_or_wrong_contract():
    with pytest.raises(ValueError):
        hand_actor_to_teacher_observation(torch.zeros(1, 109))
    bad = torch.zeros(1, 11)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError):
        teacher_action_to_hand_action(bad)
