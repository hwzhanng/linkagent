from __future__ import annotations

import torch

from simulation.ppo_tomato_grasp_v2.residual_controller import ResidualBaseControllerCfg
from simulation.ppo_tomato_grasp_v3.event_teacher import EventDrivenEnclosureTeacher, EventDrivenTeacherCfg


TARGET = (-0.12, -1.0, 0.75, -0.06, 0.30, -0.21, 0.38, 1.0, 0.28, -0.09, -0.31, 0.48, 0.16, 0.63)


def make_teacher() -> EventDrivenEnclosureTeacher:
    return EventDrivenEnclosureTeacher(
        1, "cpu", TARGET,
        ResidualBaseControllerCfg(approach_steps=2, per_digit_steps=2),
        EventDrivenTeacherCfg(reacquire_step=0.01, trust_region=0.10),
    )


def test_teacher_latches_only_on_simultaneous_physical_contact() -> None:
    teacher = make_teacher()
    step = torch.tensor([4])
    four = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.0]])
    teacher.command(step, four, torch.tensor([1.0]), torch.tensor([0.0]))
    assert not bool(teacher.latched[0])
    assert int(teacher.digit_latched[0].sum()) == 4
    five = torch.full((1, 5), 0.2)
    teacher.command(step, five, torch.tensor([1.0]), torch.tensor([0.0]))
    assert bool(teacher.latched[0])
    assert int(teacher.latch_step[0]) == 4


def test_missing_straight_digit_moves_only_inside_trust_region() -> None:
    teacher = make_teacher()
    step = torch.tensor([4])
    five = torch.full((1, 5), 0.2)
    teacher.command(step, five, torch.tensor([1.0]), torch.tensor([0.0]))
    anchor = teacher.anchor.clone()
    missing_index = five.clone()
    missing_index[:, 1] = 0.0
    for _ in range(30):
        action = teacher.command(step, missing_index, torch.tensor([1.0]), torch.tensor([0.0]))
    assert action[0, 6] > anchor[0, 6]
    assert action[0, 10] < anchor[0, 10]
    assert torch.all((action - anchor).abs() <= 0.100001)


def test_reset_clears_event_state() -> None:
    teacher = make_teacher()
    teacher.command(torch.tensor([4]), torch.full((1, 5), 0.2), torch.tensor([1.0]), torch.tensor([0.0]))
    teacher.reset(torch.tensor([0]))
    assert not bool(teacher.latched[0])
    assert not bool(teacher.digit_latched.any())
    assert int(teacher.latch_step[0]) == -1
    assert not bool(teacher.high_age.any())
    assert not bool(teacher.previous_force.any())


def test_single_high_force_frame_does_not_retreat() -> None:
    teacher = make_teacher()
    step = torch.tensor([4])
    nominal = torch.full((1, 5), 0.2)
    teacher.command(step, nominal, torch.tensor([1.0]), torch.tensor([0.0]))
    before = teacher.current.clone()
    spike = nominal.clone()
    spike[:, 2] = 2.0
    teacher.command(step, spike, torch.tensor([1.0]), torch.tensor([0.0]))
    assert torch.equal(teacher.current[:, (7, 11)], before[:, (7, 11)])


def test_falling_force_is_recovered_predictively() -> None:
    teacher = make_teacher()
    step = torch.tensor([4])
    nominal = torch.full((1, 5), 0.3)
    teacher.command(step, nominal, torch.tensor([1.0]), torch.tensor([0.0]))
    before = teacher.current.clone()
    falling = nominal.clone()
    falling[:, 2] = 0.16
    teacher.command(step, falling, torch.tensor([1.0]), torch.tensor([0.0]))
    assert teacher.current[0, 11] > before[0, 11]


def test_high_palm_share_recovers_all_under_target_digits() -> None:
    teacher = make_teacher()
    step = torch.tensor([4])
    nominal = torch.full((1, 5), 0.2)
    teacher.command(step, nominal, torch.tensor([1.0]), torch.tensor([0.0]))
    before = teacher.current.clone()
    underloaded = torch.tensor([[0.20, 0.05, 0.06, 0.20, 0.20]])
    teacher.command(step, underloaded, torch.tensor([2.0]), torch.tensor([0.0]))
    assert teacher.current[0, 6] > before[0, 6]
    assert teacher.current[0, 11] > before[0, 11]
