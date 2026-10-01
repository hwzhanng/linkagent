from __future__ import annotations

import torch

from simulation.ppo_tomato_grasp_v3.direct_joint_teacher import DirectJointEventTeacher, DirectJointTeacherCfg
from simulation.ppo_tomato_grasp_v3.direct_joint_action import v3_direct_action_to_joint_targets


def test_direct_teacher_delays_distal_until_coarse_phase() -> None:
    teacher = DirectJointEventTeacher(1, "cpu", tuple([0.5] * 14), DirectJointTeacherCfg(coarse_steps=10, distal_steps=10))
    force = torch.zeros((1, 5))
    early = teacher.command(torch.tensor([4]), force, torch.ones(1), torch.zeros(1))
    assert torch.allclose(early[:, 8:12], teacher.open_action[:, 8:12])
    late = teacher.command(torch.tensor([15]), force, torch.ones(1), torch.zeros(1))
    assert bool((late[:, 8:12] != teacher.open_action[:, 8:12]).all())


def test_direct_teacher_high_force_retreats_only_affected_distal() -> None:
    cfg = DirectJointTeacherCfg(coarse_steps=2, distal_steps=2)
    teacher = DirectJointEventTeacher(1, "cpu", tuple([0.8] * 14), cfg)
    zero = torch.zeros((1, 5))
    teacher.command(torch.tensor([4]), zero, torch.ones(1), torch.zeros(1))
    before = teacher.current.clone()
    force = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]])
    after = teacher.command(torch.tensor([5]), force, torch.ones(1), torch.zeros(1))
    assert after[0, 8] < before[0, 8]
    assert torch.all(after[0, 9:12] >= before[0, 9:12])


def test_direct_teacher_latches_each_digit_group_independently() -> None:
    cfg = DirectJointTeacherCfg(coarse_steps=4, distal_steps=4, contact_latch_steps=2)
    teacher = DirectJointEventTeacher(1, "cpu", tuple([0.8] * 14), cfg)
    zero = torch.zeros((1, 5))
    teacher.command(torch.tensor([5]), zero, torch.ones(1), torch.zeros(1))
    contact = torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.0]])
    teacher.command(torch.tensor([6]), contact, torch.ones(1), torch.zeros(1))
    latched = teacher.command(torch.tensor([7]), contact, torch.ones(1), torch.zeros(1))
    after = teacher.command(torch.tensor([8]), contact, torch.ones(1), torch.zeros(1))
    assert torch.allclose(after[:, [4, 8]], latched[:, [4, 8]])
    assert bool((after[:, [5, 9]] != latched[:, [5, 9]]).any())


def test_direct_teacher_preloads_weak_contact_instead_of_freezing() -> None:
    cfg = DirectJointTeacherCfg(
        coarse_steps=2, distal_steps=2, contact_latch_steps=1,
        contact_detect_n=(0.001,) * 5, force_lower_n=(0.1,) * 5,
    )
    teacher = DirectJointEventTeacher(1, "cpu", tuple([0.8] * 14), cfg)
    weak = torch.tensor([[0.0, 0.01, 0.0, 0.0, 0.0]])
    first = teacher.command(torch.tensor([4]), weak, torch.ones(1), torch.zeros(1))
    second = teacher.command(torch.tensor([5]), weak, torch.ones(1), torch.zeros(1))
    assert teacher.contact_latched[0, 1]
    assert bool((second[:, [4, 8]] > first[:, [4, 8]]).all())


def test_direct_teacher_probe_baseline_updates_only_on_probe() -> None:
    cfg = DirectJointTeacherCfg(
        coarse_steps=2, distal_steps=2, contact_latch_steps=1,
        lost_contact_grace_steps=0, search_probe_steps=4,
        contact_detect_n=(0.001,) * 5,
    )
    teacher = DirectJointEventTeacher(1, "cpu", tuple([0.8] * 14), cfg)
    contact = torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.0]])
    d0 = torch.full((1, 5), 0.05)
    teacher.command(torch.tensor([4]), contact, torch.ones(1), torch.zeros(1), d0)
    baseline = teacher.probe_tip_distance.clone()
    teacher.command(torch.tensor([5]), torch.zeros((1, 5)), torch.ones(1), torch.zeros(1), d0 + 0.01)
    assert torch.equal(teacher.probe_tip_distance, baseline)


def test_direct_teacher_latch_uses_measured_posture_not_lagging_command() -> None:
    cfg = DirectJointTeacherCfg(
        coarse_steps=2, distal_steps=2, contact_latch_steps=1,
        contact_detect_n=(0.001,) * 5,
    )
    teacher = DirectJointEventTeacher(1, "cpu", tuple([0.8] * 14), cfg)
    teacher.current[:] = 0.8
    measured_joint = v3_direct_action_to_joint_targets(teacher.open_action)
    expected = teacher.open_action.clone()
    contact = torch.tensor([[0.0, 0.01, 0.0, 0.0, 0.0]])
    teacher.command(
        torch.tensor([4]), contact, torch.ones(1), torch.zeros(1),
        torch.full((1, 5), 0.05), measured_joint,
    )
    assert torch.allclose(teacher.latch_reference[:, [4, 8]], expected[:, [4, 8]], atol=1e-6)


def _stagger_teacher(starts, num_envs=1):
    cfg = DirectJointTeacherCfg(
        coarse_steps=50,
        distal_steps=50,
        digit_start_steps=starts,
    )
    return DirectJointEventTeacher(num_envs, "cpu", tuple([0.6] * 14), cfg)


def _drive(teacher, steps, forces=None):
    if forces is None:
        forces = torch.zeros((teacher.num_envs, 5))
    palm = torch.ones(teacher.num_envs)
    speed = torch.zeros(teacher.num_envs)
    tip = torch.full((teacher.num_envs, 5), 0.05)
    joints = torch.full((teacher.num_envs, 16), 0.02)
    outs = []
    for s in range(steps):
        out = teacher.command(
            torch.tensor([s] * teacher.num_envs),
            forces.clone(),
            palm,
            speed,
            tip,
            joints,
        )
        outs.append(out.clone())
    return outs


def test_digit_stagger_keeps_unstarted_digits_inert() -> None:
    teacher = _stagger_teacher((0, 300, 600, 900, 1200))
    outs = _drive(teacher, 120)
    final = outs[-1]
    for ch in (4, 8, 5, 9, 6, 10, 7, 11):
        assert torch.equal(final[:, ch], teacher.open_action[:, ch])
    assert not bool(teacher.contact_latched[:, 1:].any())
    assert not bool(teacher.contact_age[:, 1:].any())


def test_digit_stagger_enters_hold_per_digit_schedule() -> None:
    teacher = _stagger_teacher((0, 100, 100, 100, 100))
    outs = _drive(teacher, 260)
    at80 = outs[80]
    # digit 0 (thumb, start 0) is mid-distal-ramp at global step 80
    assert not torch.equal(at80[:, 0], teacher.open_action[:, 0])
    # digit 1 (index, start 100) must still hold exactly the open pose
    assert torch.equal(at80[:, 4], teacher.open_action[:, 4])
    at259 = outs[-1]
    assert not torch.equal(at259[:, 0], teacher.open_action[:, 0])
    assert not torch.equal(at259[:, 4], teacher.open_action[:, 4])


def test_digit_stagger_no_cross_digit_state_contamination() -> None:
    teacher = _stagger_teacher((0, 100, 100, 100, 100), num_envs=2)
    latch_forces = torch.zeros((2, 5))
    latch_forces[0, 0] = 2.0
    outs = _drive(teacher, 200, forces=latch_forces)
    for out in outs:
        for ch in (4, 8):
            assert torch.equal(out[0, ch], out[1, ch])
    assert bool(teacher.contact_latched[0, 0])
    assert not bool(teacher.contact_latched[1, 0])
