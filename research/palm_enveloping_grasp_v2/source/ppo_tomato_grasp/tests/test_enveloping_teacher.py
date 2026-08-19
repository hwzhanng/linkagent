import torch

from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher, EnvelopingTeacherCfg


def test_teacher_waits_for_palm_then_releases_distal_stage() -> None:
    teacher = EnvelopingTeacher(
        2,
        "cpu",
        EnvelopingTeacherCfg(root_phase_steps=2, proximal_phase_steps=2, distal_phase_steps=2, support_wait_steps=2),
    )
    forces = torch.zeros((2, 5))
    palm = torch.zeros(2)
    assert torch.allclose(teacher.command(forces, palm), teacher.open_action)
    palm[:] = 1.0
    teacher.command(forces, palm)
    teacher.command(forces, palm)
    assert bool((teacher.phase >= 1).all())
    for _ in range(5):
        teacher.command(torch.full((2, 5), 0.4), palm)
    assert bool((teacher.phase >= 3).all())
    action = teacher.command(torch.full((2, 5), 0.25), palm)
    assert action.shape == (2, 14)
    assert bool((action[:, 10:].abs() >= 0).all())


def test_teacher_has_per_digit_force_feedback() -> None:
    teacher = EnvelopingTeacher(1, "cpu", EnvelopingTeacherCfg(support_wait_steps=1))
    palm = torch.ones(1)
    teacher.command(torch.zeros((1, 5)), palm)
    low = torch.tensor([[0.05, 0.05, 0.05, 0.05, 0.05]])
    high = torch.tensor([[1.5, 0.05, 0.05, 0.05, 0.05]])
    low_action = teacher.command(low, palm)
    high_action = teacher.command(high, palm)
    assert low_action[0, 5] > high_action[0, 5]


def test_thumb_first_course_releases_straight_fingers_after_coverage_window() -> None:
    teacher = EnvelopingTeacher(
        1,
        "cpu",
        EnvelopingTeacherCfg(
            root_phase_steps=2,
            thumb_first_phase_steps=3,
            proximal_phase_steps=3,
            distal_phase_steps=2,
            support_wait_steps=1,
            thumb_first_coverage_delta=(0.05, -0.08, 0.06, -0.04),
        ),
    )
    palm = torch.ones(1)
    forces = torch.zeros((1, 5))
    early = teacher.command(forces, palm)
    # The first supported command starts the thumb/coverage course, while
    # straight-finger root/residual channels remain gated.
    assert early[0, 4].abs() > 0.0 or early[0, 5].abs() > 0.0
    assert torch.allclose(early[0, 6:10], torch.zeros(4))
    for _ in range(6):
        late = teacher.command(torch.full((1, 5), 0.25), palm)
    assert bool((teacher.phase >= 2).all())
    assert bool(late[0, 6:10].abs().sum() > 0.0)
    assert torch.isfinite(late).all()
