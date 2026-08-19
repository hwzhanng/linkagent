import torch

from simulation.ppo_tomato_grasp.stage0_scripted_controller import (
    Stage0ControllerCfg,
    Stage0ScriptedController,
)


def test_controller_closes_gradually_and_latches_complete_enclosure():
    controller = Stage0ScriptedController(2, "cpu")
    no_contact = torch.zeros((2, 5))
    first = controller.command(no_contact, torch.ones(2))
    second = controller.command(no_contact, torch.ones(2))
    assert torch.linalg.vector_norm(second, dim=-1).gt(
        torch.linalg.vector_norm(first, dim=-1)
    ).all()
    contact = torch.ones((2, 5))
    for _ in range(controller.cfg.enclosure_latch_steps):
        latched = controller.command(contact, torch.ones(2))
    assert controller.enclosure_latched.all()
    assert torch.allclose(latched, controller.hold_action)


def test_controller_feedback_is_independent_per_digit():
    controller = Stage0ScriptedController(
        1,
        "cpu",
        Stage0ControllerCfg(
            target_force_scale=(1.0,) * 5,
            force_feedback_gain_per_n=0.02,
        ),
    )
    controller.progress.fill_(0.5)
    controller.enclosure_latched.fill_(True)
    controller.hold_action.copy_(controller.geometry_prior * 0.5)
    forces = torch.tensor([[0.0, 2.0, 0.7, 0.7, 0.7]])
    for _ in range(controller.cfg.missing_hysteresis_steps):
        controller.command(forces, torch.ones(1))
    assert controller.trim[0, 0] > 0.0
    assert controller.trim[0, 1] < 0.0
    assert controller.trim[0, 2] == 0.0


def test_retention_center_feedback_opposes_positive_x_drift():
    controller = Stage0ScriptedController(
        1, "cpu", Stage0ControllerCfg(retention_center_increment=0.03)
    )
    controller.enclosure_latched.fill_(True)
    controller.hold_action.copy_(controller.geometry_prior * 0.5)
    controller.command(
        torch.full((1, 5), controller.cfg.target_force_n),
        torch.ones(1),
        retention_mode=torch.ones(1, dtype=torch.bool),
        center_error_m=torch.tensor([[0.06, 0.0, 0.0]]),
    )
    assert controller.trim[0, 0] > 0.0
    assert controller.trim[0, 1:].lt(0.0).all()
