import torch

from simulation.ppo_tomato_grasp_v2.residual_controller import ResidualBaseController, ResidualBaseControllerCfg


TARGET = (
    1.0, -1.0, 0.58, -0.01, 1.0, -0.01,
    0.58, 0.99, 0.11, 0.06, -0.62, 0.65, -0.24, 0.51,
)


def test_residual_controller_has_bounded_14d_staged_output():
    controller = ResidualBaseController(2, "cpu", TARGET)
    force = torch.zeros(2, 5)
    action = controller.command(torch.tensor([0, 239]), force)
    assert action.shape == (2, 14)
    assert torch.isfinite(action).all()
    assert (action.abs() <= 1.0).all()
    # The default sequence reaches palm/root/thumb first and keeps all distal
    # channels open until their own insertion windows.
    assert torch.allclose(action[1, :10], torch.tensor(TARGET[:10]), atol=1.0e-5)
    assert torch.allclose(action[1, 10:], torch.zeros(4), atol=1.0e-5)


def test_residual_controller_freezes_active_digit_then_holds_all_digits():
    controller = ResidualBaseController(1, "cpu", TARGET)
    # Reach the coarse manifold, then force index into its lower contact band
    # for five frames.  The controller must retain that finger during the
    # remainder of the insertion phase, while the next digit remains open.
    for step in range(240):
        controller.command(torch.tensor([step]), torch.zeros(1, 5))
    measured = torch.tensor([[0.0, 0.20, 0.0, 0.0, 0.0]])
    for step in range(240, 245):
        controller.command(torch.tensor([step]), measured)
    frozen_index = float(controller.current[0, 6])
    next_step = controller.command(torch.tensor([245]), torch.zeros(1, 5))
    assert torch.isclose(next_step[0, 6], torch.tensor(frozen_index))
    # After 4*per_digit_steps the hold branch updates all fingers rather than
    # continuing to move only the little finger.
    hold = controller.command(torch.tensor([480 + 1]), torch.zeros(1, 5))
    assert torch.isfinite(hold).all()


def test_residual_controller_accepts_per_digit_insertion_and_retreat_fractions():
    cfg = ResidualBaseControllerCfg(
        insertion_step_fraction_per_digit=(0.08, 0.08, 0.02, 0.08),
        retreat_fraction_per_digit=(0.12, 0.12, 0.35, 0.12),
    )
    controller = ResidualBaseController(1, "cpu", TARGET, cfg)
    # The per-digit values are an explicit controller configuration, not a
    # terminal pose; command generation must remain finite and bounded.
    action = controller.command(torch.tensor([300]), torch.zeros(1, 5))
    assert torch.isfinite(action).all()
    assert (action.abs() <= 1.0).all()


def test_residual_controller_thumb_search_flips_only_after_low_force_period():
    cfg = ResidualBaseControllerCfg(
        approach_steps=2,
        per_digit_steps=4,
        thumb_hold_target_offset=(0.0, 0.0),
        thumb_search_amplitude=(0.0, 0.20),
        thumb_search_period_steps=3,
    )
    controller = ResidualBaseController(1, "cpu", TARGET, cfg)
    low = torch.zeros(1, 5)
    actions = [controller.command(torch.tensor([step]), low)[0, 5].item() for step in range(2, 8)]
    # The first three low-force frames use one signed probe; the next frame
    # flips the sign.  This is a bounded search, not frame-to-frame chatter.
    assert actions[2] != actions[3]
    assert max(abs(value) for value in actions) <= 1.0


def test_residual_controller_thumb_contact_latch_holds_after_contact_loss():
    cfg = ResidualBaseControllerCfg(
        approach_steps=2,
        thumb_contact_latch_force_n=0.04,
        thumb_contact_latch_steps=3,
    )
    controller = ResidualBaseController(1, "cpu", TARGET, cfg)
    controller.command(torch.tensor([0]), torch.zeros(1, 5))
    controller.command(torch.tensor([1]), torch.zeros(1, 5))
    contact = controller.command(torch.tensor([2]), torch.tensor([[0.05, 0, 0, 0, 0.]]))
    held = controller.command(torch.tensor([3]), torch.zeros(1, 5))
    assert torch.allclose(held[0, 4:6], contact[0, 4:6])
    assert int(controller.thumb_latch_remaining[0]) > 0
