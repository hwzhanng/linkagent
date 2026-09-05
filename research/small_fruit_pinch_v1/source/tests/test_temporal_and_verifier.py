from __future__ import annotations

import torch

from research.small_fruit_pinch_v1.source.temporal import CriticalDampedActionFilter
from research.small_fruit_pinch_v1.source.verifier import TimStrictVerifier
from research.small_fruit_pinch_v1.source.guided_residual_env import GuidedResidualSmallFruitPinchEnv


def _update(verifier: TimStrictVerifier, force: float, *, palm: bool = False):
    return verifier.update(
        force_n=torch.tensor([[force, force, force, 0.0, 0.0]]),
        palm_contact=torch.tensor([palm]),
        center_drift_m=torch.tensor([0.001]),
        relative_speed_m_s=torch.tensor([0.001]),
        safety_projection=torch.tensor([False]),
        self_collision=torch.tensor([False]),
    )


def test_filter_is_stateful_bounded_and_keeps_hold_response() -> None:
    filt = CriticalDampedActionFilter(1, 10, dt_s=0.05, device="cpu")
    first = filt.step(torch.ones((1, 10)), hold_phase=torch.tensor([False]))
    second = filt.step(torch.ones((1, 10)), hold_phase=torch.tensor([False]))
    assert torch.all(first.abs() < 1.0)
    assert torch.all(second > first)
    hold = filt.step(-torch.ones((1, 10)), hold_phase=torch.tensor([True]))
    assert torch.isfinite(hold).all()
    assert bool((filt.last_stats["velocity"] <= 0.15 + 1e-6).all())
    assert bool((hold != 0.0).any()), "hold 阶段不得把策略动作冻结为零"


def test_success_requires_complete_hold_not_single_contact() -> None:
    verifier = TimStrictVerifier(1, device="cpu")
    for _ in range(10):
        state = _update(verifier, 0.5)
    assert not bool(state["success"].item())
    # 前两帧只用于接触迟滞确认，不得提前计入 80 步持稳窗。
    for _ in range(72):
        state = _update(verifier, 0.5)
    assert bool(state["success"].item())
    assert not bool(state["final_success"].item())
    for _ in range(50):
        state = _update(verifier, 0.5)
    assert bool(state["final_success"].item())


def test_palm_shortcut_and_hard_overload_fail() -> None:
    verifier = TimStrictVerifier(1, device="cpu")
    for _ in range(90):
        state = _update(verifier, 0.5, palm=True)
    assert not bool(state["success"].item())
    verifier.reset()
    for _ in range(2):
        state = _update(verifier, 5.0)
    assert bool(state["hard_overload"].item())
    assert not bool(state["success"].item())


def test_residual_trust_region_is_force_directional_and_all_contact_gated() -> None:
    env = object.__new__(GuidedResidualSmallFruitPinchEnv)
    env.residual_trust_region_enabled = True
    env.residual_trust_warning_fraction = 0.60
    env.residual_trust_in_band_scale = 0.10
    env.residual_trust_zero_global = True
    env._last_physical_force_n = torch.tensor([[0.10, 0.25, 1.00, 0.0, 0.0]])
    residual = torch.ones((1, 14))
    residual[:, 5:8] = -1.0
    residual[:, 10:12] = -1.0
    contact = torch.tensor([[True, True, True, False, False]])
    gated = env._apply_residual_trust_region(residual, contact, torch.tensor([True]))
    assert gated[0, :5].abs().sum() == 0.0
    assert gated[0, 5] < 0.0  # thumb under soft-low: only close is allowed
    assert torch.isclose(gated[0, 6], torch.tensor(-0.1))
    assert gated[0, 7] == 0.0  # middle near warning: negative close is rejected
    assert torch.isclose(gated[0, 10], torch.tensor(-0.1))
    assert gated[0, 11] == 0.0
    assert torch.equal(
        env._apply_residual_trust_region(residual, contact, torch.tensor([False])),
        torch.zeros_like(residual),
    )
