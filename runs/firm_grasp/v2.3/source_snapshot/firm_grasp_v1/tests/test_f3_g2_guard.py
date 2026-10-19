from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from research.firm_grasp_v1.disturbance import pulse_wrench
from research.firm_grasp_v1.g2_guard import G2HoldGuard


def _state(speed: float = 0.0, *, released: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        fruit_in_hand_m=torch.zeros((1, 3), dtype=torch.float32),
        relative_speed_m_s=torch.tensor([speed], dtype=torch.float32),
        released=torch.tensor([released], dtype=torch.bool),
    )


def _observe(
    guard: G2HoldGuard,
    step: int,
    *,
    substep: int | None = None,
    speed: float = 0.0,
    quat: torch.Tensor | None = None,
    angular_speed: float = 0.0,
    qualified: bool = True,
    disturbed: bool = False,
    strict: bool = False,
) -> None:
    if quat is None:
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    guard.observe(
        state=_state(speed),
        qualified=torch.full((1, 6), qualified, dtype=torch.bool),
        hard_fault_any=torch.tensor([False]),
        support_zero=torch.tensor([True]),
        counter=step,
        dt=1 / 120,
        relative_quat_xyzw=quat,
        relative_angular_speed=torch.tensor([angular_speed], dtype=torch.float32),
        post_release_substep=torch.tensor([step if substep is None else substep]),
        strict_early_start=strict,
        disturbed=disturbed,
    )


def test_clock_continuity_and_policy_interval_boundary() -> None:
    guard = G2HoldGuard(1, device="cpu", initial_counter=10, decimation=2)
    _observe(guard, 11, substep=0)
    with pytest.raises(RuntimeError, match="漏采或重复"):
        _observe(guard, 13, substep=1)
    with pytest.raises(RuntimeError, match="dt 不符"):
        guard.observe(
            state=_state(),
            qualified=torch.ones((1, 6), dtype=torch.bool),
            hard_fault_any=torch.tensor([False]),
            support_zero=torch.tensor([True]),
            counter=12,
            dt=1 / 60,
            relative_quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            relative_angular_speed=torch.tensor([0.0]),
            post_release_substep=torch.tensor([1]),
            strict_early_start=False,
            disturbed=False,
        )
    with pytest.raises(RuntimeError, match="采样数错误"):
        guard.complete_policy_interval()
    _observe(guard, 12, substep=1)
    guard.complete_policy_interval()
    assert guard.interval_count == 0


def test_reset_clears_only_selected_environment_state() -> None:
    guard = G2HoldGuard(2, device="cpu", initial_counter=0)
    guard.observe(
        state=SimpleNamespace(
            fruit_in_hand_m=torch.zeros((2, 3)),
            relative_speed_m_s=torch.zeros(2),
            released=torch.ones(2, dtype=torch.bool),
        ),
        qualified=torch.ones((2, 6), dtype=torch.bool),
        hard_fault_any=torch.zeros(2, dtype=torch.bool),
        support_zero=torch.ones(2, dtype=torch.bool),
        counter=1,
        dt=1 / 120,
        relative_quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2),
        relative_angular_speed=torch.zeros(2),
        post_release_substep=torch.zeros(2, dtype=torch.long),
        strict_early_start=False,
        disturbed=False,
    )
    guard.reset(torch.tensor([0]))
    assert torch.equal(guard.streak[0], torch.zeros(6, dtype=torch.long))
    assert torch.equal(guard.streak[1], torch.ones(6, dtype=torch.long))
    assert torch.isinf(guard.drift[0]).all()
    assert guard.last_counter == 1


@pytest.mark.parametrize("start,expected", [(60, 1), (61, 0)])
def test_strict_window_start_boundary(start: int, expected: int) -> None:
    guard = G2HoldGuard(1, device="cpu", initial_counter=0)
    _observe(guard, 1, substep=start, strict=True)
    assert torch.equal(guard.streak[0], torch.full((6,), expected, dtype=torch.long))
    assert bool(guard.last_ok.all()) is bool(expected)


def test_angle_is_relative_to_each_combo_window_anchor() -> None:
    guard = G2HoldGuard(1, device="cpu", initial_counter=0)
    _observe(guard, 1, substep=0)
    ten_deg = math.radians(10.0)
    q10 = torch.tensor([[0.0, 0.0, math.sin(ten_deg / 2), math.cos(ten_deg / 2)]])
    _observe(guard, 2, substep=1, quat=q10)
    assert bool(guard.last_ok.all())
    assert torch.allclose(guard.angle, torch.full((1, 6), ten_deg), atol=2e-6)
    over = math.radians(10.1)
    q_over = torch.tensor([[0.0, 0.0, math.sin(over / 2), math.cos(over / 2)]])
    _observe(guard, 3, substep=2, quat=q_over)
    assert not bool(guard.last_ok.any())
    assert torch.equal(guard.streak, torch.zeros((1, 6), dtype=torch.long))


def test_disturbed_window_requires_recovery_from_substep_252_to_completion() -> None:
    guard = G2HoldGuard(1, device="cpu", initial_counter=0)
    for substep in range(252):
        _observe(
            guard, substep + 1, substep=substep, speed=0.05,
            strict=True, disturbed=True
        )
    assert int(guard.streak[0, 0]) == 252
    _observe(guard, 253, substep=252, speed=0.02, strict=True, disturbed=True)
    assert int(guard.streak[0, 0]) == 253
    _observe(guard, 254, substep=253, speed=0.0201, strict=True, disturbed=True)
    assert int(guard.streak[0, 0]) == 0
    # 窗口重启已超过严格早启时限，不能重新冒充覆盖扰动的窗口。
    _observe(guard, 255, substep=254, speed=0.02, strict=True, disturbed=True)
    assert int(guard.streak[0, 0]) == 0


def test_disturbed_early_window_covers_pulse_and_reaches_g2() -> None:
    guard = G2HoldGuard(1, device="cpu", initial_counter=0)
    for substep in range(360):
        speed = 0.05 if substep < 252 else 0.02
        _observe(
            guard, substep + 1, substep=substep, speed=speed,
            strict=True, disturbed=True
        )
    assert torch.equal(guard.streak, torch.full((1, 6), 360, dtype=torch.long))
    assert bool(guard.g2_success.all())


def test_pulse_wrench_has_exact_twelve_substeps_and_axis_rotation() -> None:
    quat_identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 194)
    substeps = torch.arange(179, 193, dtype=torch.long)
    quat = quat_identity[: substeps.numel()]
    force, torque = pulse_wrench(
        substeps, quat, 0.141731, 0.0335, mode="force", axis=0, sign=1.0
    )
    active = torch.linalg.vector_norm(force, dim=1) > 0
    assert int(active.sum()) == 12
    assert torch.equal(active, (substeps >= 180) & (substeps < 192))
    assert torch.allclose(force[1, 0], torch.tensor(0.1 * 0.141731 * 9.81))
    assert torch.count_nonzero(torque) == 0

    half_turn_z = torch.tensor(
        [[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]]
    )
    rotated_force, _ = pulse_wrench(
        torch.tensor([180]), half_turn_z, 1.0, 0.1,
        mode="force", axis=0, sign=1.0, force_ratio=0.1
    )
    assert torch.allclose(rotated_force, torch.tensor([[0.0, 0.981, 0.0]]), atol=1e-6)
    _, rotated_torque = pulse_wrench(
        torch.tensor([191]), half_turn_z, 1.0, 0.1,
        mode="torque", axis=1, sign=-1.0, torque_ratio=0.01
    )
    assert torch.allclose(rotated_torque, torch.tensor([[0.00981, 0.0, 0.0]]), atol=1e-6)


def test_quiet_mode_and_invalid_pulse_arguments() -> None:
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    force, torque = pulse_wrench(torch.tensor([180]), q, 0.2, 0.03, mode="quiet")
    assert torch.count_nonzero(force) == 0
    assert torch.count_nonzero(torque) == 0
    with pytest.raises(ValueError, match="mode"):
        pulse_wrench(torch.tensor([180]), q, 0.2, 0.03, mode="position")
    with pytest.raises(ValueError, match="axis"):
        pulse_wrench(torch.tensor([180]), q, 0.2, 0.03, mode="force", axis=3)


def test_pulse_wrench_accepts_scalar_and_per_environment_mass_radius() -> None:
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3)
    substeps = torch.tensor([180, 191, 192], dtype=torch.long)
    mass = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    force, torque = pulse_wrench(
        substeps, quat, mass, torch.tensor(0.05),
        mode="force", axis=0, sign=1.0
    )
    expected_force = torch.tensor([0.0981, 0.1962, 0.0])
    assert torch.allclose(force[:, 0], expected_force, atol=1e-6)
    assert torch.count_nonzero(force[:, 1:]) == 0
    assert torch.count_nonzero(torque) == 0

    force, torque = pulse_wrench(
        substeps, quat, 0.1, torch.tensor([0.05, 0.1, 0.2]),
        mode="torque", axis=2, sign=-1.0
    )
    expected_torque = torch.tensor([-0.0004905, -0.000981, 0.0])
    assert torch.allclose(torque[:, 2], expected_torque, atol=1e-7)
    assert torch.count_nonzero(torque[:, :2]) == 0
    assert torch.count_nonzero(force) == 0

    with pytest.raises(ValueError, match=r"形状为 \[N\]"):
        pulse_wrench(
            substeps, quat, torch.ones(2), 0.05,
            mode="force", axis=0
        )
