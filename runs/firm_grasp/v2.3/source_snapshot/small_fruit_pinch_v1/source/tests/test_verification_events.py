from __future__ import annotations

from types import SimpleNamespace

import torch

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchEnv


def _event_env() -> NativeSmallFruitPinchEnv:
    env = object.__new__(NativeSmallFruitPinchEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(verification_release_support=True)
    env._verifier = SimpleNamespace(success=torch.tensor([True, False]), disturbance_due=torch.tensor([False, False]))
    env._stem_detached = torch.zeros(2, dtype=torch.bool)
    env._audit_support_released = torch.zeros(2, dtype=torch.bool)
    env._audit_natural_detached = torch.zeros(2, dtype=torch.bool)
    env._audit_impulse_issued = torch.zeros(2, dtype=torch.bool)
    env._audit_impulse_step = torch.full((2,), -1, dtype=torch.long)
    env._audit_impulse_ns = torch.zeros((2, 3))
    env._audit_disturbance_directions = torch.tensor([[1., 0., 0.], [0., -1., 0.]])
    env._fruit_mass_kg = torch.tensor([0.1, 0.2])
    env.common_step_counter = 4
    return env


def test_support_release_does_not_forge_natural_stem_break() -> None:
    env = _event_env()
    env._prepare_verification_support()
    assert env._stem_detached.tolist() == [True, False]
    assert env._audit_support_released.tolist() == [True, False]
    assert env._audit_natural_detached.tolist() == [False, False]


def test_disturbance_writes_mass_scaled_momentum_only_once() -> None:
    env = _event_env()
    env._stem_detached[:] = True
    env._verifier.disturbance_due[:] = True
    writes: list[torch.Tensor] = []

    def write(*, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        writes.append(root_velocity.clone())

    env._tomato = SimpleNamespace(
        data=SimpleNamespace(root_lin_vel_w=SimpleNamespace(torch=torch.zeros((2, 3))), root_ang_vel_w=SimpleNamespace(torch=torch.zeros((2, 3)))),
        write_root_velocity_to_sim_index=write,
    )
    env._apply_verification_impulse()
    env._apply_verification_impulse()
    assert len(writes) == 1
    assert torch.allclose(writes[0][:, :3], torch.tensor([[0.122625, 0., 0.], [0., -0.122625, 0.]]))
    assert torch.allclose(env._audit_impulse_ns, torch.tensor([[0.0122625, 0., 0.], [0., -0.024525, 0.]]))
    assert env._audit_impulse_step.tolist() == [4, 4]


def test_broken_window_rearms_only_after_physics_and_only_for_affected_env() -> None:
    env = _event_env()
    env._audit_impulse_issued[:] = True
    env._audit_impulse_step[:] = 4
    env._verifier.support_hold_age = torch.tensor([0, 45])
    env._rearm_verification_impulse()
    assert env._audit_impulse_issued.tolist() == [True, True]
    env.common_step_counter = 5
    env._rearm_verification_impulse()
    assert env._audit_impulse_issued.tolist() == [False, True]
