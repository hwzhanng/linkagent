"""驱动修复、open 基线与固定时刻撤支撑的 CPU 测试（不启动 Kit/PhysX）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from research.fixed_fruit_grasp_v1 import contract as C
from research.fixed_fruit_grasp_v1.env import (
    FixedFruitGraspEnv,
    FixedFruitGraspPDActuator,
)
from research.small_fruit_pinch_v1.source.pinch_action_manifold import (
    pinch_action_to_joint_targets,
)
from research.small_fruit_pinch_v1.source.temporal import CriticalDampedActionFilter


JOINT_NAMES = ("index_joint0", "index_joint1", "thumb_joint0")
EFFORT_LIMITS = (0.15, 0.60, 0.25)
LOWER = (-0.26, 0.0, -1.05)
UPPER = (0.26, 1.66, 0.49)


def _actuator() -> FixedFruitGraspPDActuator:
    from isaaclab.actuators import IdealPDActuatorCfg

    cfg = IdealPDActuatorCfg(
        joint_names_expr=[rf"^{name}$" for name in JOINT_NAMES],
        effort_limit_sim=0.20,
        effort_limit=0.20,
        velocity_limit_sim=1.20,
        velocity_limit=1.20,
        armature=C.ACTUATOR_ARMATURE_KG_M2,
        stiffness=C.ACTUATOR_STIFFNESS,
        damping=C.ACTUATOR_DAMPING,
        friction=C.ACTUATOR_FRICTION,
        class_type=FixedFruitGraspPDActuator,
    )
    actuator = FixedFruitGraspPDActuator(cfg, list(JOINT_NAMES), slice(None), 1, "cpu")
    # 逐轴上限按 A7 口径显式覆盖，验证“逐轴”而非统一值语义。
    actuator.effort_limit = torch.tensor([EFFORT_LIMITS], dtype=torch.float32)
    return actuator


def _compute(actuator, q, qd, qdes, vdes, ff):
    from isaaclab.utils.types import ArticulationActions

    control = ArticulationActions(
        joint_positions=qdes.clone(),
        joint_velocities=vdes.clone(),
        joint_efforts=ff.clone(),
    )
    actuator.compute(control, q.clone(), qd.clone())
    return actuator.computed_effort, actuator.applied_effort


def test_limit_brake_effort_is_clipped_per_axis() -> None:
    actuator = _actuator()
    limits = torch.tensor([EFFORT_LIMITS], dtype=torch.float32)
    q = torch.tensor([[-0.36, -0.10, -1.15]])  # 全部越出下限
    qd = 494.0 * torch.ones(1, 3)
    _, applied = _compute(
        actuator, q, qd, q.clone(), qd.clone(), torch.full((1, 3), -0.5)
    )
    assert torch.all(applied.abs() <= limits + 1.0e-6)
    assert torch.allclose(applied, limits, atol=1.0e-6)
    assert torch.isfinite(applied).all()


def test_normal_domain_output_equals_clipped_pd() -> None:
    actuator = _actuator()
    limits = torch.tensor([EFFORT_LIMITS], dtype=torch.float32)
    q = torch.tensor([[0.0, 0.8, 0.2]])
    qd = torch.tensor([[0.1, -0.3, 0.2]])
    qdes = torch.tensor([[0.05, 0.83, 0.1]])
    vdes = torch.zeros_like(qd)
    ff = torch.zeros_like(qd)
    _, applied = _compute(actuator, q, qd, qdes, vdes, ff)
    expected = torch.clip(
        C.ACTUATOR_STIFFNESS * (qdes - q) + C.ACTUATOR_DAMPING * (vdes - qd) + ff,
        min=-limits,
        max=limits,
    )
    assert torch.allclose(applied, expected, atol=1.0e-6)


def test_inward_command_below_limit_not_braked() -> None:
    actuator = _actuator()
    limits = torch.tensor([EFFORT_LIMITS], dtype=torch.float32)
    q = torch.tensor([[-0.36, -0.10, -1.15]])
    qd = torch.full((1, 3), -0.4)
    ff = torch.full((1, 3), 0.05)  # 向内推：不触发制动
    _, applied = _compute(actuator, q, qd, q.clone(), qd.clone(), ff)
    assert torch.allclose(applied, torch.clip(ff, min=-limits, max=limits), atol=1.0e-6)


def _stub_env(num_envs: int = 2, hand_joints: int = 16) -> FixedFruitGraspEnv:
    env = FixedFruitGraspEnv.__new__(FixedFruitGraspEnv)
    # ``num_envs`` / ``device`` 都是 DirectRLEnv 的只读属性：用最小替身提供。
    env.sim = SimpleNamespace(device="cpu")
    env.scene = SimpleNamespace(num_envs=num_envs)
    env._is_closed = True  # 构造替身不跑 __init__，避免析构噪声
    env._open_hand_baseline = False
    env._raw_actions = torch.zeros((num_envs, C.ACTION_DIM))
    env._executed_actions = torch.zeros((num_envs, C.ACTION_DIM))
    env._previous_executed_actions = torch.zeros((num_envs, C.ACTION_DIM))
    env._filter = CriticalDampedActionFilter(
        num_envs, C.ACTION_DIM, dt_s=C.POLICY_DT_S, device="cpu"
    )
    env._last_physical_force_n = torch.zeros((num_envs, 5))
    env._force_limiter_scale = torch.ones(num_envs)
    env._verifier = SimpleNamespace(hold_age=torch.zeros(num_envs, dtype=torch.long))
    env._joint_targets = torch.zeros((num_envs, hand_joints))
    env._hand_joint_ids = list(range(hand_joints))
    env._open_hand_targets = torch.full((num_envs, hand_joints), 0.1)
    env._position_interpolation_step = torch.zeros(num_envs, dtype=torch.long)
    env._stem_break_event = torch.zeros(num_envs, dtype=torch.bool)
    env._all_env_ids = torch.arange(num_envs, dtype=torch.long)
    env._mimic_calls: list[list[int]] = []
    env._update_mimic_targets = lambda ids: env._mimic_calls.append(
        list(torch.as_tensor(ids).flatten().tolist())
    )

    def _forbidden(*args, **kwargs):
        raise AssertionError("open 基线不得写关节状态/传送关节")

    env.write_joint_position_to_sim_index = _forbidden
    env.write_joint_velocity_to_sim_index = _forbidden
    return env


def test_open_hand_baseline_writes_neutral_targets_only() -> None:
    env = _stub_env()
    env.set_open_hand_baseline(True)
    actions = torch.full((env.num_envs, C.ACTION_DIM), 0.9)
    env._pre_physics_step(actions)
    assert torch.equal(env._joint_targets, env._open_hand_targets)
    assert torch.equal(env._executed_actions, torch.zeros_like(env._executed_actions))
    # 时间滤波器仍被推进（诊断信号活跃），但没有把动作写进执行器。
    assert float(env._filter.position.abs().sum()) > 0.0
    assert env._mimic_calls == [[0, 1]]
    assert bool(env._position_interpolation_step.eq(0).all())


def test_normal_step_uses_the_14d_mapping() -> None:
    env = _stub_env()
    env.set_open_hand_baseline(False)
    actions = torch.full((env.num_envs, C.ACTION_DIM), 0.5)
    env._pre_physics_step(actions)
    expected = pinch_action_to_joint_targets(env._executed_actions)
    assert torch.allclose(env._joint_targets, expected)
    assert not torch.equal(env._joint_targets, env._open_hand_targets)


def test_open_hand_action_refuses_to_fake_an_inverse() -> None:
    env = _stub_env(1)
    env._action_effect = {"manifold_zero_target_gap_rad": 0.31}
    with pytest.raises(C.OpenHandActionUnavailable):
        env.open_hand_action()


def test_bounded_closure_reference_is_finite_and_within_safe_range() -> None:
    env = _stub_env(3)
    action = env.closure_reference_action(0.65)
    assert action.shape == (3, C.ACTION_DIM)
    assert action[:, 0].tolist() == pytest.approx([0.65, 0.65, 0.65])
    assert action[:, 4].tolist() == pytest.approx([0.65, 0.65, 0.65])
    assert bool((action[:, 1:4] == 0.0).all())  # 其余协同/残差保持中立
    targets = env.reference_joint_targets(0.65)
    assert targets.shape == (3, 16)
    assert torch.isfinite(targets).all()


def _support_stub(num_envs: int, episode_length: int, detached: bool):
    env = FixedFruitGraspEnv.__new__(FixedFruitGraspEnv)
    env.scene = SimpleNamespace(num_envs=num_envs)
    env._is_closed = True
    env.cfg = SimpleNamespace(support_release_step=40)
    env.episode_length_buf = torch.full((num_envs,), episode_length, dtype=torch.long)
    env._stem_detached = torch.full((num_envs,), detached, dtype=torch.bool)
    env._support_release_by_time = torch.zeros(num_envs, dtype=torch.bool)
    env._audit_support_released = torch.zeros(num_envs, dtype=torch.bool)
    env._audit_impulse_issued = torch.zeros(num_envs, dtype=torch.bool)
    env._audit_impulse_ever_applied = torch.zeros(num_envs, dtype=torch.bool)
    return env


def test_support_release_is_fixed_time_driven_not_success_driven() -> None:
    before = _support_stub(2, 39, False)
    before._prepare_verification_support()
    assert not bool(before._stem_detached.any())
    assert not bool(before._support_release_by_time.any())

    at_release = _support_stub(2, 40, False)
    at_release._prepare_verification_support()
    assert bool(at_release._stem_detached.all())
    assert bool(at_release._support_release_by_time.all())
    assert bool(at_release._audit_support_released.all())

    later = _support_stub(2, 120, False)
    later._prepare_verification_support()
    assert bool(later._stem_detached.all())


def test_release_flag_does_not_claim_natural_detachment() -> None:
    """已经提前断开的支撑不得被记成“按时间解除”。"""

    natural = _support_stub(1, 10, True)
    natural._prepare_verification_support()
    assert bool(natural._stem_detached.all())
    assert not bool(natural._support_release_by_time.any())


def test_verification_impulse_is_explicit_noop() -> None:
    env = _support_stub(1, 60, True)
    assert env._apply_verification_impulse() is None
    assert not bool(env._audit_impulse_issued.any())
    assert not bool(env._audit_impulse_ever_applied.any())


def test_declared_support_uses_real_wrench_and_zeroes_released_envs() -> None:
    """释放批次的物理施力归零，夹具不得逐步写果实状态。"""
    from types import SimpleNamespace
    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv
    env = _support_stub(2, 20, False)
    env.sim = SimpleNamespace(device="cpu")
    env.cfg = SimpleNamespace(sim=SimpleNamespace(gravity=(0.0, 0.0, -9.81)),
        stem_linear_natural_frequency_rad_s=20.0, stem_linear_damping_ratio=1.0,
        stem_angular_natural_frequency_rad_s=15.0, stem_angular_damping_ratio=1.0,
        maximum_stem_force_n=100.0, maximum_stem_torque_nm=0.30)
    pose = torch.tensor([[0.,0.,0.,0.,0.,0.,1.]]).repeat(2,1)
    env._fruit_axes_m = torch.full((2,3), 0.0335)
    env._fruit_mass_kg = torch.full((2,), 0.141731)
    env._stem_rest_attachment_pos_w = torch.tensor([[0.,0.,0.0335]]).repeat(2,1)
    env._stem_anchor_quat_w = pose[:,3:7].clone()
    env._last_stem_force_w = torch.zeros((2,3))
    env._last_stem_torque_w = torch.zeros((2,3))
    calls = []
    composer = SimpleNamespace(set_forces_and_torques_index=lambda **kw: calls.append(kw))
    env._tomato = SimpleNamespace(data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=pose),
        root_lin_vel_w=SimpleNamespace(torch=torch.zeros((2,3))),
        root_ang_vel_w=SimpleNamespace(torch=torch.zeros((2,3)))), permanent_wrench_composer=composer)
    env._stem_detached[1] = True
    FixedFruitGraspEnv._apply_attachment_wrench(env)
    assert calls[-1]['is_global'] is True
    assert calls[-1]['forces'][0,0,2].item() == pytest.approx(0.141731 * 9.81)
    assert not bool(calls[-1]['forces'][1].any())
    assert not bool(calls[-1]['torques'][1].any())
    env._stem_detached[:] = True
    FixedFruitGraspEnv._apply_attachment_wrench(env)
    assert not bool(calls[-1]['forces'].any())
    assert not bool(calls[-1]['torques'].any())
    assert torch.equal(pose, torch.tensor([[0.,0.,0.,0.,0.,0.,1.]]).repeat(2,1))
