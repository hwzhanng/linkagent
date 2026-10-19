"""物理子步保持守护的 CPU 语义测试（纯 torch，不启动 Kit/PhysX）。

覆盖 v1.4.0 的关键不变量：

* 连续保持只在每个真实 1/120 s 子步通过时增长，任何子步不满足立即清零；
* 粘滞硬故障在子步触发后保持到该 env reset，不被策略采样刷新覆写；
* reset 按 env 独立；dt=0 与重复 _sim_step_counter 都不记样本；
* 缺失 live 数据、非物理 dt、一个策略步内子步数不等于 decimation 时 fail-closed。
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from research.fixed_fruit_grasp_v1 import contract as C
from research.fixed_fruit_grasp_v1 import physical_hold as P


PHYSICS_DT_S = C.SIM_DT_S
DECIMATION = int(C.DECIMATION)
JOINTS = 16
FINGERS = 5


def _limits(num_envs: int = 1) -> torch.Tensor:
    """(env, joint, 2) 的关节限位；默认很宽，只有显式给值才触发 joint_limit。"""

    limits = torch.full((num_envs, JOINTS, 2), -1.0)
    limits[..., 1] = 1.0
    return limits


def _base_state(num_envs: int = 1) -> P.PhysicalSubstepState:
    """形状/类型完全合法的真实物理子步输入；用 dataclasses.replace 逐项替换。"""

    return P.PhysicalSubstepState(
        joint_pos=torch.zeros((num_envs, JOINTS)),
        joint_vel=torch.zeros((num_envs, JOINTS)),
        joint_pos_limits=_limits(num_envs),
        applied_effort=torch.full((num_envs, JOINTS), 0.05),
        effort_limits=torch.full((JOINTS,), 0.20),
        finger_force_n=torch.full((num_envs, FINGERS), 0.5),
        finger_hard_high_n=torch.tensor(C.FINGER_HARD_HIGH_N),
        fruit_pose=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * num_envs),
        fruit_linear_velocity=torch.zeros((num_envs, 3)),
        fruit_angular_velocity=torch.zeros((num_envs, 3)),
        ee_pose=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * num_envs),
        ee_linear_velocity=torch.zeros((num_envs, 3)),
        tip_positions=torch.zeros((num_envs, FINGERS, 3)),
        tip_velocities=torch.zeros((num_envs, FINGERS, 3)),
        palm_force_n=torch.zeros(num_envs),
        forbidden_proximal_n=torch.zeros((num_envs, 6)),
        previous_force_n=torch.zeros((num_envs, FINGERS)),
        executed_actions=torch.zeros((num_envs, C.ACTION_DIM)),
        raw_actions=torch.zeros((num_envs, C.ACTION_DIM)),
        previous_executed_actions=torch.zeros((num_envs, C.ACTION_DIM)),
        fruit_penetration_depth_m=torch.zeros(num_envs),
        self_penetration_depth_m=torch.zeros(num_envs),
        released=torch.ones(num_envs, dtype=torch.bool),
        required_contact=torch.ones(num_envs, dtype=torch.bool),
        relative_speed_m_s=torch.zeros(num_envs),
        fruit_in_hand_m=torch.zeros((num_envs, 3)),
        joint_limit_margin_rad=float(C.JOINT_LIMIT_MARGIN_RAD),
        instability_max_joint_velocity_rad_s=float(C.INSTABILITY_MAX_JOINT_VELOCITY_RAD_S),
        instability_max_fruit_speed_m_s=float(C.INSTABILITY_MAX_FRUIT_SPEED_M_S),
        fruit_penetration_limit_m=float(C.PENETRATION_LIMIT_M),
        self_collision_penetration_limit_m=float(C.SELF_COLLISION_PENETRATION_LIMIT_M),
        effort_guard_tolerance=0.05,
    )


def _state(**overrides: object) -> P.PhysicalSubstepState:
    return replace(_base_state(), **overrides)


def _guard(
    num_envs: int = 1,
    *,
    required_substeps: int = 120,
    enabled: bool = True,
    initial_counter: int = 0,
) -> P.PhysicalHoldGuard:
    guard = P.PhysicalHoldGuard(
        num_envs,
        device="cpu",
        physics_dt_s=PHYSICS_DT_S,
        decimation=DECIMATION,
        required_substeps=required_substeps,
        hold_max_drift_m=C.HOLD_MAX_DRIFT_M,
        hold_max_relative_speed_m_s=C.HOLD_MAX_RELATIVE_SPEED_M_S,
    )
    if enabled:
        guard.enable(initial_counter)
    return guard


def _feed(guard: P.PhysicalHoldGuard, counter: int, state: P.PhysicalSubstepState) -> bool:
    return guard.observe_substep(state, sim_step_counter=counter, scene_dt_s=PHYSICS_DT_S)


# ------------------------------------------------------------ streak 语义
def test_continuous_streak_counts_every_real_substep() -> None:
    guard = _guard()
    for index in range(120):
        assert _feed(guard, index + 1, _state()) is True
    assert guard.streak.tolist() == [120]
    assert guard.peak_streak.tolist() == [120]
    assert guard.snapshot()["streak"].tolist() == [120]


def test_single_violating_substep_clears_the_streak() -> None:
    """瞬时 fail：第 3 个子步速度越界，streak 必须清零而不是继续累积。"""

    guard = _guard()
    _feed(guard, 1, _state())
    _feed(guard, 2, _state())
    _feed(guard, 3, _state(relative_speed_m_s=torch.tensor([0.06])))
    assert guard.streak.tolist() == [0]
    assert guard.peak_streak.tolist() == [2]
    # 之后即使再连续满足，也必须从 0 重新累积到 120 才算成功。
    for index in range(3, 123):
        _feed(guard, index + 1, _state())
    assert guard.streak.tolist() == [120]


def test_transient_failure_inside_one_policy_step_is_not_hidden() -> None:
    """一个策略步 6 个子步：只有中间一个瞬时越界，20 Hz 末点无法看见。"""

    guard = _guard()
    counter = 0
    for step in range(20):
        for sub in range(DECIMATION):
            counter += 1
            speed = 0.06 if (step == 10 and sub == 2) else 0.0
            _feed(guard, counter, _state(relative_speed_m_s=torch.tensor([speed])))
        guard.complete_policy_interval()
    # 第 11 个策略步清零后，本策略步内剩余 3 个子步先重新起算，再加 9 个
    # 完整策略步（3 + 54 = 57 个子步），远低于 120。
    assert guard.streak.tolist() == [57]
    assert guard.streak.item() < guard.required_substeps


def test_required_physical_substeps_matches_one_second_at_120hz() -> None:
    assert P.required_physical_substeps(20, DECIMATION) == 120
    assert P.required_physical_substeps(60, DECIMATION) == 360
    hold = C.stage_spec("hold")
    assert hold.required_unsupported_hold_substeps == 360
    assert C.stage_spec("contact").required_unsupported_hold_substeps == 120
    with pytest.raises(P.PhysicalHoldError):
        _guard(required_substeps=0)
    with pytest.raises(P.PhysicalHoldError):
        P.required_physical_substeps(0, DECIMATION)


# ------------------------------------------------- 粘滞硬故障与 reset 独立性
def test_sticky_hard_fault_survives_policy_refresh_and_clears_on_reset() -> None:
    guard = _guard(2)
    _feed(guard, 1, _base_state(2))
    # env1 在第 2 个子步出现真实过载（v3 现场证据的 17.975 N 量级）。
    overloaded = replace(
        _base_state(2),
        finger_force_n=torch.tensor([[0.0] * FINGERS, [17.975, 0.0, 0.0, 0.0, 0.0]]),
    )
    _feed(guard, 2, overloaded)
    assert guard.hard_faults["overload"].tolist() == [False, True]
    assert guard.hard_fault_any.tolist() == [False, True]
    # 后续子步即使是完好的策略采样值，粘滞标记也必须保留。
    for counter in range(3, 10):
        _feed(guard, counter, _base_state(2))
    assert guard.hard_fault_any.tolist() == [False, True]
    assert guard.streak.tolist()[1] == 0
    # reset 只清被 reset 的 env。
    guard.reset(torch.tensor([1]))
    assert guard.hard_fault_any.tolist() == [False, False]
    assert guard.hard_faults["overload"].tolist() == [False, False]


def test_reset_clears_streak_and_peak_only_for_selected_envs() -> None:
    guard = _guard(2)
    for counter in range(1, 7):
        _feed(guard, counter, _base_state(2))
    assert guard.streak.tolist() == [6, 6]
    guard.reset(torch.tensor([0]))
    assert guard.streak.tolist() == [0, 6]
    assert guard.peak_streak.tolist() == [0, 6]
    assert guard.anchor_valid.tolist() == [False, True]
    with pytest.raises(P.PhysicalHoldError):
        guard.reset(torch.tensor([9]))


def test_hard_faults_detected_from_live_fields() -> None:
    guard = _guard()
    _feed(guard, 1, _state())
    assert guard.hard_fault_any.tolist() == [False]
    cases = (
        (2, _state(finger_force_n=torch.full((1, FINGERS), 15.01)), "overload"),
        (3, _state(joint_pos=torch.full((1, JOINTS), 2.0)), "joint_limit"),
        (4, _state(joint_vel=torch.full((1, JOINTS), 4.5)), "instability"),
        (5, _state(applied_effort=torch.full((1, JOINTS), 0.30)), "joint_effort"),
        (
            6,
            _state(
                fruit_penetration_depth_m=torch.tensor(
                    [float(C.PENETRATION_LIMIT_M) + 1.0e-3]
                )
            ),
            "penetration",
        ),
        (
            7,
            _state(
                self_penetration_depth_m=torch.tensor(
                    [float(C.SELF_COLLISION_PENETRATION_LIMIT_M) + 1.0e-3]
                )
            ),
            "self_collision",
        ),
    )
    for counter, state, name in cases:
        flags = P.compute_hard_faults(state)
        assert bool(flags[name].item()) is True, name
        _feed(guard, counter, state)
    assert guard.hard_fault_any.tolist() == [True]
    # 15.00 N 恰好等于硬门，不算过载。
    at_gate = P.compute_hard_faults(_state(finger_force_n=torch.full((1, FINGERS), 15.0)))
    assert bool(at_gate["overload"].item()) is False
    # 0.20 Nm 实际 PD effort 上限本身（ratio = 1.0）不构成 joint_effort 越界。
    at_effort_limit = P.compute_hard_faults(
        _state(applied_effort=torch.full((1, JOINTS), 0.20))
    )
    assert bool(at_effort_limit["joint_effort"].item()) is False


def test_nonfinite_live_value_triggers_nonfinite_hard_fault() -> None:
    bad = _state(joint_pos=torch.full((1, JOINTS), float("nan")))
    flags = P.compute_hard_faults(bad)
    assert bool(flags["nonfinite"].item()) is True


# ------------------------------------------------------- 计数与 fail-closed
def test_dt_zero_reset_callback_is_not_a_physical_step() -> None:
    guard = _guard()
    assert guard.observe_substep(_state(), sim_step_counter=1, scene_dt_s=0.0) is False
    assert guard.streak.tolist() == [0]
    with pytest.raises(P.PhysicalHoldError):
        guard.complete_policy_interval()  # 0 个真实子步：不完整
    # dt=0 不消耗计数器：下一个真实子步正常记账。
    assert _feed(guard, 1, _state()) is True
    assert guard.streak.tolist() == [1]


def test_duplicate_sim_step_counter_is_not_counted_twice() -> None:
    guard = _guard()
    assert _feed(guard, 1, _state()) is True
    assert _feed(guard, 1, _state()) is False
    assert guard.streak.tolist() == [1]


def test_counter_going_backwards_fails_closed() -> None:
    guard = _guard()
    _feed(guard, 1, _state())
    with pytest.raises(P.PhysicalHoldError):
        _feed(guard, 0, _state())


def test_counter_gap_fails_closed() -> None:
    """漏掉物理回调后凑满 6 个子步不能被当作连续：counter 必须严格 +1。"""

    guard = _guard()
    _feed(guard, 1, _state())
    with pytest.raises(P.PhysicalHoldError, match="跳跃"):
        _feed(guard, 3, _state())
    # 失败后守卫状态未被推进：计数器仍停在上一次真实子步。
    assert guard._last_sim_step_counter == 1
    assert guard.streak.tolist() == [1]
    assert _feed(guard, 2, _state()) is True
    assert guard.streak.tolist() == [2]


def test_dt_zero_callback_does_not_consume_the_counter_step() -> None:
    """dt=0 的 reset 回调可忽略且不修改 last，后续真实 +1 子步仍被接受。"""

    guard = _guard(initial_counter=10)
    assert guard.observe_substep(_state(), sim_step_counter=10, scene_dt_s=0.0) is False
    assert guard._last_sim_step_counter == 10
    assert _feed(guard, 11, _state()) is True
    assert guard.streak.tolist() == [1]


def test_dt_zero_after_sim_counter_reset_is_ignored() -> None:
    """dt=0 回调即使计数器被 sim.reset 归零也不改状态、不报错。"""

    guard = _guard(initial_counter=40)
    assert guard.observe_substep(_state(), sim_step_counter=0, scene_dt_s=0.0) is False
    assert guard._last_sim_step_counter == 40
    assert _feed(guard, 41, _state()) is True
    assert guard.streak.tolist() == [1]


def test_state_factory_is_only_evaluated_for_real_substeps() -> None:
    """被忽略的回调不得读取 live 缓冲（reset 期间缓冲可能不可用）。"""

    guard = _guard(initial_counter=10)

    def exploding_factory() -> P.PhysicalSubstepState:
        raise AssertionError("被忽略的回调不应求值 live 状态")

    assert (
        guard.observe_substep(exploding_factory, sim_step_counter=10, scene_dt_s=0.0)
        is False
    )
    assert (
        guard.observe_substep(
            exploding_factory, sim_step_counter=10, scene_dt_s=PHYSICS_DT_S
        )
        is False
    )
    assert guard.streak.tolist() == [0]
    # 真实 +1 子步才会求值 live 状态。
    with pytest.raises(AssertionError):
        guard.observe_substep(
            exploding_factory, sim_step_counter=11, scene_dt_s=PHYSICS_DT_S
        )


def test_non_physics_dt_fails_closed() -> None:
    """折叠 decimation 的分支（scene.update dt=step_dt）必须直接失败。"""

    guard = _guard()
    with pytest.raises(P.PhysicalHoldError, match="物理步长"):
        guard.observe_substep(_state(), sim_step_counter=1, scene_dt_s=C.POLICY_DT_S)
    with pytest.raises(P.PhysicalHoldError):
        guard.observe_substep(_state(), sim_step_counter=1, scene_dt_s=None)


def test_policy_interval_must_record_exactly_decimation_substeps() -> None:
    guard = _guard()
    for counter in range(1, DECIMATION + 1):
        _feed(guard, counter, _state())
    assert guard.complete_policy_interval() == DECIMATION
    # 第二轮只记 5 个：不完整，fail-closed。
    for counter in range(DECIMATION + 1, 2 * DECIMATION):
        _feed(guard, counter, _state())
    with pytest.raises(P.PhysicalHoldError, match="decimation"):
        guard.complete_policy_interval()


def test_missing_live_data_fails_closed() -> None:
    guard = _guard()
    with pytest.raises(P.PhysicalHoldError, match="self_penetration_depth_m"):
        _feed(guard, 1, _state(self_penetration_depth_m=None))
    with pytest.raises(P.PhysicalHoldError, match="fruit_in_hand_m"):
        _feed(guard, 1, _state(fruit_in_hand_m=None))
    assert guard.streak.tolist() == [0]


def test_guard_ignores_callbacks_before_enable() -> None:
    guard = _guard(enabled=False)
    assert (
        guard.observe_substep(_state(), sim_step_counter=1, scene_dt_s=PHYSICS_DT_S)
        is False
    )
    with pytest.raises(P.PhysicalHoldError):
        guard.complete_policy_interval()
    with pytest.raises(P.PhysicalHoldError):
        guard.enable(True)  # bool 不是合法计数器


def test_hard_fault_mask_from_mapping_fails_closed_on_missing_name() -> None:
    flags = {name: torch.tensor([True]) for name in P.HARD_FAULT_NAMES}
    assert P.hard_fault_mask_from_mapping(flags).tolist() == [True]
    del flags["overload"]
    with pytest.raises(P.PhysicalHoldError, match="overload"):
        P.hard_fault_mask_from_mapping(flags)


def test_shared_vectors_must_match_real_joint_and_finger_lengths() -> None:
    """effort_limits / finger_hard_high_n 是共享向量：长度不对必须 fail-closed。"""

    state = _state()
    state.require_complete(num_envs=1)  # 16 关节 / 5 指的默认形状合法
    with pytest.raises(P.PhysicalHoldError, match="effort_limits"):
        replace(state, effort_limits=torch.full((5,), 0.2)).require_complete(num_envs=1)
    with pytest.raises(P.PhysicalHoldError, match="effort_limits"):
        replace(
            state, effort_limits=torch.zeros((1, JOINTS))
        ).require_complete(num_envs=1)
    with pytest.raises(P.PhysicalHoldError, match="finger_hard_high_n"):
        replace(
            state, finger_hard_high_n=torch.full((3,), 15.0)
        ).require_complete(num_envs=1)
    # 共享向量按行广播：n=4 时同一份 (21,)/(5,) 直接可用。
    multi = _base_state(4)
    multi.require_complete(num_envs=4)
    flags = P.compute_hard_faults(multi)
    assert flags["overload"].tolist() == [False] * 4


def test_real_env_live_substep_state_shapes_are_accepted(monkeypatch) -> None:
    """真实 env 接口形状接缝：n=4、21 个安全关节、5 指共享向量必须被接受。

    这条测试直接调用 ``FixedFruitGraspEnv._live_physical_substep_state``，
    输入形状与真实运行时一致（joint 维 = 16 主动 + 5 跟随 = 21），避免只用
    已扩展的 toy fixture 掩盖 effort_limits/hard_high 的广播问题。
    """

    from types import SimpleNamespace

    from research.fixed_fruit_grasp_v1 import physical_contacts as PC
    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv

    num_envs = 4
    safety_joints = 21
    fingers = 5

    def wrap(tensor: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(torch=tensor)

    def bodies(values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(1).repeat(1, 1, 1)

    env = FixedFruitGraspEnv.__new__(FixedFruitGraspEnv)
    env._is_closed = True
    # num_envs / device 是父类属性（scene.num_envs / sim.device）。
    env.scene = SimpleNamespace(num_envs=num_envs)
    env.sim = SimpleNamespace(device="cpu")
    env._ee_body_id = 0
    env._fingertip_contact_body_ids = [0]
    env._safety_joint_ids = list(range(safety_joints))
    env._effort_proxy_name = "computed_torques"
    env._safety_effort_limits = torch.full((safety_joints,), C.ACTUATOR_EFFORT_LIMIT_NM)
    env._hard_high = torch.tensor(C.FINGER_HARD_HIGH_N)
    env._effort_guard_tolerance = 0.05
    env._previous_force_n = torch.zeros((num_envs, fingers))
    env._executed_actions = torch.zeros((num_envs, C.ACTION_DIM))
    env._raw_actions = torch.zeros((num_envs, C.ACTION_DIM))
    env._previous_executed_actions = torch.zeros((num_envs, C.ACTION_DIM))
    env._stem_detached = torch.zeros(num_envs, dtype=torch.bool)
    env._raw_penetration_sensors = [object()]
    env._self_penetration_sensors = [object()]
    env.cfg = SimpleNamespace(
        contact_force_threshold_n=C.CONTACT_FORCE_N,
        min_contact_fingers=C.MIN_CONTACT_FINGERS,
        joint_limit_margin_rad=C.JOINT_LIMIT_MARGIN_RAD,
        instability_max_joint_velocity_rad_s=C.INSTABILITY_MAX_JOINT_VELOCITY_RAD_S,
        instability_max_fruit_speed_m_s=C.INSTABILITY_MAX_FRUIT_SPEED_M_S,
        penetration_limit_m=C.PENETRATION_LIMIT_M,
        self_collision_penetration_limit_m=C.SELF_COLLISION_PENETRATION_LIMIT_M,
    )
    pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * num_envs)
    positions = torch.zeros((num_envs, 1, 3))
    limits = torch.full((num_envs, safety_joints, 2), -1.0)
    limits[..., 1] = 1.0
    env._robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pose_w=wrap(bodies(pose)),
            body_lin_vel_w=wrap(positions),
            body_ang_vel_w=wrap(positions),
            body_pos_w=wrap(positions),
            joint_pos=wrap(torch.zeros((num_envs, safety_joints))),
            joint_vel=wrap(torch.zeros((num_envs, safety_joints))),
            joint_pos_limits=wrap(limits),
            computed_torques=wrap(torch.full((num_envs, safety_joints), 0.05)),
        )
    )
    env._tomato = SimpleNamespace(
        data=SimpleNamespace(
            root_pose_w=wrap(pose),
            root_pos_w=wrap(torch.zeros((num_envs, 3))),
            root_lin_vel_w=wrap(torch.zeros((num_envs, 3))),
            root_ang_vel_w=wrap(torch.zeros((num_envs, 3))),
        )
    )
    env._palm_force = lambda: torch.zeros(num_envs)
    env._forbidden_proximal_force = lambda: torch.zeros((num_envs, 6))
    monkeypatch.setattr(
        PC,
        "penetration_from_sensors",
        lambda sensors, count, device: (
            torch.zeros(count, device=device),
            torch.zeros(count, dtype=torch.long, device=device),
        ),
    )

    forces = torch.full((num_envs, fingers), 0.5)
    state = FixedFruitGraspEnv._live_physical_substep_state(env, forces=forces)
    assert state.joint_pos.shape == (num_envs, safety_joints)
    assert state.applied_effort.shape == (num_envs, safety_joints)
    # 真实接口传的是共享向量，不是逐 env 扩展张量。
    assert state.effort_limits.shape == (safety_joints,)
    assert state.finger_hard_high_n.shape == (fingers,)
    assert state.finger_force_n.shape == (num_envs, fingers)
    state.require_complete(num_envs=num_envs)

    guard = _guard(num_envs)
    assert guard.observe_substep(state, sim_step_counter=1, scene_dt_s=PHYSICS_DT_S) is True
    # 支撑未撤（_stem_detached 全 False）：逐 env streak 保持 0。
    assert guard.streak.tolist() == [0] * num_envs
    assert guard.hard_fault_any.tolist() == [False] * num_envs


# ------------------------------------------------------ anchor / 漂移语义
def test_anchor_is_pinned_at_physical_streak_start() -> None:
    guard = _guard()
    _feed(guard, 1, _state(fruit_in_hand_m=torch.tensor([[0.0, 0.0, 0.0]])))
    assert guard.anchor_hand.tolist() == [[0.0, 0.0, 0.0]]
    # 漂移在容差内：anchor 不变，子步仍然通过。
    _feed(guard, 2, _state(fruit_in_hand_m=torch.tensor([[0.005, 0.0, 0.0]])))
    assert guard.anchor_hand.tolist() == [[0.0, 0.0, 0.0]]
    assert guard.streak.tolist() == [2]
    # 漂移超过 10 mm：本子步清零，anchor 在下一个子步重新固定。
    _feed(guard, 3, _state(fruit_in_hand_m=torch.tensor([[0.05, 0.0, 0.0]])))
    assert guard.streak.tolist() == [0]
    assert guard.anchor_valid.tolist() == [False]
    _feed(guard, 4, _state(fruit_in_hand_m=torch.tensor([[0.05, 0.0, 0.0]])))
    assert guard.anchor_hand[0, 0].item() == pytest.approx(0.05)
    assert guard.anchor_hand[0, 1:].tolist() == [0.0, 0.0]
    assert guard.streak.tolist() == [1]


def test_release_and_contact_requirements_gate_every_substep() -> None:
    guard = _guard()
    _feed(guard, 1, _state(released=torch.zeros(1, dtype=torch.bool)))
    assert guard.streak.tolist() == [0]
    _feed(guard, 2, _state(required_contact=torch.zeros(1, dtype=torch.bool)))
    assert guard.streak.tolist() == [0]
    _feed(guard, 3, _state())
    assert guard.streak.tolist() == [1]


# ------------------------------------------------------------- env 接线接缝
def _env_method_source(name: str) -> str:
    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv

    return textwrap.dedent(inspect.getsource(getattr(FixedFruitGraspEnv, name)))


def _called_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_get_dones_uses_physical_streak_and_interval_check() -> None:
    source = _env_method_source("_get_dones")
    assert "complete_policy_interval" in _called_names(source)
    assert "_physical_hold_guard" in source and "streak" in source
    assert "_physical_hold_required_substeps" in source
    # 旧策略步加 1 的计数必须不复存在。
    assert "_hold_ok, self._hold_age + 1" not in source


def test_refresh_task_state_merges_sticky_substep_faults_and_keeps_anchor() -> None:
    source = _env_method_source("_refresh_task_state")
    assert "guard.hard_faults" in source and "hard_fault_any" in source
    assert "anchor_valid" in source
    # anchor 不再由策略刷新写入。
    assert "_hold_anchor_hand[" not in source
    assert "rounding_mode" in source


def test_reset_clears_guard_per_environment() -> None:
    source = _env_method_source("_reset_idx")
    assert "_physical_hold_guard.reset" in source
    assert "_hold_anchor_hand[ids] =" not in source


def test_setup_scene_installs_scene_update_hook() -> None:
    source = _env_method_source("_setup_scene")
    assert "_install_physical_hold_substep_hook" in source
    hook = _env_method_source("_install_physical_hold_substep_hook")
    assert "self.scene.update" in hook and "original_update" in hook
    assert "_observe_physical_hold_substep" in hook


def test_contract_declares_v140_physical_hold_semantics() -> None:
    assert C.TASK_VERSION == "fixed_fruit_grasp_v1.6.0"
    assert C.PHYSICAL_HOLD_GUARD_FREQUENCY == "per_physics_substep"
    assert C.PHYSICAL_HOLD_HARD_FAULT_PERSISTENCE == "sticky_until_env_reset"
    assert C.PHYSICAL_HOLD_MISSING_DATA_POLICY == "fail_closed"
    semantics = C.physical_hold_semantics()
    assert semantics["guard_frequency"] == "per_physics_substep"
    assert semantics["decimation"] == DECIMATION
    for field in (
        "max_physical_hold_streak_substeps",
        "final_physical_hold_streak_substeps",
        "physical_hold_streak_required_substeps",
        "max_unsupported_hold_s_from_physical_substeps",
        "physical_hold_streak_semantics",
        "physical_substep_hard_fault_flags",
    ):
        assert field in C.EPISODE_RECORD_FIELDS


def test_contract_keeps_observation_and_physics_constants() -> None:
    """obs/action 维度、物理步长、decimation 与接触硬门数值都不变。"""

    assert C.ACTOR_OBSERVATION_DIM == 114
    assert C.CRITIC_STATE_DIM == 175
    assert C.ACTION_DIM == 14
    assert abs(C.SIM_DT_S - 1.0 / 120.0) < 1e-12
    assert C.DECIMATION == 6
    assert C.FINGER_HARD_HIGH_N == (15.0,) * 5
    assert C.CONTACT_FORCE_N == 0.08
    assert C.MIN_CONTACT_FINGERS == 2
    assert C.HOLD_MAX_RELATIVE_SPEED_M_S == 0.05
    assert C.HOLD_MAX_DRIFT_M == 0.010
    assert C.ACTUATOR_EFFORT_LIMIT_NM == 0.20
    env_source = Path(inspect.getfile(C)).parent.joinpath("env.py").read_text(
        encoding="utf-8"
    )
    # 逐物理子步守护只跳过诊断用 AABB/OBB，不跳过七类硬故障本身。
    assert "AABB/OBB clearance 矩阵" in env_source
    assert "self_collision_pairs" in env_source
