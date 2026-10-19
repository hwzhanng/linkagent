"""静态接缝测试：env 与 _EpisodeTracker 的调用签名/字段必须逐字一致。

Physics smoke 曾因 ``_EpisodeTracker.__init__`` 关键字不匹配在场景初始化后立即
失败（未进入 env step）。这类接缝不依赖 PhysX，可用 AST 静态守住。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from research.fixed_fruit_grasp_v1.env import _EpisodeTracker


def _env_source_tree() -> ast.Module:
    import research.fixed_fruit_grasp_v1.env as env_module

    return ast.parse(Path(env_module.__file__).read_text(encoding="utf-8"))


def _tracker_call_keywords(tree: ast.Module) -> list[set[str]]:
    keyword_sets: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None
        if name != _EpisodeTracker.__name__:
            continue
        keyword_sets.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return keyword_sets


def test_tracker_construction_keywords_match_signature() -> None:
    parameters = set(inspect.signature(_EpisodeTracker.__init__).parameters) - {
        "self",
        # 唯一的位置参数（num_envs）在 env 里按位置传入。
        "num_envs",
    }
    keyword_sets = _tracker_call_keywords(_env_source_tree())
    assert keyword_sets, "未找到 _EpisodeTracker(...) 调用"
    for keywords in keyword_sets:
        missing = parameters - keywords
        extra = keywords - parameters
        assert not missing, f"_EpisodeTracker 调用缺少关键字：{sorted(missing)}"
        assert not extra, f"_EpisodeTracker 调用出现未声明关键字：{sorted(extra)}"


def _tracker_state_stub(num_envs: int = 2):
    """提供 ``_tracker_state()`` 真实调用源所需的全部张量。"""

    from types import SimpleNamespace

    import torch

    from research.fixed_fruit_grasp_v1 import contract as C
    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv

    env = FixedFruitGraspEnv.__new__(FixedFruitGraspEnv)
    env.sim = SimpleNamespace(device="cpu")
    env.scene = SimpleNamespace(num_envs=num_envs)
    env._is_closed = True  # 构造替身不跑 __init__，避免析构噪声
    env._last_physical_force_n = torch.zeros((num_envs, 5))
    env._raw_finger_force_n = torch.zeros((num_envs, 5))
    env._last_palm_force_n = torch.zeros(num_envs)
    env._forbidden_contact_force_n = torch.zeros((num_envs, 3))
    env._forbidden_proximal_force_n = torch.zeros((num_envs, 6))
    env._required_contact = torch.zeros(num_envs, dtype=torch.bool)
    env._contact_mask = torch.zeros((num_envs, 5), dtype=torch.bool)
    env._hold_age = torch.zeros(num_envs, dtype=torch.long)
    env._hold_ok = torch.zeros(num_envs, dtype=torch.bool)
    env._stem_detached = torch.zeros(num_envs, dtype=torch.bool)
    env._height_gain_m = torch.zeros(num_envs)
    env._drift_m = torch.zeros(num_envs)
    env._rel_speed_m_s = torch.zeros(num_envs)
    env._joint_excursion_rad = torch.zeros(num_envs)
    env._max_applied_effort_nm = torch.zeros(num_envs)
    env._self_collision_clearance_m = torch.full((num_envs,), float("inf"))
    env._collider_distance_m = torch.full((num_envs,), float("inf"))
    env._penetration_depth_m = torch.zeros(num_envs)
    env._open_hand_baseline_flag = torch.zeros(num_envs, dtype=torch.bool)
    # v1.4.0：真实调用源读取物理子步守卫的 streak / peak / 粘滞硬故障。
    _hard_fault_names = (
        "overload",
        "nonfinite",
        "penetration",
        "joint_limit",
        "self_collision",
        "instability",
        "joint_effort",
    )
    env._physical_hold_guard = SimpleNamespace(
        streak=torch.zeros(num_envs, dtype=torch.long),
        peak_streak=torch.zeros(num_envs, dtype=torch.long),
        hard_faults={
            name: torch.zeros(num_envs, dtype=torch.bool)
            for name in _hard_fault_names
        },
    )
    env._faults = {
        name: torch.zeros(num_envs, dtype=torch.bool)
        for name in (
            "overload",
            "nonfinite",
            "penetration",
            "joint_limit",
            "self_collision",
            "instability",
            "joint_effort",
            "stem_fracture",
        )
    }
    assert C.FRUIT_RADIUS_M > 0.0
    return env


def _tracker_stub(num_envs: int = 2):
    """用真实 stage 规格构造累加器（与 env 使用同一组参数）。"""

    from research.fixed_fruit_grasp_v1 import contract as C

    spec = C.stage_spec("hold")
    return _EpisodeTracker(
        num_envs,
        device="cpu",
        policy_dt_s=C.POLICY_DT_S,
        stage=spec.stage,
        success_criterion=spec.success_criterion,
        required_hold_steps=spec.required_unsupported_hold_steps,
        stage_min_height_gain_m=spec.min_height_gain_m,
        stage_success_requires_height_gain=spec.success_requires_height_gain,
        stage_requires_lift_program=spec.requires_lift_program,
        stage_is_goal=spec.counts_as_goal,
        goal_min_height_gain_m=C.MIN_HEIGHT_GAIN_M,
        goal_status=spec.goal_status,
        action_effect_ok=True,
        open_hand_baseline=False,
        immediate_failure_steps=5,
        contact_force_threshold_n=C.CONTACT_FORCE_N,
        min_contact_fingers=C.MIN_CONTACT_FINGERS,
        required_hold_substeps=spec.required_unsupported_hold_substeps,
        sim_dt_s=C.SIM_DT_S,
    )


def test_real_tracker_state_matches_required_fields_and_is_accepted() -> None:
    """真实调用源（env._tracker_state）必须与累加器要求逐字一致，并被接受。"""

    required = set(_EpisodeTracker._REQUIRED_STATE_KEYS)
    state = _tracker_state_stub()._tracker_state()
    assert set(state) == required, (
        f"多={sorted(set(state) - required)} 少={sorted(required - set(state))}"
    )
    tracker = _tracker_stub(2)
    tracker.observe(state)  # 真实集成：不抛 KeyError，且已累加
    assert tracker.steps.tolist() == [1, 1]
    assert tracker.open_hand_steps.tolist() == [0, 0]


def test_tracker_stub_state_records_open_hand_mode() -> None:
    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv

    env = _tracker_state_stub(2)
    env._open_hand_baseline_flag.fill_(True)
    # 真实 `set_open_hand_baseline` 会同步这两处；这里直接调用真实方法验证同步。
    tracker = _tracker_stub(2)
    env._tracker = tracker
    env._open_hand_baseline = False
    FixedFruitGraspEnv.set_open_hand_baseline(env, True)
    assert bool(env._open_hand_baseline_flag.all())
    assert tracker._open_hand_baseline is True


def test_reset_pose_record_separates_requested_and_actual_position() -> None:
    """位置选择证据：请求的固定预置距离 vs 父环境自动外推后的实际落点。"""

    from types import SimpleNamespace

    import torch

    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv

    env = FixedFruitGraspEnv.__new__(FixedFruitGraspEnv)
    env._is_closed = True
    env.sim = SimpleNamespace(device="cpu")
    env.scene = SimpleNamespace(num_envs=1)
    env._ee_body_id = 0
    env._fingertip_contact_body_ids = [1, 2, 3, 4, 5]
    env.cfg = SimpleNamespace(
        closed_grasp_fruit_center_m=(-0.061, -0.013, -0.177),
        pinch_working_offset_m=(0.0, 0.0, 0.0),
        pregrasp_distance_range_m=(0.005, 0.005),
        support_release_step=40,
    )
    env._pinch_reference_center_w = torch.tensor([[0.0, 0.0, 0.2]])
    env._grasp_basis_w = torch.eye(3).repeat(1, 1, 1)
    env._tracker = SimpleNamespace(episode_index=torch.zeros(1, dtype=torch.long))
    env.last_reset_poses = []
    env.first_reset_poses = []
    env.reset_pose_count = 0
    fruit = torch.tensor([[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]])
    # 真实语料形状是 (num_envs, num_bodies, 7) / (num_envs, num_bodies, 3)。
    bodies = torch.tensor(
        [
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
            *[[float(d), 0.0, 0.12, 0.0, 0.0, 0.0, 1.0] for d in range(5)],
        ]
    ).unsqueeze(0)
    positions = torch.tensor(
        [[0.0, 0.0, 0.1], *[[float(d), 0.0, 0.12] for d in range(5)]]
    ).unsqueeze(0)
    env._tomato = SimpleNamespace(
        data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=fruit))
    )
    env._robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pose_w=SimpleNamespace(torch=bodies),
            body_pos_w=SimpleNamespace(torch=positions),
        )
    )
    FixedFruitGraspEnv._record_reset_pose(env, torch.tensor([0]))
    row = env.last_reset_poses[0]
    assert row["requested_pregrasp_distance_m"] == 0.005
    assert row["closed_grasp_fruit_center_m"] == [-0.061, -0.013, -0.177]
    # 标定中心 = ee(0,0,0.1) + 抓取基 @ (-0.061,-0.013,-0.177)
    assert row["calibrated_center_w_m"] == pytest.approx(
        [-0.061, -0.013, -0.077]
    )
    assert row["actual_offset_norm_m"] == pytest.approx(
        float(torch.tensor([0.061, 0.013, 0.277]).norm().item()), rel=1e-5
    )
    assert len(row["fingertip_frame_positions_w_m"]) == 5
    assert row["palm_position_w_m"] == pytest.approx([0.0, 0.0, 0.1])
    assert env.reset_pose_count == 1


def test_env_exposes_the_contract_surface() -> None:
    """主 smoke / worker-medium 依赖的接口面必须存在且可调用。"""

    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv, make_config

    cfg = make_config(1, "cpu", 1, "hold")
    assert cfg.stage == "hold"
    for name in (
        "task_identity",
        "set_open_hand_baseline",
        "open_hand_action",
        "closure_reference_action",
        "reference_joint_targets",
        "pop_completed_episodes",
        "episode_summary",
        "action_effect_report",
    ):
        assert callable(getattr(FixedFruitGraspEnv, name)), name
