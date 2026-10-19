"""配置/工厂/守护的 CPU 测试（导入 isaaclab，但不启动 Kit）。"""

from __future__ import annotations

import dataclasses

import pytest

from research.fixed_fruit_grasp_v1 import contract as C
from research.fixed_fruit_grasp_v1.env import (
    FixedFruitGraspCfg,
    FixedFruitGraspContactCfg,
    FixedFruitGraspPDActuator,
    make_config,
    validate_cfg,
)


def test_make_config_fixes_geometry_randomization_and_delay() -> None:
    cfg = make_config(3, "cuda:0", 11, "hold")
    assert cfg.stage == "hold"
    assert cfg.scene.num_envs == 3
    assert cfg.seed == 11
    # Isaac Lab 3.x 的真实设备绑定在 sim.device；镜像字段必须一致。
    assert cfg.sim.device == "cuda:0"
    assert cfg.device == "cuda:0"
    params = cfg.events.fruit_geometry.params
    for axis in ("x", "y", "z"):
        assert params["axis_ranges_m"][axis] == (C.FRUIT_RADIUS_M, C.FRUIT_RADIUS_M)
    assert params["density_range_kg_m3"] == (C.FRUIT_DENSITY_KG_M3,) * 2
    assert cfg.target_position_noise_m == 0.0
    assert cfg.fruit_axis_relative_noise == 0.0
    assert tuple(cfg.action_delay_steps) == (0.0, 0.0)
    assert tuple(cfg.tactile_force_gain) == (1.0, 1.0)
    assert tuple(cfg.tactile_force_bias_n) == (0.0, 0.0)
    assert cfg.pregrasp_distance_range_m[0] == cfg.pregrasp_distance_range_m[1]
    assert cfg.palm_fruit_collision_filter_enabled is False
    assert cfg.robot.spawn.articulation_props.enabled_self_collisions is True
    assert cfg.support_release_step == C.STAGE_SPECS["hold"].support_release_step
    assert cfg.required_unsupported_hold_steps == 60
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (
        C.ACTION_DIM,
        C.ACTOR_OBSERVATION_DIM,
        C.CRITIC_STATE_DIM,
    )


def test_make_config_applies_minimal_drive_repair_without_expanding_effort() -> None:
    cfg = make_config(1, "cpu", 5, "contact")
    stiffness = {
        "g20_independent": C.ACTUATOR_STIFFNESS,
        "g20_followers": C.FOLLOWER_STIFFNESS,
    }
    damping = {
        "g20_independent": C.ACTUATOR_DAMPING,
        "g20_followers": C.FOLLOWER_DAMPING,
    }
    for group in ("g20_independent", "g20_followers"):
        actuator = cfg.robot.actuators[group]
        assert actuator.class_type is FixedFruitGraspPDActuator
        assert actuator.armature == C.ACTUATOR_ARMATURE_KG_M2 == 0.01
        assert actuator.effort_limit_sim == C.ACTUATOR_EFFORT_LIMIT_NM == 0.20
        assert actuator.velocity_limit_sim == C.ACTUATOR_VELOCITY_LIMIT_RAD_S
        assert actuator.stiffness == stiffness[group]
        assert actuator.damping == damping[group]
        assert actuator.friction == C.ACTUATOR_FRICTION


def test_contact_and_hold_only_differ_in_release_window() -> None:
    hold = make_config(1, "cpu", 5, "hold")
    contact = make_config(1, "cpu", 5, "contact")
    assert contact.support_release_step == C.STAGE_SPECS["contact"].support_release_step
    assert contact.required_unsupported_hold_steps == 20
    assert hold.required_unsupported_hold_steps == 60
    # 物理合同（几何/密度/驱动/自碰撞）在阶段之间保持一致。
    assert contact.events.fruit_geometry.params == hold.events.fruit_geometry.params
    assert (
        contact.robot.spawn.articulation_props.enabled_self_collisions
        == hold.robot.spawn.articulation_props.enabled_self_collisions
    )


def test_lift_stage_fails_closed() -> None:
    with pytest.raises(C.LiftStageUnavailable):
        make_config(1, "cpu", 5, "lift")


def test_unknown_stage_and_unknown_override_rejected() -> None:
    with pytest.raises(C.UnknownStageError):
        make_config(1, "cpu", 5, "grasp")
    with pytest.raises(ValueError):
        make_config(1, "cpu", 5, "hold", not_a_field=1)


def test_explicit_fixed_initial_pose_override_is_accepted_and_recorded() -> None:
    cfg = make_config(
        1,
        "cpu",
        5,
        "hold",
        pregrasp_distance_m=0.078,
        fruit_center_local_m=(-0.058, -0.010, -0.172),
    )
    assert cfg.pregrasp_distance_range_m == (0.078, 0.078)
    assert cfg.closed_grasp_fruit_center_m == (-0.058, -0.010, -0.172)
    with pytest.raises(ValueError):
        make_config(1, "cpu", 5, "hold", fruit_center_local_m=(0.0, 0.0))


def test_validate_cfg_rejects_randomized_distance_and_device_mismatch() -> None:
    cfg = FixedFruitGraspCfg()
    cfg.pregrasp_distance_range_m = (0.05, 0.09)
    with pytest.raises(ValueError, match="确定单值"):
        validate_cfg(cfg)
    cfg = FixedFruitGraspCfg()
    cfg.device = "cpu"
    with pytest.raises(ValueError, match="device"):
        validate_cfg(cfg)
    cfg = FixedFruitGraspCfg()
    cfg.robot.spawn.articulation_props.enabled_self_collisions = False
    with pytest.raises(ValueError, match="self-collision"):
        validate_cfg(cfg)


def test_validate_cfg_requires_real_self_collisions_and_no_contact_proxy() -> None:
    """真实求解器自碰撞必须开启；几何接触代理必须关闭（两处都不能静默弱化）。"""

    cfg = FixedFruitGraspCfg()
    cfg.real_self_collisions = False
    cfg.robot.spawn.articulation_props.enabled_self_collisions = False
    with pytest.raises(ValueError, match="真实自碰撞必须开启"):
        validate_cfg(cfg)
    cfg = FixedFruitGraspCfg()
    cfg.geometric_contact_proxy_enabled = True
    with pytest.raises(ValueError, match="几何接触代理"):
        validate_cfg(cfg)
    assert FixedFruitGraspCfg().geometric_contact_proxy_enabled is False


def test_empty_self_collision_pair_resolution_fails_closed() -> None:
    """声明的自碰撞诊断对一个都解析不出来时必须报错，而不是静默降级。"""

    from types import SimpleNamespace

    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv

    env = FixedFruitGraspEnv.__new__(FixedFruitGraspEnv)
    env._robot = SimpleNamespace(body_names=["base_link", "thumb_link4"])
    env._hand_collision_body_ids = [0, 1]
    env.cfg = SimpleNamespace(self_collision_pairs=(("thumb_link4", "index_link3"),))
    with pytest.raises(C.FixedFruitGraspError, match="不得静默过滤为空"):
        env._select_self_collision_pairs()
    env._robot = SimpleNamespace(body_names=["thumb_link4", "index_link3"])
    env._hand_collision_body_ids = [0, 1]
    assert env._select_self_collision_pairs() == (("thumb_link4", "index_link3"),)
    assert env._skipped_self_collision_pairs == ()


def test_episode_length_matches_fixed_steps() -> None:
    cfg = make_config(1, "cpu", 1, "hold")
    assert cfg.episode_length_s == pytest.approx(C.EPISODE_STEPS * C.POLICY_DT_S)
    assert cfg.episode_length_s / (cfg.sim.dt * cfg.decimation) == pytest.approx(
        C.EPISODE_STEPS
    )


def test_stage_cfg_subclasses_default_to_their_stage() -> None:
    contact = FixedFruitGraspContactCfg()
    assert contact.stage == "contact"
    assert contact.support_release_step == C.STAGE_SPECS["contact"].support_release_step
    validate_cfg(contact)
    names = {field.name for field in dataclasses.fields(FixedFruitGraspCfg)}
    assert "device" in names  # 镜像字段：真实绑定仍是 sim.device
