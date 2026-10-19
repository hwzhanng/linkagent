"""固定 67 mm 单果、多指真实接触抓握保持环境（holding-only 第一版）。

设计边界（与 ``.omo/plans/ppo-grasp-first-result-20261013.md`` 一致）：

* 底座是 :class:`NativeSmallFruitPinchEnv`；复用 14D 动作映射、临界阻尼动作
  滤波与真实 PhysX 指尖接触读取。
* 不启用旧三指 Verifier 的逐帧软带/扰动合同；旧 Verifier 只保留为自己
  ``_force_slow_phase`` 钩子之外的历史属性，不参与任何判据。
* 固定 67 mm 等效直径单果、固定初始位置与朝向；尺寸/位置/质量/摩擦/
  观测噪声/动作延迟全部固定为确定值，且由 :func:`validate_cfg` 强制。
* 初始无接触；果实由原始声明的果梗弹簧支撑，在固定策略步
  ``cfg.support_release_step`` 彻底解除，不再由“已满足成功”触发。
* 真实掌面碰撞与真实自碰撞开启（``enabled_self_collisions=True``，不改 USD），
  不使用 base_link↔Tomato 过滤，也不在运行时关闭自碰撞。
* 提举需要一根明确声明的受驱动升降关节；本版本未实现，``stage='lift'``
  fail-closed。交付物是保持任务（指标 ``unsupported_hold_s``）。

本模块不启动 Kit/PhysX：只构造配置与环境对象，物理烟测由主线程执行。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Final

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_apply, quat_box_minus

from research.small_fruit_pinch_v1.source.native_env import (
    LEFT_HAND_BODY_NAMES,
    LEFT_HAND_USD,
    LeftHandDirectionalPDActuator,
    NativeSmallFruitPinchCfg,
    NativeSmallFruitPinchEnv,
)
from research.small_fruit_pinch_v1.source.pinch_action_manifold import (
    pinch_action_to_joint_targets,
)

from . import contract as C
from .physical_hold import (
    HARD_FAULT_NAMES,
    PhysicalHoldError,
    PhysicalHoldGuard,
    PhysicalSubstepState,
    required_physical_substeps,
)

_FIXED_PREGRASP_DISTANCE_M: Final = 0.065
_FAULT_NAMES: Final = (
    "overload",
    "nonfinite",
    "penetration",
    "joint_limit",
    "self_collision",
    "instability",
    "joint_effort",
)
_TRACKED_FLAG_NAMES: Final = _FAULT_NAMES + ("stem_fracture",)

# 逐物理子步守护的硬故障名字必须与策略采样硬故障完全一致；不一致意味着
# 终止/惩罚/记录三处口径分叉，直接 fail-closed。
if tuple(HARD_FAULT_NAMES) != tuple(_FAULT_NAMES):
    raise RuntimeError(
        f"物理子步硬故障名字与策略采样不一致：{HARD_FAULT_NAMES} != {_FAULT_NAMES}"
    )

_HARD_FAULT_REWARD_NAMES: Final = (
    "overload",
    "nonfinite",
    "penetration",
    "joint_limit",
    "self_collision",
    "instability",
    "joint_effort",
)
_APPROACH_SHAPING_GAMMA: Final = 0.995
_APPROACH_GAP_DECAY_M: Final = 0.025
_STABILITY_DRIFT_DECAY_M: Final = 0.02
_ASSET_ROOT: Final = Path(LEFT_HAND_USD).resolve().parent
_ASSET_FILES: Final = (
    "left_hand_l20_physx.usd",
    "configuration/left_hand_l20_physx_base.usd",
    "configuration/left_hand_l20_physx_physics.usd",
    "configuration/left_hand_l20_physx_sensor.usd",
    "left_hand_tip_pad_collision.usda",
)


def _approach_potential_from_digit_gaps(digit_gaps_m: torch.Tensor) -> torch.Tensor:
    """从 AABB 指尖距离得到有界接近势；只用于 shaping，不代表接触。"""

    if digit_gaps_m.ndim != 2 or digit_gaps_m.shape[1] < 2:
        raise ValueError("digit_gaps_m 必须是至少含两个手指的二维张量")
    per_digit = torch.exp(
        -digit_gaps_m.clamp_min(0.0) / _APPROACH_GAP_DECAY_M
    )
    return per_digit.topk(2, dim=1).values.mean(dim=1)


def _potential_difference_reward(
    potential: torch.Tensor,
    previous_potential: torch.Tensor,
    ready: torch.Tensor,
    *,
    gamma: float = _APPROACH_SHAPING_GAMMA,
) -> torch.Tensor:
    """计算 ``gamma * Phi(s') - Phi(s)``；首次只缓存势值、不发奖励。"""

    delta = torch.where(
        ready,
        float(gamma) * potential - previous_potential,
        torch.zeros_like(potential),
    )
    previous_potential.copy_(potential)
    ready.fill_(True)
    return delta


def _clear_selected_approach_potential(
    previous_potential: torch.Tensor, ready: torch.Tensor, env_ids: torch.Tensor
) -> None:
    """仅清除被 reset 的环境对应的接近势历史。"""

    ids = torch.as_tensor(
        env_ids, device=previous_potential.device, dtype=torch.long
    ).flatten()
    previous_potential[ids] = 0.0
    ready[ids] = False


def _gated_stability_reward(
    drift_m: torch.Tensor, hold_ok: torch.Tensor, released: torch.Tensor
) -> torch.Tensor:
    """稳定项要求真实撤支撑且当前帧通过接触、速度、漂移及故障守护。"""

    stability = torch.exp(-drift_m.clamp_min(0.0) / _STABILITY_DRIFT_DECAY_M)
    return stability * hold_ok.to(dtype=stability.dtype) * released.to(
        dtype=stability.dtype
    )


def _hard_fault_mask(faults: dict[str, torch.Tensor]) -> torch.Tensor:
    """与 ``_get_dones`` 同一七项集合的硬故障掩码。"""

    return torch.stack(
        [faults[name].to(dtype=torch.bool) for name in _HARD_FAULT_REWARD_NAMES],
        dim=0,
    ).any(dim=0)


def _hard_fault_penalty(hard_fault: torch.Tensor, penalty: float) -> torch.Tensor:
    return -float(penalty) * hard_fault.to(dtype=torch.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_identity() -> dict[str, Any]:
    """选定资产的身份：USD 组合层闭包内每个文件的 sha256 + 闭包哈希。

    ``left_hand_l20_physx.usd`` 本身只有约 1.6 KB，它 sublayer/reference 到
    ``configuration/`` 下的物理层与约 17 MB 的 base 载荷；只哈希根层不足以
    锁定实际资产，因此这里对资产目录内全部文件做闭包哈希。
    """

    files: dict[str, str] = {}
    missing: list[str] = []
    total_bytes = 0
    for path in sorted(_ASSET_ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(_ASSET_ROOT).as_posix()
        files[relative] = _sha256_file(path)
        total_bytes += path.stat().st_size
    for required in _ASSET_FILES:
        if required not in files:
            missing.append(required)
    manifest = "\n".join(f"{name}  {digest}" for name, digest in sorted(files.items()))
    closure_sha256 = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    declared_hash_file = _ASSET_ROOT / ".asset_hash"
    declared = (
        declared_hash_file.read_text(encoding="utf-8").strip()
        if declared_hash_file.is_file()
        else None
    )
    return {
        "root": str(_ASSET_ROOT),
        # ``usd_path`` / ``asset_config_path`` 同时供外部账本独立重算哈希。
        "usd_path": str(Path(LEFT_HAND_USD).resolve()),
        "asset_config_path": str(
            (_ASSET_ROOT / "configuration/left_hand_l20_physx_physics.usd").resolve()
        ),
        "selected": "canonical isolated left hand USD (16 active + 5 mimic followers + 27 bodies)",
        "selection_note": (
            "派生 PPV1 资产 physical_prior_v1/left_hand_ppv1.usda 要求 mimic offset=0 且依赖 "
            "NewtonMimicAPI；本任务不继承未证实的运行时后端，故不选它。"
        ),
        "not_selected": {
            "derived_ppv1_usda": "research/small_fruit_pinch_v1/assets/physical_prior_v1/left_hand_ppv1.usda",
            "tip_pad_overlay_usda": "research/small_fruit_pinch_v1/assets/left_hand_urdf/left_hand_tip_pad_collision.usda",
        },
        "files": files,
        "closure_sha256": closure_sha256,
        "closure_algorithm": "sha256 of sorted 'relative_path  sha256' manifest",
        "closure_file_count": len(files),
        "closure_bytes": int(total_bytes),
        "missing_files": missing,
        "declared_asset_hash_file": declared,
    }


def _source_identity() -> dict[str, Any]:
    """实际执行的父模块源码身份（路径 + sha256）。

    新包的包内哈希不覆盖这些真正承载环境/驱动/动作映射/滤波的父模块，
    因此这里显式列出，供外部账本独立复核。
    """

    import simulation.tomato_pick.isaac.assets as assets_module
    import simulation.tomato_pick.isaac.env as isaac_env_module
    import simulation.tomato_pick.isaac.env_cfg as isaac_cfg_module
    import simulation.tomato_pick.isaac.geometry_sampling as geometry_module
    from research.small_fruit_pinch_v1.source import native_env as pinch_native_module
    from research.small_fruit_pinch_v1.source import pinch_action_manifold as manifold_module
    from research.small_fruit_pinch_v1.source import temporal as temporal_module
    from research.small_fruit_pinch_v1.source.legacy_bridge import (
        load_legacy_hand_manifold,
    )

    sources = {
        "pinch_native_env": pinch_native_module.__file__,
        "isaac_env": isaac_env_module.__file__,
        "isaac_env_cfg": isaac_cfg_module.__file__,
        "isaac_assets": assets_module.__file__,
        "geometry_sampling": geometry_module.__file__,
        "pinch_action_manifold": manifold_module.__file__,
        "temporal_filter": temporal_module.__file__,
        "vendor_hand_manifold": load_legacy_hand_manifold().__file__,
    }
    identity: dict[str, Any] = {}
    for name, path in sources.items():
        resolved = Path(path).resolve()
        identity[name] = {
            # ``source_path`` 同时让外部账本按路径独立重算哈希。
            "source_path": str(resolved),
            "sha256": _sha256_file(resolved) if resolved.is_file() else None,
        }
    return identity


class FixedFruitGraspPDActuator(LeftHandDirectionalPDActuator):
    """canonical 左手显式 PD 驱动 + wave2 记录的两处最小驱动修复。

    1. 限位制动通道 ``brake = 0.30*|qd|`` 本身无界，接触振荡时以远超大
       授权上限的力矩绕过上游 ``_clip_effort``；这里在 brake 合成之后按
       逐轴 ``effort_limit`` 再裁剪一次（域内输出不变）。
    2. 任何非有限努力收敛为 0，避免污染 PhysX 状态。

    ``armature=0.01`` 由 cfg 显式写入（显式积分稳定性约束），effort 上限
    不扩大。
    """

    def compute(
        self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ):
        result = super().compute(control_action, joint_pos, joint_vel)
        effort = torch.nan_to_num(
            self.applied_effort, nan=0.0, posinf=0.0, neginf=0.0
        )
        effort = torch.clip(effort, min=-self.effort_limit, max=self.effort_limit)
        self.applied_effort = effort
        self.computed_effort = torch.nan_to_num(
            self.computed_effort, nan=0.0, posinf=0.0, neginf=0.0
        )
        if result.joint_efforts is not None:
            result.joint_efforts = effort
        return result


def _lift_unavailable_message() -> str:
    return (
        "lift 阶段需要一根明确声明的受驱动升降关节（USD 上的真实关节 + 驱动），"
        f"本版本未实现（LIFT_STAGE_STATUS={C.LIFT_STAGE_STATUS}）。禁止用逐帧传送手根/"
        "果实或给果实加力冒充提举。请使用 stage='contact'/'hold' 的 holding-only 任务，"
        "并按 contract.LIFT_PROGRAM 在同一 actor 上另起一轮接线。"
    )


def _apply_fixed_task_defaults(cfg: "FixedFruitGraspCfg") -> None:
    """把固定几何/物性与最小驱动修复写入 cfg。

    ``configclass`` 会把嵌套配置折叠成 ``default_factory``，因此不能从父类
    读取 ``events`` / ``robot``；这里实例化 canonical 配置后只做必要覆写。
    """

    base = NativeSmallFruitPinchCfg()
    cfg.events = deepcopy(base.events)
    params = cfg.events.fruit_geometry.params
    # 半轴与密度范围退化为单点：prestartup 采样结果与 seed 无关，
    # 于是 67 mm 等效直径与质量是确定的（质量仍由父环境计算并写回）。
    params["axis_ranges_m"] = {
        axis: (C.FRUIT_RADIUS_M, C.FRUIT_RADIUS_M) for axis in ("x", "y", "z")
    }
    params["density_range_kg_m3"] = (C.FRUIT_DENSITY_KG_M3, C.FRUIT_DENSITY_KG_M3)
    params["accepted_mass_kg"] = (C.FRUIT_MASS_KG * 0.5, C.FRUIT_MASS_KG * 2.0)
    params["maximum_axis_ratio"] = 1.0 + 1.0e-9

    cfg.stem_linear_natural_frequency_rad_s = C.STEM_LINEAR_NATURAL_FREQUENCY_RAD_S
    cfg.robot = deepcopy(base.robot)
    from isaaclab.sensors import ContactSensorCfg
    # 补齐所有27个真实刚体→果实的只读接触通道，避免只看五指而漏掉掌面/近端。
    cfg.fingertip_contacts = deepcopy(base.fingertip_contacts)
    cfg.auxiliary_contacts = deepcopy(base.auxiliary_contacts)
    existing = {c.prim_path.rsplit('/', 1)[-1] for c in [*cfg.fingertip_contacts.values(), *cfg.auxiliary_contacts.values()]}
    for body in LEFT_HAND_BODY_NAMES:
        if body not in existing:
            cfg.auxiliary_contacts['penetration_' + body] = ContactSensorCfg(
                prim_path='/World/envs/env_.*/Robot/' + body,
                update_period=1.0 / 120.0, history_length=0,
                filter_prim_paths_expr=['{ENV_REGEX_NS}/Tomato'],
                track_contact_points=False, track_pose=False,
                max_contact_data_count_per_prim=128,
            )
    for contact_cfg in [*cfg.fingertip_contacts.values(), *cfg.auxiliary_contacts.values()]:
        contact_cfg.max_contact_data_count_per_prim = 4096
    cfg.robot.spawn.articulation_props.enabled_self_collisions = bool(
        cfg.real_self_collisions
    )
    for group in ("g20_independent", "g20_followers"):
        actuator = cfg.robot.actuators[group]
        actuator.armature = C.ACTUATOR_ARMATURE_KG_M2
        actuator.class_type = FixedFruitGraspPDActuator


@configclass
class FixedFruitGraspCfg(NativeSmallFruitPinchCfg):
    """固定单果、固定初始位姿、固定时刻撤支撑的抓握保持配置。

    顶层 ``stage`` 键在 ``run_utils.normalize_contract_identity`` 中作为运行期
    字段被剔除，因此课程阶段切换允许同一物理合同下的 actor-only 迁移。
    """

    # ---- 任务身份 / 阶段（阶段差异集中在 stage 键内，见模块 docstring）----
    stage: str = C.DEFAULT_STAGE
    task_id: str = C.TASK_ID
    task_version: str = C.TASK_VERSION
    support_release_step: int = C.STAGE_SPECS[C.DEFAULT_STAGE].support_release_step
    required_unsupported_hold_steps: int = C.STAGE_SPECS[
        C.DEFAULT_STAGE
    ].required_unsupported_hold_steps

    # ---- 时序：固定整回合 ----
    episode_length_s = C.EPISODE_STEPS * C.POLICY_DT_S
    seed = 20261013
    # Isaac Lab 3.x 的 ``DirectRLEnvCfg`` 没有 ``device`` 字段：真正的绑定是
    # ``cfg.sim.device``。这里保留 ``device`` 作为镜像簿记字段，并在
    # validate_cfg 中强制两者一致，避免“写了 device 却仍在默认 cuda:0 上跑”。
    device: str = "cuda:0"

    # ---- 关闭一切随机化 / 延迟 / 观测噪声 ----
    target_position_noise_m = 0.0
    fruit_axis_relative_noise = 0.0
    tactile_force_gain = (1.0, 1.0)
    tactile_force_bias_n = (0.0, 0.0)
    action_delay_steps = (0, 0)
    pregrasp_distance_range_m = (
        _FIXED_PREGRASP_DISTANCE_M,
        _FIXED_PREGRASP_DISTANCE_M,
    )
    pregrasp_lateral_offset_range_m = (0.0, 0.0)
    pregrasp_axial_noise_m = 0.0
    pregrasp_arm_joint_noise_deg = 0.0
    minimum_initial_clearance_m = 0.002
    # 断力取固定且实际不可达值：解除完全由 support_release_step 决定。
    break_force_range_n = (
        C.STEM_DECLARED_BREAK_FORCE_N,
        C.STEM_DECLARED_BREAK_FORCE_N,
    )
    # 真实掌面参与承重：禁止 base_link↔Tomato 过滤；开启真实自碰撞。
    palm_fruit_collision_filter_enabled = False
    real_self_collisions: bool = True
    self_collision_proxy_terminates = False
    # 几何接触代理会掩盖真实 PhysX 接触；本任务只允许真实接触，validate_cfg 拒绝开启。
    geometric_contact_proxy_enabled = False

    # ---- 判据与守护 ----
    min_contact_fingers: int = C.MIN_CONTACT_FINGERS
    contact_force_threshold_n: float = C.CONTACT_FORCE_N
    hold_max_drift_m: float = C.HOLD_MAX_DRIFT_M
    hold_max_relative_speed_m_s: float = C.HOLD_MAX_RELATIVE_SPEED_M_S
    drop_height_tolerance_m: float = C.DROP_HEIGHT_TOLERANCE_M
    drop_max_drift_m: float = C.DROP_MAX_DRIFT_M
    penetration_limit_m: float = C.PENETRATION_LIMIT_M
    collider_overlap_limit_m: float = C.COLLIDER_OVERLAP_LIMIT_M
    joint_limit_margin_rad: float = C.JOINT_LIMIT_MARGIN_RAD
    self_collision_penetration_limit_m: float = C.SELF_COLLISION_PENETRATION_LIMIT_M
    instability_max_joint_velocity_rad_s: float = C.INSTABILITY_MAX_JOINT_VELOCITY_RAD_S
    instability_max_fruit_speed_m_s: float = C.INSTABILITY_MAX_FRUIT_SPEED_M_S
    self_collision_pairs: tuple[tuple[str, str], ...] = C.SELF_COLLISION_PAIRS
    immediate_failure_steps: int = 5

    # ---- 奖励（全部实时、非锁存；失接触立即归零）----
    # v1.5.0 是 reward-only 版本：相对 v1.4.0 只有下列奖励权重变化，物理、
    # 判据、120 Hz 子步守护与终结语义一律不变；数值同时写入
    # ``task_identity()["reward"]["weights"]``，逐项快照见
    # ``extras["fixed_grasp_reward_terms"]``。
    contact_reward_weight: float = 0.25
    opposition_reward_weight: float = 0.25
    band_reward_weight: float = 0.35
    closure_reward_weight: float = 0.0
    approach_reward_weight: float = 1.0
    unsupported_hold_reward_weight: float = 6.0
    stability_reward_weight: float = 0.50
    height_reward_weight: float = 0.0
    palm_only_penalty_weight: float = 0.10
    overload_penalty_weight: float = 0.30
    forbidden_penalty_weight: float = 0.0
    smoothness_penalty_weight: float = 0.12
    fault_penalty: float = 8.00
    drop_penalty: float = 6.00
    success_bonus: float = 30.0

    # ---- 运行期开关 ----
    open_hand_baseline: bool = False

    def __post_init__(self) -> None:
        _apply_fixed_task_defaults(self)


def _episode_step_limit(cfg: FixedFruitGraspCfg) -> int:
    return int(
        math.ceil(
            float(cfg.episode_length_s)
            / (float(cfg.sim.dt) * int(cfg.decimation))
        )
    )


def validate_cfg(cfg: FixedFruitGraspCfg) -> None:
    """拒绝任何偏离“固定场景 + 固定时序 + 最小驱动修复”的配置。"""

    spec = C.stage_spec(cfg.stage)
    if spec.stage == "lift":
        raise C.LiftStageUnavailable(_lift_unavailable_message())
    if int(cfg.action_space) != C.ACTION_DIM:
        raise ValueError(f"action_space 必须为 {C.ACTION_DIM}，得到 {cfg.action_space}")
    if int(cfg.observation_space) != C.ACTOR_OBSERVATION_DIM:
        raise ValueError(
            f"observation_space 必须为 {C.ACTOR_OBSERVATION_DIM}，得到 {cfg.observation_space}"
        )
    if int(cfg.state_space) != C.CRITIC_STATE_DIM:
        raise ValueError(
            f"state_space 必须为 {C.CRITIC_STATE_DIM}，得到 {cfg.state_space}"
        )
    if int(cfg.scene.num_envs) < 1:
        raise ValueError("num_envs 必须为正数")
    if str(cfg.sim.device) != str(cfg.device):
        raise ValueError(
            "device 绑定不一致：Isaac Lab 3.x 以 cfg.sim.device 为准，"
            f"但 cfg.device={cfg.device!r} 与 cfg.sim.device={cfg.sim.device!r} 不同"
        )
    steps = _episode_step_limit(cfg)
    if steps != C.EPISODE_STEPS:
        raise ValueError(
            f"episode_length_s/sim.dt/decimation 必须对应 {C.EPISODE_STEPS} 个策略步，得到 {steps}"
        )
    release_step = int(cfg.support_release_step)
    hold_steps = int(cfg.required_unsupported_hold_steps)
    if not 0 <= release_step < steps:
        raise ValueError(f"support_release_step 必须在 [0, {steps})，得到 {release_step}")
    if hold_steps < 1:
        raise ValueError("required_unsupported_hold_steps 必须为正数")
    if release_step + hold_steps > steps:
        raise ValueError(
            f"固定释放步({release_step}) + 保持窗({hold_steps}) 超过回合上限 {steps}"
        )

    for name, expected in (
        ("target_position_noise_m", 0.0),
        ("fruit_axis_relative_noise", 0.0),
        ("pregrasp_axial_noise_m", 0.0),
        ("pregrasp_arm_joint_noise_deg", 0.0),
    ):
        if float(getattr(cfg, name)) != expected:
            raise ValueError(f"{name} 必须固定为 {expected}，得到 {getattr(cfg, name)}")
    for name, expected_value in (
        ("action_delay_steps", (0, 0)),
        ("tactile_force_gain", (1.0, 1.0)),
        ("tactile_force_bias_n", (0.0, 0.0)),
        ("pregrasp_lateral_offset_range_m", (0.0, 0.0)),
        ("break_force_range_n", (C.STEM_DECLARED_BREAK_FORCE_N,) * 2),
    ):
        actual = tuple(float(item) for item in getattr(cfg, name))
        if actual != tuple(float(item) for item in expected_value):
            raise ValueError(f"{name} 必须固定为 {expected_value}，得到 {actual}")
    # 预抓取距离必须是确定单值（不随机），但具体值由固定场景选择决定：
    # 本任务不学习接近，65 mm 不是唯一合法值（后续有限场景选择计划允许调整）。
    distance = tuple(float(item) for item in cfg.pregrasp_distance_range_m)
    if len(distance) != 2 or distance[0] != distance[1]:
        raise ValueError(
            f"pregrasp_distance_range_m 必须是确定单值（不随机），得到 {distance}"
        )
    if not 0.0 < distance[0] <= 0.20:
        raise ValueError(
            f"pregrasp_distance_range_m 必须位于 (0, 0.2] m，得到 {distance[0]}"
        )
    for name in ("closed_grasp_fruit_center_m", "pinch_working_offset_m"):
        values = tuple(float(item) for item in getattr(cfg, name))
        if len(values) != 3 or not all(math.isfinite(item) for item in values):
            raise ValueError(f"{name} 必须是有限 3 元组，得到 {values}")
    if bool(cfg.palm_fruit_collision_filter_enabled):
        raise ValueError("禁止启用 base_link↔Tomato 碰撞过滤（真实掌面必须参与）")
    if bool(getattr(cfg, "geometric_contact_proxy_enabled", False)):
        raise ValueError("禁止启用几何接触代理（会掩盖真实 PhysX 接触）")
    if not bool(cfg.real_self_collisions):
        raise ValueError(
            "真实自碰撞必须开启（real_self_collisions=True）；关闭自碰撞的配置不得用于本任务"
        )
    if bool(cfg.robot.spawn.articulation_props.enabled_self_collisions) != bool(
        cfg.real_self_collisions
    ):
        raise ValueError("self-collision 配置与 real_self_collisions 不一致")

    params = cfg.events.fruit_geometry.params
    axes = params["axis_ranges_m"]
    for axis in ("x", "y", "z"):
        if tuple(float(item) for item in axes[axis]) != (C.FRUIT_RADIUS_M,) * 2:
            raise ValueError(
                f"果实半轴 {axis} 必须固定为 {C.FRUIT_RADIUS_M}，得到 {axes[axis]}"
            )
    density = tuple(float(item) for item in params["density_range_kg_m3"])
    if density != (C.FRUIT_DENSITY_KG_M3,) * 2:
        raise ValueError(f"果实密度必须固定为 {C.FRUIT_DENSITY_KG_M3}，得到 {density}")

    for group in ("g20_independent", "g20_followers"):
        actuator = cfg.robot.actuators[group]
        if float(actuator.effort_limit_sim) != C.ACTUATOR_EFFORT_LIMIT_NM:
            raise ValueError(
                f"{group}.effort_limit_sim 不得扩大：期望 {C.ACTUATOR_EFFORT_LIMIT_NM}，"
                f"得到 {actuator.effort_limit_sim}"
            )
        if float(actuator.armature) != C.ACTUATOR_ARMATURE_KG_M2:
            raise ValueError(
                f"{group}.armature 必须为最小修复值 {C.ACTUATOR_ARMATURE_KG_M2}，"
                f"得到 {actuator.armature}"
            )
        if actuator.class_type is not FixedFruitGraspPDActuator:
            raise ValueError(f"{group}.class_type 必须为 FixedFruitGraspPDActuator")


def _stage_cfg(stage: str, **kwargs: Any) -> FixedFruitGraspCfg:
    spec = C.stage_spec(stage)
    if spec.stage == "lift":
        raise C.LiftStageUnavailable(_lift_unavailable_message())
    cfg = FixedFruitGraspCfg()
    cfg.stage = spec.stage
    cfg.support_release_step = int(spec.support_release_step)
    cfg.required_unsupported_hold_steps = int(spec.required_unsupported_hold_steps)
    return cfg


def make_config(
    num_envs: int,
    device: str,
    seed: int,
    stage: str = C.DEFAULT_STAGE,
    *,
    support_release_step: int | None = None,
    pregrasp_distance_m: float | None = None,
    fruit_center_local_m: Sequence[float] | None = None,
    **overrides: Any,
) -> FixedFruitGraspCfg:
    """构造固定单果抓握配置（主线程/主 smoke 的唯一入口）。

    ``stage='contact'|'hold'`` 可行；``stage='lift'`` 立即
    :class:`LiftStageUnavailable`（提举未实现，避免混算）。

    初始位姿由两个显式、确定的量决定（都不随机）：

    * ``fruit_center_local_m`` —— 抓取基坐标下的固定果心（默认 canonical 值）；
    * ``pregrasp_distance_m`` —— 沿接近轴在闭合中心之外的固定预置距离。

    两者都随 ``task_identity()`` 记录，实际 reset 姿态另见
    ``env.last_reset_poses`` / ``env.first_reset_poses``。
    """

    spec = C.stage_spec(stage)
    if spec.stage == "lift":
        raise C.LiftStageUnavailable(_lift_unavailable_message())
    cfg = _stage_cfg(spec.stage)
    cfg.scene.num_envs = int(num_envs)
    # Isaac Lab 3.x 的真实设备绑定是 cfg.sim.device；同时写入镜像字段。
    cfg.sim.device = str(device)
    cfg.device = str(device)
    cfg.seed = int(seed)
    if support_release_step is not None:
        cfg.support_release_step = int(support_release_step)
    if pregrasp_distance_m is not None:
        value = float(pregrasp_distance_m)
        cfg.pregrasp_distance_range_m = (value, value)
    if fruit_center_local_m is not None:
        center = tuple(float(item) for item in fruit_center_local_m)
        if len(center) != 3:
            raise ValueError(f"fruit_center_local_m 必须是 3 元组，得到 {center}")
        cfg.closed_grasp_fruit_center_m = center
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"未知配置字段 {key!r}")
        setattr(cfg, key, value)
    validate_cfg(cfg)
    return cfg


class _EpisodeTracker:
    """逐回合累加器：在 reset 之前把终止快照落成 JSON-safe 行。

    只保存可以直接复核的标量/布尔统计；不保存张量、不读 reward，也不做
    任何截断（``completed_episodes`` 是 append-only 合同）。
    """

    _REQUIRED_STATE_KEYS: Final = (
        "forces_n",
        "raw_forces_n",
        "palm_force_n",
        "forbidden_total_n",
        "required_contact",
        "contact_count",
        "hold_age",
        "hold_ok",
        "support_released",
        "height_gain_m",
        "drift_m",
        "rel_speed_m_s",
        "joint_excursion_rad",
        "max_applied_effort_nm",
        "self_collision_clearance_m",
        "collider_distance_m",
        "penetration_depth_m",
        "open_hand_baseline",
        "physical_hold_streak",
        "physical_hold_peak_streak",
        "physical_hard_fault_flags",
        "faults",
    )

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device | str,
        policy_dt_s: float,
        stage: str,
        success_criterion: str,
        required_hold_steps: int,
        stage_min_height_gain_m: float,
        stage_success_requires_height_gain: bool,
        stage_requires_lift_program: bool,
        stage_is_goal: bool,
        goal_min_height_gain_m: float,
        goal_status: str,
        action_effect_ok: bool,
        open_hand_baseline: bool,
        immediate_failure_steps: int,
        contact_force_threshold_n: float,
        min_contact_fingers: int,
        required_hold_substeps: int,
        sim_dt_s: float,
    ) -> None:
        self._device = device
        self._policy_dt_s = float(policy_dt_s)
        self._stage = str(stage)
        self._success_criterion = str(success_criterion)
        self._required_hold_steps = int(required_hold_steps)
        self._stage_min_height_gain_m = float(stage_min_height_gain_m)
        self._stage_success_requires_height_gain = bool(stage_success_requires_height_gain)
        self._stage_requires_lift_program = bool(stage_requires_lift_program)
        self._stage_is_goal = bool(stage_is_goal)
        self._goal_min_height_gain_m = float(goal_min_height_gain_m)
        self._goal_status = str(goal_status)
        self._action_effect_ok = bool(action_effect_ok)
        self._open_hand_baseline = bool(open_hand_baseline)
        self._immediate_failure_steps = int(immediate_failure_steps)
        self._contact_force_threshold_n = float(contact_force_threshold_n)
        self._min_contact_fingers = int(min_contact_fingers)
        self._required_hold_substeps = int(required_hold_substeps)
        self._sim_dt_s = float(sim_dt_s)
        self._physical_hold_semantics = C.PHYSICAL_HOLD_STREAK_SEMANTICS
        count = int(num_envs)
        self.episode_index = torch.zeros(count, dtype=torch.long, device=device)
        self.steps = torch.zeros(count, dtype=torch.long, device=device)
        self.any_contact_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.contact_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.finger_contact_steps = torch.zeros(
            (count, 5), dtype=torch.long, device=device
        )
        self.required_contact_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.unsupported_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.unsupported_contact_steps = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self.palm_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.palm_only_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.forbidden_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.contact_finger_sum = torch.zeros(count, device=device)
        self.max_contact_fingers = torch.zeros(count, dtype=torch.long, device=device)
        self.peak_force = torch.zeros((count, 5), device=device)
        self.force_sum = torch.zeros((count, 5), device=device)
        self.max_hold_age = torch.zeros(count, dtype=torch.long, device=device)
        self.final_hold_age = torch.zeros(count, dtype=torch.long, device=device)
        self.max_physical_hold_streak = torch.zeros(count, dtype=torch.long, device=device)
        self.final_physical_hold_streak = torch.zeros(count, dtype=torch.long, device=device)
        self.physical_substep_hard_fault_flags = {
            name: torch.zeros(count, dtype=torch.bool, device=device)
            for name in HARD_FAULT_NAMES
        }
        self.max_height_gain = torch.zeros(count, device=device)
        self.final_height_gain = torch.zeros(count, device=device)
        self.max_drift = torch.zeros(count, device=device)
        self.max_rel_speed = torch.zeros(count, device=device)
        self.max_joint_excursion = torch.zeros(count, device=device)
        self.max_applied_effort = torch.zeros(count, device=device)
        self.min_self_collision_clearance = torch.full(
            (count,), float("inf"), device=device
        )
        self.min_collider_distance = torch.full((count,), float("inf"), device=device)
        self.max_penetration_depth = torch.zeros(count, device=device)
        self.first_fault_step = torch.full((count,), -1, dtype=torch.long, device=device)
        self.instability_step = torch.full((count,), -1, dtype=torch.long, device=device)
        self.support_release_step_actual = torch.full(
            (count,), -1, dtype=torch.long, device=device
        )
        self.flags = {
            name: torch.zeros(count, dtype=torch.bool, device=device)
            for name in _TRACKED_FLAG_NAMES
        }
        self.support_seen = torch.zeros(count, dtype=torch.bool, device=device)
        self.open_hand_steps = torch.zeros(count, dtype=torch.long, device=device)

    @torch.inference_mode()
    def observe(self, state: dict[str, torch.Tensor]) -> None:
        missing = [key for key in self._REQUIRED_STATE_KEYS if key not in state]
        if missing:
            raise KeyError(f"逐回合累加器缺少字段 {missing}")
        faults = state["faults"]
        unknown = [name for name in faults if name not in self.flags]
        if unknown:
            raise KeyError(f"未声明的故障标记 {unknown}")
        forces = state["forces_n"]
        contact = forces >= self._contact_force_threshold_n
        contact_count = state["contact_count"].to(dtype=torch.long)
        self.steps += 1
        self.any_contact_steps += (contact_count >= 1).long()
        self.contact_steps += (contact_count >= self._min_contact_fingers).long()
        self.finger_contact_steps += contact.long()
        self.required_contact_steps += state["required_contact"].long()
        self.contact_finger_sum += contact_count.to(dtype=self.contact_finger_sum.dtype)
        self.max_contact_fingers = torch.maximum(self.max_contact_fingers, contact_count)
        self.peak_force = torch.maximum(self.peak_force, forces)
        self.force_sum += forces
        palm_contact = state["palm_force_n"] >= self._contact_force_threshold_n
        self.palm_steps += palm_contact.long()
        self.palm_only_steps += (palm_contact & (contact_count == 0)).long()
        self.forbidden_steps += (state["forbidden_total_n"] >= 0.02).long()
        released = state["support_released"]
        unsupported_contact = released & state["required_contact"]
        self.unsupported_steps += released.long()
        self.unsupported_contact_steps += unsupported_contact.long()
        self.support_seen |= released
        self.open_hand_steps += state["open_hand_baseline"].to(dtype=torch.bool).long()
        newly_released = released & (self.unsupported_steps == 1)
        self.support_release_step_actual = torch.where(
            newly_released, self.steps - 1, self.support_release_step_actual
        )
        self.max_hold_age = torch.maximum(self.max_hold_age, state["hold_age"])
        self.final_hold_age.copy_(state["hold_age"])
        # 物理子步 streak：peak 在回合内单调不减，因此策略采样取最大值即
        # 等于逐物理子步峰值；sticky 硬故障同样只会累积到 reset。
        self.max_physical_hold_streak = torch.maximum(
            self.max_physical_hold_streak, state["physical_hold_peak_streak"]
        )
        self.final_physical_hold_streak.copy_(state["physical_hold_streak"])
        substep_flags = state["physical_hard_fault_flags"]
        unknown_substep = [
            name for name in substep_flags if name not in self.physical_substep_hard_fault_flags
        ]
        if unknown_substep:
            raise KeyError(f"未声明的物理子步硬故障标记 {unknown_substep}")
        for name, flag in substep_flags.items():
            self.physical_substep_hard_fault_flags[name] |= flag
        self.max_height_gain = torch.maximum(self.max_height_gain, state["height_gain_m"])
        self.final_height_gain.copy_(state["height_gain_m"])
        self.max_drift = torch.maximum(self.max_drift, state["drift_m"])
        self.max_rel_speed = torch.maximum(self.max_rel_speed, state["rel_speed_m_s"])
        self.max_joint_excursion = torch.maximum(
            self.max_joint_excursion, state["joint_excursion_rad"]
        )
        self.max_applied_effort = torch.maximum(
            self.max_applied_effort, state["max_applied_effort_nm"]
        )
        self.min_self_collision_clearance = torch.minimum(
            self.min_self_collision_clearance, state["self_collision_clearance_m"]
        )
        self.min_collider_distance = torch.minimum(
            self.min_collider_distance, state["collider_distance_m"]
        )
        self.max_penetration_depth = torch.maximum(
            self.max_penetration_depth, state["penetration_depth_m"]
        )
        for name, flag in faults.items():
            fresh = flag & ~self.flags[name]
            self.flags[name] |= flag
            if name == "instability":
                self.instability_step = torch.where(
                    fresh, self.steps - 1, self.instability_step
                )
            if name in _FAULT_NAMES:
                self.first_fault_step = torch.where(
                    fresh & (self.first_fault_step < 0),
                    self.steps - 1,
                    self.first_fault_step,
                )

    @torch.inference_mode()
    def finalize(
        self,
        env_ids: torch.Tensor,
        *,
        success: torch.Tensor,
        reason_code: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        stem_fracture: torch.Tensor,
        support_release_step: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index in (int(item) for item in env_ids.tolist()):
            steps = int(self.steps[index].item())
            if steps <= 0:
                # 初始 reset：本回合没有实际运行，不产生记录。
                continue
            episode_id = int(self.episode_index[index].item())
            self.episode_index[index] += 1
            reason_value = int(reason_code[index].item())
            succeeded = bool(success[index].item())
            first_fault = int(self.first_fault_step[index].item())
            immediate = (
                (not succeeded)
                and bool(terminated[index].item())
                and 0 <= first_fault <= self._immediate_failure_steps
            )
            unsupported = int(self.unsupported_steps[index].item())
            row: dict[str, Any] = {
                "schema_version": C.RECORD_SCHEMA_VERSION,
                "task_id": C.TASK_ID,
                "task_version": C.TASK_VERSION,
                "stage": self._stage,
                "success_criterion": self._success_criterion,
                "env_id": index,
                "episode_id": episode_id,
                "success": succeeded,
                "goal_success": False,
                "goal_status": self._goal_status,
                "reason": C.reason_name(reason_value),
                "reason_code": reason_value,
                "terminated": bool(terminated[index].item()),
                "truncated": bool(truncated[index].item()),
                "steps": steps,
                "sim_time_s": steps * self._policy_dt_s,
                "contact_fraction": (
                    int(self.contact_steps[index].item()) / steps
                ),
                "finger_contact_fraction": [
                    int(value) / steps
                    for value in self.finger_contact_steps[index].tolist()
                ],
                "required_contact_fraction": (
                    int(self.required_contact_steps[index].item()) / steps
                ),
                "unsupported_contact_fraction": (
                    int(self.unsupported_contact_steps[index].item()) / unsupported
                    if unsupported > 0
                    else 0.0
                ),
                "any_contact_fraction": int(self.any_contact_steps[index].item()) / steps,
                "mean_contact_fingers": float(
                    (self.contact_finger_sum[index] / steps).item()
                ),
                "max_contact_fingers": int(self.max_contact_fingers[index].item()),
                "palm_contact_fraction": int(self.palm_steps[index].item()) / steps,
                "palm_only_fraction": int(self.palm_only_steps[index].item()) / steps,
                "forbidden_contact_steps": int(self.forbidden_steps[index].item()),
                "peak_finger_force_n": [
                    float(value) for value in self.peak_force[index].tolist()
                ],
                "mean_finger_force_n": [
                    float(value / steps) for value in self.force_sum[index].tolist()
                ],
                "support_release_step": int(support_release_step),
                "support_release_time_s": int(support_release_step) * self._policy_dt_s,
                "support_released": bool(self.support_seen[index].item()),
                "stem_fracture_before_release": bool(
                    stem_fracture[index].item()
                ),
                "max_unsupported_hold_s": (
                    int(self.max_hold_age[index].item()) * self._policy_dt_s
                ),
                "final_unsupported_hold_s": (
                    int(self.final_hold_age[index].item()) * self._policy_dt_s
                ),
                "hold_steps_required": self._required_hold_steps,
                # v1.4.0 逐物理子步语义（便于外部核验，不改任何门限数值）。
                "max_physical_hold_streak_substeps": int(
                    self.max_physical_hold_streak[index].item()
                ),
                "final_physical_hold_streak_substeps": int(
                    self.final_physical_hold_streak[index].item()
                ),
                "physical_hold_streak_required_substeps": int(
                    self._required_hold_substeps
                ),
                "max_unsupported_hold_s_from_physical_substeps": (
                    int(self.max_physical_hold_streak[index].item()) * self._sim_dt_s
                ),
                "physical_hold_streak_semantics": self._physical_hold_semantics,
                "physical_substep_hard_fault_flags": {
                    name: bool(
                        self.physical_substep_hard_fault_flags[name][index].item()
                    )
                    for name in HARD_FAULT_NAMES
                },
                "max_height_gain_m": float(self.max_height_gain[index].item()),
                "final_height_gain_m": float(self.final_height_gain[index].item()),
                "height_gain_sufficient": bool(
                    float(self.max_height_gain[index].item())
                    >= self._goal_min_height_gain_m
                ),
                "min_height_gain_m": self._stage_min_height_gain_m,
                "goal_min_height_gain_m": self._goal_min_height_gain_m,
                "stage_success_requires_height_gain": self._stage_success_requires_height_gain,
                "stage_requires_lift_program": self._stage_requires_lift_program,
                "stage_is_goal": self._stage_is_goal,
                "max_center_drift_m": float(self.max_drift[index].item()),
                "max_relative_speed_m_s": float(self.max_rel_speed[index].item()),
                "max_joint_limit_excursion_rad": float(
                    self.max_joint_excursion[index].item()
                ),
                "max_applied_effort_nm": float(
                    self.max_applied_effort[index].item()
                ),
                "min_self_collision_clearance_m": _finite_or_none(
                    self.min_self_collision_clearance[index]
                ),
                "min_collider_distance_m": _finite_or_none(
                    self.min_collider_distance[index]
                ),
                "max_penetration_depth_m": float(
                    self.max_penetration_depth[index].item()
                ),
                "first_fault_step": first_fault,
                "instability_step": int(self.instability_step[index].item()),
                "immediate_failure": immediate,
                "failure_phase": (
                    None if succeeded else ("immediate" if immediate else "late")
                ),
                "action_effect_ok": self._action_effect_ok,
                # 逐回合如实记录：整回合都跑在 open 基线时才是 True。
                "open_hand_baseline": bool(
                    int(self.open_hand_steps[index].item()) == steps
                ),
                "open_hand_fraction": int(self.open_hand_steps[index].item()) / steps,
            }
            for name in _TRACKED_FLAG_NAMES:
                row["stem_fracture" if name == "stem_fracture" else name] = bool(
                    self.flags[name][index].item()
                )
            rows.append(row)
            self.clear(index)
        return rows

    @torch.inference_mode()
    def clear(self, env_ids: torch.Tensor | None = None) -> None:
        """清空指定环境的累加状态；``episode_index`` 不回退。"""

        ids: torch.Tensor | slice = slice(None) if env_ids is None else env_ids
        for name in (
            "steps",
            "any_contact_steps",
            "contact_steps",
            "finger_contact_steps",
            "required_contact_steps",
            "unsupported_steps",
            "unsupported_contact_steps",
            "palm_steps",
            "palm_only_steps",
            "forbidden_steps",
            "contact_finger_sum",
            "max_contact_fingers",
            "peak_force",
            "force_sum",
            "max_hold_age",
            "final_hold_age",
            "max_physical_hold_streak",
            "final_physical_hold_streak",
            "max_height_gain",
            "final_height_gain",
            "max_drift",
            "max_rel_speed",
            "max_joint_excursion",
            "max_applied_effort",
            "max_penetration_depth",
        ):
            getattr(self, name)[ids] = 0
        self.min_self_collision_clearance[ids] = float("inf")
        self.min_collider_distance[ids] = float("inf")
        self.first_fault_step[ids] = -1
        self.instability_step[ids] = -1
        self.support_release_step_actual[ids] = -1
        self.support_seen[ids] = False
        self.open_hand_steps[ids] = 0
        for flag in self.flags.values():
            flag[ids] = False
        for flag in self.physical_substep_hard_fault_flags.values():
            flag[ids] = False


def _finite_or_none(value: torch.Tensor) -> float | None:
    number = float(value.item())
    return number if math.isfinite(number) else None


class FixedFruitGraspEnv(NativeSmallFruitPinchEnv):
    """固定单果多指真实接触抓握保持环境（holding-only）。"""

    cfg: FixedFruitGraspCfg

    def _setup_scene(self) -> None:
        super()._setup_scene()
        from .collision_cooking import configure_hand_collision_cooking
        self._collision_cooking_identity = configure_hand_collision_cooking(
            self.sim.stage, self.scene.num_envs, LEFT_HAND_BODY_NAMES
        )
        # 自碰撞的硬守护也读取真实逐点 separation，不用包围盒重叠终止。
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        self._self_penetration_sensors = []
        for index, (left, right) in enumerate(self.cfg.self_collision_pairs):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path='/World/envs/env_.*/Robot/' + left,
                filter_prim_paths_expr=['/World/envs/env_.*/Robot/' + right],
                update_period=1.0 / 120.0, history_length=0,
                max_contact_data_count_per_prim=4096,
            ))
            self.scene.sensors['self_penetration_' + str(index)] = sensor
            self._self_penetration_sensors.append(sensor)
        # 逐物理子步采样的唯一安装点：包装 scene.update，不改任何 Isaac 源。
        # wrapper 只负责在原始 update 之后读取 live 状态；是否计为一个真实
        # 物理子步完全由守卫的 enabled / dt / _sim_step_counter 三条 fail-closed
        # 规则决定（dt=0 的 reset 回调与重复通知都不计数）。
        self._install_physical_hold_substep_hook()

    def _install_physical_hold_substep_hook(self) -> None:
        """构造期安装 scene.update 包装；守卫在构造完成后才 enable。"""

        if bool(getattr(self, "_physical_hold_hook_installed", False)):
            return
        self._physical_hold_hook_installed = True
        original_update = self.scene.update

        def update_then_observe(*update_args: Any, **update_kwargs: Any):
            result = original_update(*update_args, **update_kwargs)
            scene_dt_s = update_kwargs.get("dt")
            if scene_dt_s is None and update_args:
                scene_dt_s = update_args[0]
            self._observe_physical_hold_substep(scene_dt_s)
            return result

        self.scene.update = update_then_observe

    def __init__(
        self, cfg: FixedFruitGraspCfg, render_mode: str | None = None, **kwargs: Any
    ) -> None:
        validate_cfg(cfg)
        # 真实自碰撞必须在 Articulation 创建之前写入 cfg；本环境不在运行时
        # 关闭自碰撞，也不复制会关闭它的 physical_prior_adapter。
        cfg.robot.spawn.articulation_props.enabled_self_collisions = bool(
            cfg.real_self_collisions
        )
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._hard_high = self._last_physical_force_n.new_tensor(
            C.FINGER_HARD_HIGH_N
        )
        self._soft_low = self._last_physical_force_n.new_tensor(C.FINGER_SOFT_LOW_N)
        self._soft_high = self._last_physical_force_n.new_tensor(C.FINGER_SOFT_HIGH_N)
        self._action_effect = self._check_action_effect_and_limits()
        self._joint_limits = self._collect_joint_limits()
        self._effective_self_collision_pairs = self._select_self_collision_pairs()
        self._penetration_columns = self._select_penetration_columns()
        # 安全守护覆盖全部 21 个手关节（16 主动 + 5 跟随）；策略观测仍只用
        # 16 维（114D actor 合同不变）。
        self._safety_joint_ids = list(self._hand_joint_ids) + list(
            self._mimic_follower_ids
        )
        self._safety_joint_names = [
            self._robot.joint_names[index] for index in self._safety_joint_ids
        ]
        self._safety_effort_limits = self._robot.data.joint_effort_limits.torch[0][
            self._safety_joint_ids
        ]
        self._effort_guard_tolerance = 0.05
        # 在构造期就确认 effort 观测量存在（fail-closed），不要等到第一个 step。
        self._effort_proxy_name = self._select_effort_proxy_name()

        self._open_hand_targets = self._nominal_joint_targets[
            :, self._hand_joint_ids
        ].clone()
        # 逐物理子步保持守护：hold streak 与七类硬故障都在真实 1/120 s 子步上
        # 判定；这里只构造，构造完成后由 _enable_physical_hold_guard 启用。
        self._physical_hold_required_substeps = required_physical_substeps(
            C.stage_spec(cfg.stage).required_unsupported_hold_steps,
            int(cfg.decimation),
        )
        self._physical_hold_guard = PhysicalHoldGuard(
            self.num_envs,
            device=self.device,
            physics_dt_s=float(cfg.sim.dt),
            decimation=int(cfg.decimation),
            required_substeps=int(self._physical_hold_required_substeps),
            hold_max_drift_m=float(cfg.hold_max_drift_m),
            hold_max_relative_speed_m_s=float(cfg.hold_max_relative_speed_m_s),
        )
        self._open_hand_baseline = bool(cfg.open_hand_baseline)
        self._open_hand_baseline_flag = torch.full(
            (self.num_envs,), bool(cfg.open_hand_baseline), dtype=torch.bool, device=self.device
        )
        self._hold_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._approach_potential_prev = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._approach_potential_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._contact_age = torch.zeros(
            (self.num_envs, 5), dtype=torch.long, device=self.device
        )
        self._contact_mask = torch.zeros(
            (self.num_envs, 5), dtype=torch.bool, device=self.device
        )
        self._required_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._hold_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # hold anchor 由物理子步守卫拥有（物理 streak 起点）；这里只保留同一
        # 个张量的镜像引用，策略刷新不重置它。
        self._hold_anchor_hand = self._physical_hold_guard.anchor_hand
        self._hold_window_drift_m = torch.zeros(self.num_envs, device=self.device)
        self._drift_m = torch.zeros(self.num_envs, device=self.device)
        self._height_gain_m = torch.zeros(self.num_envs, device=self.device)
        self._rel_speed_m_s = torch.zeros(self.num_envs, device=self.device)
        self._joint_excursion_rad = torch.zeros(self.num_envs, device=self.device)
        self._max_applied_effort_nm = torch.zeros(self.num_envs, device=self.device)
        self._self_collision = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._self_collision_clearance_m = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self._collider_distance_m = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self._penetration_depth_m = torch.zeros(self.num_envs, device=self.device)
        self._pad_penetration_depth_m = torch.zeros(self.num_envs, device=self.device)
        by_body = {s.cfg.prim_path.rsplit('/', 1)[-1]: s for s in [*self._contact_sensors.values(), *self._aux_contact_sensors.values()]}
        if set(LEFT_HAND_BODY_NAMES) - set(by_body):
            raise C.FixedFruitGraspError('真实接触距离没有覆盖所有手部刚体')
        self._raw_penetration_sensors = [by_body[name] for name in LEFT_HAND_BODY_NAMES]
        self._whole_finger_sensors = {digit: [sensor for body, sensor in by_body.items() if body.startswith(digit + "_")] for digit in C.FINGER_NAMES}
        if any(not sensors for sensors in self._whole_finger_sensors.values()):
            raise C.FixedFruitGraspError("整指真实接触通道不完整")
        self._raw_penetration_peak_m = torch.zeros(self.num_envs, device=self.device)
        self._support_release_by_time = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._faults = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in _TRACKED_FLAG_NAMES
        }
        self._terminal_reason = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._terminal_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._drop_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._contact_ratio = torch.zeros(self.num_envs, device=self.device)
        self._band_ratio = torch.zeros(self.num_envs, device=self.device)
        self._palm_ratio = torch.zeros(self.num_envs, device=self.device)
        self._state_ready = False

        self.completed_episodes: list[dict[str, Any]] = []
        self.last_episode_records: list[dict[str, Any]] = []
        self.dropped_episode_count = 0
        self.last_reset_poses: list[dict[str, Any]] = []
        self.first_reset_poses: list[dict[str, Any]] = []
        self.reset_pose_count = 0

        spec = C.stage_spec(cfg.stage)
        self._tracker = _EpisodeTracker(
            self.num_envs,
            device=self.device,
            policy_dt_s=C.POLICY_DT_S,
            stage=spec.stage,
            success_criterion=spec.success_criterion,
            required_hold_steps=int(cfg.required_unsupported_hold_steps),
            stage_min_height_gain_m=float(spec.min_height_gain_m),
            stage_success_requires_height_gain=bool(spec.success_requires_height_gain),
            stage_requires_lift_program=bool(spec.requires_lift_program),
            stage_is_goal=bool(spec.counts_as_goal),
            goal_min_height_gain_m=C.MIN_HEIGHT_GAIN_M,
            goal_status=spec.goal_status,
            action_effect_ok=bool(self._action_effect["ok"]),
            open_hand_baseline=self._open_hand_baseline,
            immediate_failure_steps=int(cfg.immediate_failure_steps),
            contact_force_threshold_n=float(cfg.contact_force_threshold_n),
            min_contact_fingers=int(cfg.min_contact_fingers),
            required_hold_substeps=int(self._physical_hold_required_substeps),
            sim_dt_s=float(C.SIM_DT_S),
        )
        self._identity = self._build_identity()
        # 构造完成后才启用逐物理子步守护：构造期的 scene.update 属于初始化，
        # 既不是物理步，也不能被当成样本。
        self._enable_physical_hold_guard()

    # ------------------------------------------------------------------ 守护
    @torch.inference_mode()
    def _check_action_effect_and_limits(self) -> dict[str, Any]:
        probes = torch.tensor(
            ((-1.0,) * C.ACTION_DIM, (0.0,) * C.ACTION_DIM, (1.0,) * C.ACTION_DIM),
            device=self.device,
        )
        targets = pinch_action_to_joint_targets(probes)
        if not bool(torch.isfinite(targets).all()):
            raise C.FixedFruitGraspError("14D 动作映射产生非有限关节目标")
        low = targets.amin(dim=0)
        high = targets.amax(dim=0)
        span = high - low
        limits = self._robot.data.joint_pos_limits.torch[0][self._hand_joint_ids]
        if not bool(torch.isfinite(limits).all()):
            raise C.FixedFruitGraspError("机器人声明的关节硬限包含非有限值")
        outside = (low < limits[:, 0] - 1.0e-6) | (high > limits[:, 1] + 1.0e-6)
        # 关节硬限守护：非有限目标或像集退化是硬失败；可达范围略越出 USD 声明
        # 限位只记录不致命——执行器层（LeftHandDirectionalPDActuator）与 PhysX
        # 关节限位仍在物理上拦截，静态越界不会造成安全问题，但必须显式可见。
        offenders = [
            self._hand_joint_names[index]
            for index in torch.nonzero(outside, as_tuple=False).flatten().tolist()
        ]
        if float(span.mean().item()) <= 1.0e-6:
            raise C.FixedFruitGraspError("14D 动作对受控关节没有控制作用")
        neutral = self._nominal_joint_targets[0][self._hand_joint_ids]
        gap = float((targets[1] - neutral).abs().max().item())
        overshoot = (
            float(
                torch.maximum(low - limits[:, 0], limits[:, 1] - high)
                .clamp_min(0.0)
                .max()
                .item()
            )
            if offenders
            else 0.0
        )
        return {
            "ok": not offenders,
            "joint_limit_overhang_rad": overshoot,
            "joint_limit_overhang_names": offenders,
            "joint_limit_note": (
                "可达范围越出 USD 声明限位时只记录：执行器软件限位与 PhysX 仍会拦截，"
                "但该关节在越界区间的动作无作用，需要显式复核"
            ),
            "per_joint_span_rad": [float(value) for value in span.tolist()],
            "min_joint_span_rad": float(span.amin().item()),
            "mean_joint_span_rad": float(span.mean().item()),
            "manifold_zero_target_gap_rad": gap,
            "neutral_targets_rad": [float(value) for value in neutral.tolist()],
            "probe_targets_rad": [
                [float(value) for value in row] for row in targets.tolist()
            ],
        }

    @torch.inference_mode()
    def _collect_joint_limits(self) -> dict[str, list[float]]:

        limits = self._robot.data.joint_pos_limits.torch[0]
        return {
            name: [float(limits[index, 0]), float(limits[index, 1])]
            for index, name in enumerate(self._robot.joint_names)
        }

    def _select_effort_proxy_name(self) -> str:
        """选择逐轴 effort 守护使用的实际努力量（缺失即 fail-closed）。"""

        for name in ("applied_torque", "computed_torque"):
            proxy = getattr(self._robot.data, name, None)
            if proxy is not None and hasattr(proxy, "torch"):
                return name
        raise C.FixedFruitGraspError(
            "ArticulationData 缺少 applied_torque/computed_torque：无法执行逐轴 effort 守护"
        )

    def _collision_body_names(self) -> list[str]:
        return [
            self._robot.body_names[int(index)]
            for index in self._hand_collision_body_ids
        ]

    def _select_self_collision_pairs(self) -> tuple[tuple[str, str], ...]:
        available = set(self._collision_body_names())
        declared = tuple(tuple(pair) for pair in tuple(self.cfg.self_collision_pairs))
        effective = tuple(
            pair for pair in declared if pair[0] in available and pair[1] in available
        )
        skipped = tuple(pair for pair in declared if pair not in effective)
        self._skipped_self_collision_pairs = skipped
        if not effective:
            raise C.FixedFruitGraspError(
                "自碰撞诊断对在运行时一个都不存在（不得静默过滤为空）："
                f"declared={declared}"
            )
        return effective

    def _select_penetration_columns(self) -> torch.Tensor:
        names = self._collision_body_names()
        # 导入的 base_link 掌心 AABB 远大于真实掌面形状（canonical 环境已记录
        # 该诊断事实），把它从“粗穿透代理”里排除；掌面是否承重由真实 PhysX
        # 接触/力和指尖接触点深度判断。
        columns = [index for index, name in enumerate(names) if name != "base_link"]
        return torch.tensor(columns, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ 接口
    def task_identity(self) -> dict[str, Any]:
        """JSON-safe 的实际任务身份（配置/资产/判据/局限）。"""

        return deepcopy(self._identity)

    def action_effect_report(self) -> dict[str, Any]:
        """动作作用范围与关节硬限静态守护的 JSON-safe 报告。"""

        return deepcopy(self._action_effect)

    def set_open_hand_baseline(self, enabled: bool = True) -> None:
        """只写 reset 真实 neutral 关节目标，由动力驱动保持张开手。

        开启后 ``_pre_physics_step`` 忽略策略动作，把 reset 时记录的 16 个真实
        neutral 目标交给同一左手指向 PD 驱动；不写关节状态、不传送根/果实，
        也不改变固定支撑时序。用于 open 对照与“有限闭合”有界跟随烟测。
        """

        self._open_hand_baseline = bool(enabled)
        # 让逐回合记录与本回合的真实执行模式一致（身份必须如实）。
        if hasattr(self, "_open_hand_baseline_flag"):
            self._open_hand_baseline_flag.fill_(bool(enabled))
        if hasattr(self, "_tracker"):
            self._tracker._open_hand_baseline = bool(enabled)

    @property
    def open_hand_baseline_enabled(self) -> bool:
        return self._open_hand_baseline

    def open_hand_action(self) -> torch.Tensor:
        """明确拒绝：14D 流形无法精确复现 reset 的真实 neutral 张开姿态。"""

        gap = self._action_effect.get("manifold_zero_target_gap_rad")
        detail = f"当前实测最大偏差约 {gap:.4f} rad" if isinstance(gap, float) else ""
        raise C.OpenHandActionUnavailable(
            "14D 流形把 14 个动作写入 16 个关节，像集不满秩（synergy 的 max()/区间夹紧 + "
            "每指残差共享同一关节），零动作也不是 reset 的 neutral 张开姿态"
            + ("，" + detail if detail else "")
            + "。请使用 set_open_hand_baseline(True)（只写 reset 真实 neutral 目标，"
            "不经 14D 流形）作为 open 控制。"
        )

    def closure_reference_action(
        self, closure: float = 1.0, *, opposition: float | None = None
    ) -> torch.Tensor:
        """有界闭合参考动作：第 0 维全局屈曲、第 4 维拇指对掌，其余为中立。

        该动作经 14D 流形映射后的关节目标落在 canonical 安全区间内，可直接
        用于主 smoke 的“有限闭合有界跟随”对照；它不是张开手基线。
        """

        value = max(-1.0, min(1.0, float(closure)))
        oppose = value if opposition is None else max(-1.0, min(1.0, float(opposition)))
        action = torch.zeros((self.num_envs, C.ACTION_DIM), device=self.device)
        action[:, 0] = value
        action[:, 4] = oppose
        return action

    def reference_joint_targets(self, closure: float = 1.0) -> torch.Tensor:
        """``closure_reference_action`` 对应的 16 关节有界参考目标。"""

        action = self.closure_reference_action(closure)
        return pinch_action_to_joint_targets(action)

    def pop_completed_episodes(self) -> list[dict[str, Any]]:
        """取走并清空累计逐回合记录（可选的内存管理入口）。"""

        rows = list(self.completed_episodes)
        self.completed_episodes.clear()
        return rows

    def episode_summary(self) -> dict[str, Any]:
        """``completed_episodes`` 的完整分母/原因汇总。"""

        return C.summarize_records(list(self.completed_episodes))

    # ------------------------------------------------- 固定时序支撑与执行层
    @torch.inference_mode()
    def _prepare_verification_support(self) -> None:
        """在固定策略步彻底解除原始声明的果梗弹簧支撑。

        不再使用“已满足成功才释放”的历史合同：``support_release_step`` 一到，
        支撑立即解除，且不做任何补力、不改写果实姿态。
        """

        due = (self.episode_length_buf >= int(self.cfg.support_release_step)) & ~self._stem_detached
        if bool(due.any()):
            self._support_release_by_time |= due
        self._stem_detached |= due
        self._audit_support_released |= self._stem_detached

    @torch.inference_mode()
    def _apply_verification_impulse(self) -> None:
        """本任务不使用历史三指扰动脉冲（保留为显式 no-op）。"""

        return

    @torch.inference_mode()
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if not self._open_hand_baseline:
            super()._pre_physics_step(actions)
            return
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"actions 必须为 {tuple(self._raw_actions.shape)}，得到 {tuple(actions.shape)}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions 包含 NaN 或 Inf")
        self._previous_executed_actions.copy_(self._executed_actions)
        self._raw_actions.copy_(actions.clamp(-1.0, 1.0))
        # 保持时间滤波器状态活跃（诊断信号），但策略动作不进入执行器。
        self._filter.step(
            self._raw_actions,
            hold_phase=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        self._executed_actions.zero_()
        targets = self._postprocess_joint_targets(self._open_hand_targets)
        self._joint_targets[:, self._hand_joint_ids] = targets
        self._update_mimic_targets(self._all_env_ids)
        self._position_interpolation_step.zero_()
        self._stem_break_event.zero_()

    @torch.inference_mode()
    def _apply_attachment_wrench(self) -> None:
        """声明的柔顺夹具由 PhysX 外力积分；释放后严格为零，不写果实状态。"""
        pose = self._tomato.data.root_pose_w.torch
        velocity = self._tomato.data.root_lin_vel_w.torch
        angular = self._tomato.data.root_ang_vel_w.torch
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 2] = self._fruit_axes_m[:, 2]
        offset = quat_apply(pose[:, 3:7], local)
        attachment = pose[:, :3] + offset
        attachment_velocity = velocity + torch.linalg.cross(angular, offset, dim=1)
        omega = float(self.cfg.stem_linear_natural_frequency_rad_s)
        acceleration = omega**2 * (self._stem_rest_attachment_pos_w - attachment)
        acceleration -= 2 * float(self.cfg.stem_linear_damping_ratio) * omega * attachment_velocity
        acceleration -= torch.tensor(self.cfg.sim.gravity, device=self.device).unsqueeze(0)
        force = self._fruit_mass_kg.unsqueeze(1) * acceleration
        inertia = (0.4 * self._fruit_mass_kg * self._fruit_axes_m.square().mean(dim=1)).clamp_min(1e-8)
        angular_omega = float(self.cfg.stem_angular_natural_frequency_rad_s)
        torque = inertia.unsqueeze(1) * (angular_omega**2 * quat_box_minus(self._stem_anchor_quat_w, pose[:, 3:7]) - 2 * float(self.cfg.stem_angular_damping_ratio) * angular_omega * angular)
        force *= (float(self.cfg.maximum_stem_force_n) / force.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1)
        torque *= (float(self.cfg.maximum_stem_torque_nm) / torque.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1)
        attached = ~self._stem_detached
        force *= attached.unsqueeze(1)
        torque *= attached.unsqueeze(1)
        self._last_stem_force_w.copy_(force)
        self._last_stem_torque_w.copy_(torque)
        self._tomato.permanent_wrench_composer.set_forces_and_torques_index(forces=force.unsqueeze(1), torques=torque.unsqueeze(1), is_global=True)

    @torch.inference_mode()
    def _live_finger_force_n(self) -> torch.Tensor:
        """只读口径：每指取该指全部刚体 filtered fruit ``force_matrix_w`` 的最大力。

        与 ``_physical_finger_force`` 完全同口径，但不写策略步缓存，供逐物理
        子步硬故障判定复用；任一整指通道缺失直接失败（fail-closed）。
        """

        if len(self._whole_finger_sensors) != len(C.FINGER_NAMES):
            raise C.FixedFruitGraspError("整指真实接触通道不完整")
        per_digit = []
        for digit in C.FINGER_NAMES:
            sensors = self._whole_finger_sensors[digit]
            if not sensors:
                raise C.FixedFruitGraspError(f"整指 {digit} 没有真实接触传感器")
            body_forces = []
            for sensor in sensors:
                matrix = sensor.data.force_matrix_w
                if matrix is None:
                    raise PhysicalHoldError(
                        f"整指 {digit} 的 filtered fruit 接触力缺失（force_matrix_w=None）"
                    )
                body_forces.append(matrix.torch[:, 0, 0].norm(dim=1))
            per_digit.append(torch.stack(body_forces, dim=1).amax(dim=1))
        return torch.stack(per_digit, dim=1)

    @torch.inference_mode()
    def _physical_finger_force(self) -> torch.Tensor:
        # 保留旧末端指垫筛选作为诊断，训练与硬力门读取真实整指接触。
        pads = super()._physical_finger_force()
        if not hasattr(self, "_whole_finger_sensors"):
            return pads
        forces = self._live_finger_force_n()
        self._legacy_pad_force_n = pads.clone()
        self._raw_finger_force_n.copy_(forces)
        return forces

    # --------------------------------------------------------------- 状态刷新
    @torch.inference_mode()
    def _pad_contact_penetration(self, fruit_pos: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
        """指尖接触点沉入球面的深度（精细侧守护；无活接触记为 0）。"""

        radius = self._fruit_axes_m.mean(dim=1)
        depth = torch.zeros(self.num_envs, device=self.device)
        threshold = float(self.cfg.contact_force_threshold_n)
        for digit, sensor in enumerate(self._contact_sensors.values()):
            points = sensor.data.contact_pos_w
            if points is None:
                continue
            point = points.torch[:, 0, 0]
            finite = torch.isfinite(point).all(dim=1)
            value = radius - torch.linalg.vector_norm(point - fruit_pos, dim=1)
            live = finite & (forces[:, digit] >= threshold)
            depth = torch.maximum(depth, torch.where(live, value, torch.zeros_like(value)))
        return depth

    @torch.inference_mode()
    def _self_collision_clearance(self) -> torch.Tensor:
        if not self._effective_self_collision_pairs:
            raise C.FixedFruitGraspError('真实自碰撞守护没有声明刚体对')
        clearances = [
            self._hand_obb_pair_clearance_for((pair[0],), (pair[1],), self._all_env_ids)
            for pair in self._effective_self_collision_pairs
        ]
        self._self_collision_aabb_clearance_m = torch.stack(clearances, dim=1).amin(dim=1)
        from .physical_contacts import penetration_from_sensors
        depth, count = penetration_from_sensors(self._self_penetration_sensors, self.num_envs, self.device)
        self.extras['physical_self_penetration'] = {'depth_m': depth, 'points_count': count}
        # 无接触时记零穿透；不将此值解释为几何间隙。
        return -depth

    # ------------------------------------------------- 物理子步保持守护接线
    def _enable_physical_hold_guard(self) -> None:
        """构造完成后启用逐物理子步守护。

        只接受 PhysX 外部逐步 ``decimation=False`` 分支：任何把 decimation
        折叠进单次 ``sim.step`` 的后端都会让 ``scene.update`` 不再是 1/120 s
        子步回调，此时逐物理子步语义无法成立，直接 fail-closed。
        """

        cfg = self.cfg
        if bool(getattr(self, "_physics_handles_decimation", True)):
            raise C.FixedFruitGraspError(
                "物理后端把 decimation 折叠进单次 sim.step，scene.update 不再是逐 "
                "1/120 s 子步回调；v1.4.0 的物理子步保持语义无法成立（fail-closed）"
            )
        if int(cfg.decimation) != int(C.DECIMATION):
            raise C.FixedFruitGraspError(
                f"decimation 必须是声明值 {C.DECIMATION}，得到 {cfg.decimation!r}"
            )
        if abs(float(cfg.sim.dt) - float(C.SIM_DT_S)) > 1.0e-12:
            raise C.FixedFruitGraspError(
                f"固定物理步长必须是 {C.SIM_DT_S!r}，得到 {cfg.sim.dt!r}"
            )
        try:
            self._physical_hold_guard.enable(getattr(self, "_sim_step_counter", None))
        except PhysicalHoldError as exc:
            raise C.FixedFruitGraspError(str(exc)) from exc

    @torch.inference_mode()
    def _live_physical_substep_state(
        self, *, forces: torch.Tensor
    ) -> PhysicalSubstepState:
        """读取一个真实物理子步的全部 live 判定输入（只读，不写策略步缓存）。

        口径与 ``_refresh_task_state`` 一致；唯一差别是不在每个子步重复只在
        诊断中使用的 AABB/OBB clearance 矩阵。穿透与自碰撞都读 PhysX packed
        separation；自碰撞使用声明的 5 对跨指刚体真实 separation，而不是包围盒。
        """

        cfg = self.cfg
        robot = self._robot.data
        fruit = self._tomato.data
        ee_pose = robot.body_pose_w.torch[:, self._ee_body_id]
        ee_linear_velocity = robot.body_lin_vel_w.torch[:, self._ee_body_id]
        fruit_pose = fruit.root_pose_w.torch
        fruit_pos = fruit.root_pos_w.torch
        fruit_linear_velocity = fruit.root_lin_vel_w.torch
        fruit_angular_velocity = fruit.root_ang_vel_w.torch
        # 手坐标系相对速度含 ω×offset（果实位置相对 EE 原点的偏置项），与
        # 策略采样口径逐字一致。
        relative_velocity = (
            fruit_linear_velocity
            - ee_linear_velocity
            - torch.linalg.cross(
                robot.body_ang_vel_w.torch[:, self._ee_body_id],
                fruit_pos - ee_pose[:, :3],
                dim=1,
            )
        )
        relative_speed = torch.linalg.vector_norm(relative_velocity, dim=1)
        tip_positions = robot.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        tip_velocities = robot.body_lin_vel_w.torch[:, self._fingertip_contact_body_ids]
        palm_force = self._palm_force()
        from .physical_contacts import penetration_from_sensors

        fruit_depth, _ = penetration_from_sensors(
            self._raw_penetration_sensors, self.num_envs, self.device
        )
        self_depth, _ = penetration_from_sensors(
            self._self_penetration_sensors, self.num_envs, self.device
        )
        contact_threshold = float(cfg.contact_force_threshold_n)
        return PhysicalSubstepState(
            joint_pos=robot.joint_pos.torch[:, self._safety_joint_ids],
            joint_vel=robot.joint_vel.torch[:, self._safety_joint_ids],
            joint_pos_limits=robot.joint_pos_limits.torch[:, self._safety_joint_ids],
            applied_effort=getattr(robot, self._effort_proxy_name).torch[
                :, self._safety_joint_ids
            ],
            effort_limits=self._safety_effort_limits,
            finger_force_n=forces,
            finger_hard_high_n=self._hard_high,
            fruit_pose=fruit_pose,
            fruit_linear_velocity=fruit_linear_velocity,
            fruit_angular_velocity=fruit_angular_velocity,
            ee_pose=ee_pose,
            ee_linear_velocity=ee_linear_velocity,
            tip_positions=tip_positions,
            tip_velocities=tip_velocities,
            palm_force_n=palm_force,
            forbidden_proximal_n=self._forbidden_proximal_force(),
            previous_force_n=self._previous_force_n,
            executed_actions=self._executed_actions,
            raw_actions=self._raw_actions,
            previous_executed_actions=self._previous_executed_actions,
            fruit_penetration_depth_m=fruit_depth,
            self_penetration_depth_m=self_depth,
            released=self._stem_detached,
            required_contact=(forces >= contact_threshold).sum(dim=1)
            >= int(cfg.min_contact_fingers),
            relative_speed_m_s=relative_speed,
            fruit_in_hand_m=quat_apply_inverse(
                ee_pose[:, 3:7], fruit_pos - ee_pose[:, :3]
            ),
            joint_limit_margin_rad=float(cfg.joint_limit_margin_rad),
            instability_max_joint_velocity_rad_s=float(
                cfg.instability_max_joint_velocity_rad_s
            ),
            instability_max_fruit_speed_m_s=float(cfg.instability_max_fruit_speed_m_s),
            fruit_penetration_limit_m=float(cfg.penetration_limit_m),
            self_collision_penetration_limit_m=float(
                cfg.self_collision_penetration_limit_m
            ),
            effort_guard_tolerance=float(self._effort_guard_tolerance),
        )

    @torch.inference_mode()
    def _observe_physical_hold_substep(self, scene_dt_s: Any) -> bool:
        """``scene.update`` 包装回调：真实物理子步交给守卫，缺失数据 fail-closed。"""

        guard = getattr(self, "_physical_hold_guard", None)
        if guard is None or not guard.enabled:
            # 构造期 scene.update（守卫尚未 enable）：不采样、不计样本。
            return False
        def sample() -> PhysicalSubstepState:
            return self._live_physical_substep_state(forces=self._live_finger_force_n())

        return guard.observe_substep(
            sample,
            sim_step_counter=getattr(self, "_sim_step_counter", None),
            scene_dt_s=scene_dt_s,
        )

    @torch.inference_mode()
    def _refresh_task_state(self) -> None:
        cfg = self.cfg
        forces = self._physical_finger_force()
        self._last_physical_force_n.copy_(forces)
        raw = self._raw_finger_force_n
        palm_force = self._palm_force()
        self._last_palm_force_n.copy_(palm_force)
        forbidden_proximal = self._forbidden_proximal_force()

        fruit_pos = self._tomato.data.root_pos_w.torch
        fruit_pose = self._tomato.data.root_pose_w.torch
        fruit_vel = self._tomato.data.root_lin_vel_w.torch
        fruit_ang_vel = self._tomato.data.root_ang_vel_w.torch
        offset = fruit_pos - self._pinch_reference_center_w
        self._drift_m = torch.linalg.vector_norm(offset, dim=1)
        self._height_gain_m = offset[:, 2]
        ee_velocity = self._robot.data.body_lin_vel_w.torch[:, self._ee_body_id]
        ee_pose = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        tip_positions = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        tip_velocities = self._robot.data.body_lin_vel_w.torch[
            :, self._fingertip_contact_body_ids
        ]
        relative_velocity = fruit_vel - ee_velocity - torch.linalg.cross(self._robot.data.body_ang_vel_w.torch[:, self._ee_body_id], fruit_pos - ee_pose[:, :3], dim=1)
        self._rel_speed_m_s = torch.linalg.vector_norm(relative_velocity, dim=1)

        threshold = float(cfg.contact_force_threshold_n)
        contact = forces >= threshold
        self._contact_mask = contact
        contact_count = contact.sum(dim=1)
        # 包络抓握允许任意两根真实手指；掌面单独承托不计成功。
        self._required_contact = contact_count >= int(cfg.min_contact_fingers)
        self._contact_age = torch.where(
            contact, self._contact_age + 1, torch.zeros_like(self._contact_age)
        )

        joint_pos = self._robot.data.joint_pos.torch[:, self._safety_joint_ids]
        joint_vel = self._robot.data.joint_vel.torch[:, self._safety_joint_ids]
        limits = self._robot.data.joint_pos_limits.torch[:, self._safety_joint_ids]
        excursion = torch.maximum(
            limits[:, :, 0] - joint_pos, joint_pos - limits[:, :, 1]
        ).clamp_min(0.0)
        self._joint_excursion_rad = excursion.amax(dim=1)
        applied = getattr(self._robot.data, self._effort_proxy_name).torch[
            :, self._safety_joint_ids
        ]
        self._max_applied_effort_nm = applied.abs().amax(dim=1)
        effort_limit = self._safety_effort_limits
        finite_limit = torch.isfinite(effort_limit) & (effort_limit > 0.0)
        if bool(finite_limit.any()):
            ratio = applied.abs() / torch.where(
                finite_limit, effort_limit, torch.ones_like(effort_limit)
            ).clamp_min(1.0e-9)
            effort_violation = (ratio > 1.0 + self._effort_guard_tolerance).any(dim=1)
        else:
            effort_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 非有限守护覆盖策略/评论家观测实际使用的全部动态输入，而不只是关节。
        quaternion_ok = (fruit_pose[:, 3:7].norm(dim=1) > 1.0e-6) & (
            ee_pose[:, 3:7].norm(dim=1) > 1.0e-6
        )
        finite = (
            torch.isfinite(joint_pos).all(dim=1)
            & torch.isfinite(joint_vel).all(dim=1)
            & torch.isfinite(applied).all(dim=1)
            & torch.isfinite(fruit_pose).all(dim=1)
            & torch.isfinite(fruit_vel).all(dim=1)
            & torch.isfinite(fruit_ang_vel).all(dim=1)
            & torch.isfinite(ee_pose).all(dim=1)
            & torch.isfinite(ee_velocity).all(dim=1)
            & torch.isfinite(tip_positions).flatten(1).all(dim=1)
            & torch.isfinite(tip_velocities).flatten(1).all(dim=1)
            & torch.isfinite(forces).all(dim=1)
            & torch.isfinite(raw).all(dim=1)
            & torch.isfinite(palm_force)
            & torch.isfinite(forbidden_proximal).all(dim=1)
            & torch.isfinite(self._previous_force_n).all(dim=1)
            & torch.isfinite(self._executed_actions).all(dim=1)
            & torch.isfinite(self._raw_actions).all(dim=1)
            & torch.isfinite(self._previous_executed_actions).all(dim=1)
            & quaternion_ok
        )
        nonfinite = ~finite
        joint_limit = self._joint_excursion_rad > float(cfg.joint_limit_margin_rad)
        speed = joint_vel.abs().amax(dim=1)
        instability = (speed > float(cfg.instability_max_joint_velocity_rad_s)) | (
            torch.linalg.vector_norm(fruit_vel, dim=1)
            > float(cfg.instability_max_fruit_speed_m_s)
        )
        overload = (raw > self._hard_high).any(dim=1)

        matrix = self._hand_collision_clearance_matrix_for(
            fruit_pos, self._fruit_axes_m, self._all_env_ids
        )
        if self._penetration_columns.numel() > 0:
            clearance = matrix[:, self._penetration_columns].amin(dim=1)
            self._collider_distance_m = clearance + self._fruit_axes_m.mean(dim=1)
            self._penetration_depth_m = torch.relu(-clearance)
        else:
            self._collider_distance_m = torch.full_like(self._drift_m, float("inf"))
            self._penetration_depth_m = torch.zeros_like(self._drift_m)
        self._pad_penetration_depth_m = self._pad_contact_penetration(fruit_pos, forces)
        from .physical_contacts import penetration_from_sensors
        actual_depth, actual_count = penetration_from_sensors(self._raw_penetration_sensors, self.num_envs, self.device)
        self._raw_penetration_peak_m = torch.maximum(self._raw_penetration_peak_m, actual_depth)
        # AABB 是包围体，重叠不等于真实碰撞表面穿透。硬门改用原始PhysX separation。
        penetration = actual_depth > float(cfg.penetration_limit_m)
        self.extras['physical_penetration'] = {'depth_m': actual_depth, 'points_count': actual_count}

        self._self_collision_clearance_m = self._self_collision_clearance()
        self._self_collision = self._self_collision_clearance_m < -float(
            cfg.self_collision_penetration_limit_m
        )

        self._faults["overload"].copy_(overload)
        self._faults["nonfinite"].copy_(nonfinite)
        self._faults["penetration"].copy_(penetration)
        self._faults["joint_limit"].copy_(joint_limit)
        self._faults["self_collision"].copy_(self._self_collision)
        self._faults["instability"].copy_(instability)
        self._faults["joint_effort"].copy_(effort_violation)
        self._faults["stem_fracture"].copy_(
            self._stem_detached & ~self._support_release_by_time
        )
        hard_fault = (
            overload
            | nonfinite
            | penetration
            | joint_limit
            | self._self_collision
            | instability
            | effort_violation
        )
        # 物理子步粘滞硬故障并入策略采样标记：一旦任一 1/120 s 子步触发，
        # 就在这里保持为 True 直到该 env reset；本刷新不再覆写丢失。
        guard = self._physical_hold_guard
        guard_hard_any = guard.hard_fault_any
        for name in _HARD_FAULT_REWARD_NAMES:
            self._faults[name] |= guard.hard_faults[name]
        hard_fault = hard_fault | guard_hard_any

        # hold anchor 由物理子步守卫拥有：它固定在该物理 streak 的起点，
        # 策略刷新只读镜像，绝不重置（否则会掩盖子步之间的漂移）。
        fruit_in_hand = quat_apply_inverse(ee_pose[:, 3:7], fruit_pos - ee_pose[:, :3])
        drift = torch.linalg.vector_norm(fruit_in_hand - self._hold_anchor_hand, dim=1)
        self._hold_window_drift_m = torch.where(
            guard.anchor_valid, drift, torch.full_like(drift, float("inf"))
        )
        self._hold_ok = (
            self._stem_detached
            & self._required_contact
            & (self._hold_window_drift_m <= float(cfg.hold_max_drift_m))
            & (self._rel_speed_m_s <= float(cfg.hold_max_relative_speed_m_s))
            & ~hard_fault
        )
        # 日志/奖励兼容：hold_age = floor(物理子步 streak / decimation)。成功
        # 判据不再使用该策略步计数，改用真实物理 streak（见 _get_dones）。
        self._hold_age = torch.div(
            guard.streak, max(1, int(cfg.decimation)), rounding_mode="floor"
        )
        self._state_ready = True

    # ------------------------------------------------------------ MDP 钩子
    @torch.inference_mode()
    def _tracker_state(self) -> dict[str, torch.Tensor]:
        """逐回合累加器输入的唯一构造点（真实调用源）。

        字段名必须与 :attr:`_EpisodeTracker._REQUIRED_STATE_KEYS` 完全一致；
        接缝测试直接调用本方法比对，避免只测自造字典。
        """

        return {
            "forces_n": self._last_physical_force_n,
            "raw_forces_n": self._raw_finger_force_n,
            "palm_force_n": self._last_palm_force_n,
            "forbidden_total_n": self._forbidden_contact_force_n.sum(dim=1)
            + self._forbidden_proximal_force_n.sum(dim=1),
            "required_contact": self._required_contact,
            "contact_count": self._contact_mask.sum(dim=1),
            "hold_age": self._hold_age,
            "hold_ok": self._hold_ok,
            "support_released": self._stem_detached,
            "height_gain_m": self._height_gain_m,
            "drift_m": self._drift_m,
            "rel_speed_m_s": self._rel_speed_m_s,
            "joint_excursion_rad": self._joint_excursion_rad,
            "max_applied_effort_nm": self._max_applied_effort_nm,
            "self_collision_clearance_m": self._self_collision_clearance_m,
            "collider_distance_m": self._collider_distance_m,
            "penetration_depth_m": self._penetration_depth_m,
            "open_hand_baseline": self._open_hand_baseline_flag,
            # v1.4.0：真实物理子步 streak / 峰值 / 粘滞硬故障。peak 在每个
            # 回合内单调不减，因此 20 Hz 采样它取最大值就等于逐子步峰值。
            "physical_hold_streak": self._physical_hold_guard.streak,
            "physical_hold_peak_streak": self._physical_hold_guard.peak_streak,
            "physical_hard_fault_flags": self._physical_hold_guard.hard_faults,
            "faults": self._faults,
        }

    @torch.inference_mode()
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 逐物理子步采样完整性：本策略步必须恰好记录 decimation 个真实子步。
        # 折叠 decimation 的分支、漏采样、重复通知都在这里 fail-closed。
        self._physical_hold_guard.complete_policy_interval()
        self._refresh_task_state()
        cfg = self.cfg
        # 成功判据是真实物理子步 streak（1 s = 20 策略步 × 6 = 120 子步），
        # 不再由策略步 _hold_ok 加 1 得到。
        success = self._stem_detached & (
            self._physical_hold_guard.streak
            >= int(self._physical_hold_required_substeps)
        )
        self._drop_now = self._stem_detached & (
            (self._height_gain_m < -float(cfg.drop_height_tolerance_m))
            | (self._drift_m > float(cfg.drop_max_drift_m))
        )
        hard_fault = (
            self._faults["overload"]
            | self._faults["nonfinite"]
            | self._faults["penetration"]
            | self._faults["joint_limit"]
            | self._faults["self_collision"]
            | self._faults["instability"]
            | self._faults["joint_effort"]
        )
        success &= ~self._drop_now & ~hard_fault
        terminated = success | self._drop_now | hard_fault
        truncated = (self.episode_length_buf >= self.max_episode_length - 1) & ~terminated

        code = torch.zeros_like(self._terminal_reason)
        code = torch.where(success, torch.ones_like(code), code)
        code = torch.where(truncated, torch.full_like(code, 2), code)
        code = torch.where(self._drop_now, torch.full_like(code, 3), code)
        code = torch.where(self._faults["overload"], torch.full_like(code, 4), code)
        code = torch.where(self._faults["nonfinite"], torch.full_like(code, 5), code)
        code = torch.where(self._faults["penetration"], torch.full_like(code, 6), code)
        code = torch.where(self._faults["joint_limit"], torch.full_like(code, 7), code)
        code = torch.where(self._faults["self_collision"], torch.full_like(code, 8), code)
        code = torch.where(self._faults["instability"], torch.full_like(code, 10), code)
        code = torch.where(self._faults["joint_effort"], torch.full_like(code, 11), code)
        self._terminal_reason.copy_(code)
        self._terminal_success.copy_(success)

        # 供 canonical 动作滤波的阶段钩子：撤支撑后的保持窗内降低动作速率。
        self._force_slow_phase = self._hold_age > 0
        self._tracker.observe(self._tracker_state())
        self.extras["fixed_fruit_grasp"] = {
            "stage": cfg.stage,
            "support_released_count": int(self._stem_detached.sum().item()),
            "required_contact_count": int(self._required_contact.sum().item()),
            "hold_active_count": int((self._hold_age > 0).sum().item()),
            "success_count": int(success.sum().item()),
            "drop_count": int(self._drop_now.sum().item()),
            "fault_count": int(hard_fault.sum().item()),
            "open_hand_baseline": bool(self._open_hand_baseline),
        }
        return terminated, truncated

    @torch.inference_mode()
    def _get_rewards(self) -> torch.Tensor:
        if not self._state_ready:
            self._refresh_task_state()
        cfg = self.cfg
        force = self._last_physical_force_n
        raw = self._raw_finger_force_n
        contact = self._contact_mask.float()
        contact_fraction = contact.mean(dim=1)
        required = self._required_contact.float()
        in_band = ((force >= self._soft_low) & (force <= self._soft_high)).float().mean(dim=1)
        drive = self._executed_actions.abs().mean(dim=1)
        # AABB 表面距离只构成折扣势差 shaping，不计接触/成功或真实穿透。
        closest = self._hand_collision_closest_points_for(self._tomato.data.root_pos_w.torch, self._all_env_ids)
        gaps = (closest - self._tomato.data.root_pos_w.torch.unsqueeze(1)).norm(dim=2) - self._fruit_axes_m.amax(dim=1, keepdim=True)
        names = [self._robot.body_names[j] for j in self._hand_collision_body_ids]
        digit_gaps = torch.stack([gaps[:, [j for j, name in enumerate(names) if name.startswith(digit + "_")]].amin(dim=1) for digit in C.FINGER_NAMES], dim=1).clamp_min(0)
        approach_potential = _approach_potential_from_digit_gaps(digit_gaps)
        approach = _potential_difference_reward(
            approach_potential,
            self._approach_potential_prev,
            self._approach_potential_ready,
            gamma=_APPROACH_SHAPING_GAMMA,
        )
        released = self._stem_detached.float()
        hold_progress = (
            self._hold_age.float() / max(1, int(cfg.required_unsupported_hold_steps))
        ).clamp(0.0, 1.0)
        stability = _gated_stability_reward(
            self._hold_window_drift_m, self._hold_ok, self._stem_detached
        )
        palm_contact = self._last_palm_force_n >= float(cfg.contact_force_threshold_n)
        palm_only = (palm_contact & (contact.sum(dim=1) < 1.0)).float()
        overload = torch.relu(raw - self._hard_high).sum(dim=1)
        forbidden = (
            self._forbidden_contact_force_n.sum(dim=1)
            + self._forbidden_proximal_force_n.sum(dim=1)
        )
        stats = self._filter.last_stats
        action_delta = stats.get(
            "action_delta", torch.zeros_like(self._executed_actions)
        ).mean(dim=1)
        jerk = stats.get("jerk", torch.zeros_like(self._executed_actions)).mean(dim=1)
        hard_fault = _hard_fault_mask(self._faults)
        hard_fault_penalty = _hard_fault_penalty(hard_fault, cfg.fault_penalty)
        # ---- v1.5.0 逐项奖励 -------------------------------------------------
        # 每个分量的系数只在这里乘入一次，并保持 v1.4.0 原表达式的相加顺序。
        # closure 与 unsupported_hold 的乘法结合顺序与旧式不同（门因子取 0/1），
        # 因此这里只声明「CPU 上核对数值相等（容差）」，不宣称逐位等价。下面
        # 同时把本策略步已算出的张量写成诊断快照（不累加回合、不改奖励物理）。
        term_weights = {
            "approach": float(cfg.approach_reward_weight),
            "contact": float(cfg.contact_reward_weight),
            "opposition": float(cfg.opposition_reward_weight),
            "band": float(cfg.band_reward_weight),
            "closure": float(cfg.closure_reward_weight),
            "unsupported_hold": float(cfg.unsupported_hold_reward_weight),
            "stability": float(cfg.stability_reward_weight),
            "height": float(cfg.height_reward_weight),
            "palm_only_penalty": -float(cfg.palm_only_penalty_weight),
            "overload_penalty": -float(cfg.overload_penalty_weight),
            "forbidden_penalty": -float(cfg.forbidden_penalty_weight),
            "smoothness_penalty": -float(cfg.smoothness_penalty_weight),
            "hard_fault_penalty": -float(cfg.fault_penalty),
            "drop_penalty": -float(cfg.drop_penalty),
            "success_bonus": float(cfg.success_bonus),
        }
        term_unweighted = {
            "approach": approach,
            "contact": contact_fraction,
            "opposition": required,
            "band": in_band,
            "closure": drive * (1.0 - required),
            "unsupported_hold": hold_progress * released,
            "stability": stability,
            "height": torch.relu(self._height_gain_m) / C.MIN_HEIGHT_GAIN_M,
            "palm_only_penalty": palm_only,
            "overload_penalty": overload,
            "forbidden_penalty": forbidden,
            "smoothness_penalty": action_delta + 0.05 * jerk,
            "hard_fault_penalty": hard_fault.to(dtype=torch.float32),
            "drop_penalty": self._drop_now.float(),
            "success_bonus": self._terminal_success.float(),
        }
        term_values = {
            name: term_weights[name] * value
            for name, value in term_unweighted.items()
        }
        # 硬故障项沿用 v1.2 起就有的唯一表达式（回归测试断言的 helper）。
        term_values["hard_fault_penalty"] = hard_fault_penalty
        reward = sum(term_values.values())
        logs = self.extras.setdefault("log", {})
        logs.update(
            {
                "FixedGrasp/contact_fraction": float(contact_fraction.mean().item()),
                "FixedGrasp/required_contact_fraction": float(required.mean().item()),
                "FixedGrasp/in_band_fraction": float(in_band.mean().item()),
                "FixedGrasp/hold_steps": float(self._hold_age.float().mean().item()),
                "FixedGrasp/support_released_fraction": float(released.mean().item()),
                "FixedGrasp/palm_only_fraction": float(palm_only.mean().item()),
                "FixedGrasp/center_drift_m": float(self._drift_m.mean().item()),
                "FixedGrasp/height_gain_m": float(self._height_gain_m.mean().item()),
                "FixedGrasp/action_delta": float(action_delta.mean().item()),
                "FixedGrasp/fault_fraction": float(hard_fault.float().mean().item()),
                "FixedGrasp/approach_potential": float(
                    approach_potential.mean().item()
                ),
                "FixedGrasp/approach_shaping_delta": float(approach.mean().item()),
                "FixedGrasp/self_collision_fraction": float(
                    self._self_collision.float().mean().item()
                ),
                "FixedGrasp/open_hand_baseline": float(bool(self._open_hand_baseline)),
            }
        )
        # v1.5.0：末策略步的批次快照（不是整回合/整更新总回报，不做累加）。
        self.extras["fixed_grasp_reward_terms"] = {
            "scope": "last_policy_step_batch_snapshot_not_episode_sum",
            "num_envs": int(reward.numel()),
            "support_release_fraction": float(released.mean().item()),
            "total_mean": float(reward.mean().item()),
            "terms": {
                name: {
                    "weight": float(term_weights[name]),
                    "unweighted_mean": float(
                        term_unweighted[name].to(dtype=torch.float32).mean().item()
                    ),
                    "weighted_mean": float(
                        term_values[name].to(dtype=torch.float32).mean().item()
                    ),
                }
                for name in term_weights
            },
        }
        return reward

    @torch.inference_mode()
    def _get_observations(self) -> dict[str, torch.Tensor]:
        fruit_pose = self._tomato.data.root_pose_w.torch
        ee_pose = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        relative = quat_apply_inverse(
            ee_pose[:, 3:7], fruit_pose[:, :3] - ee_pose[:, :3]
        ) / 0.10
        axes = (self._fruit_axes_m / 0.040).clamp(0.0, 1.5)
        quat = fruit_pose[:, 3:7]
        # IsaacLab 3 的四元数是 XYZW；球体朝向仍按正确约定构造观测。
        x, y, z, w = quat.unbind(dim=1)
        rotation = torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y + w * z),
                2 * (x * z - w * y),
                2 * (x * y - w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z + w * x),
            ),
            dim=1,
        )
        fruit_twist = torch.cat(
            (
                quat_apply_inverse(
                    ee_pose[:, 3:7], self._tomato.data.root_lin_vel_w.torch
                )
                / 0.5,
                quat_apply_inverse(
                    ee_pose[:, 3:7], self._tomato.data.root_ang_vel_w.torch
                )
                / 2.0,
            ),
            dim=1,
        )
        joint_pos = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
        joint_vel = self._robot.data.joint_vel.torch[:, self._hand_joint_ids]
        limits = self._robot.data.joint_pos_limits.torch[:, self._hand_joint_ids]
        joint_pos_n = (
            2.0
            * (joint_pos - limits[..., 0])
            / (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6)
            - 1.0
        )
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        tip_velocity = self._robot.data.body_lin_vel_w.torch[
            :, self._fingertip_contact_body_ids
        ]
        tip_relative = quat_apply_inverse(
            fruit_pose[:, None, 3:7].expand(-1, 5, -1).reshape(-1, 4),
            (tips - fruit_pose[:, None, :3]).reshape(-1, 3),
        ).reshape(self.num_envs, 5, 3) / 0.08
        surface_error = (
            torch.linalg.vector_norm(tips - fruit_pose[:, None, :3], dim=2)
            - self._fruit_axes_m.mean(dim=1, keepdim=True)
        ) / 0.020
        in_band = (
            (self._last_physical_force_n >= self._soft_low)
            & (self._last_physical_force_n <= self._soft_high)
        ).float()
        tactile = torch.cat(
            (
                (self._last_physical_force_n / 4.0).unsqueeze(-1),
                ((self._last_physical_force_n - self._previous_force_n) / 2.0).unsqueeze(-1),
                (self._last_physical_force_n >= float(self.cfg.contact_force_threshold_n))
                .float()
                .unsqueeze(-1),
                in_band.unsqueeze(-1),
                surface_error.unsqueeze(-1),
                tip_relative,
                (torch.linalg.vector_norm(tip_velocity, dim=2) / 0.5).unsqueeze(-1),
                (self._contact_age.float() / 4.0).unsqueeze(-1),
            ),
            dim=2,
        ).reshape(self.num_envs, 50)
        policy = torch.cat(
            (
                relative,
                axes,
                rotation,
                fruit_twist,
                joint_pos_n,
                joint_vel / 2.0,
                tactile,
                self._previous_executed_actions,
            ),
            dim=1,
        ).clamp(-5.0, 5.0)
        if policy.shape[1] != C.ACTOR_OBSERVATION_DIM:
            raise RuntimeError(f"Actor observation drifted to {policy.shape[1]}D")
        self._contact_ratio = torch.where(
            self._hold_age > 0, self._contact_mask.float().mean(dim=1), self._contact_ratio
        )
        self._band_ratio = torch.where(
            self._hold_age > 0,
            ((self._last_physical_force_n >= self._soft_low) & (self._last_physical_force_n <= self._soft_high))
            .float()
            .mean(dim=1),
            self._band_ratio,
        )
        self._palm_ratio = torch.where(
            self._hold_age > 0,
            (self._last_palm_force_n >= float(self.cfg.contact_force_threshold_n)).float(),
            self._palm_ratio,
        )
        privileged = torch.cat(
            (
                self._last_physical_force_n / 4.0,
                self._last_palm_force_n.unsqueeze(1) / 2.0,
                (tips - fruit_pose[:, None, :3]).reshape(self.num_envs, -1) / 0.10,
                self._tomato.data.root_lin_vel_w.torch / 0.5,
                self._tomato.data.root_ang_vel_w.torch / 2.0,
                self._fruit_mass_kg.unsqueeze(1) / 0.25,
                self._executed_actions,
                self._raw_actions,
                (self._hold_age.float() / max(1, int(self.cfg.required_unsupported_hold_steps))).unsqueeze(1),
                self._contact_ratio.unsqueeze(1),
                self._band_ratio.unsqueeze(1),
                self._palm_ratio.unsqueeze(1),
                self._self_collision.float().unsqueeze(1),
            ),
            dim=1,
        )
        if privileged.shape[1] != 61:
            raise RuntimeError(f"Critic privileged observation drifted to {privileged.shape[1]}D")
        self._previous_force_n.copy_(self._last_physical_force_n)
        return {"policy": policy, "critic": torch.cat((policy, privileged), dim=1)}

    # --------------------------------------------------------------- reset
    @torch.inference_mode()
    def _finalize_episodes(self, env_ids: torch.Tensor) -> None:
        rows = self._tracker.finalize(
            env_ids,
            success=self._terminal_success,
            reason_code=self._terminal_reason,
            terminated=self.reset_terminated,
            truncated=self.reset_time_outs,
            stem_fracture=self._faults["stem_fracture"],
            support_release_step=int(self.cfg.support_release_step),
        )
        if rows:
            for row in rows:
                row['max_aabb_overlap_m'] = row['max_penetration_depth_m']
                row['max_penetration_depth_m'] = float(self._raw_penetration_peak_m[row['env_id']].item())
                row['penetration_source'] = 'PhysX packed per-point separation; all 27 hand bodies at policy samples'
            self.last_episode_records = rows
            self.completed_episodes.extend(rows)
        # canonical ``_log_episodes``（在 reset 内部调用）读取这些缓冲。
        self._success[env_ids] = self._terminal_success[env_ids]
        self._dropped[env_ids] = self._drop_now[env_ids]
        self._retained[env_ids] = self._terminal_success[env_ids]
        self._timed_out[env_ids] = self.reset_time_outs[env_ids]
        self._hold_streak[env_ids] = self._tracker.max_hold_age[env_ids]

    @torch.inference_mode()
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._all_env_ids
        if not hasattr(self, "_tracker"):
            # 构造尚未走完时的防御性 reset（当前 Isaac Lab 只在 reset()/step()
            # 触发，但保留与 canonical 同口径的保护）。
            super()._reset_idx(env_ids)
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        self._finalize_episodes(ids)
        super()._reset_idx(env_ids)
        # 逐 env 独立清 streak / anchor / 粘滞硬故障：只清被 reset 的环境，
        # 仍在运行的 env 保持自己的物理子步状态。
        self._physical_hold_guard.reset(ids)
        self._hold_age[ids] = 0
        _clear_selected_approach_potential(
            self._approach_potential_prev,
            self._approach_potential_ready,
            ids,
        )
        self._hold_window_drift_m[ids] = 0.0
        self._contact_age[ids] = 0
        self._contact_mask[ids] = False
        self._required_contact[ids] = False
        self._hold_ok[ids] = False
        self._drift_m[ids] = 0.0
        self._height_gain_m[ids] = 0.0
        self._rel_speed_m_s[ids] = 0.0
        self._joint_excursion_rad[ids] = 0.0
        self._max_applied_effort_nm[ids] = 0.0
        self._self_collision[ids] = False
        self._self_collision_clearance_m[ids] = float("inf")
        self._collider_distance_m[ids] = float("inf")
        self._penetration_depth_m[ids] = 0.0
        self._pad_penetration_depth_m[ids] = 0.0
        self._raw_penetration_peak_m[ids] = 0.0
        self._support_release_by_time[ids] = False
        for flag in self._faults.values():
            flag[ids] = False
        self._terminal_reason[ids] = 0
        self._terminal_success[ids] = False
        self._drop_now[ids] = False
        self._contact_ratio[ids] = 0.0
        self._band_ratio[ids] = 0.0
        self._palm_ratio[ids] = 0.0
        self._previous_force_n[ids] = 0.0
        self._open_hand_targets[ids] = self._nominal_joint_targets[ids][
            :, self._hand_joint_ids
        ]
        self._state_ready = False
        if not hasattr(self, "_force_slow_phase"):
            self._force_slow_phase = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        else:
            # 只清被 reset 的环境，保留仍在运行环境的保持阶段标志。
            self._force_slow_phase[ids] = False
        # 只清被 reset 环境的逐回合累加器：部分 done 不得丢其它 episode 统计。
        self._tracker.clear(ids)
        self._record_reset_pose(ids)
        self.extras.setdefault("log", {}).update(
            {
                "FixedGrasp/reset_count": float(len(ids)),
                "FixedGrasp/completed_episodes": float(len(self.completed_episodes)),
                "FixedGrasp/stage": 0.0,
            }
        )

    @torch.inference_mode()
    def _record_reset_pose(self, ids: torch.Tensor) -> None:
        """记录本次 reset 的实际初始姿态（固定场景选择的可复核证据）。

        父环境会在初始 clearance 不足时把果实沿接近轴自动外推，因此必须同时
        记录“请求的固定预置距离”和“实际落点相对标定闭合中心的偏移”，位置选择
        才有据可依。这里不改变任何物理量。
        """

        poses = self._tomato.data.root_pose_w.torch[ids]
        ee_pose = self._robot.data.body_pose_w.torch[ids, self._ee_body_id]
        center_g = poses.new_tensor(self.cfg.closed_grasp_fruit_center_m)
        calibrated_center_w = ee_pose[:, :3] + torch.bmm(
            self._grasp_basis_w[ids], center_g.expand(len(ids), -1).unsqueeze(2)
        ).squeeze(2)
        offset_w = poses[:, :3] - calibrated_center_w
        fingertip_positions = self._robot.data.body_pos_w.torch[
            ids[:, None], self._fingertip_contact_body_ids
        ]
        palm_position = self._robot.data.body_pos_w.torch[ids, self._ee_body_id]
        rows = [
            {
                "env_id": int(index),
                "episode_id": int(self._tracker.episode_index[index].item()),
                "fruit_world_xyz_m": [float(value) for value in pose[:3].tolist()],
                "fruit_world_quat_xyzw": [float(value) for value in pose[3:7].tolist()],
                "quaternion_order": "XYZW",
                "reference_center_w_m": [
                    float(value)
                    for value in self._pinch_reference_center_w[index].tolist()
                ],
                # 父环境内部使用的标定闭合中心（同公式重算）与真实落点偏移。
                "calibrated_center_w_m": [
                    float(value) for value in calibrated_center_w[row].tolist()
                ],
                "actual_offset_from_calibrated_center_m": [
                    float(value) for value in offset_w[row].tolist()
                ],
                "actual_offset_norm_m": float(offset_w[row].norm().item()),
                "fingertip_frame_positions_w_m": [
                    [float(value) for value in fingertip_positions[row, digit].tolist()]
                    for digit in range(fingertip_positions.shape[1])
                ],
                "palm_position_w_m": [
                    float(value) for value in palm_position[row].tolist()
                ],
                "closed_grasp_fruit_center_m": [
                    float(value) for value in self.cfg.closed_grasp_fruit_center_m
                ],
                "requested_pregrasp_distance_m": float(
                    self.cfg.pregrasp_distance_range_m[0]
                ),
                "pinch_working_offset_m": [
                    float(value) for value in self.cfg.pinch_working_offset_m
                ],
                "support_release_step": int(self.cfg.support_release_step),
            }
            for row, (index, pose) in enumerate(zip(ids.tolist(), poses))
        ]
        self.last_reset_poses = rows
        self.reset_pose_count += len(rows)
        if not self.first_reset_poses:
            self.first_reset_poses = deepcopy(rows)

    # ------------------------------------------------------------- identity
    def _build_identity(self) -> dict[str, Any]:
        cfg = self.cfg
        spec = C.stage_spec(cfg.stage)
        fruit_axes = [float(value) for value in self._fruit_axes_m[0].tolist()]
        mass = float(self._fruit_mass_kg[0].item())
        density = float(self._fruit_density_kg_m3[0].item())
        for index, value in enumerate(fruit_axes):
            if abs(value - C.FRUIT_RADIUS_M) > 1.0e-6:
                raise C.FixedFruitGraspError(
                    f"固定几何未被 prestartup event 采用：半轴[{index}]={value!r}"
                )
        if not math.isfinite(mass) or abs(mass - C.FRUIT_MASS_KG) > C.FRUIT_MASS_TOLERANCE_KG:
            raise C.FixedFruitGraspError(
                "父环境计算的实际果实质量与固定声明不一致："
                f"实际 {mass!r} kg，声明 {C.FRUIT_MASS_KG!r} kg，"
                f"容差 {C.FRUIT_MASS_TOLERANCE_KG!r} kg"
            )
        assets = _asset_identity()
        identity = {
            "contract_id": "fixed_fruit_grasp_v1",
            "task_id": cfg.task_id,
            "task_version": cfg.task_version,
            "record_schema_version": C.RECORD_SCHEMA_VERSION,
            "run": {
                "num_envs": int(self.num_envs),
                "device": str(self.device),
                "configured_sim_device": str(cfg.sim.device),
                "device_binding_note": (
                    "Isaac Lab 3.x DirectRLEnv 实际使用 cfg.sim.device；cfg.device 是镜像"
                    "簿记字段，validate_cfg 强制两者一致"
                ),
                "seed": int(cfg.seed),
            },
            # 阶段差异只放在 ``stage`` 键内：外部账本按运行期字段剔除该键后，
            # 物理合同哈希在 contact/hold 之间一致（允许 actor-only 迁移）。
            "stage": spec.to_json(),
            "physics": {
                "hand_collision_cooking": self._collision_cooking_identity,
                "policy_rate_hz": C.POLICY_RATE_HZ,
                "policy_dt_s": C.POLICY_DT_S,
                "sim_dt_s": float(cfg.sim.dt),
                "decimation": int(cfg.decimation),
                "episode_steps": C.EPISODE_STEPS,
                "episode_length_s": float(cfg.episode_length_s),
                "observation_dims": {
                    "actor": C.ACTOR_OBSERVATION_DIM,
                    "critic": C.CRITIC_STATE_DIM,
                    "action": C.ACTION_DIM,
                },
                # v1.4.0 升级：物理步长/decimation/观测维度/接触门限都不变，
                # 只有“保持计数与硬故障判定”的频率/计数/采样语义升级。
                "physical_hold_guard": {
                    "guard_frequency": C.PHYSICAL_HOLD_GUARD_FREQUENCY,
                    "guard_frequency_unit": "1/120 s 物理子步",
                    "hold_streak_semantics": C.PHYSICAL_HOLD_STREAK_SEMANTICS,
                    "hard_fault_persistence": C.PHYSICAL_HOLD_HARD_FAULT_PERSISTENCE,
                    "anchor_semantics": C.PHYSICAL_HOLD_ANCHOR_SEMANTICS,
                    "success_semantics": C.PHYSICAL_HOLD_SUCCESS_SEMANTICS,
                    "missing_data_policy": C.PHYSICAL_HOLD_MISSING_DATA_POLICY,
                    "required_hold_substeps": int(self._physical_hold_required_substeps),
                    "expected_substeps_per_policy_step": int(C.DECIMATION),
                    "scene_update_hook": (
                        "wrap_scene_update_after_original; record only when "
                        "_sim_step_counter advances and dt == physics_dt"
                    ),
                    "dt_zero_reset_callbacks_recorded": False,
                    "duplicate_scene_update_not_counted": True,
                    "sim_step_counter_delta_required": 1,
                    "sim_step_counter_gap_policy": "fail_closed",
                    "support_release_signal": (
                        "只消费原有支撑掩码 _stem_detached（attached 取反）；"
                        "guard 不重新测量支撑 wrench"
                    ),
                    "support_wrench_zero_provenance": (
                        "释放后外力/力矩为零来自原 _apply_attachment_wrench 的 attached 掩码"
                        "（fixture 撤支撑 F/T zero 代码未变），由外置 raw 采集物理证据核验；"
                        "guard 不声称自己检测了 wrench 为零"
                    ),
                    "diagnostic_aabb_obb_skipped_per_substep": True,
                    "effort_limits_source": "robot.data.joint_effort_limits[安全关节(21)]（共享向量，helper 校验长度并按行广播）",
                    "finger_hard_high_source": "C.FINGER_HARD_HIGH_N（5 指共享向量）",
                    "decimation_branch_required": "external_per_substep_decimation_false",
                },
                "support": {
                    "kind": "declared_compliant_support_physx_wrench_no_state_write",
                    "release_step": int(cfg.support_release_step),
                    "release_time_s": spec.support_release_time_s,
                    "declared_break_force_n": C.STEM_DECLARED_BREAK_FORCE_N,
                    "maximum_stem_force_n": C.STEM_MAXIMUM_FORCE_N,
                    "maximum_stem_torque_nm": C.STEM_MAXIMUM_TORQUE_NM,
                    "linear_natural_frequency_rad_s": C.STEM_LINEAR_NATURAL_FREQUENCY_RAD_S,
                    "linear_damping_ratio": C.STEM_LINEAR_DAMPING_RATIO,
                    "angular_natural_frequency_rad_s": C.STEM_ANGULAR_NATURAL_FREQUENCY_RAD_S,
                    "angular_damping_ratio": C.STEM_ANGULAR_DAMPING_RATIO,
                    "release_is_success_triggered": False,
                },
            },
            "fruit": {
                "shape": "sphere",
                "equivalent_diameter_m": 2.0 * C.FRUIT_RADIUS_M,
                "axes_m": fruit_axes,
                "density_kg_m3": density,
                "computed_mass_kg": mass,
                "expected_mass_kg": C.FRUIT_MASS_KG,
                "mass_tolerance_kg": C.FRUIT_MASS_TOLERANCE_KG,
                "mass_source": "parent_env computes 4/3*pi*a*b*c*rho from prestartup axes and density",
                "mass_declared_in_cfg_tomato_mass": float(
                    cfg.tomato.spawn.mass_props.mass
                ),
                "mass_declared_source_note": (
                    "cfg.tomato.spawn.mass_props.mass 只是 USD 初始声明值，实际质量以 "
                    "computed_mass_kg（父环境写回 PhysX）为准"
                ),
                "friction": {
                    "static": C.FRUIT_STATIC_FRICTION,
                    "dynamic": C.FRUIT_DYNAMIC_FRICTION,
                    "restitution": C.FRUIT_RESTITUTION,
                    "combine_mode": C.FRUIT_FRICTION_COMBINE_MODE,
                    "solver_default_static": C.FRUIT_SOLVER_STATIC_FRICTION,
                    "solver_default_dynamic": C.FRUIT_SOLVER_DYNAMIC_FRICTION,
                    "source": "cfg.tomato.spawn.physics_material",
                    "hand_side_material_source": "USD 未 author PhysicsMaterial，沿用模拟器默认物理材质；本层不覆写",
                },
                "sampling": "deterministic_fixed_no_randomization",
            },
            "actuation": {
                "source": "cfg.robot.actuators（canonical 孤立左手 USD 驱动）+ wave2 最小修复",
                "groups": {
                    group: {
                        "effort_limit_sim": float(cfg.robot.actuators[group].effort_limit_sim),
                        "velocity_limit_sim": float(cfg.robot.actuators[group].velocity_limit_sim),
                        "stiffness": float(cfg.robot.actuators[group].stiffness),
                        "damping": float(cfg.robot.actuators[group].damping),
                        "friction": float(cfg.robot.actuators[group].friction),
                        "armature": float(cfg.robot.actuators[group].armature),
                        "class_type": type(cfg.robot.actuators[group].class_type).__name__,
                    }
                    for group in ("g20_independent", "g20_followers")
                },
                "minimal_repair": {
                    "post_brake_effort_clip": True,
                    "armature_kg_m2": C.ACTUATOR_ARMATURE_KG_M2,
                    "source": "wave2 native_env 差异（brake 后逐轴 effort 裁剪 + armature 0.01）",
                },
                "mimic": {
                    relation: {
                        "driver": driver,
                        "multiplier": multiplier,
                        "offset_rad": offset,
                    }
                    for relation, (driver, multiplier, offset) in C.MIMIC_RELATIONS.items()
                },
                "mimic_note": (
                    "保留 canonical 软件 mimic 关系；不继承派生 PPV1 资产的 NewtonMimicAPI/"
                    "offset=0 被动零驱动路线（运行时支持未证实）"
                ),
                "joint_hard_limits_rad": self._joint_limits,
                "joint_hard_limits_source": "robot.data.joint_pos_limits（USD 关节限位）",
                "joint_software_limits_source": "LeftHandDirectionalPDActuator._JOINT_LIMITS",
            },
            "contact": {
                "force_threshold_n": C.CONTACT_FORCE_N,
                "min_contact_fingers": C.MIN_CONTACT_FINGERS,
                "soft_low_n": list(C.FINGER_SOFT_LOW_N),
                "soft_high_n": list(C.FINGER_SOFT_HIGH_N),
                "hard_high_n": list(C.FINGER_HARD_HIGH_N),
                "palm_may_carry_load": True,
                "palm_only_counts_as_success": False,
                "palm_fruit_collision_filter_enabled": bool(
                    cfg.palm_fruit_collision_filter_enabled
                ),
                "real_self_collisions_enabled": bool(
                    cfg.robot.spawn.articulation_props.enabled_self_collisions
                ),
                "self_collision_status": (
                    "ENABLED_WITH_EXPLICIT_FINE_CONVEX_DECOMPOSITION（仍需同条件评估）"
                ),
                "declared_self_collision_pairs": [list(pair) for pair in cfg.self_collision_pairs],
                "effective_self_collision_pairs": [
                    list(pair) for pair in self._effective_self_collision_pairs
                ],
                "skipped_self_collision_pairs": [
                    list(pair) for pair in self._skipped_self_collision_pairs
                ],
                "self_collision_penetration_limit_m": C.SELF_COLLISION_PENETRATION_LIMIT_M,
                "legacy_thumb_index_proxy_is_sufficient": False,
                "self_collision_guard_source": (
                    "PhysX packed separation; 5 declared cross-digit pairs; "
                    "every 1/120 s physics substep (v1.4.0)"
                ),
                "solver_self_collision_config": (
                    "cfg.robot.spawn.articulation_props.enabled_self_collisions=True（真实 PhysX 求解器自碰撞，"
                    "不在运行时关闭）；运行时实际值由主线程 physics smoke 读取 PhysxSchema 记录"
                ),
                "self_collision_diagnostic_scope": (
                    "声明的5对跨指刚体以PhysX packed逐点separation作为硬守护，OBB-SAT只诊断；"
                    "硬守护在每个真实1/120 s物理子步判定（v1.4.0），但仍不是全部手内刚体对的完整审计"
                ),
                "penetration_guard_scope": (
                    "硬门读取全部27个手部刚体→果实的PhysX packed逐点separation，每个真实1/120 s"
                    "物理子步判定（v1.4.0）；AABB重叠与接触点质心深度仅诊断，不作为真实穿透深度"
                ),
                "physical_penetration_method": (
                    "packed_separation_all_27_hand_bodies_per_physics_substep"
                ),
                "claims_out_of_scope": [
                    "完整穿透/碰撞审计",
                    "26 场景或 110 对覆盖",
                    "实机触觉标定",
                    "真机可达性/集成完成",
                ],
            },
            "reward": {
                "approach": {
                    "type": "discounted_potential_difference",
                    "gamma": _APPROACH_SHAPING_GAMMA,
                    "gamma_source": (
                        "research.small_fruit_pinch_v1.source.agents."
                        "SmallFruitPinchPPORunnerCfg.algorithm.gamma"
                    ),
                    "potential": (
                        "mean_top_2(exp(-max(digit_AABB_gap_m, 0) / "
                        f"{_APPROACH_GAP_DECAY_M}))"
                    ),
                    "first_reward_increment": 0.0,
                    "reset_scope": "selected_env_ids_only",
                    "weight": float(cfg.approach_reward_weight),
                },
                "unsupported_hold": {
                    "progress": "hold_age / required_unsupported_hold_steps",
                    "progress_weight": float(cfg.unsupported_hold_reward_weight),
                    "stability": (
                        "exp(-hold_window_drift_m / "
                        f"{_STABILITY_DRIFT_DECAY_M}) * support_released * _hold_ok"
                    ),
                    "stability_weight": float(cfg.stability_reward_weight),
                },
                "hard_fault_penalty_flags": list(_HARD_FAULT_REWARD_NAMES),
                "hard_fault_penalty": float(cfg.fault_penalty),
                # v1.5.0：reward-only 变更。旧版 identity 只列了三项权重，这里
                # 补齐全部数值权重（惩罚项按带符号贡献系数记录），便于外部
                # 账本在没有训练日志的情况下直接比对奖励配置。
                "profile": "reward_only_v1.5.0",
                "weight_sign_convention": (
                    "带符号贡献系数：奖励项为正、惩罚项为负；"
                    "weighted = weight * unweighted，unweighted 保留该项实际符号"
                    "（多数项非负，approach 为折扣势差、可为负；罚项基量非负）"
                ),
                "weights": {
                    "approach": float(cfg.approach_reward_weight),
                    "contact": float(cfg.contact_reward_weight),
                    "opposition": float(cfg.opposition_reward_weight),
                    "band": float(cfg.band_reward_weight),
                    "closure": float(cfg.closure_reward_weight),
                    "unsupported_hold": float(cfg.unsupported_hold_reward_weight),
                    "stability": float(cfg.stability_reward_weight),
                    "height": float(cfg.height_reward_weight),
                    "palm_only_penalty": -float(cfg.palm_only_penalty_weight),
                    "overload_penalty": -float(cfg.overload_penalty_weight),
                    "forbidden_penalty": -float(cfg.forbidden_penalty_weight),
                    "smoothness_penalty": -float(cfg.smoothness_penalty_weight),
                    "hard_fault_penalty": -float(cfg.fault_penalty),
                    "drop_penalty": -float(cfg.drop_penalty),
                    "success_bonus": float(cfg.success_bonus),
                },
                "per_policy_step_snapshot_key": "fixed_grasp_reward_terms",
                "per_policy_step_snapshot_scope": (
                    "last_policy_step_batch_snapshot_not_episode_sum"
                ),
            },
            "criteria": {
                "approach_reward_source": "body_AABB_gap_proxy_only_not_success_evidence",
                "contact_source": "max_filtered_fruit_force_over_all_bodies_per_digit",
                "thumb_required": False,
                "hold_drift_reference": (
                    "hand_frame_fruit_position_at_physical_substep_streak_start"
                    "（v1.4.0：120 Hz 子步 streak 起点，策略刷新不重置）"
                ),
                "hold_streak_gate": (
                    "per_physics_substep_120hz_continuous_streak_zeroed_on_any_violation"
                ),
                "hold_hard_fault_gate": (
                    "seven_hard_faults_evaluated_every_physics_substep_sticky_until_env_reset"
                ),
                "hold_max_drift_m": C.HOLD_MAX_DRIFT_M,
                "hold_max_relative_speed_m_s": C.HOLD_MAX_RELATIVE_SPEED_M_S,
                "drop_height_tolerance_m": C.DROP_HEIGHT_TOLERANCE_M,
                "drop_max_drift_m": C.DROP_MAX_DRIFT_M,
                "penetration_limit_m": C.PENETRATION_LIMIT_M,
                "collider_overlap_limit_m": C.COLLIDER_OVERLAP_LIMIT_M,
                "joint_limit_margin_rad": C.JOINT_LIMIT_MARGIN_RAD,
                "instability_max_joint_velocity_rad_s": C.INSTABILITY_MAX_JOINT_VELOCITY_RAD_S,
                "instability_max_fruit_speed_m_s": C.INSTABILITY_MAX_FRUIT_SPEED_M_S,
                "goal_min_height_gain_m": C.MIN_HEIGHT_GAIN_M,
                "goal_hold_s": C.REQUIRED_HOLD_S,
            },
            "goal": {
                "status": C.GOAL_STATUS_LIFT_NOT_IMPLEMENTED,
                "metric_this_delivery": "unsupported_hold_s",
                "goal_requires_height_gain_m": C.MIN_HEIGHT_GAIN_M,
                "goal_requires_hold_s": C.REQUIRED_HOLD_S,
                "lift_available": C.lift_stage_available(),
                "lift_stage_status": C.LIFT_STAGE_STATUS,
                "lift_program": C.LIFT_PROGRAM.to_json(),
                "note": (
                    "height_gain 与 goal_success 恒为诊断/False：需要声明的受驱动升降关节，"
                    "禁止用逐帧传送手根/果实或给果实加力制造提举"
                ),
            },
            "randomization": {
                "fruit_geometry": "fixed",
                "fruit_density": "fixed",
                "fruit_position": "fixed",
                "measured_axes_noise": float(cfg.fruit_axis_relative_noise),
                "target_position_noise_m": float(cfg.target_position_noise_m),
                "tactile_gain": list(cfg.tactile_force_gain),
                "tactile_bias_n": list(cfg.tactile_force_bias_n),
                "action_delay_steps": list(cfg.action_delay_steps),
                "arm_joint_noise_deg": float(cfg.pregrasp_arm_joint_noise_deg),
                "pregrasp_distance_range_m": list(cfg.pregrasp_distance_range_m),
                "pregrasp_lateral_offset_range_m": list(cfg.pregrasp_lateral_offset_range_m),
                "pregrasp_axial_noise_m": float(cfg.pregrasp_axial_noise_m),
                "enforced_by": "validate_cfg（任何偏离一律 fail-closed）",
            },
            "open_control": {
                "mechanism": "set_open_hand_baseline",
                "open_hand_action_available": False,
                "action_inverse_available": False,
                "reason": (
                    "14D 流形写了 16 个关节（max()/区间夹紧 + 每指残差共享同一关节），"
                    "像集不满秩，零动作不是 reset 的真实 neutral 张开姿态"
                ),
                "manifold_zero_target_gap_rad": self._action_effect.get(
                    "manifold_zero_target_gap_rad"
                ),
                "bounded_closure_reference": "closure_reference_action(closure=0..1)",
            },
            "action_effect": self._action_effect,
            "assets": assets,
            "source_dependencies": _source_identity(),
            "initial_pose": {
                "deterministic": True,
                "randomization": "none",
                "closed_grasp_fruit_center_m": [
                    float(value) for value in cfg.closed_grasp_fruit_center_m
                ],
                "pregrasp_distance_m": float(cfg.pregrasp_distance_range_m[0]),
                "pregrasp_lateral_offset_range_m": [
                    float(value) for value in cfg.pregrasp_lateral_offset_range_m
                ],
                "pregrasp_axial_noise_m": float(cfg.pregrasp_axial_noise_m),
                "pinch_working_offset_m": [
                    float(value) for value in cfg.pinch_working_offset_m
                ],
                "minimum_initial_clearance_m": float(cfg.minimum_initial_clearance_m),
                "selection_note": (
                    "预抓取距离是确定单值但不必为 65 mm：本任务不学习接近，固定场景选择"
                    "由有限场景计划给出；实际 reset 姿态见 env.last_reset_poses / "
                    "env.first_reset_poses，无接触由父环境 clearance 检查强制"
                ),
                "recorded_by": "env.last_reset_poses / env.first_reset_poses / env.reset_pose_count",
            },
            "safety_guard": {
                "guarded_joints": list(self._safety_joint_names),
                "guarded_joint_count": len(self._safety_joint_ids),
                "policy_observation_joints": len(self._hand_joint_ids),
                "note": "关节硬限/速度/非有限/effort 守护覆盖 16 主动 + 5 跟随；策略观测仍为 16 维（114D 不变）",
                "effort_limits_nm": {
                    name: (
                        float(self._safety_effort_limits[index].item())
                        if math.isfinite(float(self._safety_effort_limits[index].item()))
                        else None
                    )
                    for index, name in enumerate(self._safety_joint_names)
                },
                "effort_guard_tolerance": float(self._effort_guard_tolerance),
                "joint_limit_margin_rad": float(cfg.joint_limit_margin_rad),
                "instability_max_joint_velocity_rad_s": float(
                    cfg.instability_max_joint_velocity_rad_s
                ),
            },
            "episode_record_fields": list(C.EPISODE_RECORD_FIELDS),
            "declared_limitations": [
                "holding-only：30 mm+3 s 提举目标未实现（缺声明式受驱动升降关节）",
                "自碰撞在 canonical 孤立左手 USD 上开启但未在本机 GPU 上验证，静态姿态有自激记录",
                "粗穿透代理排除 base_link（导入掌心 AABB 远大于真实掌面形状）",
                "掌心侧摩擦/材质来自 USD 资产，本层不覆写",
                "接触力/阈值是仿真量，不是实机触觉标定",
                "本包不启动 Kit/PhysX；物理与动态验证由主线程负责",
            ],
        }
        identity["asset_config"] = {
            "usd_path": assets["usd_path"],
            "asset_config_path": assets["asset_config_path"],
            "asset_files": assets["files"],
            "fruit": identity["fruit"],
            "actuation": identity["actuation"],
            "contact": identity["contact"],
            "physics": identity["physics"],
            "criteria": identity["criteria"],
        }
        return C.json_safe(identity)


@configclass
class FixedFruitGraspContactCfg(FixedFruitGraspCfg):
    """contact 课程阶段（较晚解除支撑，独立成功不计入 30 mm 目标）。"""

    stage: str = "contact"
    support_release_step: int = C.STAGE_SPECS["contact"].support_release_step
    required_unsupported_hold_steps: int = C.STAGE_SPECS[
        "contact"
    ].required_unsupported_hold_steps


def register_tasks(rsl_rl_cfg_entry_point: str | None = None) -> list[str]:
    """注册 gym 任务（contact/hold）；lift 不注册（未实现，fail-closed）。"""

    import gymnasium as gym

    registered: list[str] = []
    for suffix, cfg_entry in (
        ("contact", "research.fixed_fruit_grasp_v1.env:FixedFruitGraspContactCfg"),
        ("hold", "research.fixed_fruit_grasp_v1.env:FixedFruitGraspCfg"),
    ):
        task_id = f"{C.TASK_ID}-{suffix}"
        if task_id in gym.registry:
            registered.append(task_id)
            continue
        kwargs: dict[str, Any] = {
            "env_cfg_entry_point": cfg_entry,
        }
        if rsl_rl_cfg_entry_point is not None:
            kwargs["rsl_rl_cfg_entry_point"] = rsl_rl_cfg_entry_point
        gym.register(
            id=task_id,
            entry_point="research.fixed_fruit_grasp_v1.env:FixedFruitGraspEnv",
            disable_env_checker=True,
            kwargs=kwargs,
        )
        registered.append(task_id)
    return registered


__all__ = [
    "FixedFruitGraspCfg",
    "FixedFruitGraspContactCfg",
    "FixedFruitGraspEnv",
    "FixedFruitGraspPDActuator",
    "make_config",
    "register_tasks",
    "validate_cfg",
]
