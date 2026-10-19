"""物理子步保持守卫：把“连续保持”和七类硬故障从 20 Hz 策略采样下移到
真实 1/120 s 物理子步（纯 torch 助手，不导入 Isaac / Kit / USD）。

背景（v3 独立审查 HOLD-P1-01）：v3 只在每个策略步末读取一次接触力、相对
速度、hold 漂移和硬故障，20 Hz 门控无法排除六个物理子步之间的瞬时失效；
同一份 120 Hz 现场证据已经确认存在 29 次速度越界与 2 次真实拇指过载被
策略采样漏掉。本模块给出可复用的最小语义：

* ``streak``：逐 env 的连续保持物理子步计数；任何子步不满足（未撤支撑、
  真实接触手指不足、相对速度/漂移越界、任一硬故障）立即清零。
* ``hard_faults``：七类硬故障的粘滞标记；一旦某个 1/120 s 子步触发，就
  粘滞到该 env reset，供下一次 ``_get_dones`` 立即终止，且不会被策略采样
  刷新（``_refresh_task_state``）覆写丢失。
* ``anchor_hand``：物理 streak 起点的手坐标系果实位置（hold anchor）；
  只在真实子步更新，策略刷新只读镜像，不重置。
* ``peak_streak``：逐回合物理子步 streak 峰值，供逐回合记录复核。

fail-closed：``enabled``/``ready`` 之外的任何调用、缺失的 live 数据、
非 1/120 s 的物理步长、重复或倒退的 ``_sim_step_counter``、以及一个策略
步内子步数不等于 ``decimation``，都直接抛 :class:`PhysicalHoldError`；
绝不在证据不全时判为通过。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch


# ------------------------------------------------------------------ 常量
HARD_FAULT_NAMES: Final = (
    "overload",
    "nonfinite",
    "penetration",
    "joint_limit",
    "self_collision",
    "instability",
    "joint_effort",
)

_DT_EPSILON_S: Final = 1.0e-9


class PhysicalHoldError(RuntimeError):
    """物理子步保持守卫的输入或时序前提不成立（fail-closed）。"""


def _require_tensor(
    name: str,
    value: Any,
    *,
    first_dim: int | None = None,
    length: int | None = None,
) -> torch.Tensor:
    """缺失/类型错误的 live 数据直接失败，不当作 0 或 False。"""

    if value is None:
        raise PhysicalHoldError(f"物理子步判定缺少 live 数据：{name}")
    if not isinstance(value, torch.Tensor):
        raise PhysicalHoldError(
            f"物理子步判定字段 {name} 必须是 torch.Tensor，得到 {type(value).__name__}"
        )
    if value.numel() == 0:
        raise PhysicalHoldError(f"物理子步判定字段 {name} 为空张量")
    if first_dim is not None and int(value.shape[0]) != int(first_dim):
        raise PhysicalHoldError(
            f"物理子步判定字段 {name} 的环境维度不是 {first_dim}：{tuple(value.shape)}"
        )
    if length is not None:
        if value.dim() != 1 or int(value.numel()) != int(length):
            raise PhysicalHoldError(
                f"物理子步判定共享向量 {name} 必须是长度 {int(length)} 的一维张量："
                f"{tuple(value.shape)}"
            )
    return value


def _require_finite_scalar(name: str, value: Any) -> float:
    if value is None:
        raise PhysicalHoldError(f"物理子步判定缺少标量：{name}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 防御性
        raise PhysicalHoldError(f"物理子步判定标量 {name} 不是数值：{value!r}") from exc
    if not math.isfinite(number):
        raise PhysicalHoldError(f"物理子步判定标量 {name} 不是有限值：{value!r}")
    return number


# ------------------------------------------------------- 七类硬故障计算口径
_TENSOR_FIELDS: Final = (
    "joint_pos",
    "joint_vel",
    "joint_pos_limits",
    "applied_effort",
    "effort_limits",
    "finger_force_n",
    "finger_hard_high_n",
    "fruit_pose",
    "fruit_linear_velocity",
    "fruit_angular_velocity",
    "ee_pose",
    "ee_linear_velocity",
    "tip_positions",
    "tip_velocities",
    "palm_force_n",
    "forbidden_proximal_n",
    "previous_force_n",
    "executed_actions",
    "raw_actions",
    "previous_executed_actions",
    "fruit_penetration_depth_m",
    "self_penetration_depth_m",
    "released",
    "required_contact",
    "relative_speed_m_s",
    "fruit_in_hand_m",
)
_SCALAR_FIELDS: Final = (
    "joint_limit_margin_rad",
    "instability_max_joint_velocity_rad_s",
    "instability_max_fruit_speed_m_s",
    "fruit_penetration_limit_m",
    "self_collision_penetration_limit_m",
    "effort_guard_tolerance",
)
# 这两个是按关节（21 个手关节 = 16 主动 + 5 跟随）/按手指（5 指）的共享
# 向量，没有 env 维；env 侧按真实 USD 形状传入，由 helper 校验长度并按行
# 广播。其余张量第一维必须是 env。
_NON_ENV_TENSOR_FIELDS: Final = ("effort_limits", "finger_hard_high_n")


@dataclass(frozen=True)
class PhysicalSubstepState:
    """一个真实物理子步的全部 live 判定输入（由 env 在 ``scene.update`` 后读取）。

    全部字段都是真实运行时缓冲的快照：接触力取整指 filtered fruit
    ``force_matrix_w``；穿透取 PhysX packed separation 汇总深度；自碰撞取
    声明的 5 对跨指刚体真实 separation；不包含任何只读诊断 AABB/OBB 计算。

形状约定：除 ``effort_limits``（与被控关节同长的共享向量，本任务为 21 个
手关节）和 ``finger_hard_high_n``（与五路整指接触同长的声明硬门向量）之外，
所有张量第一维都是 env。这两个共享向量由 helper 校验长度并按行广播，
env 侧按真实 USD 形状直接传入，不做逐 env 复制。
    """

    joint_pos: Any
    joint_vel: Any
    joint_pos_limits: Any
    applied_effort: Any
    effort_limits: Any
    finger_force_n: Any
    finger_hard_high_n: Any
    fruit_pose: Any
    fruit_linear_velocity: Any
    fruit_angular_velocity: Any
    ee_pose: Any
    ee_linear_velocity: Any
    tip_positions: Any
    tip_velocities: Any
    palm_force_n: Any
    forbidden_proximal_n: Any
    previous_force_n: Any
    executed_actions: Any
    raw_actions: Any
    previous_executed_actions: Any
    fruit_penetration_depth_m: Any
    self_penetration_depth_m: Any
    released: Any
    required_contact: Any
    relative_speed_m_s: Any
    fruit_in_hand_m: Any
    joint_limit_margin_rad: Any
    instability_max_joint_velocity_rad_s: Any
    instability_max_fruit_speed_m_s: Any
    fruit_penetration_limit_m: Any
    self_collision_penetration_limit_m: Any
    effort_guard_tolerance: Any

    def require_complete(self, *, num_envs: int) -> "PhysicalSubstepState":
        """任一预期数据缺失/类型错误即失败（fail-closed）。"""

        for name in _TENSOR_FIELDS:
            value = getattr(self, name)
            if name == "effort_limits":
                # 与被控关节同长：env 传真实 (21,) 共享向量。
                _require_tensor(name, value, length=int(self.joint_pos.shape[1]))
            elif name == "finger_hard_high_n":
                # 与五路整指接触同长：env 传声明 (5,) 硬门向量。
                _require_tensor(name, value, length=int(self.finger_force_n.shape[1]))
            else:
                _require_tensor(name, value, first_dim=num_envs)
        for name in _SCALAR_FIELDS:
            _require_finite_scalar(name, getattr(self, name))
        return self


def compute_hard_faults(state: PhysicalSubstepState) -> dict[str, torch.Tensor]:
    """按 v3 冻结源码的真实 live 字段口径计算七类硬故障（逐 env 布尔张量）。

    与 ``env._refresh_task_state`` 的差别只在执行频率：这里是每个真实
    1/120 s 物理子步，而不是每个 20 Hz 策略采样点。
    """

    joint_pos = state.joint_pos
    joint_vel = state.joint_vel
    limits = state.joint_pos_limits
    joint_limit_margin = float(state.joint_limit_margin_rad)
    num_envs = int(joint_pos.shape[0])
    device = joint_pos.device

    overload = (state.finger_force_n > state.finger_hard_high_n.reshape(1, -1)).any(dim=1)
    joint_limit = (
        torch.maximum(limits[:, :, 0] - joint_pos, joint_pos - limits[:, :, 1])
        .clamp_min(0.0)
        .amax(dim=1)
        > joint_limit_margin
    )
    instability = (
        joint_vel.abs().amax(dim=1)
        > float(state.instability_max_joint_velocity_rad_s)
    ) | (
        torch.linalg.vector_norm(state.fruit_linear_velocity, dim=1)
        > float(state.instability_max_fruit_speed_m_s)
    )

    effort_limits = state.effort_limits.reshape(1, -1)
    finite_limit = torch.isfinite(effort_limits) & (effort_limits > 0.0)
    if bool(finite_limit.any()):
        ratio = state.applied_effort.abs() / torch.where(
            finite_limit, effort_limits, torch.ones_like(effort_limits)
        ).clamp_min(1.0e-9)
        joint_effort = (ratio > 1.0 + float(state.effort_guard_tolerance)).any(dim=1)
    else:
        joint_effort = torch.zeros(num_envs, dtype=torch.bool, device=device)

    quaternion_ok = (state.fruit_pose[:, 3:7].norm(dim=1) > 1.0e-6) & (
        state.ee_pose[:, 3:7].norm(dim=1) > 1.0e-6
    )
    finite = (
        torch.isfinite(joint_pos).all(dim=1)
        & torch.isfinite(joint_vel).all(dim=1)
        & torch.isfinite(state.applied_effort).all(dim=1)
        & torch.isfinite(state.fruit_pose).all(dim=1)
        & torch.isfinite(state.fruit_linear_velocity).all(dim=1)
        & torch.isfinite(state.fruit_angular_velocity).all(dim=1)
        & torch.isfinite(state.ee_pose).all(dim=1)
        & torch.isfinite(state.ee_linear_velocity).all(dim=1)
        & torch.isfinite(state.tip_positions).flatten(1).all(dim=1)
        & torch.isfinite(state.tip_velocities).flatten(1).all(dim=1)
        & torch.isfinite(state.finger_force_n).all(dim=1)
        & torch.isfinite(state.palm_force_n)
        & torch.isfinite(state.forbidden_proximal_n).flatten(1).all(dim=1)
        & torch.isfinite(state.previous_force_n).all(dim=1)
        & torch.isfinite(state.executed_actions).all(dim=1)
        & torch.isfinite(state.raw_actions).all(dim=1)
        & torch.isfinite(state.previous_executed_actions).all(dim=1)
        & quaternion_ok
    )

    penetration = state.fruit_penetration_depth_m > float(state.fruit_penetration_limit_m)
    self_collision = state.self_penetration_depth_m > float(
        state.self_collision_penetration_limit_m
    )
    return {
        "overload": overload,
        "nonfinite": ~finite,
        "penetration": penetration,
        "joint_limit": joint_limit,
        "self_collision": self_collision,
        "instability": instability,
        "joint_effort": joint_effort,
    }


# ----------------------------------------------------------------- 守卫本体
class PhysicalHoldGuard:
    """逐 env 的物理子步 streak / 粘滞硬故障 / anchor 状态机。

    使用顺序：构造 -> :meth:`enable`（构造完成后）-> 每个真实物理子步
    :meth:`observe_substep` -> 每个策略步 :meth:`complete_policy_interval`
    （在 ``_get_dones`` 里）-> 该 env reset 时 :meth:`reset`。
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device | str,
        physics_dt_s: float,
        decimation: int,
        required_substeps: int,
        hold_max_drift_m: float,
        hold_max_relative_speed_m_s: float,
    ) -> None:
        count = int(num_envs)
        if count <= 0:
            raise PhysicalHoldError(f"物理子步守卫的 num_envs 必须为正：{num_envs!r}")
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.physics_dt_s = _require_finite_scalar("physics_dt_s", physics_dt_s)
        if self.physics_dt_s <= 0.0:
            raise PhysicalHoldError(f"physics_dt_s 必须为正：{physics_dt_s!r}")
        self.decimation = int(decimation)
        if self.decimation <= 0:
            raise PhysicalHoldError(f"decimation 必须为正：{decimation!r}")
        self.required_substeps = int(required_substeps)
        if self.required_substeps <= 0:
            raise PhysicalHoldError(f"required_substeps 必须为正：{required_substeps!r}")
        self.hold_max_drift_m = _require_finite_scalar(
            "hold_max_drift_m", hold_max_drift_m
        )
        self.hold_max_relative_speed_m_s = _require_finite_scalar(
            "hold_max_relative_speed_m_s", hold_max_relative_speed_m_s
        )
        self.num_envs = count
        self.streak = torch.zeros(count, dtype=torch.long, device=self._device)
        self.peak_streak = torch.zeros(count, dtype=torch.long, device=self._device)
        self.anchor_hand = torch.zeros((count, 3), dtype=torch.float32, device=self._device)
        self.anchor_valid = torch.zeros(count, dtype=torch.bool, device=self._device)
        self.last_hold_ok = torch.zeros(count, dtype=torch.bool, device=self._device)
        self.last_drift_m = torch.full(
            (count,), float("inf"), dtype=torch.float32, device=self._device
        )
        self.hard_faults = {
            name: torch.zeros(count, dtype=torch.bool, device=self._device)
            for name in HARD_FAULT_NAMES
        }
        self.enabled = False
        self.ready = False
        self._last_sim_step_counter: int | None = None
        self._substeps_in_interval = 0

    # ---------------------------------------------------------- 生命周期
    @property
    def hard_fault_any(self) -> torch.Tensor:
        """逐 env 的粘滞硬故障并集（在任一子步触发后保持为 True 到 reset）。"""

        any_fault = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        for flag in self.hard_faults.values():
            any_fault |= flag
        return any_fault

    def enable(self, initial_sim_step_counter: int) -> None:
        """构造完成后启用；初值必须能读到当前 ``_sim_step_counter``。"""

        if isinstance(initial_sim_step_counter, bool) or not isinstance(
            initial_sim_step_counter, int
        ):
            raise PhysicalHoldError(
                "启用物理子步守卫需要整数 _sim_step_counter，得到 "
                f"{initial_sim_step_counter!r}"
            )
        if int(initial_sim_step_counter) < 0:
            raise PhysicalHoldError(f"_sim_step_counter 不能为负：{initial_sim_step_counter!r}")
        self._last_sim_step_counter = int(initial_sim_step_counter)
        self._substeps_in_interval = 0
        self.ready = True
        self.enabled = True

    @torch.inference_mode()
    def reset(self, env_ids: Any) -> None:
        """按 env 独立清 streak / anchor / 粘滞硬故障（只清被 reset 的环境）。"""

        ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        if int(ids.min()) < 0 or int(ids.max()) >= self.num_envs:
            raise PhysicalHoldError(f"reset 的 env 索引越界：{ids.tolist()}")
        self.streak[ids] = 0
        self.peak_streak[ids] = 0
        self.anchor_hand[ids] = 0.0
        self.anchor_valid[ids] = False
        self.last_hold_ok[ids] = False
        self.last_drift_m[ids] = float("inf")
        for flag in self.hard_faults.values():
            flag[ids] = False

    # ------------------------------------------------------ 子步/策略步入口
    def observe_substep(
        self,
        state: "PhysicalSubstepState | Callable[[], PhysicalSubstepState]",
        *,
        sim_step_counter: int,
        scene_dt_s: Any,
    ) -> bool:
        """消费一个 ``scene.update`` 回调；只有真实物理子步才计数。

        返回 ``True`` 表示本回调被记为一个新的真实物理子步。以下情况返回
        ``False`` 且不修改任何逐 env 状态：

        * 守卫尚未启用（构造期 ``scene.update``）；
        * ``scene_dt_s`` 为 0（reset 回调，绝不是物理步）；
        * ``sim_step_counter`` 与上次记录相同（同一次 ``scene.update`` 的重复
          通知，不重复计样本）。

        真实物理子步的 ``_sim_step_counter`` 必须严格 +1：跨度大于 1 意味着
        中间有物理回调没有经过本守卫（漏采样），此时不允许把后续回调拼成
        “连续”，直接 fail-closed。

        ``state`` 可以是 :class:`PhysicalSubstepState`，也可以是返回它的可调用
        对象；后者只在确认本回调是真实物理子步时才求值，避免 reset（dt=0）
        回调上读取还未就绪的 live 缓冲。
        """

        if not self.enabled:
            return False
        if not self.ready or self._last_sim_step_counter is None:
            raise PhysicalHoldError("物理子步守卫在未就绪时被调用")
        if isinstance(sim_step_counter, bool) or not isinstance(sim_step_counter, int):
            raise PhysicalHoldError(
                f"物理子步守卫需要整数 _sim_step_counter，得到 {sim_step_counter!r}"
            )
        counter = int(sim_step_counter)
        if scene_dt_s is None:
            raise PhysicalHoldError("scene.update 未提供 dt，无法证明 1/120 s 物理子步")
        try:
            dt_value = float(scene_dt_s)
        except (TypeError, ValueError) as exc:  # pragma: no cover - 防御性
            raise PhysicalHoldError(f"scene.update dt 不是数值：{scene_dt_s!r}") from exc
        if not math.isfinite(dt_value):
            raise PhysicalHoldError(f"scene.update dt 不是有限值：{scene_dt_s!r}")
        if abs(dt_value) <= _DT_EPSILON_S:
            # dt=0 的 reset / 初始化回调：不是物理步，既不消耗子步计数，也不
            # 因计数器被 sim.reset 归零而误报（不修改任何状态）。
            return False
        if counter < self._last_sim_step_counter:
            raise PhysicalHoldError(
                "_sim_step_counter 倒退："
                f"{counter} < {self._last_sim_step_counter}（物理步记账不可信）"
            )
        if abs(dt_value - self.physics_dt_s) > _DT_EPSILON_S:
            raise PhysicalHoldError(
                "scene.update dt 不是固定物理步长，无法证明逐物理子步采样："
                f"dt={dt_value!r} physics_dt={self.physics_dt_s!r}"
            )
        delta = counter - self._last_sim_step_counter
        if delta == 0:
            # 同一次物理步被通知两次：不重复计样本。
            return False
        if delta != 1:
            raise PhysicalHoldError(
                "真实物理子步的 _sim_step_counter 出现跳跃，说明有回调未被采样："
                f"上次 {self._last_sim_step_counter}，本次 {counter}（delta={delta}）"
            )

        resolved = state() if callable(state) else state
        resolved.require_complete(num_envs=self.num_envs)
        self._last_sim_step_counter = counter
        self._substeps_in_interval += 1
        self._update_state(resolved)
        return True

    def complete_policy_interval(self) -> int:
        """在 ``_get_dones`` 调用：校验本策略步恰好记录 ``decimation`` 个子步。"""

        if not self.enabled:
            raise PhysicalHoldError("物理子步守卫未启用，无法确认 120 Hz 连续采样")
        recorded = int(self._substeps_in_interval)
        if recorded != self.decimation:
            raise PhysicalHoldError(
                "一个策略步内的真实物理子步数不等于 decimation："
                f"记录 {recorded}，期望 {self.decimation}"
                "（decimation 折叠分支或漏采样都不满足逐物理子步语义）"
            )
        self._substeps_in_interval = 0
        return recorded

    # -------------------------------------------------------------- 核心更新
    @torch.inference_mode()
    def _update_state(self, state: PhysicalSubstepState) -> None:
        hard_now = compute_hard_faults(state)
        for name in HARD_FAULT_NAMES:
            self.hard_faults[name] |= hard_now[name]
        sticky_any = self.hard_fault_any

        released = state.released.to(dtype=torch.bool)
        required_contact = state.required_contact.to(dtype=torch.bool)
        speed_ok = state.relative_speed_m_s <= self.hold_max_relative_speed_m_s
        fresh = self.streak == 0
        # anchor 固定在该物理 streak 的起点：只在 streak 归零后的第一个子步写入。
        self.anchor_hand[fresh] = state.fruit_in_hand_m[fresh]
        drift = torch.linalg.vector_norm(state.fruit_in_hand_m - self.anchor_hand, dim=1)
        drift = torch.where(fresh, torch.zeros_like(drift), drift)
        hold_now = (
            released
            & required_contact
            & speed_ok
            & ~sticky_any
            & (fresh | (drift <= self.hold_max_drift_m))
        )
        self.streak = torch.where(hold_now, self.streak + 1, torch.zeros_like(self.streak))
        self.peak_streak = torch.maximum(self.peak_streak, self.streak)
        self.anchor_valid = hold_now
        self.last_hold_ok = hold_now
        self.last_drift_m = torch.where(
            hold_now, drift, torch.full_like(drift, float("inf"))
        )

    # ---------------------------------------------------------------- 只读
    def snapshot(self) -> dict[str, Any]:
        """逐 env 只读快照（供记录/测试；不修改状态）。"""

        return {
            "streak": self.streak.clone(),
            "peak_streak": self.peak_streak.clone(),
            "hard_fault_any": self.hard_fault_any,
            "anchor_hand": self.anchor_hand.clone(),
            "anchor_valid": self.anchor_valid.clone(),
        }


def required_physical_substeps(required_unsupported_hold_steps: int, decimation: int) -> int:
    """把阶段要求的策略步阈值换算成真实物理子步阈值（1 s = 20*6 = 120）。"""

    steps = int(required_unsupported_hold_steps)
    dec = int(decimation)
    if steps <= 0 or dec <= 0:
        raise PhysicalHoldError(
            "策略步阈值与 decimation 必须为正："
            f"{required_unsupported_hold_steps!r}, {decimation!r}"
        )
    return steps * dec


def hard_fault_mask_from_mapping(faults: Mapping[str, Any]) -> torch.Tensor:
    """把具名硬故障映射合并为逐 env 掩码（顺序无关，缺失名字 fail-closed）。"""

    missing = [name for name in HARD_FAULT_NAMES if name not in faults]
    if missing:
        raise PhysicalHoldError(f"硬故障映射缺少名字：{missing}")
    merged: torch.Tensor | None = None
    for name in HARD_FAULT_NAMES:
        value = _require_tensor(name, faults[name])
        merged = value.to(dtype=torch.bool) if merged is None else merged | value
    assert merged is not None
    return merged


def semantics_summary() -> dict[str, Any]:
    """稳定的语义描述（写入任务身份/逐回合记录，便于外部核验）。"""

    return {
        "guard_frequency": "per_physics_substep",
        "hold_streak_unit": "physics_substeps",
        "hard_fault_persistence": "sticky_until_env_reset",
        "hard_fault_names": list(HARD_FAULT_NAMES),
        "anchor_reference": "hand_frame_fruit_position_at_physical_streak_start",
        "missing_data_policy": "fail_closed",
    }


__all__ = [
    "HARD_FAULT_NAMES",
    "PhysicalHoldError",
    "PhysicalHoldGuard",
    "PhysicalSubstepState",
    "compute_hard_faults",
    "hard_fault_mask_from_mapping",
    "required_physical_substeps",
    "semantics_summary",
]
