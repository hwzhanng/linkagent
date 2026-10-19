"""firm_grasp_v1.7.0 的支撑渐退曲线：只缩放真实外力的实际应用值。

本模块是纯函数实现，不构造仿真环境，可在 CPU 上独立复核。它定义
``support_ramp_policy_steps`` 的唯一语义：

* 默认在释放步之前的最后 20 个策略步（20 Hz ⇒ 1 s）内把声明支撑扳手
  从 1.0 线性缩到 0.0，取 0 时保留原“到点突撤”对照；
* 释放步及之后严格等于 0，斜坡窗内严格大于 0，所以渐退期间的任何
  物理子步都不可能被三指守卫计成“零支撑”成功；
* 缩放发生在果梗柔顺扳手真实写入 PhysX 之前，作用在将被施加的
  力/力矩张量上，日志缓存 ``_last_stem_force_w`` / ``_last_stem_torque_w``
  写的正是这个已缩放值；不重写元数据、不写果实/关节/根状态、不改
  重力、摩擦、PD、限位、ROI 与物理故障。
* 这是准备条件变化带来的条件收益，不是 PPO 学习收益。
"""

from __future__ import annotations

import math
from typing import Any, Final


DEFAULT_SUPPORT_RAMP_POLICY_STEPS: Final = 20
RAMP_PROFILE_ID: Final = "firm_grasp_support_release_ramp_v1"
SCOPE_NOTE: Final = (
    "准备条件收益（第5–6秒线性退出支撑），不是PPO学习收益；"
    "相同条件下的对照只能声明该条件变化，不能声明策略学习提升"
)


def _as_step_index(value: Any, name: str) -> int:
    """策略步索引必须是整数；布尔与浮点一律拒绝。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数策略步，得到 {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} 不能为负，得到 {value}")
    return int(value)


def validate_support_ramp_policy_steps(value: Any, *, release_step: Any) -> int:
    """校验渐退步数：非负整数，且不得超过释放步本身。"""

    ramp = _as_step_index(value, "support_ramp_policy_steps")
    release = _as_step_index(release_step, "support_release_step")
    if release <= 0:
        raise ValueError(f"support_release_step 必须为正数，得到 {release}")
    if ramp > release:
        raise ValueError(
            f"support_ramp_policy_steps ({ramp}) 不得超过释放步 ({release})，"
            "否则斜坡会落到准备阶段之前"
        )
    return ramp


def release_scale_for_step(
    step: Any, *, release_step: Any, ramp_policy_steps: Any
) -> float:
    """单个策略步的标量参考实现（测试与文档用，非仿真路径）。"""

    ramp = validate_support_ramp_policy_steps(
        ramp_policy_steps, release_step=release_step
    )
    index = _as_step_index(step, "policy_step")
    release = int(release_step)
    if index >= release:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, (release - index) / float(ramp))


def support_release_scale(
    policy_step: Any, *, release_step: Any, ramp_policy_steps: Any
) -> Any:
    """逐环境释放缩放因子 [N]（float32），与标量参考逐点一致。"""

    import torch

    ramp = validate_support_ramp_policy_steps(
        ramp_policy_steps, release_step=release_step
    )
    release = int(release_step)
    steps = torch.as_tensor(policy_step)
    if steps.ndim != 1:
        raise ValueError(f"policy_step 必须是 [N] 形状，得到 {tuple(steps.shape)}")
    if steps.is_floating_point() or steps.dtype == torch.bool:
        raise TypeError(f"policy_step 必须是整数策略步，得到 {steps.dtype}")
    if bool((steps < 0).any()):
        raise ValueError("policy_step 含负步索引")
    steps64 = steps.to(dtype=torch.float64)
    if ramp <= 0:
        scale = torch.ones_like(steps64)
    else:
        scale = ((float(release) - steps64) / float(ramp)).clamp(0.0, 1.0)
    scale = torch.where(steps >= release, torch.zeros_like(scale), scale)
    return scale.to(dtype=torch.float32)


def scale_support_wrench(force: Any, torque: Any, scale: Any) -> tuple[Any, Any]:
    """在真实写入之前缩放支撑力/力矩，并做 fail-closed 形状与范围校验。"""

    import torch

    for name, value in (("force", force), ("torque", torque), ("scale", scale)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} 必须使用浮点 dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} 含 NaN 或无穷值")
    if force.ndim != 2 or force.shape[1] != 3:
        raise ValueError(f"force 必须是 [N, 3] 形状，得到 {tuple(force.shape)}")
    if tuple(torque.shape) != tuple(force.shape):
        raise ValueError("torque 必须与 force 同形状")
    if scale.ndim != 1 or scale.shape[0] != force.shape[0]:
        raise ValueError(f"scale 必须是 [N] 形状，得到 {tuple(scale.shape)}")
    if float(scale.min()) < 0.0 or float(scale.max()) > 1.0:
        raise ValueError("scale 必须位于 [0, 1]")
    factor = scale.to(dtype=force.dtype).unsqueeze(1)
    return force * factor, torque.to(dtype=force.dtype) * factor


def support_ramp_identity(
    *, release_step: Any, ramp_policy_steps: Any, policy_dt_s: Any
) -> dict[str, Any]:
    """进入 checkpoint 身份与源码哈希链的显式斜坡记录。"""

    ramp = validate_support_ramp_policy_steps(
        ramp_policy_steps, release_step=release_step
    )
    release = int(release_step)
    dt_s = float(policy_dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"policy_dt_s 必须是正的有限值，得到 {policy_dt_s}")
    start_step = max(0, release - ramp)
    return {
        "profile_id": RAMP_PROFILE_ID,
        "support_ramp_policy_steps": ramp,
        "sharp_release_control": ramp == 0,
        "ramp_semantics": (
            "linear_scale_1_to_0_over_last_ramp_steps_before_release_step"
        ),
        "release_step": release,
        "release_time_s": release * dt_s,
        "ramp_start_step": start_step,
        "ramp_start_time_s": start_step * dt_s,
        "ramp_window_s": ramp * dt_s,
        "scale_at_release_step": 0.0,
        "scale_application": (
            "support_wrench_scaled_before_single_physx_permanent_wrench_write;"
            "logged_force_and_torque_buffers_equal_applied_values"
        ),
        "zero_support_semantics": (
            "exactly_zero_at_and_after_release_step;"
            "strictly_positive_inside_ramp_window,"
            "so_ramp_substeps_cannot_qualify_G1"
        ),
        "in_episode_state_writes": False,
        "gravity_friction_pd_limits_rois_and_physical_faults_unchanged": True,
        "faults_still_sticky": True,
        "legacy_prefix_preserved": False,
        "legacy_prefix_numeric_layout_preserved": True,
        "legacy_prefix_full_actor_action_semantics_preserved": False,
        "legacy_policy_prefix_dim": 161,
        "legacy_critic_prefix_dim": 222,
        "policy_dim": 162,
        "critic_dim": 223,
        "appended_clock_dim": 1,
        "appended_clock_position": {"policy": 161, "critic": 222},
        "actor_critic_action_semantics_unchanged": False,
        "action_14d_semantics_unchanged": False,
        "action_semantics_changed_from_legacy": [8, 9],
        "action14_dimensions_and_observation_prefix_formulas_preserved": True,
        "scope_note": SCOPE_NOTE,
    }
