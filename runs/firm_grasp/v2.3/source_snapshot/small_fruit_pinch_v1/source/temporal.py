"""策略动作到 PhysX 关节目标前的状态化平滑器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemporalLimits:
    """归一化动作域中的阶段性限幅。"""

    approach_speed: float = 0.45
    hold_speed: float = 0.15
    approach_acceleration: float = 4.0
    hold_acceleration: float = 1.5
    natural_frequency_hz: float = 5.0
    reversal_deadband: float = 0.008


class CriticalDampedActionFilter:
    """二阶临界阻尼、速度/加速度约束与反向死区。

    该滤波器保留每个 environment 的状态。它不把动作固定为零：在
    HOLD 中仍允许低速响应，从而能通过扰动恢复而不是静止伪装持稳。
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        dt_s: float,
        device: torch.device | str,
        limits: TemporalLimits = TemporalLimits(),
    ) -> None:
        if num_envs <= 0 or action_dim <= 0 or dt_s <= 0.0:
            raise ValueError("num_envs、action_dim 与 dt_s 必须为正数")
        self.dt_s = float(dt_s)
        self.limits = limits
        self.position = torch.zeros((num_envs, action_dim), device=device)
        self.velocity = torch.zeros_like(self.position)
        self.acceleration = torch.zeros_like(self.position)
        self.previous_acceleration = torch.zeros_like(self.position)
        self.last_stats: dict[str, torch.Tensor] = {}

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.position.zero_()
            self.velocity.zero_()
            self.acceleration.zero_()
            self.previous_acceleration.zero_()
            self.last_stats = {}
            return
        self.position[env_ids] = 0.0
        self.velocity[env_ids] = 0.0
        self.acceleration[env_ids] = 0.0
        self.previous_acceleration[env_ids] = 0.0

    def step(
        self,
        raw_action: torch.Tensor,
        *,
        hold_phase: torch.Tensor,
        speed_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if raw_action.shape != self.position.shape:
            raise ValueError(
                f"raw_action shape {tuple(raw_action.shape)} != {tuple(self.position.shape)}"
            )
        if hold_phase.shape != (raw_action.shape[0],):
            raise ValueError("hold_phase 必须为每个 environment 一个布尔值")
        raw = torch.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        hold = hold_phase.to(dtype=torch.bool, device=raw.device).unsqueeze(-1)
        speed_limit = torch.where(
            hold,
            torch.full_like(raw, self.limits.hold_speed),
            torch.full_like(raw, self.limits.approach_speed),
        )
        if speed_scale is not None:
            if speed_scale.shape != raw.shape:
                raise ValueError("speed_scale 必须与 raw_action 同形状")
            if not bool(torch.isfinite(speed_scale).all()):
                raise ValueError("speed_scale 包含 NaN 或 Inf")
            speed_limit = speed_limit * speed_scale.to(device=raw.device, dtype=raw.dtype).clamp(0.05, 1.0)
        accel_limit = torch.where(
            hold,
            torch.full_like(raw, self.limits.hold_acceleration),
            torch.full_like(raw, self.limits.approach_acceleration),
        )
        omega = 2.0 * torch.pi * self.limits.natural_frequency_hz
        desired_accel = omega * omega * (raw - self.position) - 2.0 * omega * self.velocity
        acceleration = desired_accel.clamp(-accel_limit, accel_limit)
        candidate_velocity = (self.velocity + acceleration * self.dt_s).clamp(-speed_limit, speed_limit)
        reversal = (
            (candidate_velocity * self.velocity < 0.0)
            & (candidate_velocity.abs() < self.limits.reversal_deadband)
        )
        candidate_velocity = torch.where(reversal, torch.zeros_like(candidate_velocity), candidate_velocity)
        previous_position = self.position.clone()
        next_position = (previous_position + candidate_velocity * self.dt_s).clamp(-1.0, 1.0)
        jerk = (acceleration - self.previous_acceleration) / self.dt_s
        self.previous_acceleration.copy_(acceleration)
        self.position.copy_(next_position)
        self.velocity.copy_(candidate_velocity)
        self.acceleration.copy_(acceleration)
        self.last_stats = {
            # 执行动作相对于上一策略步的变化量；这才是 PPO 动作平滑
            # 正则应约束的量。tracking_error 单列，避免把跟踪误差误记为
            # 动作突变而污染训练诊断。
            "action_delta": (next_position - previous_position).abs(),
            "tracking_error": (next_position - raw).abs(),
            "velocity": candidate_velocity.abs(),
            "acceleration": acceleration.abs(),
            "jerk": jerk.abs(),
            "reversal": reversal,
        }
        return next_position
