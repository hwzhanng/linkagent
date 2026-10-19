"""G2 抗扰抓握的逐物理子步连续窗守卫。"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


class G2HoldGuard:
    """维护六个固定三指组合各自的 G2 连续保持窗。

    ``qualified`` 由调用方按真实 PAD 接触和几何条件计算。本类只负责
    释放后连续性、漂移/速度/姿态约束、扰动恢复和逐子步时钟。
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device | str,
        initial_counter: int,
        physics_dt_s: float = 1 / 120,
        decimation: int = 6,
        max_speed: float = 0.05,
        max_drift: float = 0.01,
        max_angle_deg: float = 10.0,
        max_angular_speed: float = 0.5,
        recovery_speed: float = 0.02,
        required_substeps: int = 360,
        early_start_max_substep: int = 60,
        recovery_start_substep: int = 252,
        pulse_start_substep: int = 180,
        pulse_end_substep: int = 191,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs 必须为正数")
        if not math.isfinite(float(physics_dt_s)) or physics_dt_s <= 0:
            raise ValueError("physics_dt_s 必须为正有限数")
        if decimation <= 0 or required_substeps <= 0:
            raise ValueError("decimation 和 required_substeps 必须为正数")
        if pulse_end_substep < pulse_start_substep:
            raise ValueError("扰动脉冲子步范围无效")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.last_counter = int(initial_counter)
        self.dt = float(physics_dt_s)
        self.decimation = int(decimation)
        self.interval_count = 0
        self.max_speed = float(max_speed)
        self.max_drift = float(max_drift)
        self.max_angle_rad = math.radians(float(max_angle_deg))
        self.max_angular_speed = float(max_angular_speed)
        self.recovery_speed = float(recovery_speed)
        self.required_substeps = int(required_substeps)
        self.early_start_max_substep = int(early_start_max_substep)
        self.recovery_start_substep = int(recovery_start_substep)
        self.pulse_start_substep = int(pulse_start_substep)
        self.pulse_end_substep = int(pulse_end_substep)

        shape = (self.num_envs, 6)
        self.streak = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.peak = torch.zeros_like(self.streak)
        self.anchor = torch.zeros((*shape, 3), dtype=torch.float32, device=self.device)
        self.quat_anchor = torch.zeros((*shape, 4), dtype=torch.float64, device=self.device)
        self.last_ok = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.drift = torch.full(shape, float("inf"), dtype=torch.float32, device=self.device)
        self.angle = torch.full(shape, float("inf"), dtype=torch.float32, device=self.device)
        self.window_start_substep = torch.full(
            shape, -1, dtype=torch.long, device=self.device
        )
        self.g2_success = torch.zeros(shape, dtype=torch.bool, device=self.device)

    def _check_tensor(
        self,
        name: str,
        value: Any,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        floating: bool = False,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} 必须具有 {shape} 形状")
        if value.device != self.device:
            raise ValueError(f"{name} 必须位于 {self.device}")
        if dtype is not None and value.dtype is not dtype:
            raise TypeError(f"{name} 必须使用 {dtype} dtype")
        if floating and not value.is_floating_point():
            raise TypeError(f"{name} 必须使用浮点 dtype")
        return value

    @torch.inference_mode()
    def reset(self, ids: Tensor | list[int] | tuple[int, ...]) -> None:
        """清空指定环境的窗口状态，不改动全局物理子步时钟。"""
        self.streak[ids] = 0
        self.peak[ids] = 0
        self.anchor[ids] = 0
        self.quat_anchor[ids] = 0
        self.last_ok[ids] = False
        self.drift[ids] = float("inf")
        self.angle[ids] = float("inf")
        self.window_start_substep[ids] = -1
        self.g2_success[ids] = False

    @torch.inference_mode()
    def observe(
        self,
        *,
        state: Any,
        qualified: Tensor,
        hard_fault_any: Tensor,
        support_zero: Tensor,
        counter: int,
        dt: float,
        relative_quat_xyzw: Tensor,
        relative_angular_speed: Tensor,
        post_release_substep: Tensor,
        strict_early_start: bool,
        disturbed: bool,
    ) -> None:
        if not math.isfinite(float(dt)) or abs(float(dt) - self.dt) > 1e-9:
            raise RuntimeError("G2 守卫物理 dt 不符")
        if int(counter) != self.last_counter + 1:
            raise RuntimeError("G2 守卫物理子步漏采或重复")
        if not isinstance(strict_early_start, bool) or not isinstance(disturbed, bool):
            raise TypeError("strict_early_start 和 disturbed 必须为 bool")

        qualified = self._check_tensor(
            "qualified", qualified, (self.num_envs, 6), dtype=torch.bool
        )
        hard_fault_any = self._check_tensor(
            "hard_fault_any", hard_fault_any, (self.num_envs,), dtype=torch.bool
        )
        support_zero = self._check_tensor(
            "support_zero", support_zero, (self.num_envs,), dtype=torch.bool
        )
        released = self._check_tensor(
            "state.released", state.released, (self.num_envs,), dtype=torch.bool
        )
        position = self._check_tensor(
            "state.fruit_in_hand_m", state.fruit_in_hand_m,
            (self.num_envs, 3), floating=True
        )
        speed = self._check_tensor(
            "state.relative_speed_m_s", state.relative_speed_m_s,
            (self.num_envs,), floating=True
        )
        relative_quat_xyzw = self._check_tensor(
            "relative_quat_xyzw", relative_quat_xyzw,
            (self.num_envs, 4), floating=True
        )
        angular_speed = self._check_tensor(
            "relative_angular_speed", relative_angular_speed,
            (self.num_envs,), floating=True
        )
        post_release_substep = self._check_tensor(
            "post_release_substep", post_release_substep, (self.num_envs,)
        )
        if post_release_substep.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise TypeError("post_release_substep 必须为整数张量")

        self.last_counter = int(counter)
        self.interval_count += 1

        fresh = self.streak == 0
        position_expanded = position[:, None, :].expand_as(self.anchor)
        self.anchor.copy_(torch.where(fresh[..., None], position_expanded, self.anchor))
        drift = torch.linalg.vector_norm(position_expanded - self.anchor, dim=-1)

        q = relative_quat_xyzw.to(dtype=torch.float64)
        q_norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        q_valid = torch.isfinite(q).all(dim=-1) & torch.isfinite(q_norm[:, 0]) & (q_norm[:, 0] > 1e-12)
        identity = torch.zeros_like(q)
        identity[:, 3] = 1.0
        q_unit = torch.where(q_valid[:, None], q / q_norm.clamp_min(1e-12), identity)
        q_expanded = q_unit[:, None, :].expand_as(self.quat_anchor)
        self.quat_anchor.copy_(torch.where(fresh[..., None], q_expanded, self.quat_anchor))
        dot = (q_expanded * self.quat_anchor).sum(dim=-1).abs().clamp_(0.0, 1.0)
        angle = (2.0 * torch.acos(dot)).to(dtype=torch.float32)
        angle = torch.where(q_valid[:, None], angle, torch.full_like(angle, float("inf")))

        substep = post_release_substep.to(dtype=torch.long)
        start_substep = torch.where(
            fresh, substep[:, None].expand_as(self.window_start_substep),
            self.window_start_substep
        )
        start_allowed = (
            (not strict_early_start)
            | (~fresh)
            | (substep[:, None] <= self.early_start_max_substep)
        )
        pulse_covered = (
            (not disturbed)
            | (
                (start_substep <= self.pulse_start_substep)
                & (substep[:, None] >= self.pulse_end_substep)
            )
        )

        finite_linear = torch.isfinite(position).all(dim=-1) & torch.isfinite(speed)
        finite_angular = torch.isfinite(angular_speed) & (angular_speed >= 0)
        recovered = (
            (not disturbed)
            | (substep[:, None] < self.recovery_start_substep)
            | (speed[:, None] <= self.recovery_speed)
        )
        ok = (
            qualified
            & released[:, None]
            & support_zero[:, None]
            & ~hard_fault_any[:, None]
            & finite_linear[:, None]
            & finite_angular[:, None]
            & (speed[:, None] >= 0)
            & (speed[:, None] <= self.max_speed)
            & recovered
            & (drift <= self.max_drift)
            & (angle <= self.max_angle_rad + 1e-7)
            & (angular_speed[:, None] <= self.max_angular_speed)
            & start_allowed
        )

        next_streak = torch.where(ok, self.streak + 1, torch.zeros_like(self.streak))
        self.streak.copy_(next_streak)
        self.peak.copy_(torch.maximum(self.peak, self.streak))
        self.last_ok.copy_(ok)
        self.drift.copy_(torch.where(ok, drift, torch.full_like(drift, float("inf"))))
        self.angle.copy_(angle)
        self.window_start_substep.copy_(
            torch.where(ok, start_substep, torch.full_like(start_substep, -1))
        )
        self.g2_success.copy_(
            (self.streak >= self.required_substeps) & pulse_covered & ok
        )

    def complete_policy_interval(self) -> None:
        if self.interval_count != self.decimation:
            raise RuntimeError(
                f"G2 守卫采样数错误：{self.interval_count} != {self.decimation}"
            )
        self.interval_count = 0
