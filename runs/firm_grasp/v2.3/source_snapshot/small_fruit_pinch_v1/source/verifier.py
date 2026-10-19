"""不读取 reward 的三指捏取独立 Verifier。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch

from .contract import (
    DISTURBANCE_STEP_IN_HOLD,
    FORCE_BANDS,
    LONG_HOLD_STEPS,
    POST_DISTURBANCE_HOLD_STEPS,
    REQUIRED_DIGIT_INDICES,
    STRICT_HOLD_STEPS,
)


class VerificationState(TypedDict):
    tim_contact: torch.Tensor
    force_in_band: torch.Tensor
    hard_overload: torch.Tensor
    candidate: torch.Tensor
    hold_age: torch.Tensor
    contact_ratio: torch.Tensor
    band_ratio: torch.Tensor
    palm_ratio: torch.Tensor
    disturbance_due: torch.Tensor
    success: torch.Tensor
    final_success: torch.Tensor
    success_event: torch.Tensor
    post_hold_age: torch.Tensor
    retention_lost: torch.Tensor
    support_hold_age: torch.Tensor
    contact_acquired: torch.Tensor
    support_retained: torch.Tensor


@dataclass(slots=True)
class VerifierShapeError(ValueError):
    name: str
    expected: tuple[int, ...]
    actual: tuple[int, ...]

    def __str__(self) -> str:
        return f"{self.name} 维度必须为 {self.expected}，实际为 {self.actual}"


@dataclass(frozen=True)
class VerifierConfig:
    contact_enter_n: float = 0.08
    contact_exit_n: float = 0.04
    contact_enter_frames: int = 2
    contact_exit_frames: int = 3
    hold_steps: int = STRICT_HOLD_STEPS
    post_hold_steps: int = POST_DISTURBANCE_HOLD_STEPS
    required_contact_ratio: float = 0.85
    required_band_ratio: float = 0.75
    max_center_drift_m: float = 0.012
    max_relative_speed_m_s: float = 0.035
    max_palm_contact_ratio: float = 0.10


class TimStrictVerifier:
    """逐帧更新接触获取及脱支撑保持状态，计数器按环境独立变更。

    success 锁存 80 步合格接触获取，供控制器切换阶段。软带、支撑或
    稳定性失效会清空最终保持窗及扰动证明；恢复必须重走 120 步窗口，
    并在该窗内重新确认扰动。retention_lost 保留至完整恢复或 reset。
    硬过载、非有限观测、禁触、投影、自碰撞锁存 invalid，须 reset。
    disturbance_applied 须由实际施扰后的物理步确认；due 只是请求。
    topology_valid/physical_contact_valid 由调用方提供真实接触证明。
    final_success 只表示脱支撑保持，不代表自然脱果或完整采摘成功。
    """

    def __init__(self, num_envs: int, *, device: torch.device | str, cfg: VerifierConfig = VerifierConfig()) -> None:
        self.cfg = cfg
        self.contact = torch.zeros((num_envs, 5), dtype=torch.bool, device=device)
        self.enter_age = torch.zeros((num_envs, 5), dtype=torch.long, device=device)
        self.exit_age = torch.zeros((num_envs, 5), dtype=torch.long, device=device)
        self.hold_age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.post_hold_age = torch.zeros_like(self.hold_age)
        self.support_hold_age = torch.zeros_like(self.hold_age)
        self.contact_good = torch.zeros_like(self.hold_age)
        self.band_good = torch.zeros_like(self.hold_age)
        self.palm_contact = torch.zeros_like(self.hold_age)
        self.invalid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros_like(self.invalid)
        self.final_success = torch.zeros_like(self.success)
        self.success_event = torch.zeros_like(self.success)
        self.retention_lost = torch.zeros_like(self.success)
        self.disturbance_due = torch.zeros_like(self.success)
        self._disturbance_seen = torch.zeros_like(self.success)
        self._previous_disturbance = torch.zeros_like(self.success)

    @property
    def contact_acquired(self) -> torch.Tensor:
        return self.success

    @property
    def support_retained(self) -> torch.Tensor:
        return self.final_success

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        targets = slice(None) if env_ids is None else env_ids
        for value in (
            self.contact, self.enter_age, self.exit_age, self.hold_age,
            self.post_hold_age, self.support_hold_age, self.contact_good,
            self.band_good, self.palm_contact, self.invalid, self.success,
            self.final_success, self.success_event, self.retention_lost,
            self.disturbance_due, self._disturbance_seen, self._previous_disturbance,
        ):
            value[targets] = 0

    def update(
        self,
        *,
        force_n: torch.Tensor,
        palm_contact: torch.Tensor,
        center_drift_m: torch.Tensor,
        relative_speed_m_s: torch.Tensor,
        safety_projection: torch.Tensor,
        self_collision: torch.Tensor,
        support_removed: torch.Tensor | None = None,
        disturbance_applied: torch.Tensor | None = None,
        topology_valid: torch.Tensor | None = None,
        physical_contact_valid: torch.Tensor | None = None,
        forbidden_contact: torch.Tensor | None = None,
    ) -> VerificationState:
        # 保留调用方的独立 kwargs；先验证全部形状，避免广播污染并行环境。
        if force_n.shape != self.contact.shape:
            raise VerifierShapeError("force_n", tuple(self.contact.shape), tuple(force_n.shape))
        expected = tuple(self.success.shape)
        for name, value in (
            ("palm_contact", palm_contact), ("center_drift_m", center_drift_m),
            ("relative_speed_m_s", relative_speed_m_s), ("safety_projection", safety_projection),
            ("self_collision", self_collision), ("support_removed", support_removed),
            ("disturbance_applied", disturbance_applied), ("topology_valid", topology_valid),
            ("physical_contact_valid", physical_contact_valid), ("forbidden_contact", forbidden_contact),
        ):
            if value is not None and value.shape != expected:
                raise VerifierShapeError(name, expected, tuple(value.shape))
        absent = torch.zeros_like(self.success)
        support = absent if support_removed is None else support_removed == 1
        applied = absent if disturbance_applied is None else disturbance_applied == 1
        topology = absent if topology_valid is None else topology_valid == 1
        physical = absent if physical_contact_valid is None else physical_contact_valid == 1
        forbidden = absent if forbidden_contact is None else forbidden_contact.bool()
        finite_force = torch.isfinite(force_n)
        finite = (
            finite_force.all(dim=-1) & torch.isfinite(palm_contact)
            & torch.isfinite(center_drift_m) & torch.isfinite(relative_speed_m_s)
        )
        enter = finite_force & (force_n >= self.cfg.contact_enter_n)
        exit_ = ~finite_force | (force_n < self.cfg.contact_exit_n)
        self.enter_age = torch.where(enter, self.enter_age + 1, 0)
        self.exit_age = torch.where(exit_, self.exit_age + 1, 0)
        self.contact |= self.enter_age >= self.cfg.contact_enter_frames
        self.contact &= finite_force & ~(self.exit_age >= self.cfg.contact_exit_frames)
        required = self.contact[:, REQUIRED_DIGIT_INDICES].all(dim=-1)
        lows = force_n.new_tensor([FORCE_BANDS[index].soft_low_n for index in REQUIRED_DIGIT_INDICES])
        highs = force_n.new_tensor([FORCE_BANDS[index].soft_high_n for index in REQUIRED_DIGIT_INDICES])
        hard = force_n.new_tensor([band.hard_high_n for band in FORCE_BANDS])
        in_band = ((force_n[:, REQUIRED_DIGIT_INDICES] >= lows) & (force_n[:, REQUIRED_DIGIT_INDICES] <= highs)).all(dim=-1)
        hard_overload = (force_n > hard).any(dim=-1)
        dynamically_stable = (
            (center_drift_m <= self.cfg.max_center_drift_m)
            & (relative_speed_m_s <= self.cfg.max_relative_speed_m_s)
        )
        self.invalid |= ~finite | hard_overload | forbidden | safety_projection.bool() | self_collision.bool()
        still_candidate = (
            required & in_band & dynamically_stable & finite & topology & physical
            & ~palm_contact.bool() & ~self.invalid
        )
        was_success = self.success.clone()
        self.hold_age = torch.where(still_candidate, self.hold_age + 1, 0)
        self.contact_good = torch.where(still_candidate, self.contact_good + 1, 0)
        self.band_good = torch.where(still_candidate, self.band_good + 1, 0)
        self.palm_contact = torch.where(still_candidate & palm_contact.bool(), self.palm_contact + 1, 0)
        contact_ratio = self.contact_good.float() / self.hold_age.clamp_min(1).float()
        band_ratio = self.band_good.float() / self.hold_age.clamp_min(1).float()
        palm_ratio = self.palm_contact.float() / self.hold_age.clamp_min(1).float()
        self.success |= (
            (self.hold_age >= max(STRICT_HOLD_STEPS, self.cfg.hold_steps))
            & (contact_ratio >= self.cfg.required_contact_ratio)
            & (band_ratio >= self.cfg.required_band_ratio)
            & (palm_ratio < self.cfg.max_palm_contact_ratio) & ~self.invalid
        )
        support_candidate = still_candidate & support
        self.retention_lost |= was_success & (~still_candidate | ((self.support_hold_age > 0) & ~support))
        # 窗口开始时为 True 或上一窗口遗留的 True 均不是本窗内的新施扰。
        confirmed = support_candidate & (self.support_hold_age >= DISTURBANCE_STEP_IN_HOLD) & applied & ~self._previous_disturbance
        self._previous_disturbance = applied.clone()
        self.support_hold_age = torch.where(support_candidate, self.support_hold_age + 1, 0)
        self._disturbance_seen = support_candidate & (self._disturbance_seen | confirmed)
        # 确认当帧不计为扰动“后”的恢复步；重复脉冲重新开始恢复计时。
        self.post_hold_age = torch.where(
            self._disturbance_seen & ~confirmed, self.post_hold_age + 1, 0,
        )
        self.disturbance_due = (self.support_hold_age >= DISTURBANCE_STEP_IN_HOLD) & ~self._disturbance_seen
        previous_final = self.final_success.clone()
        self.final_success = (
            self.success & (self.support_hold_age >= LONG_HOLD_STEPS)
            & self._disturbance_seen
            & (self.post_hold_age >= max(POST_DISTURBANCE_HOLD_STEPS, self.cfg.post_hold_steps))
        )
        self.retention_lost &= ~self.final_success
        self.success_event = self.final_success & ~previous_final
        return {
            "tim_contact": required,
            "force_in_band": in_band,
            "hard_overload": hard_overload,
            "candidate": still_candidate,
            "hold_age": self.hold_age,
            "contact_ratio": contact_ratio,
            "band_ratio": band_ratio,
            "palm_ratio": palm_ratio,
            "disturbance_due": self.disturbance_due,
            "success": self.success,
            "final_success": self.final_success,
            "success_event": self.success_event,
            "post_hold_age": self.post_hold_age,
            "retention_lost": self.retention_lost,
            "support_hold_age": self.support_hold_age,
            "contact_acquired": self.contact_acquired,
            "support_retained": self.support_retained,
        }
