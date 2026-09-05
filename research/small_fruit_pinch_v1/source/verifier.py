"""不读取 reward 的三指捏取独立 Verifier。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contract import (
    DISTURBANCE_STEP_IN_HOLD,
    FORCE_BANDS,
    POST_DISTURBANCE_HOLD_STEPS,
    REQUIRED_DIGIT_INDICES,
    STRICT_HOLD_STEPS,
)


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
    """拇指—食指—中指真实接触、低力保持和无掌托捷径验证。"""

    def __init__(self, num_envs: int, *, device: torch.device | str, cfg: VerifierConfig = VerifierConfig()) -> None:
        self.cfg = cfg
        self.contact = torch.zeros((num_envs, 5), dtype=torch.bool, device=device)
        self.enter_age = torch.zeros((num_envs, 5), dtype=torch.long, device=device)
        self.exit_age = torch.zeros((num_envs, 5), dtype=torch.long, device=device)
        self.hold_age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.post_hold_age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.contact_good = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.band_good = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.palm_contact = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.invalid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.final_success = torch.zeros_like(self.success)
        self.success_event = torch.zeros_like(self.success)
        self.retention_lost = torch.zeros_like(self.success)
        self.disturbance_due = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        targets = slice(None) if env_ids is None else env_ids
        self.contact[targets] = False
        self.enter_age[targets] = 0
        self.exit_age[targets] = 0
        self.hold_age[targets] = 0
        self.post_hold_age[targets] = 0
        self.contact_good[targets] = 0
        self.band_good[targets] = 0
        self.palm_contact[targets] = 0
        self.invalid[targets] = False
        self.success[targets] = False
        self.final_success[targets] = False
        self.success_event[targets] = False
        self.retention_lost[targets] = False
        self.disturbance_due[targets] = False

    def update(
        self,
        *,
        force_n: torch.Tensor,
        palm_contact: torch.Tensor,
        center_drift_m: torch.Tensor,
        relative_speed_m_s: torch.Tensor,
        safety_projection: torch.Tensor,
        self_collision: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if force_n.shape != self.contact.shape:
            raise ValueError("force_n 必须为 (num_envs, 5)")
        n = force_n.shape[0]
        expected = (n,)
        if any(value.shape != expected for value in (palm_contact, center_drift_m, relative_speed_m_s, safety_projection, self_collision)):
            raise ValueError("Verifier 标量输入必须为 (num_envs,)")
        enter = force_n >= self.cfg.contact_enter_n
        exit_ = force_n < self.cfg.contact_exit_n
        self.enter_age = torch.where(enter, self.enter_age + 1, torch.zeros_like(self.enter_age))
        self.exit_age = torch.where(exit_, self.exit_age + 1, torch.zeros_like(self.exit_age))
        self.contact |= self.enter_age >= self.cfg.contact_enter_frames
        self.contact &= ~(self.exit_age >= self.cfg.contact_exit_frames)

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
        still_candidate = required & dynamically_stable & ~hard_overload & ~safety_projection.bool() & ~self_collision.bool()
        was_success = self.success.clone()
        new_hold = still_candidate
        self.hold_age = torch.where(new_hold, self.hold_age + 1, torch.zeros_like(self.hold_age))
        self.contact_good = torch.where(new_hold & required, self.contact_good + 1, torch.zeros_like(self.contact_good))
        self.band_good = torch.where(new_hold & in_band, self.band_good + 1, torch.zeros_like(self.band_good))
        self.palm_contact = torch.where(new_hold & palm_contact.bool(), self.palm_contact + 1, self.palm_contact)
        self.invalid |= hard_overload | safety_projection.bool() | self_collision.bool()
        # 严格持稳窗达到后仍要经历一段连续保持；一旦接触/动态稳定性丢失，
        # 本回合不能把先前的成功 latch 当作最终成功，避免提前终止和虚高
        # 成功率。接触迟滞只影响 required 的更新，不绕过此连续保持门。
        self.retention_lost |= was_success & ~still_candidate
        self.disturbance_due = (self.hold_age == DISTURBANCE_STEP_IN_HOLD) & ~self.success
        completed = self.hold_age >= self.cfg.hold_steps
        contact_ratio = self.contact_good.float() / self.hold_age.clamp_min(1).float()
        band_ratio = self.band_good.float() / self.hold_age.clamp_min(1).float()
        palm_ratio = self.palm_contact.float() / self.hold_age.clamp_min(1).float()
        passed = completed & (contact_ratio >= self.cfg.required_contact_ratio) & (band_ratio >= self.cfg.required_band_ratio) & (palm_ratio < self.cfg.max_palm_contact_ratio) & ~self.invalid
        self.success |= passed
        self.post_hold_age = torch.where(
            self.success & still_candidate & ~self.retention_lost,
            self.post_hold_age + 1,
            torch.zeros_like(self.post_hold_age),
        )
        previous_final = self.final_success.clone()
        self.final_success = (
            self.success
            & ~self.retention_lost
            & (self.post_hold_age >= self.cfg.post_hold_steps)
        )
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
        }
