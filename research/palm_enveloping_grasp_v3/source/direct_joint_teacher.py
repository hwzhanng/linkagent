"""Observable staged teacher for the independent v3 flexion contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simulation.ppo_tomato_grasp_v3.direct_joint_action import joint_targets_to_v3_direct_action


_RESET_ACTIVE_JOINT_RAD = (
    0.6752, 0.02745098, 0.02745098, 0.02745098, 0.02745098,
    -0.9420, -0.13254902, -0.04180392, 0.04588235, 0.16619608,
    1.3248, 0.02529412, 0.02117647, 0.02117647, 0.02117647, 0.02117647,
)


@dataclass(frozen=True)
class DirectJointTeacherCfg:
    coarse_steps: int = 260
    distal_steps: int = 160
    force_lower_n: tuple[float, float, float, float, float] = (0.08, 0.12, 0.12, 0.10, 0.08)
    force_upper_n: tuple[float, float, float, float, float] = (0.70, 0.85, 0.85, 0.75, 0.65)
    retreat_action_step: float = 0.008
    hold_closure_margin: float = 0.10
    palm_support_threshold_n: float = 0.20
    precontact_fraction: float = 0.55
    contact_latch_steps: int = 2
    lost_contact_grace_steps: int = 4
    reacquire_action_step: float = 0.003
    max_closure_extension: float = 0.18
    contact_detect_n: tuple[float, float, float, float, float] = (0.001, 0.001, 0.001, 0.001, 0.001)
    contact_latch_start_step: int = 0
    search_probe_steps: int = 4
    high_force_dwell_steps: int = 1
    digit_start_steps: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.coarse_steps <= 0 or self.distal_steps <= 0:
            raise ValueError("direct teacher phase steps must be positive")
        if len(self.force_lower_n) != 5 or len(self.force_upper_n) != 5:
            raise ValueError("direct teacher force bands must contain five values")
        if any(lo <= 0 or hi < lo for lo, hi in zip(self.force_lower_n, self.force_upper_n)):
            raise ValueError("direct teacher force bands are invalid")
        if not 0 < self.retreat_action_step <= 0.1:
            raise ValueError("direct teacher retreat step is invalid")
        if not 0 <= self.hold_closure_margin <= 0.25:
            raise ValueError("direct teacher closure margin is invalid")
        if not 0.0 < self.precontact_fraction < 1.0:
            raise ValueError("precontact fraction must lie in (0,1)")
        if self.contact_latch_steps <= 0 or self.lost_contact_grace_steps < 0:
            raise ValueError("contact latch timing is invalid")
        if not 0.0 < self.reacquire_action_step <= 0.05:
            raise ValueError("reacquire action step is invalid")
        if not 0.0 <= self.max_closure_extension <= 0.4:
            raise ValueError("maximum closure extension is invalid")
        if len(self.contact_detect_n) != 5 or any(value <= 0.0 for value in self.contact_detect_n):
            raise ValueError("contact detection thresholds are invalid")
        if self.contact_latch_start_step < 0:
            raise ValueError("contact latch start step must be non-negative")
        if self.search_probe_steps <= 0:
            raise ValueError("search probe steps must be positive")
        if self.high_force_dwell_steps <= 0:
            raise ValueError("high force dwell steps must be positive")
        if self.digit_start_steps is not None:
            if len(self.digit_start_steps) != 5 or any(int(v) < 0 for v in self.digit_start_steps):
                raise ValueError("digit start steps must contain five non-negative values")


class DirectJointEventTeacher:
    """Root placement followed by independently force-banded fingertip curl."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        target_action: tuple[float, ...],
        cfg: DirectJointTeacherCfg | None = None,
    ) -> None:
        if len(target_action) != 14:
            raise ValueError("direct teacher target must contain 14 values")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg or DirectJointTeacherCfg()
        self.target = torch.tensor(target_action, device=self.device).view(1, 14).expand(num_envs, -1).clone()
        reset_joint = torch.tensor(_RESET_ACTIVE_JOINT_RAD, device=self.device).view(1, 16)
        self.open_action = joint_targets_to_v3_direct_action(reset_joint).expand(num_envs, -1).clone()
        self.current = self.open_action.clone()
        self.contact_age = torch.zeros((num_envs, 5), dtype=torch.long, device=self.device)
        self.lost_contact_age = torch.zeros((num_envs, 5), dtype=torch.long, device=self.device)
        self.contact_latched = torch.zeros((num_envs, 5), dtype=torch.bool, device=self.device)
        self.latch_reference = self.open_action.clone()
        self.search_sign = torch.ones((num_envs, 5), device=self.device)
        # Distance is sampled only at probe boundaries.  Updating this every
        # policy step makes a probe compare against the immediately preceding
        # frame and can miss a clear multi-step worsening trend.
        self.probe_tip_distance = torch.full((num_envs, 5), float("inf"), device=self.device)
        self.search_active = torch.zeros((num_envs, 5), dtype=torch.bool, device=self.device)
        self.high_force_age = torch.zeros((num_envs, 5), dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.current[ids] = self.open_action[ids]
        self.contact_age[ids] = 0
        self.lost_contact_age[ids] = 0
        self.contact_latched[ids] = False
        self.latch_reference[ids] = self.open_action[ids]
        self.search_sign[ids] = 1.0
        self.probe_tip_distance[ids] = float("inf")
        self.search_active[ids] = False
        self.high_force_age[ids] = 0

    @torch.inference_mode()
    def command(
        self,
        episode_step: torch.Tensor,
        measured_force_n: torch.Tensor,
        palm_force_n: torch.Tensor,
        fruit_linear_speed_m_s: torch.Tensor,
        tip_distance_m: torch.Tensor | None = None,
        active_joint_position_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del fruit_linear_speed_m_s
        reset_ids = episode_step <= 0
        if bool(reset_ids.any()):
            self.reset(torch.nonzero(reset_ids, as_tuple=False).flatten())
        cfg = self.cfg
        palm_ready = palm_force_n >= float(cfg.palm_support_threshold_n)
        coarse_fraction = ((episode_step.float() + 1.0) / float(cfg.coarse_steps)).clamp(0.0, 1.0)
        # Fan/spread is established before flexion.  Digit channels only move
        # to a conservative pre-contact fraction in this phase so a thin
        # native terminal collider cannot be crossed before tactile latching.
        spread_channels = [12, 13]
        desired_spread = self.open_action[:, spread_channels] + coarse_fraction.unsqueeze(-1) * (
            self.target[:, spread_channels] - self.open_action[:, spread_channels]
        )
        self.current[:, spread_channels] = torch.where(
            palm_ready.unsqueeze(-1), desired_spread, self.current[:, spread_channels]
        )
        distal_fraction = (
            (episode_step.float() - float(cfg.coarse_steps) + 1.0) / float(cfg.distal_steps)
        ).clamp(0.0, 1.0)
        lower = measured_force_n.new_tensor(cfg.force_lower_n)
        upper = measured_force_n.new_tensor(cfg.force_upper_n)
        detect = measured_force_n.new_tensor(cfg.contact_detect_n)
        latch_window = episode_step >= int(cfg.contact_latch_start_step)
        if cfg.digit_start_steps is not None:
            starts = torch.tensor(
                [int(v) for v in cfg.digit_start_steps], device=episode_step.device
            ).view(1, 5)
            started = episode_step.view(-1, 1) > starts
        else:
            started = None
        active = (measured_force_n >= detect) & latch_window.unsqueeze(-1)
        if started is not None:
            # Unstarted digits stay fully inert: no contact aging, no latching,
            # no lost-contact growth, so late fingers cannot inherit state from
            # earlier phases of the same episode.
            active = active & started
        high_raw = measured_force_n > upper
        self.high_force_age = torch.where(
            high_raw, self.high_force_age + 1, torch.zeros_like(self.high_force_age)
        )
        self.contact_age = torch.where(active, self.contact_age + 1, torch.zeros_like(self.contact_age))
        lost_grow = ~active if started is None else (~active) & started
        self.lost_contact_age = torch.where(
            active,
            torch.zeros_like(self.lost_contact_age),
            torch.where(lost_grow, self.lost_contact_age + 1, torch.zeros_like(self.lost_contact_age)),
        )
        newly_latched = (~self.contact_latched) & (self.contact_age >= int(cfg.contact_latch_steps))
        self.contact_latched |= newly_latched
        measured_action = None
        if active_joint_position_rad is not None:
            if active_joint_position_rad.shape != (self.num_envs, 16):
                raise ValueError("active joint position must have shape (num_envs, 16)")
            measured_action = joint_targets_to_v3_direct_action(active_joint_position_rad)
        if tip_distance_m is not None:
            if tip_distance_m.shape != (self.num_envs, 5):
                raise ValueError("tip distance must have shape (num_envs, 5)")
            probe = (episode_step % int(cfg.search_probe_steps)) == 0
            low_force = measured_force_n < lower
            lost_after_grace = self.lost_contact_age > int(cfg.lost_contact_grace_steps)
            probing_search = self.contact_latched & low_force & lost_after_grace & self.search_active
            worsening = (
                probing_search
                & torch.isfinite(self.probe_tip_distance)
                & (tip_distance_m > self.probe_tip_distance + 5.0e-5)
                & probe.unsqueeze(-1)
            )
            self.search_sign = torch.where(worsening, -self.search_sign, self.search_sign)
            update_probe = probe.unsqueeze(-1) & (
                probing_search | newly_latched
            )
            self.probe_tip_distance = torch.where(
                update_probe, tip_distance_m, self.probe_tip_distance
            )
        digit_channels = ([0, 1, 2, 3], [4, 8], [5, 9], [6, 10], [7, 11])
        pre_fraction = float(cfg.precontact_fraction)
        for digit, channels in enumerate(digit_channels):
            open_value = self.open_action[:, channels]
            target_value = self.target[:, channels]
            precontact = open_value + pre_fraction * (target_value - open_value)
            if digit > 0:
                # Straight-finger distal flexion is intentionally reserved
                # for phase two so its participation is explicit/auditable.
                precontact = precontact.clone()
                precontact[:, 1] = open_value[:, 1]
            if cfg.digit_start_steps is not None:
                start_step = int(cfg.digit_start_steps[digit])
                local_step = (episode_step - start_step).clamp_min(0)
                d_coarse = ((local_step.float() + 1.0) / float(cfg.coarse_steps)).clamp(0.0, 1.0)
                d_distal = (
                    (local_step.float() - float(cfg.coarse_steps) + 1.0) / float(cfg.distal_steps)
                ).clamp(0.0, 1.0)
                phase_step = local_step
                ramped = torch.where(
                    (phase_step < cfg.coarse_steps).unsqueeze(-1),
                    open_value + d_coarse.unsqueeze(-1) * (precontact - open_value),
                    precontact + d_distal.unsqueeze(-1) * (target_value - precontact),
                )
                # Before its start step a finger must hold exactly the open
                # pose: clamp-to-zero alone would leak the first coarse step.
                planned = torch.where(
                    (episode_step > start_step).unsqueeze(-1),
                    ramped,
                    open_value.expand_as(ramped),
                )
            else:
                d_coarse = coarse_fraction
                d_distal = distal_fraction
                planned = torch.where(
                    (episode_step < cfg.coarse_steps).unsqueeze(-1),
                    open_value + coarse_fraction.unsqueeze(-1) * (precontact - open_value),
                    precontact + distal_fraction.unsqueeze(-1) * (target_value - precontact),
                )
            direction = torch.sign(target_value - open_value)
            if bool(newly_latched[:, digit].any()):
                rows = newly_latched[:, digit].nonzero(as_tuple=False).flatten()
                # The implicit drive can lag its command by tens of policy
                # steps.  Capture the inverse-mapped measured posture at the
                # actual contact event; retaining the planned command would
                # make the drive continue through the fruit after latching.
                if measured_action is not None:
                    self.current[rows[:, None], channels] = measured_action[rows[:, None], channels]
                self.latch_reference[rows[:, None], channels] = self.current[rows[:, None], channels]
            # Per-digit hold phase: with staggered starts a late finger must
            # not enter search/hold before its own distal ramp completes.
            hold_phase = d_distal >= 1.0
            extended_target = (target_value + float(cfg.max_closure_extension) * direction).clamp(-1.0, 1.0)
            low = measured_force_n[:, digit] < lower[digit]
            high = self.high_force_age[:, digit] >= int(cfg.high_force_dwell_steps)
            contact_now = active[:, digit]
            in_band = (~low) & (~high)
            grace_hold = self.contact_latched[:, digit] & (~contact_now) & (
                self.lost_contact_age[:, digit] <= int(cfg.lost_contact_grace_steps)
            )
            stable_hold = self.contact_latched[:, digit] & contact_now & in_band
            value = torch.where(
                (grace_hold | stable_hold).unsqueeze(-1), self.current[:, channels], planned
            )
            search_weight = torch.ones_like(direction)
            if digit > 0:
                search_weight[:, 0] = 0.25
            reacquire = self.current[:, channels] + (
                float(cfg.reacquire_action_step)
                * self.search_sign[:, digit].unsqueeze(-1)
                * direction
                * search_weight
            )
            anchor_low = (self.latch_reference[:, channels] - float(cfg.max_closure_extension)).clamp(-1.0, 1.0)
            anchor_high = (self.latch_reference[:, channels] + float(cfg.max_closure_extension)).clamp(-1.0, 1.0)
            reacquire = reacquire.maximum(anchor_low).minimum(anchor_high)
            # A weak but real tomato contact is not a reason to freeze.  Add a
            # small preload in the known closure direction until the measured
            # force enters its per-digit band.  Geometry hill-climbing is used
            # only after the contact has actually been lost beyond the grace
            # period.
            preload = self.current[:, channels] + float(cfg.reacquire_action_step) * direction * search_weight
            preload = preload.maximum(anchor_low).minimum(anchor_high)
            weak_contact = self.contact_latched[:, digit] & contact_now & low
            value = torch.where(weak_contact.unsqueeze(-1), preload, value)
            lost_search = self.contact_latched[:, digit] & low & (~contact_now) & (~grace_hold)
            value = torch.where(
                lost_search.unsqueeze(-1),
                reacquire,
                value,
            )
            first_search = self.current[:, channels] + float(cfg.reacquire_action_step) * direction
            first_search = torch.where(
                direction >= 0.0,
                torch.minimum(first_search, extended_target),
                torch.maximum(first_search, extended_target),
            )
            value = torch.where((hold_phase & low & ~self.contact_latched[:, digit]).unsqueeze(-1), first_search, value)
            retreat = self.current[:, channels] + float(cfg.retreat_action_step) * (open_value - self.current[:, channels])
            value = torch.where(high.unsqueeze(-1), retreat, value)
            self.current[:, channels] = torch.where(
                palm_ready.unsqueeze(-1), value, self.current[:, channels]
            )
            self.search_active[:, digit] = lost_search
        return self.current.clamp(-1.0, 1.0).clone()


__all__ = ["DirectJointEventTeacher", "DirectJointTeacherCfg"]
