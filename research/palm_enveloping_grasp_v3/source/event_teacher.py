"""Event-driven, force-balanced five-finger retention teacher."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simulation.ppo_tomato_grasp_v2.residual_controller import ResidualBaseController, ResidualBaseControllerCfg


@dataclass(frozen=True)
class EventDrivenTeacherCfg:
    contact_threshold_n: float = 0.04
    force_lower_n: tuple[float, float, float, float, float] = (0.12, 0.15, 0.15, 0.12, 0.10)
    force_upper_n: tuple[float, float, float, float, float] = (0.65, 0.80, 0.80, 0.70, 0.65)
    reacquire_step: float = 0.012
    retreat_step: float = 0.018
    trust_region: float = 0.18
    thumb_reacquire_direction: tuple[float, float] = (0.0, -1.0)
    load_share_max: float = 0.70
    search_period_steps: int = 12
    fruit_speed_hold_m_s: float = 0.08
    high_force_dwell_steps: int = 3
    force_prediction_gain: float = 1.0
    balance_target_n: tuple[float, float, float, float, float] = (0.10, 0.12, 0.12, 0.10, 0.08)
    digit_latch_start_step: int = 0
    palm_latch_threshold_n: float = 0.30

    def __post_init__(self) -> None:
        if len(self.force_lower_n) != 5 or len(self.force_upper_n) != 5:
            raise ValueError("force bands must contain five digits")
        if any(low <= 0 or high < low for low, high in zip(self.force_lower_n, self.force_upper_n)):
            raise ValueError("invalid event-teacher force bands")
        if not 0 < self.reacquire_step <= self.trust_region <= 0.5:
            raise ValueError("reacquire_step/trust_region are invalid")
        if not 0 < self.retreat_step <= 0.5:
            raise ValueError("retreat_step is invalid")
        if not 0 < self.load_share_max < 1:
            raise ValueError("load_share_max must lie in (0,1)")
        if self.search_period_steps <= 0:
            raise ValueError("search_period_steps must be positive")
        if self.high_force_dwell_steps <= 0:
            raise ValueError("high_force_dwell_steps must be positive")
        if not 0.0 <= self.force_prediction_gain <= 4.0:
            raise ValueError("force_prediction_gain must lie in [0,4]")
        if len(self.balance_target_n) != 5 or any(value <= 0 for value in self.balance_target_n):
            raise ValueError("balance_target_n must contain five positive values")
        if self.digit_latch_start_step < 0:
            raise ValueError("digit_latch_start_step must be non-negative")
        if self.palm_latch_threshold_n <= 0:
            raise ValueError("palm_latch_threshold_n must be positive")


class EventDrivenEnclosureTeacher:
    """Use candidate123 only to reach contact, then regulate contact events.

    The first simultaneous five-finger physical contact is an event, not a
    terminal pose.  The controller freezes the coarse manifold at that event
    and thereafter adjusts each digit independently inside a small action
    trust region.  This keeps all policy decisions observable from joint,
    tactile, palm-load, and fruit-speed channels.
    """

    _CHANNELS = ((4, 5), (6, 10), (7, 11), (8, 12), (9, 13))

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        target_action: tuple[float, ...],
        base_cfg: ResidualBaseControllerCfg,
        cfg: EventDrivenTeacherCfg | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg or EventDrivenTeacherCfg()
        self.base = ResidualBaseController(num_envs, device, target_action, base_cfg)
        self.latched = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.digit_latched = torch.zeros((num_envs, 5), dtype=torch.bool, device=self.device)
        self.latch_step = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.current = self.base.open_action.clone()
        self.anchor = self.base.open_action.clone()
        self.lost_age = torch.zeros((num_envs, 5), dtype=torch.long, device=self.device)
        self.high_age = torch.zeros((num_envs, 5), dtype=torch.long, device=self.device)
        self.previous_force = torch.zeros((num_envs, 5), dtype=torch.float32, device=self.device)
        target = self.base.target[0]
        open_action = self.base.open_action[0]
        straight_direction = []
        for root, distal in self._CHANNELS[1:]:
            straight_direction.append((
                float(torch.sign(target[root] - open_action[root]).item()),
                float(torch.sign(target[distal] - open_action[distal]).item()),
            ))
        self.direction = torch.tensor(
            (self.cfg.thumb_reacquire_direction, *straight_direction),
            dtype=torch.float32,
            device=self.device,
        )

    @torch.inference_mode()
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.base.reset(env_ids)
        self.latched[ids] = False
        self.digit_latched[ids] = False
        self.latch_step[ids] = -1
        self.current[ids] = self.base.open_action[ids]
        self.anchor[ids] = self.base.open_action[ids]
        self.lost_age[ids] = 0
        self.high_age[ids] = 0
        self.previous_force[ids] = 0.0

    @torch.inference_mode()
    def command(
        self,
        episode_step: torch.Tensor,
        measured_force_n: torch.Tensor,
        palm_force_n: torch.Tensor,
        fruit_linear_speed_m_s: torch.Tensor,
    ) -> torch.Tensor:
        if measured_force_n.shape != (self.num_envs, 5):
            raise ValueError("measured_force_n must have shape (num_envs,5)")
        base_action = self.base.command(episode_step, measured_force_n)
        active = measured_force_n >= float(self.cfg.contact_threshold_n)
        latch_eligible = (
            (episode_step >= int(self.cfg.digit_latch_start_step))
            & (palm_force_n >= float(self.cfg.palm_latch_threshold_n))
        )
        newly_digit_latched = active & latch_eligible.unsqueeze(-1) & ~self.digit_latched
        for digit, channels in enumerate(self._CHANNELS):
            new_digit = newly_digit_latched[:, digit]
            if bool(new_digit.any()):
                for channel in channels:
                    self.anchor[new_digit, channel] = base_action[new_digit, channel]
                    self.current[new_digit, channel] = base_action[new_digit, channel]
                self.previous_force[new_digit, digit] = measured_force_n[new_digit, digit]
        self.digit_latched |= newly_digit_latched
        newly_latched = self.digit_latched.all(-1) & ~self.latched
        if bool(newly_latched.any()):
            self.anchor[newly_latched, :4] = base_action[newly_latched, :4]
            self.current[newly_latched, :4] = base_action[newly_latched, :4]
            self.latch_step[newly_latched] = episode_step[newly_latched].to(torch.long)
        self.latched |= newly_latched
        # Fingers not yet in contact continue following the nominal approach;
        # fingers that have touched switch independently to force retention.
        for digit, channels in enumerate(self._CHANNELS):
            waiting = ~self.digit_latched[:, digit]
            for channel in channels:
                self.current[waiting, channel] = base_action[waiting, channel]
        if not bool(self.digit_latched.any()):
            self.current[:, :4] = base_action[:, :4]
            return self.current.clone()

        lower = measured_force_n.new_tensor(self.cfg.force_lower_n)
        upper = measured_force_n.new_tensor(self.cfg.force_upper_n)
        balance_target = measured_force_n.new_tensor(self.cfg.balance_target_n)
        force_slope = measured_force_n - self.previous_force
        predicted_force = measured_force_n + float(self.cfg.force_prediction_gain) * force_slope
        low = (measured_force_n < lower) | (predicted_force < lower)
        high = measured_force_n > upper
        self.lost_age = torch.where(low & self.digit_latched, self.lost_age + 1, torch.zeros_like(self.lost_age))
        self.high_age = torch.where(high & self.digit_latched, self.high_age + 1, torch.zeros_like(self.high_age))
        digit_sum = measured_force_n.sum(-1)
        palm_share = palm_force_n / (palm_force_n + digit_sum).clamp_min(1.0e-6)
        underloaded = self.latched & (palm_share > float(self.cfg.load_share_max))
        weakest = measured_force_n.argmin(-1)
        below_balance_target = measured_force_n < balance_target
        moving_fast = fruit_linear_speed_m_s > float(self.cfg.fruit_speed_hold_m_s)

        for digit, channels in enumerate(self._CHANNELS):
            select_weakest = underloaded & (weakest == digit)
            # A high palm share is a total digit-load problem.  Correct every
            # under-target digit together; select the weakest only as a
            # fallback when all five happen to exceed their balance targets.
            coordinated_recovery = underloaded & (below_balance_target[:, digit] | select_weakest)
            advance = self.digit_latched[:, digit] & (low[:, digit] | coordinated_recovery)
            # Contact forces lag action changes.  A one-frame peak must not
            # trigger an immediate retreat that removes the contact two frames
            # later.  The hard safety termination remains independent of this
            # soft-band dwell.
            retreat = self.digit_latched[:, digit] & (self.high_age[:, digit] >= int(self.cfg.high_force_dwell_steps)) & ~advance
            direction = self.direction[digit]
            # The thumb searches the opposite side of its contact anchor when
            # absent; straight digits use their audited closure directions.
            if digit == 0:
                sign = torch.where(
                    (self.lost_age[:, digit] // int(self.cfg.search_period_steps)).remainder(2) == 0,
                    torch.ones(self.num_envs, device=self.device),
                    -torch.ones(self.num_envs, device=self.device),
                )
            else:
                sign = torch.ones(self.num_envs, device=self.device)
            for local, channel in enumerate(channels):
                delta = float(self.cfg.reacquire_step) * direction[local] * sign
                # When the fruit is already moving, use only 25% of the
                # correction to avoid converting contact recovery into impact.
                delta = torch.where(moving_fast, 0.25 * delta, delta)
                value = torch.where(advance, self.current[:, channel] + delta, self.current[:, channel])
                value = torch.where(
                    retreat,
                    value - float(self.cfg.retreat_step) * direction[local],
                    value,
                )
                low_bound = self.anchor[:, channel] - float(self.cfg.trust_region)
                high_bound = self.anchor[:, channel] + float(self.cfg.trust_region)
                self.current[:, channel] = torch.maximum(torch.minimum(value, high_bound), low_bound).clamp(-1.0, 1.0)
        # Coarse global/spread channels are held at the contact event; only
        # the five independent digit pairs remain active.
        self.current[~self.latched, :4] = base_action[~self.latched, :4]
        self.current[self.latched, :4] = self.anchor[self.latched, :4]
        self.previous_force[self.digit_latched] = measured_force_n[self.digit_latched]
        return self.current.clone()


__all__ = ["EventDrivenEnclosureTeacher", "EventDrivenTeacherCfg"]
