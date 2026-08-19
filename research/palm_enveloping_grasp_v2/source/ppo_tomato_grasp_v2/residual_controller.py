"""Observable staged base controller for the v2 residual-policy branch.

The controller is simulation-only.  It supplies a bounded, auditable action
manifold (palm/root/thumb approach followed by one straight finger at a time)
while a PPO actor is restricted to a small residual around that manifold.  It
does not encode a terminal pose or a hardware command path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResidualBaseControllerCfg:
    approach_steps: int = 240
    per_digit_steps: int = 60
    # Keep the thumb opposition pair open until a configurable point in the
    # finger-insertion schedule.  A delayed thumb is an explicit timing
    # ablation: closing it during the coarse palm/root approach can move the
    # supported fruit away from the four-finger enclosure corridor.
    thumb_delay_steps: int = 0
    thumb_ramp_steps: int = 60
    # Optional thumb-first sub-phase.  During this window only the thumb
    # opposition/residual pair (and an optional fraction of global flexion)
    # advances; straight-finger roots remain at the open command so the
    # supported fruit is not pushed away before the counterforce is present.
    thumb_first_steps: int = 0
    thumb_first_global_scale: float = 0.0
    # Match the validated contact_closed_loop_probe defaults.  The small
    # approach interpolation and stronger force-band retreat are deliberate:
    # a fast coarse jump made the fruit leave the palm before the thumb could
    # become a counterforce.
    approach_smoothing: float = 1.0
    insertion_step_fraction: float = 0.08
    retreat_fraction: float = 0.12
    # Optional per-digit overrides (index, middle, ring, little).  These are
    # deliberately separate from the global fractions so a geometrically
    # crowded digit can insert more slowly or retreat more readily without
    # changing the already validated behaviour of the other three digits.
    insertion_step_fraction_per_digit: tuple[float, float, float, float] | None = None
    retreat_fraction_per_digit: tuple[float, float, float, float] | None = None
    # Palm/root/thumb first is the intended enclosure sequence.  A full
    # simultaneous ramp remains available as an explicit ablation, but it
    # couples four distal impulses into the support surface and is not the
    # default residual manifold.
    simultaneous_reach: bool = False
    # These are the same broad contact-band limits used by the validated
    # closed-loop probe.  They are deliberately not terminal-force targets:
    # below the lower edge the active finger advances a small amount, while
    # above the upper edge only that finger retreats.
    lower_force_n: tuple[float, float, float, float] = (0.18, 0.18, 0.18, 0.18)
    upper_force_n: tuple[float, float, float, float] = (0.80, 0.80, 0.80, 0.80)
    # The thumb has its own measured force band during retention.  A small
    # optional action-space offset lets the controller search beyond the
    # coarse reference when the thumb loses contact; the default is zero so
    # archived residual runs retain their exact behaviour.
    thumb_lower_force_n: float = 0.10
    thumb_upper_force_n: float = 0.80
    thumb_hold_target_offset: tuple[float, float] = (0.0, 0.0)
    thumb_insertion_step_fraction: float = 0.04
    thumb_retreat_fraction: float = 0.08
    # Optional bounded two-sided search around the thumb hold target.  The
    # probe is only active while the measured thumb force is below the lower
    # band; it alternates the signed offset after a finite number of low-force
    # frames, so it can reacquire a contact corridor without forcing a fixed
    # terminal pose.  Defaults preserve archived controller behaviour.
    thumb_search_amplitude: tuple[float, float] = (0.0, 0.0)
    thumb_search_period_steps: int = 24
    # If positive, straight-finger insertion is held until the measured thumb
    # force has remained in-band for this many policy frames. Zero preserves
    # the historical timing path.
    thumb_first_contact_gate_steps: int = 0
    # Optional measured-contact latch.  Once the thumb reaches the physical
    # contact threshold, hold its current action for a bounded number of
    # policy frames instead of immediately probing past the narrow contact
    # corridor.  Zero preserves the archived force-band behaviour.
    thumb_contact_latch_force_n: float = 0.0
    thumb_contact_latch_steps: int = 0
    thumb_contact_latch_freeze_coarse: bool = False

    def __post_init__(self) -> None:
        for name in ("insertion_step_fraction_per_digit", "retreat_fraction_per_digit"):
            values = getattr(self, name)
            if values is None:
                continue
            if len(values) != 4 or any(not 0.0 < float(value) <= 1.0 for value in values):
                raise ValueError(f"{name} must contain four values in (0,1]")
        if not 0.0 <= self.thumb_lower_force_n <= self.thumb_upper_force_n:
            raise ValueError("thumb force band must satisfy 0 <= lower <= upper")
        if self.thumb_first_steps < 0:
            raise ValueError("thumb_first_steps must be non-negative")
        if not 0.0 <= self.thumb_first_global_scale <= 1.0:
            raise ValueError("thumb_first_global_scale must lie in [0,1]")
        if len(self.thumb_hold_target_offset) != 2 or any(abs(float(v)) > 1.0 for v in self.thumb_hold_target_offset):
            raise ValueError("thumb_hold_target_offset must contain two values in [-1,1]")
        if len(self.thumb_search_amplitude) != 2 or any(abs(float(v)) > 1.0 for v in self.thumb_search_amplitude):
            raise ValueError("thumb_search_amplitude must contain two values in [-1,1]")
        if self.thumb_search_period_steps <= 0:
            raise ValueError("thumb_search_period_steps must be positive")
        if self.thumb_first_contact_gate_steps < 0:
            raise ValueError("thumb_first_contact_gate_steps must be non-negative")
        if self.thumb_contact_latch_force_n < 0.0:
            raise ValueError("thumb_contact_latch_force_n must be non-negative")
        if self.thumb_contact_latch_steps < 0:
            raise ValueError("thumb_contact_latch_steps must be non-negative")
        if not 0.0 < self.thumb_insertion_step_fraction <= 1.0 or not 0.0 < self.thumb_retreat_fraction <= 1.0:
            raise ValueError("thumb insertion/retreat fractions must lie in (0,1]")


class ResidualBaseController:
    """Vectorized open-loop/force-band stage controller for 14-D actions."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        target_action: tuple[float, ...],
        cfg: ResidualBaseControllerCfg | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if len(target_action) != 14 or any(abs(float(value)) > 1.000001 for value in target_action):
            raise ValueError("target_action must contain 14 values in [-1,1]")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg or ResidualBaseControllerCfg()
        self.target = torch.tensor(target_action, dtype=torch.float32, device=self.device).view(1, 14).expand(num_envs, -1).clone()
        self.open_action = torch.zeros((num_envs, 14), dtype=torch.float32, device=self.device)
        self.open_action[:, :10] = torch.tensor(
            (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
            dtype=torch.float32,
            device=self.device,
        )
        self.current = self.open_action.clone()
        self.contact_streak = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.frozen = torch.zeros((num_envs, 4), dtype=torch.bool, device=self.device)
        self.thumb_low_streak = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.thumb_search_sign = torch.ones((num_envs,), dtype=torch.float32, device=self.device)
        self.thumb_gate_streak = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.thumb_latch_remaining = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.reset()

    @torch.inference_mode()
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.current[ids] = self.open_action[ids]
        self.contact_streak[ids] = 0.0
        self.frozen[ids] = False
        self.thumb_low_streak[ids] = 0
        self.thumb_search_sign[ids] = 1.0
        self.thumb_gate_streak[ids] = 0
        self.thumb_latch_remaining[ids] = 0

    @torch.inference_mode()
    def command(
        self,
        episode_step: torch.Tensor,
        measured_force_n: torch.Tensor,
    ) -> torch.Tensor:
        if episode_step.shape != (self.num_envs,):
            raise ValueError("episode_step must have shape (num_envs,)")
        if measured_force_n.shape != (self.num_envs, 5):
            raise ValueError("measured_force_n must have shape (num_envs,5)")
        step = episode_step.to(device=self.device, dtype=torch.float32)
        reset_ids = step <= 0.0
        if bool(reset_ids.any()):
            self.reset(torch.nonzero(reset_ids, as_tuple=False).flatten())

        cfg = self.cfg
        approach = max(int(cfg.approach_steps), 1)
        digit_steps = max(int(cfg.per_digit_steps), 1)
        approach_width = 14 if cfg.simultaneous_reach else 10
        thumb_first = min(int(cfg.thumb_first_steps), max(approach - 1, 0))
        if thumb_first > 0:
            thumb_phase = step < float(thumb_first)
            thumb_fraction = ((step + 1.0) / float(thumb_first)).clamp(0.0, 1.0).unsqueeze(-1)
            remaining = max(approach - thumb_first, 1)
            coarse_fraction = (
                (step - float(thumb_first) + 1.0) / float(remaining)
            ).clamp(0.0, 1.0).unsqueeze(-1)
            desired_coarse = self.open_action[:, :approach_width].expand_as(
                self.current[:, :approach_width]
            ).clone()
            # The thumb pair is intentionally the only active pair during
            # the first phase.  Keep global flexion at the open value unless
            # explicitly requested by a calibrated prior.
            desired_coarse[:, 4:6] = self.open_action[:, 4:6] + thumb_fraction * (
                self.target[:, 4:6] - self.open_action[:, 4:6]
            )
            desired_coarse[:, 0:1] = self.open_action[:, 0:1] + (
                thumb_fraction * float(cfg.thumb_first_global_scale)
                * (self.target[:, 0:1] - self.open_action[:, 0:1])
            )
            post_target = self.open_action[:, :approach_width] + coarse_fraction * (
                self.target[:, :approach_width] - self.open_action[:, :approach_width]
            )
            desired_coarse = torch.where(thumb_phase.unsqueeze(-1), desired_coarse, post_target)
        else:
            thumb_phase = torch.zeros_like(step, dtype=torch.bool)
            fraction = ((step + 1.0) / float(approach)).clamp(0.0, 1.0).unsqueeze(-1)
            desired_coarse = self.open_action[:, :approach_width] + fraction * (
                self.target[:, :approach_width] - self.open_action[:, :approach_width]
            )
        thumb_delay = max(int(cfg.thumb_delay_steps), 0)
        if thumb_delay > 0:
            thumb_ramp = max(int(cfg.thumb_ramp_steps), 1)
            thumb_fraction = ((step - float(thumb_delay) + 1.0) / float(thumb_ramp)).clamp(0.0, 1.0).unsqueeze(-1)
            desired_coarse[:, 4:6] = self.open_action[:, 4:6] + thumb_fraction * (
                self.target[:, 4:6] - self.open_action[:, 4:6]
            )
        alpha = float(cfg.approach_smoothing)
        before_approach = step < float(approach)
        coarse_previous = self.current[:, :approach_width].clone()
        if cfg.simultaneous_reach:
            self.current[:, :approach_width] = self.current[:, :approach_width] + alpha * (
                desired_coarse - self.current[:, :approach_width]
            )
        else:
            # Once the coarse ramp has completed, keep the thumb channels
            # under the measured force-band controller below.  Reassigning
            # all ten coarse channels to ``desired_coarse`` on every hold
            # step would overwrite the thumb correction immediately, making
            # its effective action a single 4%-of-offset nudge per frame.
            if bool(before_approach.any()):
                self.current[before_approach, :approach_width] = (
                    self.current[before_approach, :approach_width]
                    + alpha * (
                        desired_coarse[before_approach]
                        - self.current[before_approach, :approach_width]
                    )
                )
            after_approach = ~before_approach
            if bool(after_approach.any()):
                self.current[after_approach, :4] = desired_coarse[after_approach, :4]
                self.current[after_approach, 6:approach_width] = desired_coarse[
                    after_approach, 6:approach_width
                ]
        # Distal channels stay open during the palm/root/thumb approach.
        before_insertion = step < float(approach)
        if not cfg.simultaneous_reach:
            self.current[before_insertion, 10:] = self.open_action[before_insertion, 10:]

        # After the coarse approach, insert index -> middle -> ring -> little
        # one at a time.  Each digit owns its proximal and distal residual;
        # high force retreats only that pair, while already inserted digits
        # remain available for hold feedback.
        elapsed = (step - float(approach)).clamp_min(0.0)
        active_digit = torch.floor(elapsed / float(digit_steps)).long().clamp(0, 3)
        insertion_phase = (elapsed < float(4 * digit_steps)) & (step >= float(approach))
        lower = measured_force_n.new_tensor(cfg.lower_force_n)
        upper = measured_force_n.new_tensor(cfg.upper_force_n)
        # Once the coarse palm/root/thumb approach is complete, close-loop the
        # thumb independently.  This prevents the sequential four-finger
        # insertion from silently turning the enclosure back into a
        # four-finger pinch.  The offset is bounded and opt-in; all force and
        # joint safety gates remain enforced by the environment.
        # Thumb retention is allowed as soon as the coarse approach ends,
        # including while the four straight fingers are inserted.  Waiting
        # until all four insertion windows finish leaves the thumb open for
        # most of the episode; a supported fruit can drift away during that
        # interval and the later force-band update has no contact to recover.
        hold_update = step >= float(approach)
        thumb_force = measured_force_n[:, 0]
        thumb_low = thumb_force < float(cfg.thumb_lower_force_n)
        thumb_high = thumb_force >= float(cfg.thumb_upper_force_n)
        latch_enabled = int(cfg.thumb_contact_latch_steps) > 0 and float(cfg.thumb_contact_latch_force_n) > 0.0
        latch_contact = hold_update & (thumb_force >= float(cfg.thumb_contact_latch_force_n))
        if latch_enabled:
            self.thumb_latch_remaining = torch.where(
                latch_contact,
                torch.full_like(self.thumb_latch_remaining, int(cfg.thumb_contact_latch_steps)),
                (self.thumb_latch_remaining - 1).clamp_min(0),
            )
        else:
            self.thumb_latch_remaining.zero_()
        thumb_latched = self.thumb_latch_remaining > 0
        gate_enabled = int(cfg.thumb_first_contact_gate_steps) > 0
        if gate_enabled:
            thumb_in_band = hold_update & ((~thumb_low & ~thumb_high) | thumb_latched)
            self.thumb_gate_streak = torch.where(
                thumb_in_band,
                self.thumb_gate_streak + 1,
                torch.zeros_like(self.thumb_gate_streak),
            )
            thumb_gate_ready = self.thumb_gate_streak >= int(cfg.thumb_first_contact_gate_steps)
            insertion_phase &= thumb_gate_ready
        else:
            thumb_gate_ready = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        # When the explicit thumb gate is enabled, also hold the coarse
        # palm/root channels until the thumb has established contact. This
        # closes the remaining path in which finger insertion was paused but
        # the root ramp could still push the supported fruit away.
        if gate_enabled:
            blocked_coarse = (~before_approach) & ~thumb_gate_ready
            if bool(blocked_coarse.any()):
                self.current[blocked_coarse, :4] = coarse_previous[blocked_coarse, :4]
                self.current[blocked_coarse, 6:approach_width] = coarse_previous[blocked_coarse, 6:approach_width]
        if latch_enabled and cfg.thumb_contact_latch_freeze_coarse and bool(thumb_latched.any()):
            # A thumb contact can be lost because the still-ramping palm/root
            # channels move the supported fruit out of its narrow opposition
            # corridor.  During the bounded latch, preserve the previous
            # coarse command while leaving the thumb pair under its own
            # safety/force logic.
            self.current[thumb_latched, :4] = coarse_previous[thumb_latched, :4]
            self.current[thumb_latched, 6:approach_width] = coarse_previous[thumb_latched, 6:approach_width]
        # A low-force streak drives an optional, symmetric local probe.  The
        # sign flips only at the configured period, avoiding frame-to-frame
        # chatter and keeping the search within a small action trust region.
        low_update = hold_update & thumb_low
        self.thumb_low_streak = torch.where(
            low_update,
            self.thumb_low_streak + 1,
            torch.zeros_like(self.thumb_low_streak),
        )
        search_enabled = any(abs(float(v)) > 0.0 for v in cfg.thumb_search_amplitude)
        if search_enabled:
            flip = low_update & (self.thumb_low_streak > 0) & (
                self.thumb_low_streak.remainder(int(cfg.thumb_search_period_steps)) == 0
            )
            self.thumb_search_sign = torch.where(
                flip, -self.thumb_search_sign, self.thumb_search_sign
            )
            probe = measured_force_n.new_tensor(cfg.thumb_search_amplitude).view(1, 2)
            probe = self.thumb_search_sign.unsqueeze(-1) * probe
        else:
            probe = measured_force_n.new_zeros((self.num_envs, 2))
        thumb_target = self.target[:, 4:6] + measured_force_n.new_tensor(cfg.thumb_hold_target_offset) + probe
        thumb_target = thumb_target.clamp(-1.0, 1.0)
        thumb_advance = self.current[:, 4:6] + float(cfg.thumb_insertion_step_fraction) * (
            thumb_target - self.current[:, 4:6]
        )
        thumb_retreat = self.current[:, 4:6] + float(cfg.thumb_retreat_fraction) * (
            self.open_action[:, 4:6] - self.current[:, 4:6]
        )
        thumb_next = torch.where(
            thumb_high.unsqueeze(-1),
            thumb_retreat,
            torch.where(
                thumb_latched.unsqueeze(-1),
                self.current[:, 4:6],
                torch.where(thumb_low.unsqueeze(-1), thumb_advance, self.current[:, 4:6]),
            ),
        )
        self.current[:, 4:6] = torch.where(hold_update.unsqueeze(-1), thumb_next, self.current[:, 4:6])
        for digit in range(4):
            # During insertion only the currently active finger is allowed to
            # move.  Once all four windows have elapsed, switch to a shared
            # hold band so each digit can make its own low-force correction.
            eligible = insertion_phase & (active_digit == digit)
            hold_update = (~insertion_phase) & (step >= float(approach)) & thumb_gate_ready
            force = measured_force_n[:, digit + 1]
            # Do not consume a digit's contact streak while it is merely
            # passing through the coarse approach.  The closed-loop probe
            # starts each streak at the beginning of that digit's insertion
            # window; counting approach contact here froze fingers before
            # their turn and made the residual evaluator diverge from the
            # audited controller path.
            eligible_or_hold = eligible | hold_update
            active_contact = (force >= lower[digit]) & eligible_or_hold
            self.contact_streak[:, digit] = torch.where(
                active_contact,
                self.contact_streak[:, digit] + 1.0,
                torch.where(
                    eligible_or_hold,
                    torch.zeros_like(self.contact_streak[:, digit]),
                    self.contact_streak[:, digit],
                ),
            )
            self.frozen[:, digit] |= eligible_or_hold & (self.contact_streak[:, digit] >= 5.0)
            eligible_current = eligible & ~self.frozen[:, digit]
            low = force < lower[digit]
            high = force >= upper[digit]
            insertion_fraction = (
                float(cfg.insertion_step_fraction_per_digit[digit])
                if cfg.insertion_step_fraction_per_digit is not None
                else float(cfg.insertion_step_fraction)
            )
            retreat_fraction = (
                float(cfg.retreat_fraction_per_digit[digit])
                if cfg.retreat_fraction_per_digit is not None
                else float(cfg.retreat_fraction)
            )
            for channel in (6 + digit, 10 + digit):
                advance = self.current[:, channel] + insertion_fraction * (
                    self.target[:, channel] - self.current[:, channel]
                )
                retreat = self.current[:, channel] + retreat_fraction * (
                    self.open_action[:, channel] - self.current[:, channel]
                )
                next_value = torch.where(high, retreat, torch.where(low, advance, self.current[:, channel]))
                self.current[:, channel] = torch.where(
                    eligible_current, next_value, self.current[:, channel]
                )
                # During the hold phase all four fingers are independently
                # force-banded, matching the closed-loop diagnostic probe.
                self.current[:, channel] = torch.where(
                    hold_update, next_value, self.current[:, channel]
                )
        return self.current.clamp(-1.0, 1.0).clone()


__all__ = ["ResidualBaseController", "ResidualBaseControllerCfg"]
