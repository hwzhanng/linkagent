"""Weak staged teacher for palm-supported enclosure demonstrations.

The teacher is not a replay of a fixed successful pose.  It only supplies a
safe curriculum prior: wait for palm support, move roots/ thumb opposition,
curl proximal joints, then release independent distal residuals and trim each
finger from its own force feedback.  PPO remains free to discover a different
joint trajectory and final pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EnvelopingTeacherCfg:
    palm_support_force_n: float = 0.30
    contact_threshold_n: float = 0.12
    target_force_n: tuple[float, float, float, float, float] = (0.45, 0.32, 0.32, 0.28, 0.28)
    high_force_n: tuple[float, float, float, float, float] = (1.80, 1.50, 1.50, 1.35, 1.35)
    root_phase_steps: int = 12
    proximal_phase_steps: int = 40
    distal_phase_steps: int = 90
    support_wait_steps: int = 5
    tactile_gain: float = 0.08
    tactile_deadband_n: float = 0.06
    max_feedback_step: float = 0.03
    distal_base_action: float = 0.15
    # Optional 4D per-finger distal weak prior discovered by a bounded
    # simulation-only action search.  It is a staged target, not a replayed
    # successful pose: the teacher still ramps it, applies tactile trim, and
    # is blended out of PPO rollouts after the warm-up window.
    distal_prior_action: tuple[float, float, float, float] | None = None
    # When non-zero, use the audited complete prior as one smooth approach
    # trajectory.  This is a simulation-only warm-start mode: it preserves
    # tactile trim and PPO hand-off, but avoids a phase schedule whose distal
    # release timing can inject a different object impulse than the validated
    # low-speed candidate path.
    prior_ramp_steps: int = 0
    # Optional simulation-only force-band warm start.  It mirrors the audited
    # contact_closed_loop_probe: ramp to the candidate, then advance/retreat
    # only the matching proximal+distal pair from raw link4 force.  This is a
    # teacher trajectory for PPO, not a deploy-time controller or a fixed
    # success pose.
    force_band_enabled: bool = False
    force_band_lower_n: tuple[float, float, float, float] = (0.10, 0.10, 0.10, 0.10)
    force_band_upper_n: tuple[float, float, float, float] = (4.0, 3.0, 3.0, 2.5)
    force_band_step: float = 0.02
    force_band_retreat: float = 0.02
    # Optional palm-first coverage course.  The course is a directional,
    # bounded prior rather than a terminal-pose replay: opposition/balance /
    # spread are introduced after the short root pre-position, then the
    # straight-finger channels are released.  Zero keeps the historical
    # staged teacher exactly unchanged.
    thumb_first_phase_steps: int = 0
    thumb_first_global_scale: float = 0.35
    thumb_first_coverage_delta: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    motion_smoothing_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.thumb_first_phase_steps < 0:
            raise ValueError("thumb_first_phase_steps must be non-negative")
        if not 0.0 < self.thumb_first_global_scale <= 1.0:
            raise ValueError("thumb_first_global_scale must lie in (0, 1]")
        if not 0.0 < self.motion_smoothing_alpha <= 1.0:
            raise ValueError("motion_smoothing_alpha must lie in (0, 1]")
        if len(self.thumb_first_coverage_delta) != 4:
            raise ValueError("thumb_first_coverage_delta must contain four values")
        if any(abs(float(value)) > 0.30 for value in self.thumb_first_coverage_delta):
            raise ValueError("thumb_first_coverage_delta values must lie in [-0.30, 0.30]")


class EnvelopingTeacher:
    """Vectorized 14D staged teacher with independent distal trim."""

    ACTION_DIM = 14

    def __init__(self, num_envs: int, device: torch.device | str, cfg: EnvelopingTeacherCfg | None = None):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or EnvelopingTeacherCfg()
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.phase_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.palm_streak = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.support_latched = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.trim = torch.zeros((num_envs, 5), device=self.device)
        self.distal_trim = torch.zeros((num_envs, 4), device=self.device)
        # Three nearby P1-feasible branches are blended as weak priors.  The
        # teacher never selects one final pose forever: tactile trim and the
        # policy's residual noise remain active after this approach path.
        root_prefixes = torch.tensor(
            (
                (0.46, -0.18, -0.72, -0.83, 0.94, 0.13, -0.91, 0.35, -0.96, 0.10),
                (0.21, 0.44, -0.12, -0.74, 0.89, -0.25, -0.30, 0.66, -0.83, -0.09),
                (0.48, -0.25, -0.09, -0.56, 1.00, -0.01, -0.73, 0.64, -0.65, -0.04),
            ),
            device=self.device,
        )
        root_prefix = root_prefixes[torch.arange(num_envs, device=self.device) % 3]
        self.root_prior = torch.zeros((num_envs, self.ACTION_DIM), device=self.device)
        self.root_prior[:, :10] = root_prefix
        residual_prefixes = torch.tensor(
            (
                (0.13, -0.91, 0.35, -0.96, 0.10),
                (-0.25, -0.30, 0.66, -0.83, -0.09),
                (-0.01, -0.73, 0.64, -0.65, -0.04),
            ),
            device=self.device,
        )
        self.residual_prior = residual_prefixes[torch.arange(num_envs, device=self.device) % 3].clone()
        open_prefix = torch.tensor(
            (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
            device=self.device,
        )
        self.open_action = torch.zeros((num_envs, self.ACTION_DIM), device=self.device)
        self.open_action[:, :10] = open_prefix
        self.force_action = self.open_action.clone()
        self.previous_action = self.open_action.clone()

    @property
    def thumb_first_enabled(self) -> bool:
        return self.cfg.thumb_first_phase_steps > 0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        with torch.inference_mode(False):
            self.phase = self.phase.clone()
            self.phase_step = self.phase_step.clone()
            self.palm_streak = self.palm_streak.clone()
            self.support_latched = self.support_latched.clone()
            self.trim = self.trim.clone()
            self.distal_trim = self.distal_trim.clone()
            self.force_action = self.force_action.clone()
            self.previous_action = self.previous_action.clone()
            self.phase[ids] = 0
            self.phase_step[ids] = 0
            self.palm_streak[ids] = 0
            self.support_latched[ids] = False
            self.trim[ids] = 0.0
            self.distal_trim[ids] = 0.0
            self.force_action[ids] = self.open_action[ids]
            self.previous_action[ids] = self.open_action[ids]

    def _smooth_action(self, target: torch.Tensor) -> torch.Tensor:
        target = target.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        alpha = float(self.cfg.motion_smoothing_alpha)
        action = self.previous_action + alpha * (target - self.previous_action)
        action = action.clamp(-1.0, 1.0)
        self.previous_action = action
        return action

    @torch.inference_mode()
    def command(
        self,
        digit_force_n: torch.Tensor,
        palm_force_n: torch.Tensor,
        episode_step: torch.Tensor | None = None,
        measured_digit_force_n: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if digit_force_n.shape != (self.num_envs, 5):
            raise ValueError("digit_force_n must have shape (num_envs, 5)")
        if palm_force_n.shape != (self.num_envs,):
            raise ValueError("palm_force_n must have shape (num_envs,)")
        cfg = self.cfg
        supported = palm_force_n >= cfg.palm_support_force_n
        self.support_latched |= supported
        self.palm_streak = torch.where(supported, self.palm_streak + 1, torch.zeros_like(self.palm_streak))
        ready = self.palm_streak >= cfg.support_wait_steps
        self.phase = torch.where((self.phase == 0) & ready, torch.ones_like(self.phase), self.phase)
        self.phase_step = torch.where(self.phase > 0, self.phase_step + 1, self.phase_step)
        root_duration = cfg.root_phase_steps + (cfg.thumb_first_phase_steps if self.thumb_first_enabled else 0)
        self.phase = torch.where(
            (self.phase == 1) & (self.phase_step >= root_duration),
            torch.full_like(self.phase, 2),
            self.phase,
        )
        self.phase = torch.where(
            (self.phase == 2) & (self.phase_step >= root_duration + cfg.proximal_phase_steps),
            torch.full_like(self.phase, 3),
            self.phase,
        )

        root_fraction = (self.phase_step.float() / max(1, root_duration)).clamp(0.0, 1.0)
        proximal_fraction = (
            (self.phase_step.float() - root_duration) / max(1, cfg.proximal_phase_steps)
        ).clamp(0.0, 1.0)
        distal_fraction = (
            (self.phase_step.float() - root_duration - cfg.proximal_phase_steps)
            / max(1, cfg.distal_phase_steps)
        ).clamp(0.0, 1.0)
        action = self.open_action.clone()
        direct_prior = None
        if self.cfg.prior_ramp_steps > 0 and cfg.distal_prior_action is not None:
            direct_prior = torch.cat(
                (
                    self.root_prior[:, :10],
                    digit_force_n.new_tensor(cfg.distal_prior_action).unsqueeze(0).expand(self.num_envs, -1),
                ),
                dim=-1,
            )
            # The deployable actor cannot observe the teacher's private
            # ``phase_step``.  When a complete prior is used for BC, clock the
            # ramp from the environment episode step (which is part of the
            # actor observation) instead of the teacher's hidden palm-streak
            # clock.  The fallback keeps the standalone teacher API backward
            # compatible for unit tests and diagnostics.
            ramp_clock = self.phase_step.float() if episode_step is None else episode_step.float()
            direct_fraction = (ramp_clock / float(cfg.prior_ramp_steps)).clamp(0.0, 1.0)
            action = self.open_action + direct_fraction.unsqueeze(-1) * (direct_prior - self.open_action)
            if cfg.force_band_enabled:
                if measured_digit_force_n is None or measured_digit_force_n.shape != (self.num_envs, 5):
                    raise ValueError("force-band teacher requires measured_digit_force_n with shape (num_envs, 5)")
                measured = measured_digit_force_n[:, 1:]
                ramp_clock = self.phase_step.float() if episode_step is None else episode_step.float()
                ramp_done = ramp_clock >= float(cfg.prior_ramp_steps)
                force_action = torch.where(
                    ramp_done.unsqueeze(-1), self.force_action, action
                )
                lower = measured.new_tensor(cfg.force_band_lower_n)
                upper = measured.new_tensor(cfg.force_band_upper_n)
                step_fraction = float(cfg.force_band_step)
                retreat_fraction = float(cfg.force_band_retreat)
                for digit in range(4):
                    force = measured[:, digit]
                    low = force < lower[digit]
                    high = force >= upper[digit]
                    for channel in (6 + digit, 10 + digit):
                        advance = force_action[:, channel] + step_fraction * (
                            direct_prior[:, channel] - force_action[:, channel]
                        )
                        retreat = force_action[:, channel] + retreat_fraction * (
                            self.open_action[:, channel] - force_action[:, channel]
                        )
                        force_action[:, channel] = torch.where(
                            ramp_done & high,
                            retreat,
                            torch.where(ramp_done & low, advance, force_action[:, channel]),
                        )
                force_action = force_action.clamp(-1.0, 1.0)
                force_action[~self.support_latched] = self.open_action[~self.support_latched]
                self.force_action = force_action
                return self._smooth_action(force_action)
        if direct_prior is None:
            if self.thumb_first_enabled:
                # Explicitly separate the thumb/coverage subspace from the
                # straight-finger subspace.  During the first short root
                # window only the thumb-facing synergy coordinates move.  A
                # small signed coverage delta is then ramped in, after which
                # the remaining root/proximal channels are released.  All
                # values stay bounded and tactile trim remains active below.
                root_fraction = (self.phase_step.float() / max(1, cfg.root_phase_steps)).clamp(0.0, 1.0)
                thumb_fraction = (
                    (self.phase_step.float() - cfg.root_phase_steps)
                    / max(1, cfg.thumb_first_phase_steps)
                ).clamp(0.0, 1.0)
                straight_fraction = proximal_fraction
                action = self.open_action.clone()
                thumb_channels = (0, 2, 3, 4, 5)
                action[:, list(thumb_channels)] = self.open_action[:, list(thumb_channels)] + root_fraction.unsqueeze(-1) * (
                    self.root_prior[:, list(thumb_channels)] - self.open_action[:, list(thumb_channels)]
                )
                action[:, 0] = self.open_action[:, 0] + root_fraction * cfg.thumb_first_global_scale * (
                    self.root_prior[:, 0] - self.open_action[:, 0]
                )
                delta = self.root_prior.new_tensor(cfg.thumb_first_coverage_delta)
                action[:, 2:6] += thumb_fraction.unsqueeze(-1) * delta.unsqueeze(0)
                straight_channels = (1, 6, 7, 8, 9)
                action[:, list(straight_channels)] = self.open_action[:, list(straight_channels)] + straight_fraction.unsqueeze(-1) * (
                    self.root_prior[:, list(straight_channels)] - self.open_action[:, list(straight_channels)]
                )
            else:
                action = torch.where(
                    (self.phase >= 1).unsqueeze(-1),
                    self.open_action + root_fraction.unsqueeze(-1) * (self.root_prior - self.open_action),
                    action,
                )
        else:
            action = torch.where(
                torch.zeros_like(self.phase, dtype=torch.bool).unsqueeze(-1),
                self.open_action,
                action,
            )
        # Keep the per-digit coarse residuals zero during root prepositioning;
        # they are released only after the proximal phase.
        if direct_prior is None:
            if self.thumb_first_enabled:
                # Channel 5 is the thumb residual and belongs to the early
                # opposition window; the four straight-finger residuals stay
                # gated until the proximal phase.
                action[:, 6:10] = torch.where(
                    (self.phase >= 2).unsqueeze(-1), self.residual_prior[:, 1:], torch.zeros_like(self.residual_prior[:, 1:])
                )
            else:
                action[:, 5:10] = torch.where(
                    (self.phase >= 2).unsqueeze(-1), self.residual_prior, torch.zeros_like(self.residual_prior)
                )
        proximal_offsets = torch.tensor((0.08, 0.16, 0.05, 0.04, 0.02), device=self.device)
        if self.thumb_first_enabled:
            # Keep the initial thumb coverage window free of a global curl
            # impulse; the small offsets are released with the proximal
            # finger phase instead.
            proximal_offsets = proximal_offsets * torch.tensor((0.0, 1.0, 1.0, 1.0, 1.0), device=self.device)
        action[:, :5] += proximal_fraction.unsqueeze(-1) * proximal_offsets
        contact = digit_force_n >= cfg.contact_threshold_n
        target = digit_force_n.new_tensor(cfg.target_force_n)
        high = digit_force_n.new_tensor(cfg.high_force_n)
        error = target - digit_force_n
        feedback = torch.where(error.abs() >= cfg.tactile_deadband_n, cfg.tactile_gain * error, torch.zeros_like(error))
        feedback = feedback.clamp(-cfg.max_feedback_step, cfg.max_feedback_step)
        self.trim += feedback * contact.float()
        self.trim -= 0.5 * cfg.max_feedback_step * (digit_force_n > high).float()
        self.trim.clamp_(-0.30, 0.30)
        action[:, 5:10] = (
            action[:, 5:10] + self.trim
        ).clamp(-1.0, 1.0)
        # Independent distal residuals become available only in phase 3.
        # Before contact, distal joints advance through a slow phase ramp;
        # tactile feedback itself is applied only on valid contact channels.
        # This prevents the teacher from integrating a large force error while
        # the fingers are still in free space and then slamming the fruit.
        self.distal_trim += (
            cfg.tactile_gain * error[:, 1:] * contact[:, 1:].float()
        ).clamp(-cfg.max_feedback_step, cfg.max_feedback_step)
        self.distal_trim -= 0.5 * cfg.max_feedback_step * (digit_force_n[:, 1:] > high[1:]).float()
        self.distal_trim.clamp_(-0.65, 0.65)
        if cfg.distal_prior_action is None:
            distal_target = cfg.distal_base_action + self.distal_trim
        else:
            distal_target = digit_force_n.new_tensor(cfg.distal_prior_action).unsqueeze(0) + self.distal_trim
        if direct_prior is None:
            action[:, 10:] = distal_fraction.unsqueeze(-1) * distal_target
        else:
            # Keep the direct prior ramp, then apply only bounded tactile trim
            # so feedback cannot turn the warm start into a pose replay.
            action[:, 10:] = (action[:, 10:] + self.distal_trim).clamp(-1.0, 1.0)
        if direct_prior is None:
            action[self.phase == 0] = self.open_action[self.phase == 0]
        else:
            # Keep the palm-first safety invariant without depending on the
            # hidden teacher phase: a direct-prior ramp is released only when
            # the current observable palm channel has reached support.
            action[~self.support_latched] = self.open_action[~self.support_latched]
        return self._smooth_action(action)


__all__ = ["EnvelopingTeacher", "EnvelopingTeacherCfg"]
