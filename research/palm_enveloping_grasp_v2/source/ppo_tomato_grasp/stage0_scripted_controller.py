"""Force-feedback scripted controller for the Stage-0 PhysX gate.

The ten-dimensional pose below is a weak geometry prior discovered in the
converted left-hand workspace.  The environment resets the joints at the true
open-manifold pose, then its calibrated joint-rate limiter safely moves toward
the zero-action pregrasp and this prior.  The prior is never executed as an
open-loop final pose: coarse closure freezes at the first complete enclosure
and each digit residual then follows its own contact force.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Stage0ControllerCfg:
    contact_threshold_n: float = 0.12
    palm_support_force_n: float = 0.25
    target_force_n: float = 0.65
    # The opposed thumb carries a larger share of the lateral equilibrium;
    # treating every reporter channel as an identical force setpoint pushes
    # the fruit away from the thumb side on this hand geometry.
    target_force_scale: tuple[float, float, float, float, float] = (1.0,) * 5
    high_force_n: float = 1.50
    progress_per_step: float = 0.020
    minimum_progress_per_step: float = 0.008
    tiny_progress_per_step: float = 0.020
    size_condition_low_m: float = 0.032
    size_condition_high_m: float = 0.035
    prelatch_seek_increment: float = 0.003
    seek_increment: float = 0.0005
    backoff_increment: float = 0.002
    force_feedback_gain_per_n: float = 0.0
    force_feedback_deadband_n: float = 0.08
    force_feedback_max_increment: float = 0.015
    one_sided_force_feedback: bool = False
    retention_seek_increment: float = 0.01
    retention_all_digit_increment: float = 0.0
    retention_global_increment: float = 0.0
    retention_global_limit: float = 0.25
    retention_center_deadband_m: float = 0.050
    retention_center_increment: float = 0.0
    retention_balance_increment: float = 0.0
    retention_balance_limit: float = 0.40
    missing_hysteresis_steps: int = 4
    enclosure_latch_steps: int = 1
    trim_limit: float = 0.45


class Stage0ScriptedController:
    """Vectorized coarse-to-contact controller with independent digit trim."""

    def __init__(self, num_envs: int, device: torch.device | str, cfg: Stage0ControllerCfg | None = None):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.cfg = cfg or Stage0ControllerCfg()
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.geometry_prior = torch.tensor(
            (
                0.2671942,
                -0.0992608,
                0.0454599,
                -0.8494038,
                1.0,
                -0.1388997,
                -0.8474188,
                0.7692336,
                -0.9704037,
                -0.3135286,
            ),
            device=self.device,
        ).expand(num_envs, -1).clone()
        # Radius-32 mm endpoint from a safe 64-environment PhysX search
        # (five digits + palm, 20-step stage-1 hold).  Interpolation between
        # this and ``geometry_prior`` is a weak size prior, not a final pose:
        # per-digit tactile search and force feedback remain active.
        self.small_geometry_prior = torch.tensor(
            (
                0.1623066,
                -0.2356111,
                0.00530165,
                -0.8699145,
                0.9777843,
                -0.0491881,
                -0.7238308,
                0.5984016,
                -0.8817820,
                -0.1348379,
            ),
            device=self.device,
        ).expand(num_envs, -1).clone()
        # 29--30 mm CEM reachability search, retained only as a weak approach
        # path. The search found five-digit+palm contact candidates, while the
        # controller still requires live tactile latching and independent
        # digit correction before it may hold or pick.
        self.tiny_geometry_prior = torch.tensor(
            (
                0.3039185,
                -0.2679365,
                0.2610287,
                -0.8804674,
                1.0000000,
                -0.1864282,
                -0.7756650,
                0.0020886,
                -0.7655179,
                -0.2075170,
            ),
            device=self.device,
        ).expand(num_envs, -1).clone()
        self.medium_geometry_prior = torch.tensor(
            (
                0.1623066,
                -0.2356111,
                0.00530165,
                -0.8699145,
                0.9777843,
                -0.0491881,
                -0.7238308,
                0.5984016,
                -0.8817820,
                -0.1348379,
            ),
            device=self.device,
        ).expand(num_envs, -1).clone()
        self.open_action = torch.tensor(
            (-1.00, 0.00, 0.00, 0.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00),
            device=self.device,
        ).expand(num_envs, -1).clone()
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset all controller state, or only environments reset by PhysX."""
        if not hasattr(self, "progress"):
            self.progress = torch.zeros(self.num_envs, device=self.device)
            self.trim = torch.zeros((self.num_envs, 5), device=self.device)
            self.enclosure_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.complete_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.hold_action = torch.zeros((self.num_envs, 10), device=self.device)
            self.missing_steps = torch.zeros((self.num_envs, 5), dtype=torch.long, device=self.device)
            self.retention_boost = torch.zeros(self.num_envs, device=self.device)
            self.balance_boost = torch.zeros(self.num_envs, device=self.device)
            return
        ids = slice(None) if env_ids is None else env_ids
        self.progress[ids] = 0.0
        self.trim[ids] = 0.0
        self.enclosure_latched[ids] = False
        self.complete_steps[ids] = 0
        self.hold_action[ids] = 0.0
        self.missing_steps[ids] = 0
        self.retention_boost[ids] = 0.0
        self.balance_boost[ids] = 0.0

    @torch.inference_mode()
    def command(
        self,
        digit_force_n: torch.Tensor,
        palm_force_n: torch.Tensor,
        retention_mode: torch.Tensor | None = None,
        center_error_m: torch.Tensor | None = None,
        object_radius_m: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if digit_force_n.shape != (self.num_envs, 5):
            raise ValueError("digit_force_n must have shape (num_envs, 5)")
        if palm_force_n.shape != (self.num_envs,):
            raise ValueError("palm_force_n must have shape (num_envs,)")
        if retention_mode is None:
            retention_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if retention_mode.shape != (self.num_envs,):
            raise ValueError("retention_mode must have shape (num_envs,)")
        if center_error_m is not None and center_error_m.shape != (self.num_envs, 3):
            raise ValueError("center_error_m must have shape (num_envs, 3)")
        if object_radius_m is not None and object_radius_m.shape != (self.num_envs,):
            raise ValueError("object_radius_m must have shape (num_envs,)")

        cfg = self.cfg
        contact = digit_force_n >= cfg.contact_threshold_n
        complete = contact.all(dim=-1) & (palm_force_n >= cfg.palm_support_force_n)

        closing = ~self.enclosure_latched
        if object_radius_m is None:
            progress_increment = torch.full_like(self.progress, cfg.progress_per_step)
            geometry_prior = self.geometry_prior
        else:
            size_fraction = (
                (object_radius_m - cfg.size_condition_low_m)
                / (cfg.size_condition_high_m - cfg.size_condition_low_m)
            ).clamp(0.0, 1.0)
            progress_increment = cfg.minimum_progress_per_step + size_fraction * (
                cfg.progress_per_step - cfg.minimum_progress_per_step
            )
            tiny = object_radius_m < 0.031
            progress_increment = torch.where(
                tiny,
                torch.full_like(progress_increment, cfg.tiny_progress_per_step),
                progress_increment,
            )
            small = object_radius_m < 0.03275
            large = object_radius_m >= 0.03425
            geometry_prior = torch.where(
                tiny.unsqueeze(-1),
                self.tiny_geometry_prior,
                torch.where(
                    small.unsqueeze(-1),
                    self.small_geometry_prior,
                    torch.where(
                        large.unsqueeze(-1),
                        self.geometry_prior,
                        self.medium_geometry_prior,
                    ),
                ),
            )
        self.progress = torch.where(
            closing,
            (self.progress + progress_increment).clamp_max(1.0),
            self.progress,
        )
        coarse = geometry_prior * self.progress.unsqueeze(-1)

        # Once the safe coarse path has completed, independently advance only
        # missing digits.  Previously an ellipsoid that did not happen to put
        # all five reporters over threshold on the fixed prior could wait
        # forever, even though the 10D contract explicitly provides one
        # residual per digit.
        prelatch_search = (~self.enclosure_latched) & (self.progress >= 1.0)
        self.trim += (
            cfg.prelatch_seek_increment
            * prelatch_search.unsqueeze(-1).float()
            * (~contact).float()
        )
        self.trim -= (
            cfg.backoff_increment
            * prelatch_search.unsqueeze(-1).float()
            * (digit_force_n > cfg.high_force_n).float()
        )
        self.trim.clamp_(-cfg.trim_limit, cfg.trim_limit)
        coarse_adjusted = coarse.clone()
        coarse_adjusted[:, 5:] = (
            coarse_adjusted[:, 5:] + self.trim
        ).clamp(-1.0, 1.0)

        # Do not chase isolated contacts during the approach: doing so changes
        # the coupled path and can sweep the freely resting fruit off the palm.
        # The calibrated coarse path runs only until the first simultaneous
        # enclosure; independent tactile feedback begins after that latch.

        self.complete_steps = torch.where(
            complete & ~self.enclosure_latched,
            self.complete_steps + 1,
            torch.zeros_like(self.complete_steps),
        )
        newly_complete = (
            self.complete_steps >= cfg.enclosure_latch_steps
        ) & ~self.enclosure_latched
        if bool(newly_complete.any()):
            # Preserve the per-digit tactile search that actually produced
            # the complete enclosure.  Latching ``coarse`` here discarded
            # those residuals on the very next frame and could release a
            # just-found contact before or immediately after stem break.
            self.hold_action[newly_complete] = coarse_adjusted[newly_complete]
            self.enclosure_latched |= newly_complete

        hold = self.hold_action.clone()
        missing = (~contact) & self.enclosure_latched.unsqueeze(-1)
        self.missing_steps = torch.where(
            missing, self.missing_steps + 1, torch.zeros_like(self.missing_steps)
        )
        hold_seek = (self.missing_steps >= cfg.missing_hysteresis_steps) | (
            missing & retention_mode.unsqueeze(-1)
        )
        hold_backoff = (digit_force_n > cfg.high_force_n) & self.enclosure_latched.unsqueeze(-1)
        seek_increment = torch.where(
            retention_mode,
            torch.full_like(palm_force_n, cfg.retention_seek_increment),
            torch.full_like(palm_force_n, cfg.seek_increment),
        )
        # Once an enclosure has been found, balance each digit around the
        # tactile target instead of merely waiting for a hard overload.  The
        # old controller defined ``target_force_n`` but never used it, so a
        # strong thumb could keep pushing the fruit out of the opposing
        # fingers.  Contact loss still uses the explicit seek path below.
        target_force = digit_force_n.new_tensor(cfg.target_force_scale) * cfg.target_force_n
        force_error = target_force - digit_force_n
        feedback_active = (
            self.enclosure_latched.unsqueeze(-1)
            & ~newly_complete.unsqueeze(-1)
            & contact
            & (force_error.abs() > cfg.force_feedback_deadband_n)
        )
        feedback_delta = (
            cfg.force_feedback_gain_per_n * force_error
        ).clamp(
            -cfg.force_feedback_max_increment,
            cfg.force_feedback_max_increment,
        )
        if cfg.one_sided_force_feedback:
            feedback_delta.clamp_min_(0.0)
        self.trim += feedback_delta * feedback_active.float()
        self.trim += seek_increment.unsqueeze(-1) * hold_seek.float()
        self.trim += (
            cfg.retention_all_digit_increment
            * retention_mode.unsqueeze(-1).float()
            * self.enclosure_latched.unsqueeze(-1).float()
        )
        if center_error_m is not None:
            # On the imported left hand the thumb approaches from +x and the
            # four opposing fingers from -x.  Use the simulator/vision centre
            # estimate to remove slow lateral drift after stem release.  This
            # complements tactile force balance: force magnitude alone has no
            # information about the direction in which a sphere is rolling.
            x_error = center_error_m[:, 0]
            centered = x_error.abs() <= cfg.retention_center_deadband_m
            direction = torch.sign(x_error) * (~centered).float() * retention_mode.float()
            center_step = cfg.retention_center_increment * direction
            self.trim[:, 0] += center_step
            self.trim[:, 1:] -= center_step.unsqueeze(-1) / 3.0
            y_error = center_error_m[:, 1]
            y_outside = y_error.abs() > cfg.retention_center_deadband_m
            self.balance_boost += (
                torch.sign(y_error)
                * y_outside.float()
                * retention_mode.float()
                * cfg.retention_balance_increment
            )
            self.balance_boost.clamp_(
                -cfg.retention_balance_limit, cfg.retention_balance_limit
            )
        self.trim -= cfg.backoff_increment * hold_backoff.float()
        self.trim.clamp_(-cfg.trim_limit, cfg.trim_limit)
        hold[:, 5:] = (self.hold_action[:, 5:] + self.trim).clamp(-1.0, 1.0)
        self.retention_boost = torch.where(
            retention_mode,
            (self.retention_boost + cfg.retention_global_increment).clamp_max(
                cfg.retention_global_limit
            ),
            self.retention_boost,
        )
        hold[:, 0] = (hold[:, 0] + self.retention_boost).clamp(-1.0, 1.0)
        hold[:, 2] = (hold[:, 2] + self.balance_boost).clamp(-1.0, 1.0)
        return torch.where(
            self.enclosure_latched.unsqueeze(-1), hold, coarse_adjusted
        ).clamp(-1.0, 1.0)


__all__ = ("Stage0ControllerCfg", "Stage0ScriptedController")
