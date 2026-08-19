"""PhysX environment variant for strict palm-supported five-finger enclosure.

This module is intentionally separate from the archived 10D tomato runner.
It reuses the validated scene/physics setup, opts into the 14D action adapter,
and adds an online geometry gate based on filtered PhysX contact positions.
Isaac Sim 4.5 exposes finite ``contact_pos_w`` data when contact-point
tracking is enabled, but contact manifolds can be truncated in large batches;
the environment therefore averages finite manifold points and explicitly
falls back to fingertip kinematics per digit when a measured point is absent.
The verifier remains authoritative and every report exposes the measured-point
rate instead of silently treating the fallback as a real contact point.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from simulation.ppo_tomato_grasp.enveloping_verifier import _contact_geometry_metrics
from simulation.ppo_tomato_grasp.physx_tomato_env import (
    PhysxTomatoGraspEnv,
    PhysxTomatoGraspEnvCfg,
)


@configclass
class EnvelopingPhysxTomatoGraspEnvCfg(PhysxTomatoGraspEnvCfg):
    """G0/G1 configuration; all values remain simulation-only priors."""

    action_space = 14
    strict_enveloping_enabled = True
    strict_enveloping_require_stem_break = False
    training_stage = 1
    episode_length_s = 10.0
    strict_angular_coverage_min_deg = 200.0
    strict_hold_steps = 10
    strict_palm_support_streak_steps = 5
    strict_palm_first_margin_steps = 3
    strict_distal_participation_delta_rad = 0.03
    strict_proximal_dominance_delta_rad = 0.15
    # G0 uses the P1 central work area and a broad enough size band to expose
    # the 70--100 mm geometry without allowing unreachable edge positions.
    tomato_rest_xy = (0.040, 0.000)
    position_jitter = (0.002, 0.002, 0.001)
    strict_reward_weight = 2.0


class EnvelopingPhysxTomatoGraspEnv(PhysxTomatoGraspEnv):
    """14D hand-only PhysX task with an online strict enclosure gate."""

    cfg: EnvelopingPhysxTomatoGraspEnvCfg

    def __init__(self, cfg: EnvelopingPhysxTomatoGraspEnvCfg, render_mode: str | None = None, **kwargs):
        # DirectRLEnv may reset once during the parent constructor.  The
        # guarded reset hook below therefore initializes strict buffers only
        # after the parent scene and articulation have been constructed.
        super().__init__(cfg, render_mode, **kwargs)
        envs = self.num_envs
        device = self.device
        with torch.inference_mode(False):
            self.strict_first_palm = torch.full((envs,), -1, dtype=torch.long, device=device)
            self.strict_first_three = torch.full((envs,), -1, dtype=torch.long, device=device)
            self.strict_palm_streak = torch.zeros(envs, dtype=torch.long, device=device)
            self.strict_hold_streak = torch.zeros(envs, dtype=torch.long, device=device)
            self.strict_hold_max = torch.zeros(envs, dtype=torch.long, device=device)
            self.strict_coverage_max = torch.zeros(envs, device=device)
            self.strict_opposition_min = torch.ones(envs, device=device)
            self.strict_hemisphere_max = torch.ones(envs, device=device)
            # Unlike coverage (where larger is better), the hemisphere score
            # is a *minimum* target.  Keep a separate minimum diagnostic;
            # the historical ``strict_hemisphere_max`` buffer is retained for
            # checkpoint/state compatibility but must not be used as a pass
            # metric (it was initialized to 1 and therefore could never pass).
            self.strict_hemisphere_min = torch.ones(envs, device=device)
            self.strict_coverage_current = torch.zeros(envs, device=device)
            self.strict_opposition_current = torch.ones(envs, device=device)
            self.strict_hemisphere_current = torch.ones(envs, device=device)
            self.strict_contact_geometry_measured = torch.zeros(
                (envs, 5), dtype=torch.bool, device=device
            )
            self.strict_distal_participation = torch.zeros((envs, 4), dtype=torch.bool, device=device)
            self.strict_proximal_only = torch.zeros(envs, dtype=torch.bool, device=device)
            self.strict_success_latched = torch.zeros(envs, dtype=torch.bool, device=device)
            self.strict_initial_joint_pos = self.robot.data.joint_pos[:, self.active_joint_ids].detach().clone()
        # Parent reset has already placed the hand at a valid open pose.  Run
        # the guarded reset once more so all strict buffers match that pose.
        self._reset_idx(self.robot._ALL_INDICES)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "strict_first_palm"):
            return
        with torch.inference_mode(False):
            # Isaac Lab invokes reset hooks from an inference context.  Clone
            # the bookkeeping buffers once here so later partial resets are
            # ordinary mutable tensors rather than inference tensors.
            self.strict_first_palm = self.strict_first_palm.clone()
            self.strict_first_three = self.strict_first_three.clone()
            self.strict_palm_streak = self.strict_palm_streak.clone()
            self.strict_hold_streak = self.strict_hold_streak.clone()
            self.strict_hold_max = self.strict_hold_max.clone()
            self.strict_coverage_max = self.strict_coverage_max.clone()
            self.strict_opposition_min = self.strict_opposition_min.clone()
            self.strict_hemisphere_max = self.strict_hemisphere_max.clone()
            self.strict_hemisphere_min = self.strict_hemisphere_min.clone()
            self.strict_coverage_current = self.strict_coverage_current.clone()
            self.strict_opposition_current = self.strict_opposition_current.clone()
            self.strict_hemisphere_current = self.strict_hemisphere_current.clone()
            self.strict_contact_geometry_measured = self.strict_contact_geometry_measured.clone()
            self.strict_distal_participation = self.strict_distal_participation.clone()
            self.strict_proximal_only = self.strict_proximal_only.clone()
            self.strict_success_latched = self.strict_success_latched.clone()
            self.strict_initial_joint_pos = self.strict_initial_joint_pos.clone()
            self.strict_first_palm[env_ids] = -1
            self.strict_first_three[env_ids] = -1
            self.strict_palm_streak[env_ids] = 0
            self.strict_hold_streak[env_ids] = 0
            self.strict_hold_max[env_ids] = 0
            self.strict_coverage_max[env_ids] = 0.0
            self.strict_opposition_min[env_ids] = 1.0
            self.strict_hemisphere_max[env_ids] = 1.0
            self.strict_hemisphere_min[env_ids] = 1.0
            self.strict_coverage_current[env_ids] = 0.0
            self.strict_opposition_current[env_ids] = 1.0
            self.strict_hemisphere_current[env_ids] = 1.0
            self.strict_contact_geometry_measured[env_ids] = False
            self.strict_distal_participation[env_ids] = False
            self.strict_proximal_only[env_ids] = False
            self.strict_success_latched[env_ids] = False
            self.strict_initial_joint_pos[env_ids] = self.robot.data.joint_pos[env_ids][:, self.active_joint_ids]

    @torch.inference_mode()
    def _update_strict_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        forces = self.contact_forces
        palm_support = self.palm_force >= self.cfg.palm_support_force_n
        digit_active = forces >= self.cfg.contact_threshold_n
        simultaneous = digit_active.sum(-1)
        step = self.episode_length_buf.to(torch.long)
        self.strict_palm_streak = torch.where(
            palm_support, self.strict_palm_streak + 1, torch.zeros_like(self.strict_palm_streak)
        )
        first_palm_now = (self.strict_first_palm < 0) & (
            self.strict_palm_streak >= self.cfg.strict_palm_support_streak_steps
        )
        self.strict_first_palm = torch.where(first_palm_now, step - self.cfg.strict_palm_support_streak_steps + 1, self.strict_first_palm)
        first_three_now = (self.strict_first_three < 0) & (simultaneous >= 3)
        self.strict_first_three = torch.where(first_three_now, step, self.strict_first_three)

        center = self.tomato.data.root_pos_w
        tips, measured_points_valid = self._geometry_contact_points()
        normal = torch.zeros((self.num_envs, 3), device=self.device)
        normal[:, 2] = 1.0
        coverage, opposition, hemisphere = _contact_geometry_metrics(
            tips.unsqueeze(0), digit_active.unsqueeze(0), center.unsqueeze(0), normal.unsqueeze(0)
        )
        coverage = coverage[0]
        opposition = opposition[0]
        hemisphere = hemisphere[0]
        self.strict_coverage_current = coverage
        self.strict_opposition_current = opposition
        self.strict_hemisphere_current = hemisphere
        self.strict_contact_geometry_measured = measured_points_valid
        self.strict_coverage_max = torch.maximum(self.strict_coverage_max, coverage)
        self.strict_opposition_min = torch.minimum(self.strict_opposition_min, opposition)
        self.strict_hemisphere_max = torch.maximum(self.strict_hemisphere_max, hemisphere)
        eligible_geometry = simultaneous >= 3
        self.strict_hemisphere_min = torch.where(
            eligible_geometry,
            torch.minimum(self.strict_hemisphere_min, hemisphere),
            self.strict_hemisphere_min,
        )

        joint = self.robot.data.joint_pos[:, self.active_joint_ids]
        delta = (joint - self.strict_initial_joint_pos).abs()
        distal = delta[:, [12, 13, 14, 15]] >= self.cfg.strict_distal_participation_delta_rad
        self.strict_distal_participation |= distal
        root_delta = delta[:, [1, 2, 3, 4]]
        distal_delta = delta[:, [12, 13, 14, 15]]
        # A staged enclosure is expected to pass through a proximal-only
        # *phase*.  Do not latch that transient as an episode failure: the
        # final verifier compares the completed joint excursion and coverage,
        # and only rejects a closure that remains proximal-only at evaluation.
        self.strict_proximal_only = (
            (root_delta.mean(-1) >= self.cfg.strict_proximal_dominance_delta_rad)
            & (distal_delta.mean(-1) < self.cfg.strict_distal_participation_delta_rad)
            & (self.strict_coverage_max < self.cfg.strict_angular_coverage_min_deg)
        )

        load_share = self.palm_force / (self.palm_force + forces.sum(-1)).clamp_min(1.0e-6)
        force_safe = (forces <= self.cfg.safe_force_n).all(-1) & (
            self.palm_force <= 1.25 * self.cfg.safe_force_n
        )
        strict_now = (
            (normal[:, 2] >= self.cfg.strict_palm_up_cos_min)
            & palm_support
            & (simultaneous == 5)
            & (coverage >= self.cfg.strict_angular_coverage_min_deg)
            & (opposition <= self.cfg.strict_thumb_opposition_cos_max)
            & (hemisphere <= self.cfg.strict_same_hemisphere_score_max)
            & (load_share >= self.cfg.strict_palm_load_share_min)
            & (load_share <= self.cfg.strict_palm_load_share_max)
            & force_safe
            & ~self.dropped
        )
        self.strict_hold_streak = torch.where(
            strict_now, self.strict_hold_streak + 1, torch.zeros_like(self.strict_hold_streak)
        )
        self.strict_hold_max = torch.maximum(self.strict_hold_max, self.strict_hold_streak)
        margin = self.strict_first_three - self.strict_first_palm
        distal_ready = self.strict_distal_participation.all(-1) | self.cfg.retention_only
        ready = (
            (self.strict_first_palm >= 0)
            & (self.strict_first_three >= 0)
            & (margin >= self.cfg.strict_palm_first_margin_steps)
            & (self.strict_hold_streak >= self.cfg.strict_hold_steps)
            & distal_ready
            & ~self.strict_proximal_only
        )
        if self.cfg.strict_enveloping_require_stem_break:
            ready &= self.stem_broke
        newly = ready & ~self.strict_success_latched
        self.strict_success_latched |= ready
        return strict_now, newly

    @torch.inference_mode()
    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        if not self.cfg.strict_enveloping_enabled:
            return reward
        strict_now, newly = self._update_strict_geometry()
        # Parent rewards provide dense proximity/contact shaping.  This extra
        # term makes the final objective geometry-aware without replacing the
        # well-tested physics and actuator terms.  Crucially, parent success is
        # overwritten by the strict latch before _get_dones runs.
        # Dense topology shaping closes the gap between ordinary five-touch
        # and a true enclosure.  Geometry is evaluated as a progress signal,
        # never as a replacement for the strict latch below.
        eligible = (self.contact_forces >= self.cfg.contact_threshold_n).sum(-1) >= 3
        coverage_progress = (
            self.strict_coverage_current / max(float(self.cfg.strict_angular_coverage_min_deg), 1.0)
        ).clamp(0.0, 1.0)
        opposition_progress = (-self.strict_opposition_current).clamp(0.0, 1.0)
        # The pass threshold is intentionally strict (<= 0.85), but using
        # ``threshold - score`` as the only shaping signal gives a completely
        # same-side contact pattern (score ~= 1) zero gradient.  Reward the
        # full radial spread instead: it is zero for coincident directions,
        # grows continuously as contacts occupy the opposite hemisphere, and
        # still leaves the verifier threshold authoritative for success.
        hemisphere_progress = (1.0 - self.strict_hemisphere_current).clamp(0.0, 1.0)
        # Make the four straight-finger distal DOFs useful before their
        # thresholded participation flag flips.  A continuous excursion
        # signal avoids the old proximal-only local optimum and preserves the
        # hard all-four gate in the verifier.
        current_joint = self.robot.data.joint_pos[:, self.active_joint_ids]
        distal_delta = (current_joint - self.strict_initial_joint_pos).abs()[:, [12, 13, 14, 15]]
        distal_progress = (
            distal_delta / max(float(self.cfg.strict_distal_participation_delta_rad), 1.0e-3)
        ).clamp(0.0, 1.0).mean(-1)
        topology_shape = eligible.float() * (
            0.45 * coverage_progress
            + 0.25 * opposition_progress
            + 0.40 * hemisphere_progress
            + 0.20 * distal_progress
        )
        reward = reward + 0.8 * topology_shape
        # Keep the distal DOFs learnable even when the full strict latch is
        # still sparse.  This is gated by at least three active fingers and
        # scales with actual joint excursion, so free-space oscillation does
        # not earn reward and the hard all-four participation gate remains in
        # force for success.
        reward = reward + self.cfg.strict_distal_reward_weight * eligible.float() * distal_progress
        reward = reward + self.cfg.strict_reward_weight * strict_now.float()
        reward = reward + self.cfg.success_bonus * newly.float()
        self.success = self.strict_success_latched.clone()
        log = self.extras.setdefault("log", {})
        log["strict_envelop_success_rate"] = self.strict_success_latched.float().mean()
        # ``DirectRLEnv.step`` resets successful environments immediately
        # after this method.  Keep an event-valued metric so collectors and
        # PPO diagnostics do not mistake the post-reset latch for a failure.
        log["strict_new_success_rate"] = newly.float().mean()
        log["strict_palm_first_rate"] = (self.strict_first_palm >= 0).float().mean()
        log["strict_hold_rate"] = (self.strict_hold_max >= self.cfg.strict_hold_steps).float().mean()
        log["strict_hold_max_mean"] = self.strict_hold_max.float().mean()
        log["strict_hold_max"] = self.strict_hold_max.max()
        log["strict_coverage_mean_deg_proxy"] = self.strict_coverage_max.mean()
        log["strict_coverage_max_deg_proxy"] = self.strict_coverage_max.max()
        log["strict_first_three_rate"] = (self.strict_first_three >= 0).float().mean()
        log["strict_distal_all_rate"] = self.strict_distal_participation.all(-1).float().mean()
        log["strict_opposition_pass_rate"] = (self.strict_opposition_min <= self.cfg.strict_thumb_opposition_cos_max).float().mean()
        log["strict_hemisphere_pass_rate"] = (
            self.strict_hemisphere_min <= self.cfg.strict_same_hemisphere_score_max
        ).float().mean()
        log["strict_contact_geometry_measured_rate"] = self.strict_contact_geometry_measured.float().mean()
        current_load_share = self.palm_force / (self.palm_force + self.contact_forces.sum(-1)).clamp_min(1.0e-6)
        log["strict_load_share_pass_rate"] = (
            (current_load_share >= self.cfg.strict_palm_load_share_min)
            & (current_load_share <= self.cfg.strict_palm_load_share_max)
        ).float().mean()
        log["strict_force_safe_rate"] = (
            (self.contact_forces <= self.cfg.safe_force_n).all(-1)
            & (self.palm_force <= 1.25 * self.cfg.safe_force_n)
        ).float().mean()
        log["strict_all_five_rate"] = (
            (self.contact_forces >= self.cfg.contact_threshold_n).sum(-1) == 5
        ).float().mean()
        log["strict_proximal_only_rate"] = self.strict_proximal_only.float().mean()
        log["strict_distal_participation_index"] = self.strict_distal_participation[:, 0].float().mean()
        log["strict_distal_participation_middle"] = self.strict_distal_participation[:, 1].float().mean()
        log["strict_distal_participation_ring"] = self.strict_distal_participation[:, 2].float().mean()
        log["strict_distal_participation_little"] = self.strict_distal_participation[:, 3].float().mean()
        return reward


__all__ = ["EnvelopingPhysxTomatoGraspEnv", "EnvelopingPhysxTomatoGraspEnvCfg"]
