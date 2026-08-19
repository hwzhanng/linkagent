"""Strict, CPU-testable verifier for palm-supported five-finger enclosure.

The verifier is deliberately independent from Isaac Lab and PPO.  It consumes
an episode trace and returns both per-step diagnostics and per-episode gates.
The same function can therefore be used in unit tests, teacher-data audits,
training-time termination, and final evaluation without changing semantics.

Contact geometry is supplied explicitly.  A caller using fingertip positions
as a temporary proxy must set ``contact_geometry_proxy=True`` in its report;
this module does not silently turn missing contact points into success.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


STRAIGHT_FINGER_ROOT_INDICES = (1, 2, 3, 4)
STRAIGHT_FINGER_DISTAL_INDICES = (12, 13, 14, 15)


@dataclass(frozen=True)
class EnvelopingVerifierConfig:
    """Thresholds are simulation priors, not real-device safety limits."""

    palm_up_cos_min: float = math.cos(math.radians(10.0))
    palm_support_force_n: float = 0.50
    digit_force_thresholds_n: tuple[float, float, float, float, float] = (0.20, 0.20, 0.20, 0.20, 0.20)
    palm_support_streak_steps: int = 5
    palm_first_margin_steps: int = 3
    strict_hold_steps: int = 10
    angular_coverage_min_deg: float = 200.0
    thumb_opposition_cos_max: float = -0.35
    same_hemisphere_score_max: float = 0.85
    palm_load_share_min: float = 0.05
    palm_load_share_max: float = 0.70
    distal_participation_delta_rad: float = 0.03
    proximal_dominance_delta_rad: float = 0.15
    pressure_limits_n: tuple[float, float, float, float, float] = (12.0, 10.0, 10.0, 9.0, 9.0)
    require_postbreak_hold: bool = False
    postbreak_hold_steps: int = 20

    def __post_init__(self) -> None:
        if len(self.digit_force_thresholds_n) != 5 or len(self.pressure_limits_n) != 5:
            raise ValueError("digit thresholds and pressure limits must contain five values")
        if self.palm_support_streak_steps <= 0 or self.strict_hold_steps <= 0:
            raise ValueError("streak lengths must be positive")
        if self.postbreak_hold_steps <= 0:
            raise ValueError("postbreak_hold_steps must be positive")
        if not 0.0 <= self.palm_load_share_min <= self.palm_load_share_max <= 1.0:
            raise ValueError("invalid palm load-share range")


def _check_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}")
    for actual, expected in zip(value.shape, shape):
        if expected >= 0 and actual != expected:
            raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
    if not torch.isfinite(torch.nan_to_num(value)).all():
        raise ValueError(f"{name} must contain finite values (use contact_valid for masked points)")


def _first_streak(mask: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first index and maximum streak for a (T,N) boolean mask."""

    if mask.ndim != 2:
        raise ValueError("mask must have shape (T, N)")
    steps, envs = mask.shape
    first = torch.full((envs,), -1, dtype=torch.long, device=mask.device)
    current = torch.zeros(envs, dtype=torch.long, device=mask.device)
    maximum = torch.zeros_like(current)
    for step in range(steps):
        current = torch.where(mask[step], current + 1, torch.zeros_like(current))
        maximum = torch.maximum(maximum, current)
        hit = (first < 0) & (current >= length)
        first = torch.where(hit, torch.full_like(first, step - length + 1), first)
    return first, maximum


def _first_true(mask: torch.Tensor) -> torch.Tensor:
    steps, envs = mask.shape
    indices = torch.arange(steps, device=mask.device).unsqueeze(1).expand(steps, envs)
    masked = torch.where(mask, indices, torch.full_like(indices, steps))
    result = masked.min(dim=0).values
    return torch.where(result < steps, result, torch.full_like(result, -1))


def _contact_geometry_metrics(
    contact_points_w: torch.Tensor,
    contact_valid: torch.Tensor,
    fruit_center_w: torch.Tensor,
    palm_normal_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return coverage angle, opposition cosine, and hemisphere score (T,N)."""

    steps, envs = contact_valid.shape[:2]
    device = contact_points_w.device
    coverage = torch.zeros((steps, envs), dtype=contact_points_w.dtype, device=device)
    opposition = torch.ones_like(coverage)
    hemisphere = torch.ones_like(coverage)
    world_x = torch.zeros((steps, envs, 3), dtype=contact_points_w.dtype, device=device)
    world_x[..., 0] = 1.0
    world_y = torch.zeros_like(world_x)
    world_y[..., 1] = 1.0
    normal = torch.nn.functional.normalize(palm_normal_w, dim=-1)
    basis_u = world_x - (world_x * normal).sum(-1, keepdim=True) * normal
    fallback = torch.linalg.vector_norm(basis_u, dim=-1, keepdim=True) < 1.0e-5
    basis_u = torch.where(
        fallback,
        world_y - (world_y * normal).sum(-1, keepdim=True) * normal,
        basis_u,
    )
    basis_u = torch.nn.functional.normalize(basis_u, dim=-1)
    basis_v = torch.nn.functional.normalize(torch.linalg.cross(normal, basis_u, dim=-1), dim=-1)
    vectors = contact_points_w - fruit_center_w.unsqueeze(-2)
    radial = torch.nn.functional.normalize(vectors, dim=-1)

    for step in range(steps):
        for env in range(envs):
            valid = contact_valid[step, env]
            count = int(valid.sum().item())
            if count < 3:
                continue
            dirs = radial[step, env, valid]
            angle = torch.atan2(
                (dirs * basis_v[step, env]).sum(-1),
                (dirs * basis_u[step, env]).sum(-1),
            )
            angle = torch.sort(angle).values
            gaps = torch.diff(angle)
            wrap = angle[0] + 2.0 * math.pi - angle[-1]
            max_gap = torch.cat((gaps, wrap.reshape(1))).max()
            coverage[step, env] = (2.0 * math.pi - max_gap) * 180.0 / math.pi
            mean_other = torch.nn.functional.normalize(dirs[1:].mean(0), dim=0)
            opposition[step, env] = torch.dot(dirs[0], mean_other)
            hemisphere[step, env] = torch.linalg.vector_norm(dirs.mean(0))
    return coverage, opposition, hemisphere


def verify_enveloping_trace(
    *,
    palm_normal_w: torch.Tensor,
    fruit_center_w: torch.Tensor,
    palm_force_n: torch.Tensor,
    digit_force_n: torch.Tensor,
    contact_points_w: torch.Tensor,
    contact_valid: torch.Tensor,
    joint_pos_rad: torch.Tensor,
    cfg: EnvelopingVerifierConfig | None = None,
    pressure_safe: torch.Tensor | None = None,
    damage_ok: torch.Tensor | None = None,
    dropped: torch.Tensor | None = None,
    postbreak_start_step: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Verify a batch of episode traces.

    All time-series tensors use shape ``(T, N, ...)``.  ``postbreak_start_step``
    is optional because P2/P5 can verify the enclosure before a stem exists;
    when ``cfg.require_postbreak_hold`` is true it becomes mandatory.
    """

    cfg = cfg or EnvelopingVerifierConfig()
    if palm_normal_w.ndim != 3 or palm_normal_w.shape[-1] != 3:
        raise ValueError("palm_normal_w must have shape (T, N, 3)")
    steps, envs = palm_normal_w.shape[:2]
    _check_shape("fruit_center_w", fruit_center_w, (steps, envs, 3))
    _check_shape("palm_force_n", palm_force_n, (steps, envs))
    _check_shape("digit_force_n", digit_force_n, (steps, envs, 5))
    if contact_points_w.shape != (steps, envs, 5, 3):
        raise ValueError("contact_points_w must have shape (T, N, 5, 3)")
    if contact_valid.shape != (steps, envs, 5) or contact_valid.dtype != torch.bool:
        raise ValueError("contact_valid must be boolean with shape (T, N, 5)")
    _check_shape("joint_pos_rad", joint_pos_rad, (steps, envs, 16))
    if pressure_safe is None:
        pressure_safe = torch.ones((steps, envs), dtype=torch.bool, device=palm_force_n.device)
    if damage_ok is None:
        damage_ok = torch.ones_like(pressure_safe)
    if dropped is None:
        dropped = torch.zeros_like(pressure_safe)
    for name, value in (("pressure_safe", pressure_safe), ("damage_ok", damage_ok), ("dropped", dropped)):
        if value.shape != (steps, envs) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean with shape (T, N)")
    if cfg.require_postbreak_hold and postbreak_start_step is None:
        raise ValueError("postbreak_start_step is required when postbreak hold is enabled")
    if postbreak_start_step is None:
        postbreak_start_step = torch.full((envs,), -1, dtype=torch.long, device=palm_force_n.device)
    if postbreak_start_step.shape != (envs,):
        raise ValueError("postbreak_start_step must have shape (N,)")

    up = palm_normal_w.new_tensor((0.0, 0.0, 1.0))
    palm_up_cos = (torch.nn.functional.normalize(palm_normal_w, dim=-1) * up).sum(-1)
    palm_up_ok = palm_up_cos >= cfg.palm_up_cos_min
    palm_supported = palm_force_n >= cfg.palm_support_force_n
    thresholds = palm_force_n.new_tensor(cfg.digit_force_thresholds_n)
    digit_active = contact_valid & (digit_force_n >= thresholds)
    simultaneous = digit_active.sum(-1)
    all_five = simultaneous == 5
    first_palm, palm_support_max_streak = _first_streak(palm_supported, cfg.palm_support_streak_steps)
    first_three = _first_true(simultaneous >= 3)
    first_five = _first_true(all_five)
    palm_first_margin = torch.where(
        (first_palm >= 0) & (first_three >= 0),
        first_three - first_palm,
        torch.full_like(first_palm, -1),
    )

    coverage, opposition, hemisphere = _contact_geometry_metrics(
        contact_points_w, contact_valid, fruit_center_w, palm_normal_w
    )
    total_digit_force = digit_force_n.sum(-1)
    palm_load_share = palm_force_n / (palm_force_n + total_digit_force).clamp_min(1.0e-6)
    max_pressure = palm_force_n.new_tensor(cfg.pressure_limits_n)
    force_safe_now = (digit_force_n <= max_pressure).all(-1)
    force_safe = pressure_safe & force_safe_now

    joint_delta = (joint_pos_rad - joint_pos_rad[0:1]).abs().amax(0)
    root_delta = joint_delta[:, list(STRAIGHT_FINGER_ROOT_INDICES)]
    distal_delta = joint_delta[:, list(STRAIGHT_FINGER_DISTAL_INDICES)]
    distal_participation = distal_delta >= cfg.distal_participation_delta_rad
    proximal_only = (
        (root_delta.mean(-1) >= cfg.proximal_dominance_delta_rad)
        & (distal_delta.mean(-1) < cfg.distal_participation_delta_rad)
        & (coverage.amax(0) < cfg.angular_coverage_min_deg)
    )

    step_ok = (
        palm_up_ok
        & palm_supported
        & all_five
        & (coverage >= cfg.angular_coverage_min_deg)
        & (opposition <= cfg.thumb_opposition_cos_max)
        & (hemisphere <= cfg.same_hemisphere_score_max)
        & (palm_load_share >= cfg.palm_load_share_min)
        & (palm_load_share <= cfg.palm_load_share_max)
        & force_safe
        & damage_ok
        & ~dropped
    )
    _, envelope_hold_max = _first_streak(step_ok, cfg.strict_hold_steps)
    # Geometry metrics are undefined before three fingers provide a usable
    # radial span.  Their neutral defaults (coverage=0, hemisphere=1) must not
    # poison episode maxima/minima during the intentional staged approach.
    geometry_eligible = simultaneous >= 3
    coverage_for_episode = torch.where(geometry_eligible, coverage, torch.zeros_like(coverage))
    opposition_for_episode = torch.where(geometry_eligible, opposition, torch.ones_like(opposition))
    hemisphere_for_episode = torch.where(geometry_eligible, hemisphere, torch.zeros_like(hemisphere))
    postbreak_hold_max = torch.zeros_like(envelope_hold_max)
    postbreak_ok = torch.ones(envs, dtype=torch.bool, device=palm_force_n.device)
    if cfg.require_postbreak_hold:
        postbreak_ok = torch.zeros_like(postbreak_ok)
        for env in range(envs):
            start = int(postbreak_start_step[env])
            if start < 0 or start >= steps:
                continue
            _, postbreak_hold_max[env] = _first_streak(step_ok[start:, env : env + 1], cfg.postbreak_hold_steps)
            postbreak_ok[env] = postbreak_hold_max[env] >= cfg.postbreak_hold_steps

    strict_success = (
        (first_palm >= 0)
        & (first_three >= 0)
        & (palm_first_margin >= cfg.palm_first_margin_steps)
        & (envelope_hold_max >= cfg.strict_hold_steps)
        & (coverage_for_episode.amax(0) >= cfg.angular_coverage_min_deg)
        & (opposition_for_episode.amin(0) <= cfg.thumb_opposition_cos_max)
        & step_ok.any(0)
        & distal_participation.all(-1)
        & ~proximal_only
        & postbreak_ok
    )

    reasons: list[list[str]] = []
    for env in range(envs):
        env_reasons: list[str] = []
        if first_palm[env] < 0:
            env_reasons.append("palm_support_missing")
        if first_three[env] >= 0 and palm_first_margin[env] < cfg.palm_first_margin_steps:
            env_reasons.append("palm_not_first")
        if envelope_hold_max[env] < cfg.strict_hold_steps:
            env_reasons.append("envelope_hold_short")
        if coverage_for_episode[:, env].max() < cfg.angular_coverage_min_deg:
            env_reasons.append("under_enclosure")
        if (opposition_for_episode[:, env].min() > cfg.thumb_opposition_cos_max):
            env_reasons.append("thumb_opposition_missing")
        if distal_participation[env].logical_not().any():
            env_reasons.append("distal_joint_idle")
        if proximal_only[env]:
            env_reasons.append("proximal_only_closure")
        if not force_safe[:, env].all():
            env_reasons.append("overload_or_impact")
        if dropped[:, env].any():
            env_reasons.append("drop_or_slip")
        if cfg.require_postbreak_hold and not postbreak_ok[env]:
            env_reasons.append("postbreak_hold_short")
        reasons.append(env_reasons or ["pass"])

    return {
        "strict_envelop_success": strict_success,
        "palm_up_rate": palm_up_ok.all(0),
        "palm_support_first_step": first_palm,
        "first_three_digit_step": first_three,
        "first_all_five_step": first_five,
        "palm_first_margin_steps": palm_first_margin,
        "palm_support_max_streak": palm_support_max_streak,
        "envelope_hold_max_streak": envelope_hold_max,
        "postbreak_hold_max_streak": postbreak_hold_max,
        "max_angular_coverage_deg": coverage_for_episode.amax(0),
        "min_thumb_opposition_cos": opposition_for_episode.amin(0),
        "max_same_hemisphere_score": hemisphere_for_episode.amax(0),
        "palm_load_share_mean": palm_load_share.mean(0),
        "palm_load_share_min": palm_load_share.amin(0),
        "distal_joint_participation": distal_participation,
        "root_joint_delta_rad": root_delta,
        "distal_joint_delta_rad": distal_delta,
        "proximal_only_closure": proximal_only,
        "step_ok": step_ok,
        "failure_reasons": reasons,
        "contact_geometry_proxy": False,
    }


__all__ = [
    "EnvelopingVerifierConfig",
    "STRAIGHT_FINGER_DISTAL_INDICES",
    "STRAIGHT_FINGER_ROOT_INDICES",
    "verify_enveloping_trace",
]
