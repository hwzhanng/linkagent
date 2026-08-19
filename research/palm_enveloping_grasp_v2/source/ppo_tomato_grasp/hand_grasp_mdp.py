"""Pure-Torch reward and success state machine for gentle hand-only grasping.

The damage score is a bounded simulation diagnostic, not a probability of
real tomato damage.  Numerical guards are deliberately asymmetric by digit
and require sustained exposure or severe overload before termination.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


REWARD_VERSION = "gentle_enclosure_v2_action_and_joint_soft_limits"
FINGER_FORCE_CEILING_N = (6.0, 7.0, 7.0, 6.0, 5.0)
PALM_FORCE_CEILING_N = 8.0
CONTACT_THRESHOLD_N = 0.25
PALM_CONTACT_THRESHOLD_N = 0.25
MAXIMUM_CENTER_DISTANCE_M = 0.055
MAXIMUM_RELATIVE_SPEED_M_S = 0.08
SUCCESS_HOLD_STEPS = 20
DAMAGE_SUCCESS_LIMIT = 0.75
SEVERE_OVERLOAD_FACTOR = 1.25


@dataclass(frozen=True)
class HandGraspState:
    best_proximity: torch.Tensor
    best_contact: torch.Tensor
    best_opposition: torch.Tensor
    best_enclosure: torch.Tensor
    best_retention: torch.Tensor
    best_hold: torch.Tensor
    hold_streak: torch.Tensor
    previous_force_n: torch.Tensor
    peak_force_n: torch.Tensor
    force_exposure_n_s: torch.Tensor
    impact_exposure_n: torch.Tensor
    damage_score: torch.Tensor


@dataclass(frozen=True)
class HandGraspTransition:
    reward: torch.Tensor
    state: HandGraspState
    success: torch.Tensor
    terminated: torch.Tensor
    metrics: dict[str, torch.Tensor]
    reward_terms: dict[str, torch.Tensor]


def initial_hand_grasp_state(reference: torch.Tensor) -> HandGraspState:
    """Allocate state for a batch identified by a one-dimensional tensor."""

    if not isinstance(reference, torch.Tensor) or reference.ndim != 1:
        raise ValueError("reference must be a one-dimensional tensor")
    if not torch.isfinite(reference).all():
        raise ValueError("reference must be finite")
    scalar = torch.zeros_like(reference)
    force = torch.zeros((*reference.shape, 5), device=reference.device, dtype=reference.dtype)
    return HandGraspState(
        best_proximity=scalar.clone(),
        best_contact=scalar.clone(),
        best_opposition=scalar.clone(),
        best_enclosure=scalar.clone(),
        best_retention=scalar.clone(),
        best_hold=scalar.clone(),
        hold_streak=torch.zeros_like(reference, dtype=torch.long),
        previous_force_n=force.clone(),
        peak_force_n=force.clone(),
        force_exposure_n_s=force.clone(),
        impact_exposure_n=force.clone(),
        damage_score=scalar.clone(),
    )


def _progress(score: torch.Tensor, previous_best: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bounded = score.clamp(0.0, 1.0)
    next_best = torch.maximum(previous_best.clamp(0.0, 1.0), bounded)
    return next_best - previous_best, next_best


def _validate_inputs(
    state: HandGraspState,
    surface_gap_m: torch.Tensor,
    physical_finger_force_n: torch.Tensor,
    physical_palm_force_n: torch.Tensor,
    center_distance_m: torch.Tensor,
    relative_speed_m_s: torch.Tensor,
    slip_detected: torch.Tensor,
    dropped: torch.Tensor,
    self_collision: torch.Tensor,
    action: torch.Tensor,
    action_delta: torch.Tensor,
    normalized_joint_limit_proximity: torch.Tensor,
) -> None:
    batch = state.best_proximity.shape
    scalar_tensors = {
        "physical_palm_force_n": physical_palm_force_n,
        "center_distance_m": center_distance_m,
        "relative_speed_m_s": relative_speed_m_s,
        "slip_detected": slip_detected,
        "dropped": dropped,
        "self_collision": self_collision,
        "normalized_joint_limit_proximity": normalized_joint_limit_proximity,
    }
    if surface_gap_m.shape != (*batch, 5):
        raise ValueError("surface_gap_m must have shape (N, 5)")
    if physical_finger_force_n.shape != (*batch, 5):
        raise ValueError("physical_finger_force_n must have shape (N, 5)")
    if action_delta.shape != (*batch, 10):
        raise ValueError("action_delta must have shape (N, 10)")
    if action.shape != (*batch, 10):
        raise ValueError("action must have shape (N, 10)")
    for name, value in scalar_tensors.items():
        if value.shape != batch:
            raise ValueError(f"{name} must have shape {batch}")
    finite = (
        torch.isfinite(surface_gap_m).all()
        and torch.isfinite(physical_finger_force_n).all()
        and torch.isfinite(physical_palm_force_n).all()
        and torch.isfinite(center_distance_m).all()
        and torch.isfinite(relative_speed_m_s).all()
        and torch.isfinite(action).all()
        and torch.isfinite(action_delta).all()
        and torch.isfinite(normalized_joint_limit_proximity).all()
    )
    if not finite:
        raise ValueError("MDP inputs must be finite")


def hand_grasp_transition(
    state: HandGraspState,
    *,
    surface_gap_m: torch.Tensor,
    physical_finger_force_n: torch.Tensor,
    physical_palm_force_n: torch.Tensor,
    center_distance_m: torch.Tensor,
    relative_speed_m_s: torch.Tensor,
    slip_detected: torch.Tensor,
    dropped: torch.Tensor,
    self_collision: torch.Tensor,
    action: torch.Tensor,
    action_delta: torch.Tensor,
    normalized_joint_limit_proximity: torch.Tensor,
    dt_s: float = 0.05,
    force_ceiling_n: tuple[float, ...] = FINGER_FORCE_CEILING_N,
    palm_force_ceiling_n: float = PALM_FORCE_CEILING_N,
    contact_threshold_n: float = CONTACT_THRESHOLD_N,
    success_hold_steps: int = SUCCESS_HOLD_STEPS,
) -> HandGraspTransition:
    """Advance the bounded progress reward and physical success oracle."""

    _validate_inputs(
        state,
        surface_gap_m,
        physical_finger_force_n,
        physical_palm_force_n,
        center_distance_m,
        relative_speed_m_s,
        slip_detected,
        dropped,
        self_collision,
        action,
        action_delta,
        normalized_joint_limit_proximity,
    )
    if dt_s <= 0.0 or success_hold_steps < 1:
        raise ValueError("dt_s and success_hold_steps must be positive")
    if len(force_ceiling_n) != 5 or any(value <= 0.0 for value in force_ceiling_n):
        raise ValueError("force_ceiling_n must contain five positive values")
    if palm_force_ceiling_n <= 0.0 or contact_threshold_n <= 0.0:
        raise ValueError("force thresholds must be positive")

    force = physical_finger_force_n.clamp_min(0.0)
    palm_force = physical_palm_force_n.clamp_min(0.0)
    ceiling = force.new_tensor(force_ceiling_n)
    normalized_contact = (force / contact_threshold_n).clamp(0.0, 1.0)
    contact = force >= contact_threshold_n
    thumb_contact = contact[:, 0]
    nonthumb_count = contact[:, 1:].sum(dim=1)
    finger_support = thumb_contact & (nonthumb_count >= 2)
    palm_contact = palm_force >= PALM_CONTACT_THRESHOLD_N
    palm_support = palm_contact & thumb_contact & (nonthumb_count >= 1)
    secure_enclosure = finger_support | palm_support

    raw_proximity = torch.exp(-surface_gap_m.clamp_min(0.0) / 0.020).mean(dim=1)
    # A distant reset must not earn a one-off positive reward merely because
    # exp(-distance) is mathematically nonzero.
    proximity_score = ((raw_proximity - 0.05) / 0.95).clamp(0.0, 1.0)
    contact_score = torch.topk(normalized_contact, k=3, dim=1).values.mean(dim=1)
    strongest_two_nonthumb = torch.topk(normalized_contact[:, 1:], k=2, dim=1).values
    finger_enclosure_score = normalized_contact[:, 0] * strongest_two_nonthumb.mean(dim=1)
    palm_score = (palm_force / PALM_CONTACT_THRESHOLD_N).clamp(0.0, 1.0)
    palm_enclosure_score = (
        normalized_contact[:, 0]
        * normalized_contact[:, 1:].amax(dim=1)
        * palm_score
    )
    opposition_score = normalized_contact[:, 0] * normalized_contact[:, 1:].amax(dim=1)
    enclosure_score = torch.maximum(finger_enclosure_score, palm_enclosure_score)

    force_rate = (force - state.previous_force_n).clamp_min(0.0) / dt_s
    peak_force = torch.maximum(state.peak_force_n, force)
    safe_exposure_start = 0.65 * ceiling
    force_exposure = state.force_exposure_n_s + (force - safe_exposure_start).clamp_min(0.0) * dt_s
    impact_increment = (force_rate - 30.0).clamp_min(0.0) * dt_s
    impact_exposure = state.impact_exposure_n + impact_increment
    peak_ratio = (
        (peak_force - safe_exposure_start)
        / ((SEVERE_OVERLOAD_FACTOR - 0.65) * ceiling)
    ).clamp(0.0, 1.0)
    exposure_budget = 0.50 * ceiling
    exposure_ratio = (force_exposure / exposure_budget).clamp(0.0, 1.0)
    impact_ratio = (impact_exposure / (1.5 * ceiling)).clamp(0.0, 1.0)
    per_finger_damage = 0.50 * peak_ratio + 0.35 * exposure_ratio + 0.15 * impact_ratio
    damage_score = torch.maximum(state.damage_score, per_finger_damage.amax(dim=1))

    within_force_limit = (force <= ceiling).all(dim=1) & (palm_force <= palm_force_ceiling_n)
    severe_overload = (force >= SEVERE_OVERLOAD_FACTOR * ceiling).any(dim=1)
    severe_overload |= palm_force >= SEVERE_OVERLOAD_FACTOR * palm_force_ceiling_n
    kinematic_retained = (
        (center_distance_m <= MAXIMUM_CENTER_DISTANCE_M)
        & (relative_speed_m_s <= MAXIMUM_RELATIVE_SPEED_M_S)
    )
    retained = (
        secure_enclosure
        & kinematic_retained
        & within_force_limit
        & ~slip_detected.bool()
        & (damage_score < DAMAGE_SUCCESS_LIMIT)
        & ~dropped.bool()
        & ~self_collision.bool()
    )
    hold_streak = torch.where(retained, state.hold_streak + 1, torch.zeros_like(state.hold_streak))
    success = retained & (hold_streak >= success_hold_steps)
    terminated = success | dropped.bool() | self_collision.bool() | severe_overload | (damage_score >= 1.0)

    kinematic_score = torch.exp(-center_distance_m / 0.035) * torch.exp(-relative_speed_m_s / 0.050)
    retention_score = enclosure_score * kinematic_score * (~slip_detected.bool()).float()
    hold_score = (hold_streak.float() / float(success_hold_steps)).clamp(0.0, 1.0)

    proximity_progress, best_proximity = _progress(proximity_score, state.best_proximity)
    contact_progress, best_contact = _progress(contact_score, state.best_contact)
    opposition_progress, best_opposition = _progress(opposition_score, state.best_opposition)
    enclosure_progress, best_enclosure = _progress(enclosure_score, state.best_enclosure)
    retention_progress, best_retention = _progress(retention_score, state.best_retention)
    hold_progress, best_hold = _progress(hold_score, state.best_hold)
    damage_increment = (damage_score - state.damage_score).clamp_min(0.0)
    overload_ratio = ((force - ceiling).clamp_min(0.0) / ceiling).mean(dim=1)
    action_limit_proximity = ((action.abs() - 0.80).clamp_min(0.0) / 0.20).clamp_max(1.0)
    joint_limit_proximity = normalized_joint_limit_proximity.clamp(0.0, 1.0)

    terms = {
        "proximity_progress": 2.0 * proximity_progress,
        "contact_progress": 4.0 * contact_progress,
        "opposition_progress": 6.0 * opposition_progress,
        "enclosure_progress": 8.0 * enclosure_progress,
        "retention_progress": 8.0 * retention_progress,
        "hold_progress": 10.0 * hold_progress,
        "success": 30.0 * success.float(),
        "time_cost": -0.02 * (~terminated).float(),
        "action_rate": -0.005 * action_delta.square().mean(dim=1),
        "action_effort": -0.02 * action.square().mean(dim=1),
        "action_soft_limit": -0.50 * action_limit_proximity.square().mean(dim=1),
        "joint_soft_limit": -1.00 * joint_limit_proximity.square(),
        "damage_increment": -15.0 * damage_increment,
        "overload": -4.0 * overload_ratio,
        "slip": -2.0 * slip_detected.float(),
        "drop": -20.0 * dropped.float(),
        "self_collision": -25.0 * self_collision.float(),
        "severe_overload": -20.0 * severe_overload.float(),
    }
    reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
    next_state = HandGraspState(
        best_proximity=best_proximity,
        best_contact=best_contact,
        best_opposition=best_opposition,
        best_enclosure=best_enclosure,
        best_retention=best_retention,
        best_hold=best_hold,
        hold_streak=hold_streak,
        previous_force_n=force,
        peak_force_n=peak_force,
        force_exposure_n_s=force_exposure,
        impact_exposure_n=impact_exposure,
        damage_score=damage_score,
    )
    metrics = {
        "proximity_score": proximity_score,
        "contact_score": contact_score,
        "opposition_score": opposition_score,
        "enclosure_score": enclosure_score,
        "retention_score": retention_score,
        "finger_support": finger_support,
        "palm_support": palm_support,
        "secure_enclosure": secure_enclosure,
        "kinematic_retained": kinematic_retained,
        "within_force_limit": within_force_limit,
        "severe_overload": severe_overload,
        "damage_score": damage_score,
        "hold_streak": hold_streak,
    }
    return HandGraspTransition(
        reward=reward,
        state=next_state,
        success=success,
        terminated=terminated,
        metrics=metrics,
        reward_terms=terms,
    )


__all__ = [
    "CONTACT_THRESHOLD_N",
    "DAMAGE_SUCCESS_LIMIT",
    "FINGER_FORCE_CEILING_N",
    "HandGraspState",
    "HandGraspTransition",
    "MAXIMUM_CENTER_DISTANCE_M",
    "MAXIMUM_RELATIVE_SPEED_M_S",
    "PALM_FORCE_CEILING_N",
    "REWARD_VERSION",
    "SEVERE_OVERLOAD_FACTOR",
    "SUCCESS_HOLD_STEPS",
    "hand_grasp_transition",
    "initial_hand_grasp_state",
]
