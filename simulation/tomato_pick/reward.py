"""Outcome-aligned hierarchical reward for tomato grasp-and-detach PPO."""

from __future__ import annotations

import torch


# Progress is paid once while an episode remains alive, then fully recovered on
# every terminal failure.  This is the second-stage objective for a mature
# warm-start policy: it preserves dense temporal credit within successful
# trajectories without making reach, contact, loading, or partial retention a
# profitable terminal outcome of their own.
REACH_PROGRESS_WEIGHT = 5.0
GRASP_PROGRESS_WEIGHT = 10.0
STEM_LOAD_PROGRESS_WEIGHT = 20.0
STABLE_DETACH_WEIGHT = 40.0
RETENTION_PROGRESS_WEIGHT = 20.0
SUCCESS_WEIGHT = 100.0
TERMINAL_FAILURE_WEIGHT = -100.0
SHAPING_GAMMA = 0.99
# With this cost on every non-terminal transition, waiting T steps and then
# failing has discounted return exactly TERMINAL_FAILURE_WEIGHT for every T:
#   sum(gamma**t * -1, t=0..T-1) + gamma**T * -100 == -100.
# Safe timeout therefore cannot dominate an earlier physical attempt merely
# because PPO discounts delayed negative outcomes.
TIME_COST_WEIGHT = (1.0 - SHAPING_GAMMA) * TERMINAL_FAILURE_WEIGHT


def best_ever_progress(
    score: torch.Tensor,
    previous_best: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-repeatable progress of a bounded score and its new maximum."""

    bounded_score = score.clamp(0.0, 1.0)
    bounded_best = previous_best.clamp(0.0, 1.0)
    next_best = torch.maximum(bounded_best, bounded_score)
    return next_best - bounded_best, next_best


def milestone_credit_term(
    score: torch.Tensor,
    previous_best: torch.Tensor,
    *,
    weight: float,
    terminal_failure: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pay bounded best-ever progress once, except on a failing transition."""

    progress, next_best = best_ever_progress(score, previous_best)
    credit = weight * progress
    paid = torch.where(terminal_failure.bool(), torch.zeros_like(credit), credit)
    return paid, next_best


def update_recoverable_credit(
    previous_balance: torch.Tensor,
    milestone_credit: torch.Tensor,
    *,
    success: torch.Tensor,
    terminal_failure: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return failure recovery and the next gamma-correct credit balance.

    Before step ``t``, ``previous_balance`` is the amount whose payment at
    step ``t`` exactly cancels all earlier milestone rewards under gamma.
    Non-terminal credit therefore advances as ``(balance + credit) / gamma``.
    On failure the old balance is recovered immediately; progress from the
    same failing transition was never paid.  Success keeps every milestone.
    """

    terminal = success.bool() | terminal_failure.bool()
    recovery = torch.where(
        terminal_failure.bool(), -previous_balance, torch.zeros_like(previous_balance)
    )
    next_balance = torch.where(
        terminal,
        torch.zeros_like(previous_balance),
        (previous_balance + milestone_credit) / SHAPING_GAMMA,
    )
    return recovery, next_balance


def opposed_contact_score(
    finger_force_n: torch.Tensor,
    *,
    full_contact_force_n: float,
) -> torch.Tensor:
    """Soft thumb-plus-one-support-finger grasp score in ``[0, 1]``.

    This is the learning-time ``good finger contact`` condition used by mature
    dexterous-grasp tasks.  It is deliberately less restrictive than the final
    task outcome: a two-sided contact is learnable, while detach-and-retain is
    still the only way to receive the large task reward.
    """

    if finger_force_n.ndim != 2 or finger_force_n.shape[1] != 5:
        raise ValueError("finger_force_n must have shape (num_envs, 5)")
    if full_contact_force_n <= 0.0:
        raise ValueError("full_contact_force_n must be positive")
    normalized = (finger_force_n / full_contact_force_n).clamp(0.0, 1.0)
    thumb = normalized[:, 0]
    strongest_support = normalized[:, 1:].amax(dim=1)
    return thumb * strongest_support


def grasp_contact_score(
    finger_force_n: torch.Tensor,
    *,
    full_contact_force_n: float,
) -> torch.Tensor:
    """Continuous contact-count plus opposed-grasp quality in ``[0, 1]``.

    Individual finger contact supplies a small early learning signal, as in
    NVIDIA's DexSuite.  Most of the score is reserved for thumb opposition, so
    pushing with non-thumb fingers cannot look like a completed grasp.
    """

    if finger_force_n.ndim != 2 or finger_force_n.shape[1] != 5:
        raise ValueError("finger_force_n must have shape (num_envs, 5)")
    if full_contact_force_n <= 0.0:
        raise ValueError("full_contact_force_n must be positive")
    normalized = (finger_force_n / full_contact_force_n).clamp(0.0, 1.0)
    contact_count_score = normalized.mean(dim=1)
    opposed = opposed_contact_score(
        finger_force_n, full_contact_force_n=full_contact_force_n
    )
    return 0.2 * contact_count_score + 0.8 * opposed


def break_grasp_quality(
    finger_force_n: torch.Tensor,
    *,
    contact_force_n: float,
) -> torch.Tensor:
    """Quality of the hand contact at the instant the stem breaks.

    A single fingertip collision receives no detach credit.  Contact from any
    two fingers receives partial credit, while thumb opposition receives full
    credit.  This mirrors the good-contact definition used by mature
    dexterous-grasp tasks without prescribing a particular finger pair.
    """

    if finger_force_n.ndim != 2 or finger_force_n.shape[1] != 5:
        raise ValueError("finger_force_n must have shape (num_envs, 5)")
    if contact_force_n <= 0.0:
        raise ValueError("contact_force_n must be positive")
    contact = finger_force_n >= contact_force_n
    any_two = contact.sum(dim=1) >= 2
    thumb_opposed = contact[:, 0] & contact[:, 1:].any(dim=1)
    return 0.4 * any_two.float() + 0.6 * thumb_opposed.float()


def tomato_pick_reward(
    *,
    reach_score: torch.Tensor,
    previous_best_reach: torch.Tensor,
    grasp_score: torch.Tensor,
    previous_best_grasp: torch.Tensor,
    stem_load_score: torch.Tensor,
    previous_best_stem_load: torch.Tensor,
    stable_detach_score: torch.Tensor,
    previous_best_stable_detach: torch.Tensor,
    retention_score: torch.Tensor,
    previous_best_retention: torch.Tensor,
    previous_credit_balance: torch.Tensor,
    success: torch.Tensor,
    terminal_failure: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Compute the complete minimal-v3 task reward.

    Best-ever state prevents oscillation farming.  Every milestone remains
    dense on successful trajectories and is recovered exactly on failure.  A
    matched time cost makes a no-progress timeout equal to every other failure.
    """

    terminal = success.bool() | terminal_failure.bool()
    reach_term, next_reach = milestone_credit_term(
        reach_score,
        previous_best_reach,
        weight=REACH_PROGRESS_WEIGHT,
        terminal_failure=terminal_failure,
    )
    grasp_term, next_grasp = milestone_credit_term(
        grasp_score,
        previous_best_grasp,
        weight=GRASP_PROGRESS_WEIGHT,
        terminal_failure=terminal_failure,
    )
    stem_load_term, next_stem_load = milestone_credit_term(
        stem_load_score,
        previous_best_stem_load,
        weight=STEM_LOAD_PROGRESS_WEIGHT,
        terminal_failure=terminal_failure,
    )
    stable_detach_term, next_stable_detach = milestone_credit_term(
        stable_detach_score,
        previous_best_stable_detach,
        weight=STABLE_DETACH_WEIGHT,
        terminal_failure=terminal_failure,
    )
    retention_term, next_retention = milestone_credit_term(
        retention_score,
        previous_best_retention,
        weight=RETENTION_PROGRESS_WEIGHT,
        terminal_failure=terminal_failure,
    )
    # All shaping is temporary credit.  Keeping early reach/grasp/load credit on
    # failed trajectories made the mature policy trade thumb opposition for
    # faster approach/pull actions, increasing both drop and excess force.
    recoverable_credit = (
        reach_term
        + grasp_term
        + stem_load_term
        + stable_detach_term
        + retention_term
    )
    credit_recovery, next_credit_balance = update_recoverable_credit(
        previous_credit_balance,
        recoverable_credit,
        success=success,
        terminal_failure=terminal_failure,
    )
    terms = {
        "time_cost": torch.where(
            terminal, torch.zeros_like(reach_term), torch.full_like(reach_term, TIME_COST_WEIGHT)
        ),
        "reach_progress": reach_term,
        "grasp_progress": grasp_term,
        "stem_load_progress": stem_load_term,
        # Detach itself is not rewarded.  Credit appears only after the fruit
        # has remained stably held with opposed contact for several policy
        # steps, making shallow collision-induced breaks unprofitable.
        "stable_detach": stable_detach_term,
        "retention_progress": retention_term,
        "credit_recovery": credit_recovery,
        "success": SUCCESS_WEIGHT * success.float(),
        "terminal_failure": TERMINAL_FAILURE_WEIGHT * terminal_failure.float(),
    }
    reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
    return (
        reward,
        next_reach,
        next_grasp,
        next_stem_load,
        next_stable_detach,
        next_retention,
        next_credit_balance,
        terms,
    )


__all__ = [
    "GRASP_PROGRESS_WEIGHT",
    "REACH_PROGRESS_WEIGHT",
    "RETENTION_PROGRESS_WEIGHT",
    "SHAPING_GAMMA",
    "STABLE_DETACH_WEIGHT",
    "STEM_LOAD_PROGRESS_WEIGHT",
    "SUCCESS_WEIGHT",
    "TERMINAL_FAILURE_WEIGHT",
    "TIME_COST_WEIGHT",
    "best_ever_progress",
    "break_grasp_quality",
    "grasp_contact_score",
    "opposed_contact_score",
    "milestone_credit_term",
    "tomato_pick_reward",
    "update_recoverable_credit",
]
