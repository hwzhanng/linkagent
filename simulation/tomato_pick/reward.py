"""Curriculum reward for from-scratch tomato-picking PPO.

The discovery endpoint preserves the positive hierarchy that made the first
successful policy learnable.  The aligned endpoint keeps the bounded front-end
progress, but replaces immediate detach and repeated hold credit with stable
detach, best-ever hold progress, and a much larger final outcome margin.

The curriculum is a smooth function of PPO update count.  It is deliberately
zero until the historical first-success regime has been passed, so late
failure penalties cannot teach a random policy to avoid the fruit.
"""

from __future__ import annotations

import torch


SHAPING_GAMMA = 0.99

# Discovery endpoint: the seven-term positive hierarchy documented by the run
# that produced model_1450.  All progress scores are bounded in [0, 1].
REACH_PROGRESS_WEIGHT = 5.0
GRASP_PROGRESS_WEIGHT = 10.0
STEM_LOAD_PROGRESS_WEIGHT = 20.0
BREAK_GRASPED_WEIGHT = 40.0
DISCOVERY_SUCCESS_WEIGHT = 100.0
DISCOVERY_TIME_COST_WEIGHT = -0.01

# Aligned endpoint.  Front-end reach/grasp/load progress remains available;
# only the vulnerable detach/hold/outcome channel changes.
OPPOSED_PROGRESS_WEIGHT = 5.0
STABLE_DETACH_WEIGHT = 40.0
RETENTION_PROGRESS_WEIGHT = 40.0
ALIGNED_SUCCESS_WEIGHT = 200.0
ALIGNED_TIME_COST_WEIGHT = -0.10
DROP_FAILURE_WEIGHT = -40.0
EXCESS_FORCE_FAILURE_WEIGHT = -60.0
OUT_OF_BOUNDS_FAILURE_WEIGHT = -60.0
PRE_DETACH_TIMEOUT_WEIGHT = -10.0
POST_DETACH_TIMEOUT_WEIGHT = -30.0

# Update-count schedule.  Update 29 was the first historical success, while
# useful success density appeared only hundreds of updates later.  Therefore
# the objective is exactly the discovery endpoint through update 250, changes
# smoothly through update 1300, and remains fully aligned thereafter.
CURRICULUM_POINTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (250.0, 0.0),
    (600.0, 0.35),
    (1000.0, 0.75),
    (1300.0, 1.0),
)


def best_ever_progress(
    score: torch.Tensor,
    previous_best: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-repeatable progress and the updated bounded best score."""

    bounded_score = score.clamp(0.0, 1.0)
    bounded_best = previous_best.clamp(0.0, 1.0)
    next_best = torch.maximum(bounded_best, bounded_score)
    return next_best - bounded_best, next_best


def alignment_alpha_for_iteration(iteration: float) -> float:
    """Return a smooth, monotone reward-alignment coefficient in [0, 1]."""

    value = max(0.0, float(iteration))
    for (left_iteration, left_alpha), (right_iteration, right_alpha) in zip(
        CURRICULUM_POINTS[:-1], CURRICULUM_POINTS[1:]
    ):
        if value <= right_iteration:
            if right_iteration == left_iteration:
                return float(right_alpha)
            fraction = (value - left_iteration) / (right_iteration - left_iteration)
            fraction = min(1.0, max(0.0, fraction))
            # Smoothstep avoids a gradient-scale corner at curriculum knots.
            fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            return float(left_alpha + fraction * (right_alpha - left_alpha))
    return float(CURRICULUM_POINTS[-1][1])


def opposed_contact_score(
    finger_force_n: torch.Tensor,
    *,
    full_contact_force_n: float,
) -> torch.Tensor:
    """Soft thumb-plus-one-support-finger contact score in [0, 1]."""

    if finger_force_n.ndim != 2 or finger_force_n.shape[1] != 5:
        raise ValueError("finger_force_n must have shape (num_envs, 5)")
    if full_contact_force_n <= 0.0:
        raise ValueError("full_contact_force_n must be positive")
    normalized = (finger_force_n / full_contact_force_n).clamp(0.0, 1.0)
    return normalized[:, 0] * normalized[:, 1:].amax(dim=1)


def grasp_contact_score(
    finger_force_n: torch.Tensor,
    *,
    full_contact_force_n: float,
) -> torch.Tensor:
    """Continuous score of the second-strongest fingertip contact.

    Reach covers the first contact.  This term therefore rewards adding a
    second finger without requiring that it already be the thumb; opposition
    is represented separately and receives its own progress credit.
    """

    normalized = (finger_force_n / full_contact_force_n).clamp(0.0, 1.0)
    return torch.topk(normalized, k=2, dim=1).values[:, 1]


def break_grasp_quality(
    finger_force_n: torch.Tensor,
    *,
    contact_force_n: float,
) -> torch.Tensor:
    """Return detach grasp quality in {0, 0.4, 1}."""

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
    opposed_score: torch.Tensor,
    previous_best_opposed: torch.Tensor,
    stem_load_score: torch.Tensor,
    previous_best_stem_load: torch.Tensor,
    stable_detach_score: torch.Tensor,
    previous_best_stable_detach: torch.Tensor,
    retention_score: torch.Tensor,
    previous_best_retention: torch.Tensor,
    break_grasped_score: torch.Tensor,
    success: torch.Tensor,
    dropped: torch.Tensor,
    excess_force: torch.Tensor,
    out_of_bounds: torch.Tensor,
    timed_out: torch.Tensor,
    stem_detached: torch.Tensor,
    alignment_alpha: float,
    break_grasped_alignment_scale: float | None = None,
    time_cost_alignment_scale: float | None = None,
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
    """Compute one transition of the Clean-v5 curriculum objective."""

    if not 0.0 <= alignment_alpha <= 1.0:
        raise ValueError("alignment_alpha must lie in [0, 1]")

    reach_progress, next_reach = best_ever_progress(
        reach_score, previous_best_reach
    )
    grasp_progress, next_grasp = best_ever_progress(
        grasp_score, previous_best_grasp
    )
    opposed_progress, next_opposed = best_ever_progress(
        opposed_score, previous_best_opposed
    )
    stem_progress, next_stem = best_ever_progress(
        stem_load_score, previous_best_stem_load
    )
    stable_detach_progress, next_stable_detach = best_ever_progress(
        stable_detach_score, previous_best_stable_detach
    )
    retention_progress, next_retention = best_ever_progress(
        retention_score, previous_best_retention
    )

    alpha = float(alignment_alpha)
    discovery = 1.0 - alpha
    if break_grasped_alignment_scale is None:
        break_credit_scale = discovery
    else:
        break_credit_scale = float(break_grasped_alignment_scale)
        if not 0.0 <= break_credit_scale <= 1.0:
            raise ValueError(
                "break_grasped_alignment_scale must lie in [0, 1]"
            )
    if time_cost_alignment_scale is None:
        time_cost_alpha = alpha
    else:
        time_cost_alpha = float(time_cost_alignment_scale)
        if not 0.0 <= time_cost_alpha <= 1.0:
            raise ValueError("time_cost_alignment_scale must lie in [0, 1]")
    time_cost_discovery = 1.0 - time_cost_alpha
    success_weight = (
        discovery * DISCOVERY_SUCCESS_WEIGHT + alpha * ALIGNED_SUCCESS_WEIGHT
    )
    timeout_weight = torch.where(
        stem_detached.bool(),
        torch.full_like(reach_score, POST_DETACH_TIMEOUT_WEIGHT),
        torch.full_like(reach_score, PRE_DETACH_TIMEOUT_WEIGHT),
    )
    terminal = (
        success.bool()
        | dropped.bool()
        | excess_force.bool()
        | out_of_bounds.bool()
        | timed_out.bool()
    )
    terms = {
        # This bounded front-end bridge never disappears, so contact remains
        # distinguishable from waiting even after outcome alignment is strong.
        "reach_progress": REACH_PROGRESS_WEIGHT * reach_progress,
        "grasp_progress": GRASP_PROGRESS_WEIGHT * grasp_progress,
        "stem_load_progress": STEM_LOAD_PROGRESS_WEIGHT * stem_progress,
        # Discovery pays maintained opposition and the actual break event.
        "break_grasped": break_credit_scale
        * BREAK_GRASPED_WEIGHT
        * break_grasped_score.clamp(0.0, 1.0),
        # Opposition and hold duration are new-best progress from the first
        # update onward.  A first correct trajectory still receives the same
        # dense increments; repeated, regressed, or oscillating states get 0.
        "opposed_progress": OPPOSED_PROGRESS_WEIGHT * opposed_progress,
        "stable_detach": alpha
        * STABLE_DETACH_WEIGHT
        * stable_detach_progress,
        "retention_progress": RETENTION_PROGRESS_WEIGHT * retention_progress,
        # At gamma=.99, -0.1 on every non-terminal step plus -10 on a
        # pre-detach timeout makes waiting for any duration worth exactly -10.
        "time_cost": (
            time_cost_discovery * DISCOVERY_TIME_COST_WEIGHT
            + time_cost_alpha * ALIGNED_TIME_COST_WEIGHT
        ) * (~terminal).float(),
        "success": success_weight * success.float(),
        "drop_failure": alpha * DROP_FAILURE_WEIGHT * dropped.float(),
        "excess_force_failure": alpha
        * EXCESS_FORCE_FAILURE_WEIGHT
        * excess_force.float(),
        "out_of_bounds_failure": alpha
        * OUT_OF_BOUNDS_FAILURE_WEIGHT
        * out_of_bounds.float(),
        "timeout_failure": alpha * timeout_weight * timed_out.float(),
    }
    reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
    return (
        reward,
        next_reach,
        next_grasp,
        next_opposed,
        next_stem,
        next_stable_detach,
        next_retention,
        terms,
    )


__all__ = [
    "ALIGNED_SUCCESS_WEIGHT",
    "ALIGNED_TIME_COST_WEIGHT",
    "BREAK_GRASPED_WEIGHT",
    "CURRICULUM_POINTS",
    "DISCOVERY_TIME_COST_WEIGHT",
    "DISCOVERY_SUCCESS_WEIGHT",
    "DROP_FAILURE_WEIGHT",
    "EXCESS_FORCE_FAILURE_WEIGHT",
    "GRASP_PROGRESS_WEIGHT",
    "OPPOSED_PROGRESS_WEIGHT",
    "OUT_OF_BOUNDS_FAILURE_WEIGHT",
    "POST_DETACH_TIMEOUT_WEIGHT",
    "PRE_DETACH_TIMEOUT_WEIGHT",
    "REACH_PROGRESS_WEIGHT",
    "RETENTION_PROGRESS_WEIGHT",
    "SHAPING_GAMMA",
    "STABLE_DETACH_WEIGHT",
    "STEM_LOAD_PROGRESS_WEIGHT",
    "alignment_alpha_for_iteration",
    "best_ever_progress",
    "break_grasp_quality",
    "grasp_contact_score",
    "opposed_contact_score",
    "tomato_pick_reward",
]
