"""Simulation-only 14D action adapter for palm-supported enclosure.

The archived 10D manifold remains untouched.  This adapter preserves its
first ten coordinates and adds one independent distal residual for each
straight finger (index, middle, ring, little).  The residuals affect only the
distal joints, so a policy can curl fingertips after root/proximal
pre-positioning without silently moving the roots again.
"""

from __future__ import annotations

import torch

from simulation.ppo_tomato_grasp.hand_manifold import (
    DEFAULT_RESIDUAL_SPAN_FRACTION,
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    hybrid_action_to_joint_targets,
    synergy_to_joint_targets,
)


ENVELOPING_ACTION_DIM = 14
ENVELOPING_ACTION_ORDER = (
    "global_finger_flexion",
    "proximal_distal_curl_shape",
    "radial_ulnar_flexion_balance",
    "finger_spread",
    "thumb_opposition",
    "thumb_residual",
    "index_residual",
    "middle_residual",
    "ring_residual",
    "little_residual",
    "index_distal_residual",
    "middle_distal_residual",
    "ring_distal_residual",
    "little_distal_residual",
)
STRAIGHT_FINGER_DISTAL_JOINT_INDICES = (12, 13, 14, 15)
# The first v2 enclosure probe showed that a 0.25 span left the straight
# fingertip bodies 4--6 cm from a 37.5 mm fruit while link3 already carried
# load.  v2 may use a wider *simulation-only* residual, still clamped to the
# calibrated joint limits below.  The archived 10D manifold never calls this
# adapter and therefore keeps its historical range.
DEFAULT_DISTAL_RESIDUAL_SPAN_FRACTION = 0.25


def _require_action(action: torch.Tensor) -> None:
    if not isinstance(action, torch.Tensor):
        raise TypeError("action must be a torch.Tensor")
    if action.ndim < 1 or action.shape[-1] != ENVELOPING_ACTION_DIM:
        raise ValueError(f"action must end with dimension {ENVELOPING_ACTION_DIM}")
    if not torch.isfinite(action).all():
        raise ValueError("action must contain only finite values")


def enveloping_action_to_joint_targets(
    action: torch.Tensor,
    *,
    residual_span_fraction: float = DEFAULT_RESIDUAL_SPAN_FRACTION,
    distal_residual_span_fraction: float = DEFAULT_DISTAL_RESIDUAL_SPAN_FRACTION,
    per_digit_root_residual_span_fraction: float | None = None,
) -> torch.Tensor:
    """Map 14D enclosure actions to canonical 16-joint targets.

    ``action[..., :10]`` is exactly the archived hybrid mapper.  The final
    four values are independent distal residuals and are applied only to
    joints 12--15.  All outputs are clamped to the simulation safe intervals.
    """

    _require_action(action)
    if not 0.0 <= distal_residual_span_fraction <= 0.75:
        raise ValueError("distal_residual_span_fraction must lie in [0, 0.75]")
    if per_digit_root_residual_span_fraction is None:
        root_span = float(residual_span_fraction)
    else:
        root_span = float(per_digit_root_residual_span_fraction)
        if not 0.0 <= root_span <= 0.75:
            raise ValueError("per_digit_root_residual_span_fraction must lie in [0, 0.75]")
    # Keep the archived 10D mapper bitwise-equivalent when no v2 override is
    # supplied.  The optional branch gives only index/middle/ring/little root
    # residuals a wider simulation-only reach window; thumb residual and all
    # distal channels retain their original separate spans.
    if per_digit_root_residual_span_fraction is None:
        targets = hybrid_action_to_joint_targets(
            action[..., :10], residual_span_fraction=residual_span_fraction
        )
    else:
        targets = synergy_to_joint_targets(action[..., :5])
        lower = action.new_tensor(LEFT_SAFE_LOWER_RAD)
        upper = action.new_tensor(LEFT_SAFE_UPPER_RAD)
        span = upper - lower
        residual = action[..., 5:10].clamp(-1.0, 1.0)
        # Keep thumb residual semantics and straight-finger distal residuals
        # unchanged; widen only the four straight-finger root/proximal axes.
        targets[..., 0] += residual[..., 0] * float(residual_span_fraction) * span[0]
        targets[..., 11] += residual[..., 0] * float(residual_span_fraction) * span[11]
        for digit_index, joint_index in enumerate((1, 2, 3, 4), start=1):
            targets[..., joint_index] += residual[..., digit_index] * root_span * span[joint_index]
            targets[..., joint_index + 11] += residual[..., digit_index] * float(residual_span_fraction) * span[joint_index + 11]
        targets = torch.maximum(lower, torch.minimum(upper, targets))
    lower = action.new_tensor(LEFT_SAFE_LOWER_RAD)
    upper = action.new_tensor(LEFT_SAFE_UPPER_RAD)
    span = upper - lower
    distal_residual = action[..., 10:].clamp(-1.0, 1.0)
    for component, joint_index in enumerate(STRAIGHT_FINGER_DISTAL_JOINT_INDICES):
        targets[..., joint_index] += (
            distal_residual[..., component]
            * distal_residual_span_fraction
            * span[joint_index]
        )
    return torch.maximum(lower, torch.minimum(upper, targets))


def distal_residual_jacobian(
    action: torch.Tensor,
    *,
    distal_residual_span_fraction: float = DEFAULT_DISTAL_RESIDUAL_SPAN_FRACTION,
) -> torch.Tensor:
    """Return the analytic distal-residual sensitivity for audit/tests.

    Shape is ``(..., 4, 16)``; each row contains exactly one non-zero distal
    joint entry away from a hard limit.
    """

    _require_action(action)
    jac = torch.zeros((*action.shape[:-1], 4, 16), dtype=action.dtype, device=action.device)
    span = action.new_tensor(LEFT_SAFE_UPPER_RAD) - action.new_tensor(LEFT_SAFE_LOWER_RAD)
    for component, joint_index in enumerate(STRAIGHT_FINGER_DISTAL_JOINT_INDICES):
        jac[..., component, joint_index] = distal_residual_span_fraction * span[joint_index]
    return jac


__all__ = [
    "DEFAULT_DISTAL_RESIDUAL_SPAN_FRACTION",
    "ENVELOPING_ACTION_DIM",
    "ENVELOPING_ACTION_ORDER",
    "STRAIGHT_FINGER_DISTAL_JOINT_INDICES",
    "distal_residual_jacobian",
    "enveloping_action_to_joint_targets",
]
