"""Deployable tomato-pick progress checks and normalized action validation."""

from __future__ import annotations

from proto import PickPhase

from .contracts import ACTION_DIM


CONTACT_FORCE_THRESHOLD_N = 0.60
CONTACT_MINIMUM_SUPPORT_FINGERS = 1
CONTACT_STREAK_STEPS = 2
GRASP_MINIMUM_SUPPORT_FINGERS = 2
# CONTACT only proves that a fingertip has reached the fruit.  Before DETACH,
# the grasp must also be able to support the heaviest randomized tomato at the
# minimum randomized friction.  About 6.2 N total normal load supports 0.188 kg
# at friction 0.30; use 6.5 N and let the G20 distribute that load naturally
# across unequal fingers.  Per-finger force safety remains enforced separately.
GRASP_MINIMUM_TOTAL_NORMAL_FORCE_N = 6.50
# Calibrated against the G20 collision geometry: thumb plus two support pads
# overlap at roughly 0.15 measured synergy.  Requiring 0.25 makes the fingers
# pass that enclosure and lose thumb contact before the condition can become
# true.  Contact-force requirements below remain the primary grasp evidence.
GRASP_MINIMUM_CLOSURE = 0.15
# Successful simulated grasps keep the loaded G20 close to its neutral
# opposition/spread pre-shape.  Failed grasps that merely accumulate force
# consistently splay the fingers and roll the thumb away before pulling.  The
# joint-encoder bounds below are deployable on the real hand and prevent that
# false-positive geometry without exposing simulator-only contact vectors to
# the actor or Skill contract.
GRASP_MINIMUM_THUMB_OPPOSITION = -0.15
GRASP_MAXIMUM_FINGER_SPREAD = 0.32


def supervise_action_values(
    values: tuple[float, ...],
    phase: PickPhase,
) -> tuple[float, ...]:
    """Return all policy actions unchanged after contract validation.

    ``phase`` is retained in the V1 interface as an observable progress label.
    It must not replace, mask, rescale, or otherwise prescribe a policy action.
    Safety limits are applied later by the Cartesian and hand controllers.
    """

    if len(values) != ACTION_DIM:
        raise ValueError(f"tomato-pick action must contain {ACTION_DIM} values")
    if not isinstance(phase, PickPhase):
        raise ValueError(f"unsupported phase {phase!r}")
    result = tuple(float(value) for value in values)
    if any(value < -1.0 or value > 1.0 for value in result):
        raise ValueError("normalized tomato-pick actions must stay in [-1, 1]")
    return result


def contact_ready(normal_force_n: tuple[float | None, ...]) -> bool:
    """CONTACT requires one stable support finger; thumb-only is insufficient."""

    if len(normal_force_n) != 5:
        raise ValueError("normal_force_n must use thumb-first five-finger order")
    return sum(
        force is not None and force >= CONTACT_FORCE_THRESHOLD_N
        for force in normal_force_n[1:]
    ) >= CONTACT_MINIMUM_SUPPORT_FINGERS


def grasp_ready(
    normal_force_n: tuple[float | None, ...],
    *,
    closure: float,
    thumb_opposition: float = 0.0,
    finger_spread: float = 0.0,
) -> bool:
    """GRASP requires load, enclosure, and a deployable G20 pre-shape."""

    if len(normal_force_n) != 5:
        raise ValueError("normal_force_n must use thumb-first five-finger order")
    thumb = normal_force_n[0]
    support_count = sum(
        force is not None and force >= CONTACT_FORCE_THRESHOLD_N
        for force in normal_force_n[1:]
    )
    total_normal_force_n = sum(
        max(float(force), 0.0) for force in normal_force_n if force is not None
    )
    return (
        thumb is not None
        and thumb >= CONTACT_FORCE_THRESHOLD_N
        and support_count >= GRASP_MINIMUM_SUPPORT_FINGERS
        and total_normal_force_n >= GRASP_MINIMUM_TOTAL_NORMAL_FORCE_N
        and closure >= GRASP_MINIMUM_CLOSURE
        and thumb_opposition >= GRASP_MINIMUM_THUMB_OPPOSITION
        and finger_spread <= GRASP_MAXIMUM_FINGER_SPREAD
    )


__all__ = [
    "CONTACT_FORCE_THRESHOLD_N",
    "CONTACT_MINIMUM_SUPPORT_FINGERS",
    "CONTACT_STREAK_STEPS",
    "GRASP_MINIMUM_CLOSURE",
    "GRASP_MINIMUM_SUPPORT_FINGERS",
    "GRASP_MINIMUM_THUMB_OPPOSITION",
    "GRASP_MINIMUM_TOTAL_NORMAL_FORCE_N",
    "GRASP_MAXIMUM_FINGER_SPREAD",
    "contact_ready",
    "grasp_ready",
    "supervise_action_values",
]
