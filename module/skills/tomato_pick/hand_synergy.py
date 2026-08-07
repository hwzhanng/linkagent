"""Five-dimensional, absolute G20 grasp manifold.

The coordinates follow the eigengrasp structure used by mature dexterous
grasping systems: one overall flexion coordinate plus independent shape,
radial/ulnar balance, spread, and thumb-opposition coordinates.  The policy
therefore retains continuous control of the grasp while avoiding sixteen
independent joint actions that almost never explore a coherent enclosure.
"""

from __future__ import annotations

from typing import Sequence


HAND_JOINT_LIMITS = (
    (0.016471, 0.016471, 0.675294),
    (0.032549, 0.032549, 1.562353),
    (0.032549, 0.032549, 1.627451),
    (0.032549, 0.032549, 1.627451),
    (0.032549, 0.032549, 1.562353),
    (-0.35, 1.025843, 1.05),
    (-0.24, 0.132549, 0.24),
    (-0.24, 0.041804, 0.24),
    (-0.24, -0.045883, 0.24),
    (-0.24, -0.166196, 0.24),
    (-1.30, -0.093824, -0.02),
    (0.025882, 0.025882, 0.931765),
    (0.026667, 0.026667, 1.333333),
    (0.026667, 0.026667, 1.333333),
    (0.026667, 0.026667, 1.306667),
    (0.026667, 0.026667, 1.333333),
)
HAND_LOWER_RAD = tuple(item[0] for item in HAND_JOINT_LIMITS)
HAND_NEUTRAL_RAD = tuple(item[1] for item in HAND_JOINT_LIMITS)
HAND_UPPER_RAD = tuple(item[2] for item in HAND_JOINT_LIMITS)
# G20 channel 5 is thumb side-swing.  Its large sweep was explicitly marked
# unsafe during hardware calibration, so the grasp manifold keeps it at the
# calibrated neutral pose.  Channel 10 is the thumb cross-swing/roll axis and
# is fixed at mid travel.  The policy instead controls root flexion (channel
# 0 / thumb_joint2) over the first half of its mechanical travel and tip
# flexion (channel 15 / thumb_joint3) independently.
THUMB_SIDE_JOINT_INDEX = 5
THUMB_ROLL_JOINT_INDEX = 10
THUMB_ROOT_JOINT_INDEX = 0
THUMB_TIP_JOINT_INDEX = 11
THUMB_SIDE_SAFE_RAD = HAND_NEUTRAL_RAD[THUMB_SIDE_JOINT_INDEX]
THUMB_ROLL_FIXED_RAD = 0.5 * (
    HAND_LOWER_RAD[THUMB_ROLL_JOINT_INDEX]
    + HAND_UPPER_RAD[THUMB_ROLL_JOINT_INDEX]
)
THUMB_ROOT_MAX_FRACTION = 0.5
PREGRASP_FLEXION_FRACTION = 0.15
PREGRASP_SYNERGY = (0.0, 0.0, 0.0, 0.0, 0.0)
# A positive curl-shape coordinate closes the distal joints more and the
# proximal joints less.  The asymmetric gains reflect the larger distal range
# needed to wrap a sphere without folding the proximal links into the palm.
PROXIMAL_CURL_SHAPE_GAIN = 0.20
DISTAL_CURL_SHAPE_GAIN = 0.30
# Positive balance closes the radial pair (index/middle) more than the ulnar
# pair (ring/little); negative balance does the opposite.  This is the missing
# mode in the old manifold, where all four fingers were forced to close
# identically even after one side contacted the fruit first.
RADIAL_ULNAR_BALANCE_GAIN = 0.22
RADIAL_ULNAR_BALANCE_PATTERN = (1.0, 0.45, -0.45, -1.0)
FINGER_SPREAD_PATTERN = (-1.0, -0.35, 0.35, 1.0)


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _centered_target(index: int, value: float) -> float:
    lower, neutral, upper = HAND_JOINT_LIMITS[index]
    centered = _clip(value)
    return (
        neutral + centered * (upper - neutral)
        if centered >= 0.0
        else neutral + (-centered) * (lower - neutral)
    )


def _centered_unit(value: float, neutral: float = PREGRASP_FLEXION_FRACTION) -> float:
    """Map zero to pregrasp while preserving the full open/closed range."""

    centered = _clip(value)
    return (
        neutral + centered * (1.0 - neutral)
        if centered >= 0.0
        else neutral + (-centered) * (0.0 - neutral)
    )


def g20_synergy_to_joint_targets(values: Sequence[float]) -> tuple[float, ...]:
    """Map normalized grasp coordinates to absolute sixteen-joint targets."""

    if len(values) != 5:
        raise ValueError("G20 grasp synergy must contain exactly five values")
    global_flexion, curl_shape, radial_ulnar_balance, spread, thumb_opposition = (
        _clip(value) for value in values
    )
    global_flexion_unit = _centered_unit(global_flexion)
    finger_proximal_units = tuple(
        _clip(
            global_flexion_unit
            - PROXIMAL_CURL_SHAPE_GAIN * curl_shape
            + RADIAL_ULNAR_BALANCE_GAIN * radial_ulnar_balance * factor,
            0.0,
            1.0,
        )
        for factor in RADIAL_ULNAR_BALANCE_PATTERN
    )
    finger_distal_units = tuple(
        _clip(
            global_flexion_unit
            + DISTAL_CURL_SHAPE_GAIN * curl_shape
            + RADIAL_ULNAR_BALANCE_GAIN * radial_ulnar_balance * factor,
            0.0,
            1.0,
        )
        for factor in RADIAL_ULNAR_BALANCE_PATTERN
    )
    thumb_unit = _centered_unit(thumb_opposition)
    thumb_root_unit = THUMB_ROOT_MAX_FRACTION * thumb_unit
    thumb_tip_unit = thumb_unit

    targets = [0.0] * 16
    lower, _, upper = HAND_JOINT_LIMITS[THUMB_ROOT_JOINT_INDEX]
    targets[THUMB_ROOT_JOINT_INDEX] = lower + thumb_root_unit * (upper - lower)
    for index, unit in zip(range(1, 5), finger_proximal_units):
        lower, _, upper = HAND_JOINT_LIMITS[index]
        targets[index] = lower + unit * (upper - lower)
    targets[THUMB_SIDE_JOINT_INDEX] = THUMB_SIDE_SAFE_RAD
    targets[THUMB_ROLL_JOINT_INDEX] = THUMB_ROLL_FIXED_RAD
    for index, factor in enumerate(FINGER_SPREAD_PATTERN, start=6):
        targets[index] = _centered_target(index, spread * factor)
    lower, _, upper = HAND_JOINT_LIMITS[THUMB_TIP_JOINT_INDEX]
    targets[THUMB_TIP_JOINT_INDEX] = lower + thumb_tip_unit * (upper - lower)
    for target_index, unit in zip(range(12, 16), finger_distal_units):
        lower, _, upper = HAND_JOINT_LIMITS[target_index]
        targets[target_index] = lower + unit * (upper - lower)
    return tuple(
        max(HAND_LOWER_RAD[index], min(HAND_UPPER_RAD[index], value))
        for index, value in enumerate(targets)
    )


__all__ = [
    "HAND_JOINT_LIMITS",
    "HAND_LOWER_RAD",
    "HAND_NEUTRAL_RAD",
    "HAND_UPPER_RAD",
    "DISTAL_CURL_SHAPE_GAIN",
    "FINGER_SPREAD_PATTERN",
    "PREGRASP_FLEXION_FRACTION",
    "PREGRASP_SYNERGY",
    "PROXIMAL_CURL_SHAPE_GAIN",
    "RADIAL_ULNAR_BALANCE_GAIN",
    "RADIAL_ULNAR_BALANCE_PATTERN",
    "THUMB_ROLL_FIXED_RAD",
    "THUMB_ROLL_JOINT_INDEX",
    "THUMB_ROOT_JOINT_INDEX",
    "THUMB_ROOT_MAX_FRACTION",
    "THUMB_SIDE_JOINT_INDEX",
    "THUMB_SIDE_SAFE_RAD",
    "THUMB_TIP_JOINT_INDEX",
    "g20_synergy_to_joint_targets",
]
