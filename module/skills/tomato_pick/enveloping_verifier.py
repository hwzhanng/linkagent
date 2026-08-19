"""Independent offline verifier for palm-supported five-finger enclosure.

The caller supplies explicit contact/geometry evidence from simulation or a
recorded episode.  Thresholds here are evaluation priors, not hardware force
limits, and a successful result never authorizes real motion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class EnvelopingVerifierConfig:
    palm_support_streak_steps: int = 5
    palm_first_margin_steps: int = 3
    strict_hold_steps: int = 10
    angular_coverage_min_deg: float = 200.0
    thumb_opposition_cos_max: float = -0.35
    same_hemisphere_score_max: float = 0.85
    palm_load_share_min: float = 0.05
    palm_load_share_max: float = 0.70
    distal_participation_delta_rad: float = 0.03
    require_postbreak_hold: bool = False
    postbreak_hold_steps: int = 20

    def __post_init__(self) -> None:
        for field_name in (
            "palm_support_streak_steps",
            "palm_first_margin_steps",
            "strict_hold_steps",
            "postbreak_hold_steps",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be an integer > 0")
        if not 0.0 <= self.palm_load_share_min <= self.palm_load_share_max <= 1.0:
            raise ValueError("invalid palm load-share range")


@dataclass(frozen=True, slots=True)
class EnvelopingTraceFrame:
    palm_supported: bool
    digit_contact: tuple[bool, bool, bool, bool, bool]
    angular_coverage_deg: float
    thumb_opposition_cos: float
    same_hemisphere_score: float
    palm_load_share: float
    distal_joint_delta_rad: tuple[float, float, float, float]
    pressure_safe: bool = True
    damage_ok: bool = True
    dropped: bool = False
    postbreak: bool = False

    def __post_init__(self) -> None:
        if len(self.digit_contact) != 5 or any(not isinstance(item, bool) for item in self.digit_contact):
            raise ValueError("digit_contact must contain five bools")
        if len(self.distal_joint_delta_rad) != 4:
            raise ValueError("distal_joint_delta_rad must contain four values")
        for field_name in (
            "angular_coverage_deg",
            "thumb_opposition_cos",
            "same_hemisphere_score",
            "palm_load_share",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        distal = tuple(float(item) for item in self.distal_joint_delta_rad)
        if any(not math.isfinite(item) or item < 0.0 for item in distal):
            raise ValueError("distal_joint_delta_rad must contain finite non-negative values")
        object.__setattr__(self, "distal_joint_delta_rad", distal)
        if not 0.0 <= self.palm_load_share <= 1.0:
            raise ValueError("palm_load_share must lie in [0, 1]")
        for field_name in ("palm_supported", "pressure_safe", "damage_ok", "dropped", "postbreak"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")


@dataclass(frozen=True, slots=True)
class EnvelopingVerification:
    strict_success: bool
    failure_reasons: tuple[str, ...]
    first_palm_support_step: int | None
    first_three_digit_step: int | None
    first_all_five_step: int | None
    palm_first_margin_steps: int | None
    envelope_hold_max_streak: int
    postbreak_hold_max_streak: int
    max_angular_coverage_deg: float
    min_thumb_opposition_cos: float
    max_same_hemisphere_score: float
    distal_joint_participation: tuple[bool, bool, bool, bool]
    hardware_release_allowed: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _first_streak(mask: tuple[bool, ...], length: int) -> tuple[int | None, int]:
    first: int | None = None
    current = 0
    maximum = 0
    for index, active in enumerate(mask):
        current = current + 1 if active else 0
        maximum = max(maximum, current)
        if first is None and current >= length:
            first = index - length + 1
    return first, maximum


def _first_true(mask: tuple[bool, ...]) -> int | None:
    return next((index for index, value in enumerate(mask) if value), None)


def verify_enveloping_trace(
    frames: Iterable[EnvelopingTraceFrame],
    config: EnvelopingVerifierConfig | None = None,
) -> EnvelopingVerification:
    """Apply one stable set of gates to simulation or replayed evidence."""

    cfg = config or EnvelopingVerifierConfig()
    trace = tuple(frames)
    if not trace:
        raise ValueError("enveloping trace must contain at least one frame")

    support = tuple(frame.palm_supported for frame in trace)
    three_digits = tuple(sum(frame.digit_contact) >= 3 for frame in trace)
    five_digits = tuple(all(frame.digit_contact) for frame in trace)
    first_palm, _ = _first_streak(support, cfg.palm_support_streak_steps)
    first_three = _first_true(three_digits)
    first_five = _first_true(five_digits)
    margin = first_three - first_palm if first_palm is not None and first_three is not None else None

    step_ok = tuple(
        frame.palm_supported
        and all(frame.digit_contact)
        and frame.angular_coverage_deg >= cfg.angular_coverage_min_deg
        and frame.thumb_opposition_cos <= cfg.thumb_opposition_cos_max
        and frame.same_hemisphere_score <= cfg.same_hemisphere_score_max
        and cfg.palm_load_share_min <= frame.palm_load_share <= cfg.palm_load_share_max
        and frame.pressure_safe
        and frame.damage_ok
        and not frame.dropped
        for frame in trace
    )
    _, envelope_hold = _first_streak(step_ok, cfg.strict_hold_steps)
    postbreak_mask = tuple(ok and frame.postbreak for ok, frame in zip(step_ok, trace))
    _, postbreak_hold = _first_streak(postbreak_mask, cfg.postbreak_hold_steps)
    distal_max = tuple(max(frame.distal_joint_delta_rad[index] for frame in trace) for index in range(4))
    distal_participation = tuple(value >= cfg.distal_participation_delta_rad for value in distal_max)

    reasons: list[str] = []
    if first_palm is None:
        reasons.append("palm_support_missing")
    if first_three is None:
        reasons.append("three_digit_contact_missing")
    elif margin is None or margin < cfg.palm_first_margin_steps:
        reasons.append("palm_not_first")
    if first_five is None:
        reasons.append("five_digit_contact_missing")
    if envelope_hold < cfg.strict_hold_steps:
        reasons.append("envelope_hold_short")
    if max(frame.angular_coverage_deg for frame in trace) < cfg.angular_coverage_min_deg:
        reasons.append("under_enclosure")
    if min(frame.thumb_opposition_cos for frame in trace) > cfg.thumb_opposition_cos_max:
        reasons.append("thumb_opposition_missing")
    if not all(distal_participation):
        reasons.append("distal_joint_idle")
    if any(not frame.pressure_safe or not frame.damage_ok for frame in trace):
        reasons.append("overload_or_damage")
    if any(frame.dropped for frame in trace):
        reasons.append("drop_or_slip")
    if cfg.require_postbreak_hold and postbreak_hold < cfg.postbreak_hold_steps:
        reasons.append("postbreak_hold_short")

    return EnvelopingVerification(
        strict_success=not reasons,
        failure_reasons=tuple(reasons or ("pass",)),
        first_palm_support_step=first_palm,
        first_three_digit_step=first_three,
        first_all_five_step=first_five,
        palm_first_margin_steps=margin,
        envelope_hold_max_streak=envelope_hold,
        postbreak_hold_max_streak=postbreak_hold,
        max_angular_coverage_deg=max(frame.angular_coverage_deg for frame in trace),
        min_thumb_opposition_cos=min(frame.thumb_opposition_cos for frame in trace),
        max_same_hemisphere_score=max(frame.same_hemisphere_score for frame in trace),
        distal_joint_participation=distal_participation,
    )


__all__ = [
    "EnvelopingTraceFrame",
    "EnvelopingVerification",
    "EnvelopingVerifierConfig",
    "verify_enveloping_trace",
]
