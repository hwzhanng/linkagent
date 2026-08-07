"""Deployable contract for the minimal tomato-picking policy.

The policy is deliberately small: six Cartesian palm increments and five
absolute G20 grasp coordinates.  It never observes simulator-only contact or
stem state and it never owns the global approach to the plant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence


INTERFACE_VERSION = "3.1.0"
OBSERVATION_LAYOUT_ID = "tomato_pick_actor_66d_v3"
ACTION_LAYOUT_ID = "tomato_pick_cartesian_g20_eigengrasp_11d_v4"
OBSERVATION_DIM = 66
ACTION_DIM = 11
CRITIC_STATE_DIM = 83
POLICY_RATE_HZ = 20
FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
ARM_JOINT_ORDER = ("1", "2", "3", "4", "5", "6", "7")
G20_ACTIVE_RAW_INDICES = (*range(11), *range(15, 20))
G20_JOINT_ORDER = (
    "thumb_joint2",
    "index_joint1",
    "middle_joint1",
    "ring_joint1",
    "little_joint1",
    "thumb_joint0",
    "index_joint0",
    "middle_joint0",
    "ring_joint0",
    "little_joint0",
    "thumb_joint1",
    "thumb_joint3",
    "index_joint2",
    "middle_joint2",
    "ring_joint2",
    "little_joint2",
)
HAND_SYNERGY_ORDER = (
    "global_finger_flexion",
    "proximal_distal_curl_shape",
    "radial_ulnar_flexion_balance",
    "finger_spread",
    "thumb_opposition",
)
OBSERVATION_BLOCKS = MappingProxyType(
    {
        "fruit_center_in_palm": slice(0, 3),
        "fruit_semi_axes": slice(3, 6),
        "pull_axis_in_palm": slice(6, 9),
        "ee_twist_in_palm": slice(9, 15),
        "hand_joint_position": slice(15, 31),
        "hand_joint_velocity": slice(31, 47),
        "finger_pressure": slice(47, 52),
        "gravity_in_palm": slice(52, 55),
        "previous_action": slice(55, 66),
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _floats(value: object, *, field_name: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a numeric sequence")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a numeric sequence") from exc
    if len(values) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    return tuple(
        _finite(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(values)
    )


@dataclass(frozen=True, slots=True)
class PolicyManifest:
    """Fields required to reproduce a deployable policy boundary."""

    policy_id: str
    checkpoint_sha256: str
    hand_joint_mapper_id: str
    tactile_feature_extractor_id: str
    training_asset_id: str
    action_scales: tuple[float, ...]
    hand_position_lower_rad: tuple[float, ...]
    hand_position_upper_rad: tuple[float, ...]
    hand_velocity_scale_rad_s: tuple[float, ...]
    fruit_axis_lower_m: tuple[float, ...]
    fruit_axis_upper_m: tuple[float, ...]
    max_state_age_s: tuple[float, ...]
    max_force_age_s: float
    normalization_stats: Mapping[str, Any] = field(default_factory=dict)
    interface_version: str = INTERFACE_VERSION
    observation_layout_id: str = OBSERVATION_LAYOUT_ID
    action_layout_id: str = ACTION_LAYOUT_ID
    policy_rate_hz: int = POLICY_RATE_HZ
    grasp_frame_convention: str = "z_plus_away_from_plant"

    def __post_init__(self) -> None:
        for name in (
            "policy_id",
            "hand_joint_mapper_id",
            "tactile_feature_extractor_id",
            "training_asset_id",
            "grasp_frame_convention",
        ):
            object.__setattr__(
                self, name, _non_empty(getattr(self, name), field_name=name)
            )
        sha = _non_empty(self.checkpoint_sha256, field_name="checkpoint_sha256")
        if not _SHA256_PATTERN.fullmatch(sha):
            raise ValueError("checkpoint_sha256 must contain 64 lowercase hex characters")
        object.__setattr__(self, "checkpoint_sha256", sha)
        if self.interface_version != INTERFACE_VERSION:
            raise ValueError(f"interface_version must be {INTERFACE_VERSION}")
        if self.observation_layout_id != OBSERVATION_LAYOUT_ID:
            raise ValueError(f"observation_layout_id must be {OBSERVATION_LAYOUT_ID}")
        if self.action_layout_id != ACTION_LAYOUT_ID:
            raise ValueError(f"action_layout_id must be {ACTION_LAYOUT_ID}")
        if self.policy_rate_hz != POLICY_RATE_HZ:
            raise ValueError(f"policy_rate_hz must be {POLICY_RATE_HZ}")
        scales = _floats(self.action_scales, field_name="action_scales", length=6)
        if any(value <= 0.0 for value in scales):
            raise ValueError("Cartesian action scales must be positive")
        object.__setattr__(self, "action_scales", scales)
        lower = _floats(
            self.hand_position_lower_rad,
            field_name="hand_position_lower_rad",
            length=16,
        )
        upper = _floats(
            self.hand_position_upper_rad,
            field_name="hand_position_upper_rad",
            length=16,
        )
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("every hand lower limit must be below its upper limit")
        object.__setattr__(self, "hand_position_lower_rad", lower)
        object.__setattr__(self, "hand_position_upper_rad", upper)
        velocity = _floats(
            self.hand_velocity_scale_rad_s,
            field_name="hand_velocity_scale_rad_s",
            length=16,
        )
        if any(value <= 0.0 for value in velocity):
            raise ValueError("hand velocity scales must be positive")
        object.__setattr__(self, "hand_velocity_scale_rad_s", velocity)
        fruit_lower = _floats(
            self.fruit_axis_lower_m, field_name="fruit_axis_lower_m", length=3
        )
        fruit_upper = _floats(
            self.fruit_axis_upper_m, field_name="fruit_axis_upper_m", length=3
        )
        if any(lo <= 0.0 or lo >= hi for lo, hi in zip(fruit_lower, fruit_upper)):
            raise ValueError("fruit axis limits must be positive and increasing")
        object.__setattr__(self, "fruit_axis_lower_m", fruit_lower)
        object.__setattr__(self, "fruit_axis_upper_m", fruit_upper)
        ages = _floats(self.max_state_age_s, field_name="max_state_age_s", length=3)
        if any(value <= 0.0 for value in ages):
            raise ValueError("state age limits must be positive")
        object.__setattr__(self, "max_state_age_s", ages)
        force_age = _finite(self.max_force_age_s, field_name="max_force_age_s")
        if force_age <= 0.0:
            raise ValueError("max_force_age_s must be positive")
        object.__setattr__(self, "max_force_age_s", force_age)
        if not isinstance(self.normalization_stats, Mapping):
            raise TypeError("normalization_stats must be a mapping")
        object.__setattr__(
            self, "normalization_stats", MappingProxyType(dict(self.normalization_stats))
        )

    @classmethod
    def mock_default(cls) -> "PolicyManifest":
        from .hand_synergy import HAND_LOWER_RAD, HAND_UPPER_RAD

        return cls(
            policy_id="mock_tomato_pick_v3",
            checkpoint_sha256="0" * 64,
            hand_joint_mapper_id="g20_eigengrasp_5d_v3_thumb_roll_fixed",
            tactile_feature_extractor_id="g20_pressure_5d_v1",
            training_asset_id="linkbot_g20_tomato_v1",
            action_scales=(0.004, 0.004, 0.006, 0.04, 0.04, 0.06),
            hand_position_lower_rad=HAND_LOWER_RAD,
            hand_position_upper_rad=HAND_UPPER_RAD,
            hand_velocity_scale_rad_s=(2.0,) * 16,
            fruit_axis_lower_m=(0.025, 0.025, 0.025),
            fruit_axis_upper_m=(0.035, 0.035, 0.035),
            # The Agent-side target may arrive from a 5 Hz RGB-D perception
            # loop; robot and hand state remain locally sampled at control rate.
            max_state_age_s=(0.50, 0.10, 0.10),
            max_force_age_s=0.15,
            normalization_stats={
                "relative_position_scale_m": 0.15,
                "linear_velocity_scale_m_s": 0.5,
                "angular_velocity_scale_rad_s": 2.0,
                "pressure_full_scale_n": 20.0,
            },
        )


@dataclass(frozen=True, slots=True)
class TomatoPickObservation:
    """The exact deployable 66-value Actor input."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _floats(self.values, field_name="values", length=OBSERVATION_DIM),
        )

    @classmethod
    def pack(
        cls,
        *,
        fruit_center_in_palm: Sequence[float],
        fruit_semi_axes_norm: Sequence[float],
        pull_axis_in_palm: Sequence[float],
        ee_twist_in_palm_norm: Sequence[float],
        hand_joint_position_norm: Sequence[float],
        hand_joint_velocity_norm: Sequence[float],
        finger_pressure_norm: Sequence[float],
        gravity_in_palm: Sequence[float],
        previous_action: Sequence[float],
    ) -> "TomatoPickObservation":
        blocks = (
            _floats(fruit_center_in_palm, field_name="fruit_center_in_palm", length=3),
            _floats(fruit_semi_axes_norm, field_name="fruit_semi_axes_norm", length=3),
            _floats(pull_axis_in_palm, field_name="pull_axis_in_palm", length=3),
            _floats(ee_twist_in_palm_norm, field_name="ee_twist_in_palm_norm", length=6),
            _floats(hand_joint_position_norm, field_name="hand_joint_position_norm", length=16),
            _floats(hand_joint_velocity_norm, field_name="hand_joint_velocity_norm", length=16),
            _floats(finger_pressure_norm, field_name="finger_pressure_norm", length=5),
            _floats(gravity_in_palm, field_name="gravity_in_palm", length=3),
            _floats(previous_action, field_name="previous_action", length=ACTION_DIM),
        )
        values = tuple(value for block in blocks for value in block)
        if len(values) != OBSERVATION_DIM:  # pragma: no cover
            raise AssertionError("observation layout arithmetic changed")
        return cls(values)

    def block(self, name: str) -> tuple[float, ...]:
        try:
            item = OBSERVATION_BLOCKS[name]
        except KeyError as exc:
            raise KeyError(f"unknown observation block {name!r}") from exc
        return self.values[item]


@dataclass(frozen=True, slots=True)
class TomatoPickAction:
    """Six palm increments followed by five absolute grasp coordinates."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = _floats(self.values, field_name="values", length=ACTION_DIM)
        if any(value < -1.0 or value > 1.0 for value in values):
            raise ValueError("action values must lie in [-1, 1]")
        object.__setattr__(self, "values", values)

    @classmethod
    def zero(cls) -> "TomatoPickAction":
        return cls((0.0,) * ACTION_DIM)

    @property
    def delta_xyz(self) -> tuple[float, float, float]:
        return self.values[0:3]  # type: ignore[return-value]

    @property
    def delta_rotation_vector(self) -> tuple[float, float, float]:
        return self.values[3:6]  # type: ignore[return-value]

    @property
    def hand_synergy(self) -> tuple[float, float, float, float, float]:
        return self.values[6:11]  # type: ignore[return-value]


__all__ = [
    "ACTION_DIM",
    "ACTION_LAYOUT_ID",
    "ARM_JOINT_ORDER",
    "CRITIC_STATE_DIM",
    "FINGER_ORDER",
    "G20_ACTIVE_RAW_INDICES",
    "G20_JOINT_ORDER",
    "HAND_SYNERGY_ORDER",
    "INTERFACE_VERSION",
    "OBSERVATION_BLOCKS",
    "OBSERVATION_DIM",
    "OBSERVATION_LAYOUT_ID",
    "POLICY_RATE_HZ",
    "PolicyManifest",
    "TomatoPickAction",
    "TomatoPickObservation",
]
