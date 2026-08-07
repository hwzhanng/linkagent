"""Single active deployable tomato-pick policy contract.

The policy owns a local six-dimensional end-effector increment and every
independently commanded G20 joint.  Mechanical mimic joints and reserved SDK
slots are deliberately outside the learned action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence


OBSERVATION_DIM = 113
ACTION_DIM = 22
OBSERVATION_LAYOUT_ID = "tomato_pick_observation_113d"
ACTION_LAYOUT_ID = "tomato_pick_action_22d"
INTERFACE_VERSION = "2.0.0"
PHASE_ORDER = ("contact", "grasp", "detach", "hold")
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
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _floats(
    value: object,
    *,
    field_name: str,
    length: int,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a numeric sequence")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a numeric sequence") from exc
    if len(items) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    return tuple(
        _finite(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(items)
    )


def _positive_floats(
    value: object,
    *,
    field_name: str,
    length: int,
) -> tuple[float, ...]:
    items = _floats(value, field_name=field_name, length=length)
    if any(item <= 0.0 for item in items):
        raise ValueError(f"{field_name} values must be > 0")
    return items


def _strings(
    value: object,
    *,
    field_name: str,
    length: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a string sequence")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a string sequence") from exc
    if len(items) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    return tuple(
        _non_empty(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(items)
    )


@dataclass(frozen=True, slots=True)
class PolicyManifest:
    """Metadata that makes checkpoint/normalization/mapping drift detectable."""

    policy_id: str
    checkpoint_sha256: str
    hand_joint_mapper_id: str
    tactile_feature_extractor_id: str
    training_asset_id: str
    domain_randomization_id: str
    action_scales: tuple[float, ...]
    phase_step_limits: tuple[int, ...]
    arm_position_lower_rad: tuple[float, ...]
    arm_position_upper_rad: tuple[float, ...]
    arm_velocity_scale_rad_s: tuple[float, ...]
    hand_position_lower_rad: tuple[float, ...]
    hand_position_upper_rad: tuple[float, ...]
    hand_velocity_scale_rad_s: tuple[float, ...]
    fruit_axis_lower_m: tuple[float, ...]
    fruit_axis_upper_m: tuple[float, ...]
    max_state_age_s: tuple[float, ...]
    max_force_age_s: float
    max_matrix_age_s: float
    normalization_stats: Mapping[str, Any] = field(default_factory=dict)
    interface_version: str = INTERFACE_VERSION
    observation_layout_id: str = OBSERVATION_LAYOUT_ID
    action_layout_id: str = ACTION_LAYOUT_ID
    policy_rate_hz: int = 20
    arm_joint_order: tuple[str, ...] = ARM_JOINT_ORDER
    finger_order: tuple[str, ...] = FINGER_ORDER
    grasp_frame_convention: str = "z_plus_away_from_stem"

    def __post_init__(self) -> None:
        for field_name in (
            "policy_id",
            "hand_joint_mapper_id",
            "tactile_feature_extractor_id",
            "training_asset_id",
            "domain_randomization_id",
            "interface_version",
            "observation_layout_id",
            "action_layout_id",
            "grasp_frame_convention",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_empty(getattr(self, field_name), field_name=field_name),
            )
        sha = _non_empty(self.checkpoint_sha256, field_name="checkpoint_sha256")
        if not _SHA256_PATTERN.fullmatch(sha):
            raise ValueError("checkpoint_sha256 must be 64 lowercase hex characters")
        object.__setattr__(self, "checkpoint_sha256", sha)
        if self.interface_version != INTERFACE_VERSION:
            raise ValueError(f"interface_version must be {INTERFACE_VERSION}")
        if self.observation_layout_id != OBSERVATION_LAYOUT_ID:
            raise ValueError(f"observation_layout_id must be {OBSERVATION_LAYOUT_ID}")
        if self.action_layout_id != ACTION_LAYOUT_ID:
            raise ValueError(f"action_layout_id must be {ACTION_LAYOUT_ID}")
        if self.policy_rate_hz != 20 or isinstance(self.policy_rate_hz, bool):
            raise ValueError("policy_rate_hz must be 20 for the active interface")
        object.__setattr__(
            self,
            "action_scales",
            _positive_floats(
                self.action_scales,
                field_name="action_scales",
                length=ACTION_DIM,
            ),
        )
        steps = tuple(self.phase_step_limits)
        if len(steps) != 4 or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in steps
        ):
            raise ValueError("phase_step_limits must contain four integers > 0")
        object.__setattr__(self, "phase_step_limits", steps)
        lower = _floats(
            self.arm_position_lower_rad,
            field_name="arm_position_lower_rad",
            length=7,
        )
        upper = _floats(
            self.arm_position_upper_rad,
            field_name="arm_position_upper_rad",
            length=7,
        )
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("every arm lower limit must be below its upper limit")
        object.__setattr__(self, "arm_position_lower_rad", lower)
        object.__setattr__(self, "arm_position_upper_rad", upper)
        object.__setattr__(
            self,
            "arm_velocity_scale_rad_s",
            _positive_floats(
                self.arm_velocity_scale_rad_s,
                field_name="arm_velocity_scale_rad_s",
                length=7,
            ),
        )
        hand_lower = _floats(
            self.hand_position_lower_rad,
            field_name="hand_position_lower_rad",
            length=16,
        )
        hand_upper = _floats(
            self.hand_position_upper_rad,
            field_name="hand_position_upper_rad",
            length=16,
        )
        if any(lo >= hi for lo, hi in zip(hand_lower, hand_upper)):
            raise ValueError("every hand lower limit must be below its upper limit")
        object.__setattr__(self, "hand_position_lower_rad", hand_lower)
        object.__setattr__(self, "hand_position_upper_rad", hand_upper)
        object.__setattr__(
            self,
            "hand_velocity_scale_rad_s",
            _positive_floats(
                self.hand_velocity_scale_rad_s,
                field_name="hand_velocity_scale_rad_s",
                length=16,
            ),
        )
        fruit_lower = _positive_floats(
            self.fruit_axis_lower_m,
            field_name="fruit_axis_lower_m",
            length=3,
        )
        fruit_upper = _positive_floats(
            self.fruit_axis_upper_m,
            field_name="fruit_axis_upper_m",
            length=3,
        )
        if any(lo >= hi for lo, hi in zip(fruit_lower, fruit_upper)):
            raise ValueError("every fruit axis lower bound must be below its upper bound")
        object.__setattr__(self, "fruit_axis_lower_m", fruit_lower)
        object.__setattr__(self, "fruit_axis_upper_m", fruit_upper)
        object.__setattr__(
            self,
            "max_state_age_s",
            _positive_floats(
                self.max_state_age_s,
                field_name="max_state_age_s",
                length=3,
            ),
        )
        for field_name in ("max_force_age_s", "max_matrix_age_s"):
            value = _finite(getattr(self, field_name), field_name=field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be > 0")
            object.__setattr__(self, field_name, value)
        arm_order = _strings(
            self.arm_joint_order,
            field_name="arm_joint_order",
            length=7,
        )
        finger_order = _strings(
            self.finger_order,
            field_name="finger_order",
            length=5,
        )
        if arm_order != ARM_JOINT_ORDER:
            raise ValueError(f"arm_joint_order must be {ARM_JOINT_ORDER}")
        if finger_order != FINGER_ORDER:
            raise ValueError(f"finger_order must be {FINGER_ORDER}")
        object.__setattr__(self, "arm_joint_order", arm_order)
        object.__setattr__(self, "finger_order", finger_order)
        if not isinstance(self.normalization_stats, Mapping):
            raise TypeError("normalization_stats must be a mapping")
        if any(not isinstance(key, str) for key in self.normalization_stats):
            raise TypeError("normalization_stats keys must be strings")
        object.__setattr__(
            self,
            "normalization_stats",
            MappingProxyType(dict(self.normalization_stats)),
        )

    @classmethod
    def mock_default(cls) -> "PolicyManifest":
        """Return the deterministic C1 manifest; it is not a trained checkpoint."""

        return cls(
            policy_id="mock_tomato_pick_v1",
            checkpoint_sha256="0" * 64,
            hand_joint_mapper_id="mock_g20_synergy_v1",
            tactile_feature_extractor_id="g20_pressure_features",
            training_asset_id="mock_no_asset",
            domain_randomization_id="mock_none",
            action_scales=(
                0.003,
                0.003,
                0.004,
                0.040,
                0.040,
                0.050,
                *((0.025,) * 16),
            ),
            phase_step_limits=(30, 30, 40, 20),
            arm_position_lower_rad=(-math.pi,) * 7,
            arm_position_upper_rad=(math.pi,) * 7,
            arm_velocity_scale_rad_s=(1.0,) * 7,
            hand_position_lower_rad=(-1.0,) * 16,
            hand_position_upper_rad=(1.0,) * 16,
            hand_velocity_scale_rad_s=(1.0,) * 16,
            fruit_axis_lower_m=(0.025, 0.025, 0.025),
            fruit_axis_upper_m=(0.035, 0.035, 0.035),
            max_state_age_s=(5.0, 0.10, 0.10),
            max_force_age_s=0.15,
            max_matrix_age_s=0.35,
            normalization_stats={"force_full_scale_n": 20.0},
        )


@dataclass(frozen=True, slots=True)
class TomatoPickObservation:
    """Exactly 113 deployable float32-compatible values in the frozen order."""

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
        target_pose_error: Sequence[float],
        ee_twist: Sequence[float],
        arm_joint_position_norm: Sequence[float],
        arm_joint_velocity_norm: Sequence[float],
        hand_joint_position_norm: Sequence[float],
        hand_joint_velocity_norm: Sequence[float],
        finger_pressure_norm: Sequence[float],
        finger_pressure_rate_norm: Sequence[float],
        gravity_direction: Sequence[float],
        previous_action: Sequence[float],
        target_arm_hand_valid: Sequence[float],
        force_valid: Sequence[float],
        target_arm_hand_age_norm: Sequence[float],
        force_age_norm: Sequence[float],
        fruit_axes_norm: Sequence[float],
        episode_progress: float,
    ) -> "TomatoPickObservation":
        """Pack named blocks and preserve the normative index layout."""

        blocks = (
            _floats(target_pose_error, field_name="target_pose_error", length=6),
            _floats(ee_twist, field_name="ee_twist", length=6),
            _floats(
                arm_joint_position_norm,
                field_name="arm_joint_position_norm",
                length=7,
            ),
            _floats(
                arm_joint_velocity_norm,
                field_name="arm_joint_velocity_norm",
                length=7,
            ),
            _floats(
                hand_joint_position_norm,
                field_name="hand_joint_position_norm",
                length=16,
            ),
            _floats(
                hand_joint_velocity_norm,
                field_name="hand_joint_velocity_norm",
                length=16,
            ),
            _floats(
                finger_pressure_norm,
                field_name="finger_pressure_norm",
                length=5,
            ),
            _floats(
                finger_pressure_rate_norm,
                field_name="finger_pressure_rate_norm",
                length=5,
            ),
            _floats(gravity_direction, field_name="gravity_direction", length=3),
            _floats(previous_action, field_name="previous_action", length=ACTION_DIM),
            _floats(
                target_arm_hand_valid,
                field_name="target_arm_hand_valid",
                length=3,
            ),
            _floats(force_valid, field_name="force_valid", length=5),
            _floats(
                target_arm_hand_age_norm,
                field_name="target_arm_hand_age_norm",
                length=3,
            ),
            _floats(force_age_norm, field_name="force_age_norm", length=5),
            _floats(fruit_axes_norm, field_name="fruit_axes_norm", length=3),
            (_finite(episode_progress, field_name="episode_progress"),),
        )
        values = tuple(value for block in blocks for value in block)
        if len(values) != OBSERVATION_DIM:  # pragma: no cover - frozen arithmetic.
            raise AssertionError("observation packing no longer totals 113")
        return cls(values)

    def block(self, start: int, stop: int) -> tuple[float, ...]:
        """Return a read-only slice for policy and golden tests."""

        return self.values[start:stop]


@dataclass(frozen=True, slots=True)
class TomatoPickAction:
    """Twenty-two normalized outputs in the only active semantic order."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = _floats(self.values, field_name="values", length=ACTION_DIM)
        if any(not -1.0 <= value <= 1.0 for value in values):
            raise ValueError("action values must be in [-1,1]")
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
    def delta_hand_joints(self) -> tuple[float, ...]:
        return self.values[6:22]


__all__ = [
    "ACTION_DIM",
    "ACTION_LAYOUT_ID",
    "ARM_JOINT_ORDER",
    "FINGER_ORDER",
    "G20_ACTIVE_RAW_INDICES",
    "G20_JOINT_ORDER",
    "INTERFACE_VERSION",
    "OBSERVATION_DIM",
    "OBSERVATION_LAYOUT_ID",
    "PHASE_ORDER",
    "PolicyManifest",
    "TomatoPickAction",
    "TomatoPickObservation",
]
