"""Physical G20 mapping derived from the connected-hand motion calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from proto import ArmSide, HandSynergy


ACTIVE_CHANNELS = (*range(11), *range(15, 20))


@dataclass(frozen=True, slots=True)
class G20ChannelCalibration:
    logical_index: int
    joint: str
    raw_low: int
    raw_high: int
    left_rad_low: float
    left_rad_high: float
    mirror_right: bool
    kind: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "G20ChannelCalibration":
        raw_range = _pair(value.get("raw_range"), name="raw_range")
        rad_range = _pair(value.get("left_rad_range"), name="left_rad_range")
        logical_index = value.get("logical_index")
        if isinstance(logical_index, bool) or not isinstance(logical_index, int):
            raise TypeError("logical_index must be an integer")
        if logical_index not in ACTIVE_CHANNELS:
            raise ValueError(f"G20 channel {logical_index} is not an active channel")
        raw_low, raw_high = raw_range
        if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_range):
            raise TypeError("raw_range must contain integers")
        if not 0 <= raw_low < raw_high <= 255:
            raise ValueError("raw_range must be ordered within 0..255")
        left_rad_low, left_rad_high = (float(item) for item in rad_range)
        if not left_rad_low < left_rad_high:
            raise ValueError("left_rad_range must be strictly increasing")
        joint = value.get("joint")
        if not isinstance(joint, str) or not joint:
            raise ValueError("joint must be a non-empty string")
        mirror_right = value.get("mirror_right")
        if not isinstance(mirror_right, bool):
            raise TypeError("mirror_right must be a bool")
        kind = value.get("kind")
        if kind not in {"flexion", "yaw_roll"}:
            raise ValueError("kind must be flexion or yaw_roll")
        return cls(
            logical_index,
            joint,
            raw_low,
            raw_high,
            left_rad_low,
            left_rad_high,
            mirror_right,
            kind,
        )

    def rad_range(self, side: ArmSide) -> tuple[float, float]:
        if side is ArmSide.RIGHT and self.mirror_right:
            return (-self.left_rad_high, -self.left_rad_low)
        return (self.left_rad_low, self.left_rad_high)

    def raw_to_rad(self, side: ArmSide, raw: int) -> float:
        # The verified overlay marks every active channel as inverted: raw_low
        # is the high-radian endpoint and raw_high is the low-radian endpoint.
        fraction = (raw - self.raw_low) / (self.raw_high - self.raw_low)
        left_rad = self.left_rad_high + fraction * (
            self.left_rad_low - self.left_rad_high
        )
        return (
            -left_rad
            if side is ArmSide.RIGHT and self.mirror_right
            else left_rad
        )

    def rad_to_raw(self, side: ArmSide, rad: float) -> int:
        rad_low, rad_high = self.rad_range(side)
        if rad < rad_low - 1.0e-9 or rad > rad_high + 1.0e-9:
            raise ValueError(
                f"{side.value} {self.joint} command {rad:.6f} rad is outside "
                f"verified range [{rad_low:.6f}, {rad_high:.6f}]"
            )
        rad = min(rad_high, max(rad_low, rad))
        left_rad = -rad if side is ArmSide.RIGHT and self.mirror_right else rad
        fraction = (self.left_rad_high - left_rad) / (
            self.left_rad_high - self.left_rad_low
        )
        return round(self.raw_low + fraction * (self.raw_high - self.raw_low))


class CalibratedG20Mapper:
    """Map the 16 independent joints to the SDK's verified 20-value vector."""

    def __init__(
        self,
        *,
        mapping_id: str,
        channels: Sequence[G20ChannelCalibration],
        reserved_channels: Sequence[int],
    ) -> None:
        if not isinstance(mapping_id, str) or not mapping_id:
            raise ValueError("mapping_id must be a non-empty string")
        prepared = tuple(channels)
        if tuple(item.logical_index for item in prepared) != ACTIVE_CHANNELS:
            raise ValueError("channels must follow active G20 order 0..10,15..19")
        if tuple(reserved_channels) != (11, 12, 13, 14):
            raise ValueError("reserved_channels must be exactly 11..14")
        self.mapping_id = mapping_id
        self._channels = prepared

    @classmethod
    def from_file(cls, path: str | Path) -> "CalibratedG20Mapper":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("G20 motion calibration schema_version must be 1")
        return cls(
            mapping_id=raw.get("mapping_id"),
            channels=tuple(
                G20ChannelCalibration.from_mapping(item)
                for item in raw.get("channels", ())
            ),
            reserved_channels=raw.get("reserved_channels", ()),
        )

    @staticmethod
    def _side(side: ArmSide) -> ArmSide:
        return side if isinstance(side, ArmSide) else ArmSide(side)

    def to_g20_joint_positions(
        self, side: ArmSide, joint_positions_rad: tuple[float, ...]
    ) -> tuple[int, ...]:
        side = self._side(side)
        values = tuple(joint_positions_rad)
        if len(values) != 16:
            raise ValueError("joint_positions_rad must contain 16 values")
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("joint_positions_rad must contain finite numbers")
        logical = [255] * 20
        for channel, value in zip(self._channels, values):
            logical[channel.logical_index] = channel.rad_to_raw(side, float(value))
        return tuple(logical)

    def from_g20_joint_positions(
        self, side: ArmSide, logical_raw: tuple[int, ...]
    ) -> tuple[float, ...]:
        side = self._side(side)
        values = _logical_raw(logical_raw)
        return tuple(
            channel.raw_to_rad(side, values[channel.logical_index])
            for channel in self._channels
        )

    def to_g20_logical(self, side: ArmSide, synergy: HandSynergy) -> tuple[int, ...]:
        side = self._side(side)
        if not isinstance(synergy, HandSynergy):
            raise TypeError("synergy must be a HandSynergy")
        if synergy != HandSynergy(0.0, 0.0, 0.0):
            raise ValueError(
                "physical G20 synergy is only calibrated for the neutral open command; "
                "use 16D joint positions for motion"
            )
        logical = [255] * 20
        for channel in self._channels:
            logical[channel.logical_index] = (
                channel.raw_high
                if channel.kind == "flexion"
                else round((channel.raw_low + channel.raw_high) / 2)
            )
        return tuple(logical)

    def to_sim_independent(
        self, side: ArmSide, synergy: HandSynergy
    ) -> tuple[float, ...]:
        logical = self.to_g20_logical(side, synergy)
        return self.from_g20_joint_positions(side, logical)

    def from_g20_state(
        self, side: ArmSide, logical_raw: tuple[int, ...]
    ) -> HandSynergy:
        self._side(side)
        values = _logical_raw(logical_raw)
        closure = sum(
            max(
                0.0,
                min(
                    1.0,
                    (channel.raw_high - values[channel.logical_index])
                    / (channel.raw_high - channel.raw_low),
                ),
            )
            for channel in self._channels
            if channel.kind == "flexion"
        ) / 10.0
        return HandSynergy(closure, 0.0, 0.0)


def _pair(value: object, *, name: str) -> tuple[Any, Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{name} must contain exactly two values")
    return value[0], value[1]


def _logical_raw(value: tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(value)
    if len(values) != 20 or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
        for item in values
    ):
        raise ValueError("logical_raw must contain 20 integers in 0..255")
    return values


__all__ = ["CalibratedG20Mapper", "G20ChannelCalibration"]
