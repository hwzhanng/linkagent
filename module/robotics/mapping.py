"""Versioned C1 hand-synergy mapping; not a physical G20 calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from proto import ArmSide, HandSynergy


@runtime_checkable
class HandSynergyMapper(Protocol):
    @property
    def mapping_id(self) -> str: ...
    def to_g20_logical(self, side: ArmSide, synergy: HandSynergy) -> tuple[int, ...]: ...
    def to_sim_independent(self, side: ArmSide, synergy: HandSynergy) -> tuple[float, ...]: ...
    def from_g20_state(self, side: ArmSide, logical_raw: tuple[int, ...]) -> HandSynergy: ...
    def to_g20_joint_positions(
        self, side: ArmSide, joint_positions_rad: tuple[float, ...]
    ) -> tuple[int, ...]: ...
    def from_g20_joint_positions(
        self, side: ArmSide, logical_raw: tuple[int, ...]
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class MockHandSynergyMapper:
    """Deterministic dimensional bridge for mocks and contract tests only."""

    mapping_id: str = "mock_g20_synergy_v1"

    @staticmethod
    def _side(side: ArmSide) -> ArmSide:
        return side if isinstance(side, ArmSide) else ArmSide(side)

    def to_g20_logical(self, side: ArmSide, synergy: HandSynergy) -> tuple[int, ...]:
        self._side(side)
        if not isinstance(synergy, HandSynergy):
            raise TypeError("synergy must be a HandSynergy")
        closure_raw = round(255 - synergy.closure * 200)
        values = [255] * 20
        for index in range(5):
            values[index] = closure_raw
            values[index + 15] = closure_raw
        spread_center = round(128 + synergy.finger_spread * 40)
        values[5] = round(128 + synergy.thumb_opposition * 40)
        for index in range(6, 10):
            values[index] = spread_center
        values[10] = round(128 + synergy.thumb_opposition * 80)
        # G20 logical channels 11..14 are reserved and remain at the safe
        # mock calibration value. They are never treated as extra joints.
        values[11:15] = [255, 255, 255, 255]
        return tuple(max(0, min(255, value)) for value in values)

    def to_sim_independent(self, side: ArmSide, synergy: HandSynergy) -> tuple[float, ...]:
        self._side(side)
        if not isinstance(synergy, HandSynergy):
            raise TypeError("synergy must be a HandSynergy")
        spread = synergy.finger_spread
        closure = synergy.closure
        opposition = synergy.thumb_opposition
        # Frozen active G20 raw order: 0..10 then 15..19.  Reserved raw slots
        # 11..14 and the five mechanical followers never enter this vector.
        values = [
            closure * 1.00,  # thumb_joint2
            closure * 1.20,  # index_joint1
            closure * 1.20,  # middle_joint1
            closure * 1.20,  # ring_joint1
            closure * 1.20,  # little_joint1
            opposition * 0.40,  # thumb_joint0
            spread * -1.5 * 0.08,  # index_joint0
            spread * -0.5 * 0.08,  # middle_joint0
            spread * 0.5 * 0.08,  # ring_joint0
            spread * 1.5 * 0.08,  # little_joint0
            opposition * 0.40,  # thumb_joint1
            closure * 1.00,  # thumb_joint3
            closure * 1.00,  # index_joint2
            closure * 1.00,  # middle_joint2
            closure * 1.00,  # ring_joint2
            closure * 1.00,  # little_joint2
        ]
        if len(values) != 16:  # pragma: no cover - frozen mapping arithmetic.
            raise AssertionError("sim independent hand mapping must be 16D")
        return tuple(values)

    def from_g20_state(self, side: ArmSide, logical_raw: tuple[int, ...]) -> HandSynergy:
        self._side(side)
        values = tuple(logical_raw)
        if len(values) != 20 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in values
        ):
            raise ValueError("logical_raw must contain 20 integers in 0..255")
        closure_channels = values[0:5] + values[15:20]
        closure = max(0.0, min(1.0, (255 - sum(closure_channels) / 10.0) / 200.0))
        opposition = max(-1.0, min(1.0, (values[10] - 128) / 80.0))
        spread = max(-1.0, min(1.0, (sum(values[6:10]) / 4.0 - 128) / 40.0))
        return HandSynergy(closure, opposition, spread)

    def to_g20_joint_positions(
        self, side: ArmSide, joint_positions_rad: tuple[float, ...]
    ) -> tuple[int, ...]:
        self._side(side)
        if len(joint_positions_rad) != 16:
            raise ValueError("joint_positions_rad must contain 16 values")
        active_indices = (*range(11), *range(15, 20))
        logical = [255] * 20
        for raw_index, value in zip(active_indices, joint_positions_rad):
            logical[raw_index] = round(128.0 + max(-1.0, min(1.0, value)) * 127.0)
        return tuple(logical)

    def from_g20_joint_positions(
        self, side: ArmSide, logical_raw: tuple[int, ...]
    ) -> tuple[float, ...]:
        self._side(side)
        if len(logical_raw) != 20:
            raise ValueError("logical_raw must contain 20 values")
        active_indices = (*range(11), *range(15, 20))
        return tuple((logical_raw[index] - 128.0) / 127.0 for index in active_indices)


__all__ = ["HandSynergyMapper", "MockHandSynergyMapper"]
