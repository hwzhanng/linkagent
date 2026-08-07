"""Raw SDK gateway contracts kept below the neutral robot Ports."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from proto import ArmSide, CommandResult, ExecutionHandle


def _floats(value: object, *, length: int, field_name: str) -> tuple[float, ...]:
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence") from exc
    if len(values) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    result: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{field_name} values must be real numbers")
        converted = float(item)
        if not math.isfinite(converted):
            raise ValueError(f"{field_name} values must be finite")
        result.append(converted)
    return tuple(result)


def _time(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0")
    return value


def _optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be null or a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _optional_time(value: object, *, field_name: str) -> int | None:
    return None if value is None else _time(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class SdkPoseRpy:
    position_m: tuple[float, ...]
    rpy_rad: tuple[float, ...]
    frame_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _floats(self.position_m, length=3, field_name="position_m"))
        object.__setattr__(self, "rpy_rad", _floats(self.rpy_rad, length=3, field_name="rpy_rad"))
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        object.__setattr__(self, "frame_id", self.frame_id.strip())


@dataclass(frozen=True, slots=True)
class ArmSdkState:
    """Gateway sample; ``ee_twist`` is expressed in ``ee_pose_rpy.frame_id``."""

    side: ArmSide
    joint_positions_rad: tuple[float, ...]
    ee_pose_rpy: SdkPoseRpy
    source_sequence: int
    sample_monotonic_ns: int
    received_monotonic_ns: int
    joint_velocities_rad_s: tuple[float, ...] | None = None
    ee_twist: tuple[float, ...] | None = None
    enabled: bool = True
    run_state: str = "idle"
    valid: bool = True
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.side, ArmSide):
            object.__setattr__(self, "side", ArmSide(self.side))
        object.__setattr__(self, "joint_positions_rad", _floats(self.joint_positions_rad, length=7, field_name="joint_positions_rad"))
        if not isinstance(self.ee_pose_rpy, SdkPoseRpy):
            raise TypeError("ee_pose_rpy must be SdkPoseRpy")
        for field_name in ("source_sequence", "sample_monotonic_ns", "received_monotonic_ns"):
            object.__setattr__(self, field_name, _time(getattr(self, field_name), field_name=field_name))
        if self.received_monotonic_ns < self.sample_monotonic_ns:
            raise ValueError("received_monotonic_ns must be >= sample_monotonic_ns")
        if self.joint_velocities_rad_s is not None:
            object.__setattr__(self, "joint_velocities_rad_s", _floats(self.joint_velocities_rad_s, length=7, field_name="joint_velocities_rad_s"))
        if self.ee_twist is not None:
            object.__setattr__(self, "ee_twist", _floats(self.ee_twist, length=6, field_name="ee_twist"))
        if not isinstance(self.enabled, bool) or not isinstance(self.valid, bool):
            raise TypeError("enabled and valid must be bools")
        if not isinstance(self.run_state, str) or not self.run_state.strip():
            raise ValueError("run_state must be a non-empty string")


@dataclass(frozen=True, slots=True)
class G20SdkState:
    side: ArmSide
    logical_raw: tuple[int, ...]
    temperatures_c: tuple[float, ...]
    fault_codes: tuple[int, ...]
    source_sequence: int
    sample_monotonic_ns: int
    received_monotonic_ns: int
    valid: bool = True
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.side, ArmSide):
            object.__setattr__(self, "side", ArmSide(self.side))
        logical = tuple(self.logical_raw)
        if len(logical) != 20 or any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255 for item in logical):
            raise ValueError("logical_raw must contain 20 integers in 0..255")
        object.__setattr__(self, "logical_raw", logical)
        object.__setattr__(self, "temperatures_c", _floats(self.temperatures_c, length=20, field_name="temperatures_c"))
        faults = tuple(self.fault_codes)
        if len(faults) != 20 or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in faults
        ):
            raise ValueError("fault_codes must contain 20 integers >= 0")
        object.__setattr__(self, "fault_codes", faults)
        for field_name in ("source_sequence", "sample_monotonic_ns", "received_monotonic_ns"):
            object.__setattr__(self, field_name, _time(getattr(self, field_name), field_name=field_name))
        if self.received_monotonic_ns < self.sample_monotonic_ns:
            raise ValueError("received_monotonic_ns must be >= sample_monotonic_ns")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")


@dataclass(frozen=True, slots=True)
class G20FingerSample:
    normal_force_n: float | None
    tangential_force_n: float | None
    tangential_direction_raw: float | None
    matrix_weight_raw: tuple[tuple[int, ...], ...] | None
    approach_raw: float | None
    force_sample_monotonic_ns: int | None
    matrix_sample_monotonic_ns: int | None
    force_valid: bool
    matrix_valid: bool

    def __post_init__(self) -> None:
        for field_name in (
            "normal_force_n",
            "tangential_force_n",
            "tangential_direction_raw",
            "approach_raw",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_float(getattr(self, field_name), field_name=field_name),
            )
        direction = self.tangential_direction_raw
        if direction is not None and not 0.0 <= direction <= 255.0:
            raise ValueError("tangential_direction_raw must be in 0..255")
        matrix = self.matrix_weight_raw
        if matrix is not None:
            rows = tuple(tuple(row) for row in matrix)
            if len(rows) != 12 or any(len(row) != 6 for row in rows):
                raise ValueError("matrix_weight_raw must have shape 12x6")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for row in rows
                for value in row
            ):
                raise ValueError("matrix_weight_raw values must be uint8 integers")
            object.__setattr__(self, "matrix_weight_raw", rows)
        for field_name in (
            "force_sample_monotonic_ns",
            "matrix_sample_monotonic_ns",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_time(getattr(self, field_name), field_name=field_name),
            )
        if not isinstance(self.force_valid, bool) or not isinstance(self.matrix_valid, bool):
            raise TypeError("force_valid and matrix_valid must be bools")
        if self.force_valid and self.force_sample_monotonic_ns is None:
            raise ValueError("valid force data requires force_sample_monotonic_ns")
        if self.matrix_valid and (
            self.matrix_sample_monotonic_ns is None or self.matrix_weight_raw is None
        ):
            raise ValueError("valid matrix data requires matrix and sample time")


@dataclass(frozen=True, slots=True)
class G20TactileSample:
    side: ArmSide
    fingers: tuple[G20FingerSample, ...]
    calibration_id: str
    feature_extractor_id: str
    received_monotonic_ns: int
    source_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.side, ArmSide):
            object.__setattr__(self, "side", ArmSide(self.side))
        fingers = tuple(self.fingers)
        if len(fingers) != 5 or not all(isinstance(item, G20FingerSample) for item in fingers):
            raise TypeError("fingers must contain exactly five G20FingerSample values")
        object.__setattr__(self, "fingers", fingers)
        for field_name in ("calibration_id", "feature_extractor_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        object.__setattr__(
            self,
            "received_monotonic_ns",
            _time(self.received_monotonic_ns, field_name="received_monotonic_ns"),
        )
        object.__setattr__(
            self,
            "source_sequence",
            _time(self.source_sequence, field_name="source_sequence"),
        )


@runtime_checkable
class ArmSdkGateway(Protocol):
    def open(self, *, read_only: bool) -> None: ...
    def read_state(self, side: ArmSide) -> ArmSdkState: ...
    def fk(self, side: ArmSide, joints_rad: tuple[float, ...]) -> tuple[bool, SdkPoseRpy | None, str | None]: ...
    def ik(self, side: ArmSide, pose: SdkPoseRpy, seed_rad: tuple[float, ...]) -> tuple[bool, tuple[float, ...], str | None]: ...
    def submit_move_j(self, *, owner_id: str, command_id: str, side: ArmSide, joints_rad: tuple[float, ...], speed: float, acceleration: float, timeout_s: float) -> ExecutionHandle: ...
    def submit_move_jp(self, *, owner_id: str, command_id: str, side: ArmSide, pose: SdkPoseRpy, speed: float, acceleration: float, timeout_s: float) -> ExecutionHandle: ...
    def submit_move_l(self, *, owner_id: str, command_id: str, side: ArmSide, pose: SdkPoseRpy, speed: float, acceleration: float, timeout_s: float) -> ExecutionHandle: ...
    def submit_follow_p(self, *, owner_id: str, command_id: str, side: ArmSide, pose: SdkPoseRpy, timeout_s: float) -> ExecutionHandle: ...
    def get_result(self, command_id: str) -> CommandResult: ...
    def cancel(self, *, owner_id: str, command_id: str) -> CommandResult: ...
    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult: ...
    def close(self) -> None: ...


@runtime_checkable
class G20SdkGateway(Protocol):
    def open(self, side: ArmSide, *, read_only: bool) -> None: ...
    def read_state(self, side: ArmSide) -> G20SdkState: ...
    def read_tactile(self, side: ArmSide) -> G20TactileSample: ...
    def submit_logical(self, *, owner_id: str, command_id: str, side: ArmSide, logical_raw: tuple[int, ...], timeout_s: float) -> ExecutionHandle: ...
    def hold(self, *, owner_id: str, side: ArmSide) -> CommandResult: ...
    def stop(self, *, owner_id: str, side: ArmSide, reason: str) -> CommandResult: ...
    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult: ...
    def get_result(self, command_id: str) -> CommandResult: ...
    def close(self) -> None: ...


__all__ = [
    "ArmSdkGateway",
    "ArmSdkState",
    "G20FingerSample",
    "G20SdkGateway",
    "G20SdkState",
    "G20TactileSample",
    "SdkPoseRpy",
]
