"""SDK-neutral robot ports and command DTOs for C1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from proto import (
    ArmSide,
    ArmState,
    CommandResult,
    ExecutionHandle,
    HandState,
    HandSynergy,
    Pose,
    TactileFrame,
)

from .control import ControlLease


def _command_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("command_id must be a non-empty string")
    return value.strip()


def _side(value: object) -> ArmSide:
    if isinstance(value, ArmSide):
        return value
    try:
        return ArmSide(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("side must be left or right") from exc


def _positive(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{field_name} must be finite and > 0")
    return converted


def _joints(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("joint_positions_rad must be a seven-value sequence")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("joint_positions_rad must be a seven-value sequence") from exc
    if len(items) != 7:
        raise ValueError("joint_positions_rad must contain exactly seven values")
    result: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"joint_positions_rad[{index}] must be a real number")
        converted = float(item)
        if not math.isfinite(converted):
            raise ValueError(f"joint_positions_rad[{index}] must be finite")
        result.append(converted)
    return tuple(result)


def _hand_joints(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("hand joint positions must be a sixteen-value sequence")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("hand joint positions must be a sixteen-value sequence") from exc
    if len(items) != 16:
        raise ValueError("hand joint positions must contain exactly sixteen values")
    result = tuple(float(item) for item in items)
    if any(not math.isfinite(item) for item in result):
        raise ValueError("hand joint positions must be finite")
    return result


@dataclass(frozen=True, slots=True)
class JointMoveCommand:
    command_id: str
    side: ArmSide
    joint_positions_rad: tuple[float, ...]
    speed: float
    acceleration: float
    timeout_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _command_id(self.command_id))
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "joint_positions_rad", _joints(self.joint_positions_rad))
        for field_name in ("speed", "acceleration", "timeout_s"):
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), field_name=field_name),
            )


@dataclass(frozen=True, slots=True)
class PoseMoveCommand:
    command_id: str
    side: ArmSide
    target_pose: Pose
    speed: float
    acceleration: float
    timeout_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _command_id(self.command_id))
        object.__setattr__(self, "side", _side(self.side))
        if not isinstance(self.target_pose, Pose):
            raise TypeError("target_pose must be a Pose")
        for field_name in ("speed", "acceleration", "timeout_s"):
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), field_name=field_name),
            )


@dataclass(frozen=True, slots=True)
class LinearMoveCommand(PoseMoveCommand):
    """MoveL semantic command."""


@dataclass(frozen=True, slots=True)
class FollowPoseCommand:
    command_id: str
    side: ArmSide
    target_pose: Pose
    timeout_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _command_id(self.command_id))
        object.__setattr__(self, "side", _side(self.side))
        if not isinstance(self.target_pose, Pose):
            raise TypeError("target_pose must be a Pose")
        object.__setattr__(
            self,
            "timeout_s",
            _positive(self.timeout_s, field_name="timeout_s"),
        )


@dataclass(frozen=True, slots=True)
class HandSynergyCommand:
    command_id: str
    side: ArmSide
    target: HandSynergy
    timeout_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _command_id(self.command_id))
        object.__setattr__(self, "side", _side(self.side))
        if not isinstance(self.target, HandSynergy):
            raise TypeError("target must be a HandSynergy")
        object.__setattr__(
            self,
            "timeout_s",
            _positive(self.timeout_s, field_name="timeout_s"),
        )


@dataclass(frozen=True, slots=True)
class HandJointPositionCommand:
    """Command the sixteen independent G20 channels in frozen SDK order."""

    command_id: str
    side: ArmSide
    joint_positions_rad: tuple[float, ...]
    timeout_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _command_id(self.command_id))
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(
            self,
            "joint_positions_rad",
            _hand_joints(self.joint_positions_rad),
        )
        object.__setattr__(
            self,
            "timeout_s",
            _positive(self.timeout_s, field_name="timeout_s"),
        )


@dataclass(frozen=True, slots=True)
class FKResult:
    success: bool
    pose: Pose | None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if self.success != isinstance(self.pose, Pose):
            raise ValueError("successful FK requires pose; failed FK must not provide one")


@dataclass(frozen=True, slots=True)
class IKResult:
    success: bool
    joint_positions_rad: tuple[float, ...]
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        joints = _joints(self.joint_positions_rad) if self.success else tuple(self.joint_positions_rad)
        if not self.success and joints:
            raise ValueError("failed IK must return an empty joint tuple")
        object.__setattr__(self, "joint_positions_rad", joints)


@dataclass(frozen=True, slots=True)
class HandSession:
    side: ArmSide
    read_only: bool
    opened_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _side(self.side))
        if not isinstance(self.read_only, bool):
            raise TypeError("read_only must be a bool")
        if (
            isinstance(self.opened_monotonic_ns, bool)
            or not isinstance(self.opened_monotonic_ns, int)
            or self.opened_monotonic_ns < 0
        ):
            raise ValueError("opened_monotonic_ns must be an integer >= 0")


@runtime_checkable
class ArmControlPort(Protocol):
    def get_state(self, side: ArmSide) -> ArmState: ...
    def fk(self, side: ArmSide, joint_positions_rad: tuple[float, ...]) -> FKResult: ...
    def ik(self, side: ArmSide, target_pose: Pose, seed_joint_positions_rad: tuple[float, ...]) -> IKResult: ...
    def move_joint(self, lease: ControlLease, command: JointMoveCommand) -> ExecutionHandle: ...
    def move_pose(self, lease: ControlLease, command: PoseMoveCommand) -> ExecutionHandle: ...
    def move_linear(self, lease: ControlLease, command: LinearMoveCommand) -> ExecutionHandle: ...
    def follow_pose(self, lease: ControlLease, command: FollowPoseCommand) -> ExecutionHandle: ...
    def get_result(self, command_id: str) -> CommandResult: ...
    def cancel(self, lease: ControlLease, command_id: str) -> CommandResult: ...
    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult: ...
    def close(self) -> None: ...


@runtime_checkable
class HandControlPort(Protocol):
    def open(self, *, read_only: bool) -> HandSession: ...
    def get_state(self, side: ArmSide) -> HandState: ...
    def command_synergy(self, lease: ControlLease, command: HandSynergyCommand) -> ExecutionHandle: ...
    def command_joint_positions(
        self, lease: ControlLease, command: HandJointPositionCommand
    ) -> ExecutionHandle: ...
    def hold(self, lease: ControlLease, side: ArmSide) -> CommandResult: ...
    def stop(self, lease: ControlLease, side: ArmSide, reason: str) -> CommandResult: ...
    def get_result(self, command_id: str) -> CommandResult: ...
    def close(self) -> None: ...


@runtime_checkable
class TactilePort(Protocol):
    def get_frame(self, side: ArmSide) -> TactileFrame: ...
    def close(self) -> None: ...


__all__ = [
    "ArmControlPort",
    "FKResult",
    "FollowPoseCommand",
    "HandControlPort",
    "HandJointPositionCommand",
    "HandSession",
    "HandSynergyCommand",
    "IKResult",
    "JointMoveCommand",
    "LinearMoveCommand",
    "PoseMoveCommand",
    "TactilePort",
]
