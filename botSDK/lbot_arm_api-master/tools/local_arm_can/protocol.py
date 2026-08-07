"""Recovered Linkbot A7_V2 joint-motor CAN protocol.

The implementation was reconstructed from the unstripped AArch64
``lbot_lkls_i1_hal/lbot_arm_driver`` shipped on the controller and checked
against passive ``candump -x`` captures.  This module has no ROS dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
import struct
import time
from typing import Iterable, Literal


RIGHT_BASE_ID = 0x33
LEFT_BASE_ID = 0x3D
JOINT_COUNT = 7
MOTOREVO_IDS = frozenset((0x38, 0x39, 0x42, 0x43))

CLASSIC_POSITION_RANGE = (-10.0, 10.0)
CLASSIC_VELOCITY_RANGE = (-10.0, 10.0)
CLASSIC_EFFORT_RANGE = (-80.0, 80.0)

EVO_POSITION_RANGE = (-12.5, 12.5)
EVO_VELOCITY_RANGE = (-65.0, 65.0)
EVO_EFFORT_RANGE = (-30.0, 30.0)

_CANDUMP_RE = re.compile(
    r"^(?:\((?P<timestamp>[0-9.]+)\)\s+)?"
    r"(?P<interface>[A-Za-z0-9_.:-]+)\s+"
    r"(?P<can_id>[0-9A-Fa-f]{1,8})#(?P<data>[0-9A-Fa-f]*)"
    r"(?:\s+(?P<direction>[RT]))?\s*$"
)


class ProtocolError(ValueError):
    """Raised when a frame or command cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class CanFrame:
    can_id: int
    data: bytes = b""
    timestamp: float | None = None
    direction: Literal["R", "T"] | None = None
    interface: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.can_id <= 0x7FF:
            raise ProtocolError(f"only 11-bit CAN IDs are supported: {self.can_id:#x}")
        if len(self.data) > 8:
            raise ProtocolError(f"classic CAN payload exceeds 8 bytes: {len(self.data)}")

    @property
    def dlc(self) -> int:
        return len(self.data)

    def candump(self) -> str:
        direction = f" {self.direction}" if self.direction else ""
        interface = self.interface or "can?"
        return f"{interface} {self.can_id:03X}#{self.data.hex().upper()}{direction}"


@dataclass(frozen=True, slots=True)
class JointTelemetry:
    can_id: int
    joint_index: int
    motor_type: Literal["classic", "motorevo"]
    position: float
    velocity: float
    effort: float
    status: int
    error: int | None
    timestamp: float


@dataclass(frozen=True, slots=True)
class TemperatureTelemetry:
    can_id: int
    joint_index: int
    temperature: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class HandshakeReply:
    can_id: int
    joint_index: int
    ok: bool
    motor_type: Literal["classic", "motorevo"]
    timestamp: float


DecodedEvent = JointTelemetry | TemperatureTelemetry | HandshakeReply


@dataclass(slots=True)
class JointStateSlot:
    can_id: int
    position: float | None = None
    velocity: float | None = None
    effort: float | None = None
    temperature: float | None = None
    status: int | None = None
    error: int | None = None
    last_state_monotonic: float | None = None
    last_temperature_monotonic: float | None = None
    state_frames: int = 0
    temperature_frames: int = 0


@dataclass(slots=True)
class ArmState:
    base_id: int = RIGHT_BASE_ID
    stale_after: float = 0.25
    started_monotonic: float = field(default_factory=time.monotonic)
    rx_frames: int = 0
    tx_frames: int = 0
    unknown_frames: int = 0
    joints: list[JointStateSlot] = field(init=False)

    def __post_init__(self) -> None:
        validate_base_id(self.base_id)
        self.joints = [JointStateSlot(self.base_id + index) for index in range(JOINT_COUNT)]

    def update(self, event: DecodedEvent, *, now_monotonic: float | None = None) -> None:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        slot = self.joints[event.joint_index]
        if isinstance(event, JointTelemetry):
            slot.position = event.position
            slot.velocity = event.velocity
            slot.effort = event.effort
            slot.status = event.status
            slot.error = event.error
            slot.last_state_monotonic = now
            slot.state_frames += 1
        elif isinstance(event, TemperatureTelemetry):
            slot.temperature = event.temperature
            slot.last_temperature_monotonic = now
            slot.temperature_frames += 1

    def snapshot(self, *, now_monotonic: float | None = None) -> dict:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        joint_rows: list[dict] = []
        for index, slot in enumerate(self.joints):
            age = None
            if slot.last_state_monotonic is not None:
                age = max(0.0, now - slot.last_state_monotonic)
            row = asdict(slot)
            row.update(
                joint_index=index,
                joint_number=index + 1,
                motor_type=motor_type(slot.can_id),
                state_age=age,
                healthy=age is not None and age <= self.stale_after,
            )
            row.pop("last_state_monotonic")
            row.pop("last_temperature_monotonic")
            joint_rows.append(row)
        healthy = [row["healthy"] for row in joint_rows]
        return {
            "base_id": self.base_id,
            "base_id_hex": f"0x{self.base_id:03X}",
            "rx_frames": self.rx_frames,
            "tx_frames": self.tx_frames,
            "unknown_frames": self.unknown_frames,
            "healthy_joint_count": sum(healthy),
            "all_joints_healthy": all(healthy),
            "joints": joint_rows,
        }


def validate_base_id(base_id: int) -> None:
    if base_id not in (RIGHT_BASE_ID, LEFT_BASE_ID):
        raise ProtocolError(
            f"unsupported arm base ID {base_id:#x}; expected right 0x33 or left 0x3D"
        )


def motor_type(can_id: int) -> Literal["classic", "motorevo"]:
    return "motorevo" if can_id in MOTOREVO_IDS else "classic"


def joint_index(can_id: int, base_id: int) -> int | None:
    index = can_id - base_id
    return index if 0 <= index < JOINT_COUNT else None


def uint_to_float(raw: int, minimum: float, maximum: float, bits: int) -> float:
    maximum_raw = (1 << bits) - 1
    if not 0 <= raw <= maximum_raw:
        raise ProtocolError(f"{raw} does not fit in {bits} bits")
    return minimum + raw * (maximum - minimum) / maximum_raw


def float_to_uint(value: float, minimum: float, maximum: float, bits: int) -> int:
    if not math.isfinite(value):
        raise ProtocolError("CAN command values must be finite")
    clamped = min(maximum, max(minimum, value))
    maximum_raw = (1 << bits) - 1
    return int((clamped - minimum) * maximum_raw / (maximum - minimum))


def parse_candump_line(line: str) -> CanFrame:
    match = _CANDUMP_RE.match(line.strip())
    if not match:
        raise ProtocolError(f"invalid candump line: {line!r}")
    data_hex = match.group("data")
    if len(data_hex) % 2:
        raise ProtocolError(f"odd-length CAN payload: {data_hex!r}")
    timestamp = match.group("timestamp")
    return CanFrame(
        can_id=int(match.group("can_id"), 16),
        data=bytes.fromhex(data_hex),
        timestamp=float(timestamp) if timestamp else None,
        direction=match.group("direction"),
        interface=match.group("interface"),
    )


def decode_frame(frame: CanFrame, base_id: int = RIGHT_BASE_ID) -> DecodedEvent | None:
    validate_base_id(base_id)
    timestamp = time.time() if frame.timestamp is None else frame.timestamp

    index = joint_index(frame.can_id, base_id)
    if index is not None and frame.dlc >= 1:
        kind = motor_type(frame.can_id)
        if kind == "classic" and frame.dlc == 8 and frame.data[0] == 0x96:
            position_raw = (frame.data[1] << 8) | frame.data[2]
            velocity_raw = (frame.data[3] << 4) | (frame.data[4] >> 4)
            effort_raw = ((frame.data[4] & 0x0F) << 8) | frame.data[5]
            return JointTelemetry(
                can_id=frame.can_id,
                joint_index=index,
                motor_type=kind,
                position=uint_to_float(position_raw, *CLASSIC_POSITION_RANGE, 16),
                velocity=uint_to_float(velocity_raw, *CLASSIC_VELOCITY_RANGE, 12),
                effort=uint_to_float(effort_raw, *CLASSIC_EFFORT_RANGE, 12),
                status=frame.data[7],
                error=frame.data[7],
                timestamp=timestamp,
            )
        if kind == "classic" and frame.dlc >= 5 and frame.data[0] == 0x5F:
            return TemperatureTelemetry(
                can_id=frame.can_id,
                joint_index=index,
                temperature=struct.unpack("<f", frame.data[1:5])[0],
                timestamp=timestamp,
            )
        if kind == "classic" and frame.dlc >= 2 and frame.data[1] == 0x01:
            return HandshakeReply(frame.can_id, index, True, kind, timestamp)
        if kind == "motorevo" and frame.dlc == 8:
            if frame.data[:7] == b"\xff" * 7 and frame.data[7] in (0xFA, 0xFC, 0xFD, 0xFE):
                return None
            position_raw = (frame.data[1] << 8) | frame.data[2]
            velocity_raw = (frame.data[3] << 4) | (frame.data[4] >> 4)
            effort_raw = ((frame.data[4] & 0x0F) << 8) | frame.data[5]
            return JointTelemetry(
                can_id=frame.can_id,
                joint_index=index,
                motor_type=kind,
                position=uint_to_float(position_raw, *EVO_POSITION_RANGE, 16),
                velocity=uint_to_float(velocity_raw, *EVO_VELOCITY_RANGE, 12),
                effort=uint_to_float(effort_raw, *EVO_EFFORT_RANGE, 12),
                status=frame.data[6],
                error=frame.data[7],
                timestamp=timestamp,
            )

    # Motorevo configuration replies use motor_id + 0x600 and echo the
    # original motor ID in data[0].
    if frame.dlc >= 3 and frame.can_id >= 0x600:
        echoed_id = frame.data[0]
        index = joint_index(echoed_id, base_id)
        if index is not None and motor_type(echoed_id) == "motorevo":
            ok = frame.can_id == echoed_id + 0x600 and frame.data[1:3] == b"\x0b\x01"
            return HandshakeReply(echoed_id, index, ok, "motorevo", timestamp)
    return None


def handshake_frame(can_id: int) -> CanFrame:
    if motor_type(can_id) == "motorevo":
        return CanFrame(can_id + 0x600, bytes((0x67, 0x0A, 0, 0, 0, 0, 0x04, 0x76)))
    return CanFrame(can_id, b"\x00")


def state_query_frame(can_id: int) -> CanFrame:
    if motor_type(can_id) == "motorevo":
        return CanFrame(can_id, b"\xff\xff\xff\xff\xff\xff\xff\xfa")
    return CanFrame(can_id, b"\x06")


def temperature_query_frame(can_id: int) -> CanFrame | None:
    if motor_type(can_id) == "motorevo":
        return None
    return CanFrame(can_id, b"\x5f")


def set_auto_report_frame(can_id: int, period_ms: float) -> CanFrame | None:
    if motor_type(can_id) == "motorevo":
        return None
    return CanFrame(can_id, b"\x95" + struct.pack("<f", period_ms))


def control_mode_frame(can_id: int, mode: int = 4) -> CanFrame:
    if motor_type(can_id) == "motorevo":
        if mode not in (1, 4):
            raise ProtocolError("Motorevo supports recovered control modes 1 and 4 only")
        mode_byte = 0x01 if mode == 4 else 0x02
        return CanFrame(
            can_id + 0x600,
            bytes((0x67, 0x0B, mode_byte, 0, 0, 0, 0x15, 0x76)),
        )
    if not 0 <= mode <= 0xFF:
        raise ProtocolError("classic control mode must fit in one byte")
    return CanFrame(can_id, bytes((0x07, mode)))


def enable_frame(can_id: int) -> CanFrame:
    if motor_type(can_id) == "motorevo":
        return CanFrame(can_id, b"\xff\xff\xff\xff\xff\xff\xff\xfc")
    return CanFrame(can_id, b"\x2a\x01")


def disable_frame(can_id: int) -> CanFrame:
    if motor_type(can_id) == "motorevo":
        return CanFrame(can_id, b"\xff\xff\xff\xff\xff\xff\xff\xfd")
    return CanFrame(can_id, b"\x2a\x00")


def set_home_frame(can_id: int) -> CanFrame:
    if motor_type(can_id) == "motorevo":
        return CanFrame(can_id, b"\xff\xff\xff\xff\xff\xff\xff\xfe")
    return CanFrame(can_id, b"\x87")


def save_parameters_frame(can_id: int) -> CanFrame | None:
    if motor_type(can_id) == "motorevo":
        return None
    return CanFrame(can_id, b"\x0d")


def classic_float_command(can_id: int, command: int, value: float) -> CanFrame:
    if motor_type(can_id) != "classic":
        raise ProtocolError(f"command {command:#x} is a classic-motor command")
    if not 0 <= command <= 0xFF or not math.isfinite(value):
        raise ProtocolError("invalid float command")
    return CanFrame(can_id, bytes((command,)) + struct.pack("<f", value))


def position_frame(
    can_id: int,
    position: float,
    *,
    evo_parameters: Iterable[float] | None = None,
) -> CanFrame:
    if motor_type(can_id) == "classic":
        return classic_float_command(can_id, 0x0A, position)

    if evo_parameters is None:
        raise ProtocolError(
            "Motorevo position commands require six explicit parameters; "
            "implicit gains are intentionally forbidden"
        )
    parameters = tuple(float(value) for value in evo_parameters)
    if len(parameters) != 6:
        raise ProtocolError("Motorevo position command requires exactly six parameters")
    ranges = (
        EVO_VELOCITY_RANGE,
        (0.0, 500.0),
        (0.0, 5.0),
        (0.0, 500.0),
        (0.0, 250.0),
        (0.0, 0.05),
    )
    position_raw = float_to_uint(position, *EVO_POSITION_RANGE, 16)
    packed = [
        float_to_uint(value, minimum, maximum, 8)
        for value, (minimum, maximum) in zip(parameters, ranges, strict=True)
    ]
    return CanFrame(
        can_id,
        bytes((position_raw >> 8, position_raw & 0xFF, *packed)),
    )


def recovered_classic_parameter_frame(can_id: int, name: str, value: float) -> CanFrame:
    commands = {
        "current": 0x08,
        "velocity": 0x09,
        "position": 0x0A,
        "pp": 0x12,
        "pi": 0x13,
        "pd": 0x14,
        "current_limit": 0x58,
        "velocity_limit": 0x5A,
        "position_max": 0x83,
        "position_min": 0x84,
        "auto_report": 0x95,
    }
    try:
        command = commands[name]
    except KeyError as error:
        raise ProtocolError(f"unknown recovered classic parameter: {name}") from error
    return classic_float_command(can_id, command, value)
