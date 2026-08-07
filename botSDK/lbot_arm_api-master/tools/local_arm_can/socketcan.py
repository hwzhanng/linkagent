"""Small, safety-gated SocketCAN transport."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import time

from .protocol import CanFrame, ProtocolError


_CAN_FRAME = struct.Struct("=IB3x8s")
_INTERFACE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x000007FF


@dataclass(frozen=True, slots=True)
class InterfaceInfo:
    name: str
    exists: bool
    up: bool
    kind: str | None
    bitrate: int | None
    operstate: str | None
    driver: str | None
    device: str | None


def list_can_interfaces() -> list[InterfaceInfo]:
    result = subprocess.run(
        ["ip", "-details", "-json", "link", "show"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    rows = json.loads(result.stdout or "[]")
    interfaces: list[InterfaceInfo] = []
    for row in rows:
        # Some iproute2 builds return one empty JSON object per non-CAN link
        # for ``link show type can``.  Read all links and filter explicitly so
        # an empty object can never be mistaken for a usable CAN adapter.
        if row.get("linkinfo", {}).get("info_kind") != "can":
            continue
        name = row.get("ifname", "")
        details = row.get("linkinfo", {}).get("info_data", {})
        device_path = Path("/sys/class/net") / name / "device"
        driver_path = device_path / "driver"
        interfaces.append(
            InterfaceInfo(
                name=name,
                exists=True,
                up="UP" in row.get("flags", []),
                kind=row.get("linkinfo", {}).get("info_kind"),
                bitrate=details.get("bittiming", {}).get("bitrate")
                or details.get("bitrate"),
                operstate=row.get("operstate"),
                driver=driver_path.resolve().name if driver_path.exists() else None,
                device=str(device_path.resolve()) if device_path.exists() else None,
            )
        )
    return interfaces


def interface_info(interface: str) -> InterfaceInfo:
    validate_interface_name(interface)
    for item in list_can_interfaces():
        if item.name == interface:
            return item
    return InterfaceInfo(interface, False, False, None, None, None, None, None)


def validate_interface_name(interface: str) -> None:
    if not _INTERFACE_RE.fullmatch(interface):
        raise ProtocolError(f"invalid CAN interface name: {interface!r}")


def configure_interface(
    interface: str,
    bitrate: int = 1_000_000,
    *,
    allow_can0: bool = False,
) -> None:
    """Configure an explicitly selected interface; never called implicitly."""
    validate_interface_name(interface)
    if interface == "can0" and not allow_can0:
        raise ProtocolError(
            "can0 is protected because it belongs to the existing local left-hand chain"
        )
    if bitrate != 1_000_000:
        raise ProtocolError("the recovered arm protocol requires exactly 1 Mbit/s")
    subprocess.run(["ip", "link", "set", "dev", interface, "down"], check=True)
    subprocess.run(
        ["ip", "link", "set", "dev", interface, "type", "can", "bitrate", str(bitrate)],
        check=True,
    )
    subprocess.run(["ip", "link", "set", "dev", interface, "txqueuelen", "1000"], check=True)
    subprocess.run(["ip", "link", "set", "dev", interface, "up"], check=True)


class SocketCan:
    def __init__(
        self,
        interface: str,
        *,
        tx_enabled: bool = False,
        allow_can0: bool = False,
        timeout: float = 0.1,
    ) -> None:
        validate_interface_name(interface)
        if interface == "can0" and not allow_can0:
            raise ProtocolError(
                "can0 is protected because it belongs to the existing local left-hand chain; "
                "select the right-arm adapter's separate interface"
            )
        self.interface = interface
        self.tx_enabled = tx_enabled
        self._socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self._socket.settimeout(timeout)
        self._socket.bind((interface,))

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "SocketCan":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def receive(self) -> CanFrame | None:
        try:
            raw = self._socket.recv(_CAN_FRAME.size)
        except TimeoutError:
            return None
        if len(raw) != _CAN_FRAME.size:
            raise OSError(f"short CAN frame: {len(raw)} bytes")
        raw_id, dlc, payload = _CAN_FRAME.unpack(raw)
        if raw_id & (CAN_EFF_FLAG | CAN_RTR_FLAG | CAN_ERR_FLAG):
            return None
        if dlc > 8:
            raise OSError(f"invalid classic CAN DLC: {dlc}")
        return CanFrame(
            can_id=raw_id & CAN_SFF_MASK,
            data=payload[:dlc],
            timestamp=time.time(),
            direction="R",
            interface=self.interface,
        )

    def send(self, frame: CanFrame) -> None:
        if not self.tx_enabled:
            raise PermissionError("CAN transmission is locked; start with explicit TX authorization")
        payload = frame.data.ljust(8, b"\x00")
        written = self._socket.send(_CAN_FRAME.pack(frame.can_id, frame.dlc, payload))
        if written != _CAN_FRAME.size:
            raise OSError(f"short CAN write: {written} bytes")
