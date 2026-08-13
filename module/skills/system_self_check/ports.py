"""Read-only health ports used by the harvest admission gate."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from proto import (
    CameraHealthSnapshot,
    EmergencyStopHealthSnapshot,
    VisionWorkerHealthSnapshot,
)


@runtime_checkable
class CameraHealthPort(Protocol):
    def snapshot(self) -> CameraHealthSnapshot:
        """Return current RGB-D and SDK health without starting a device."""

    def close(self) -> None: ...


@runtime_checkable
class EmergencyStopHealthPort(Protocol):
    def snapshot(self) -> EmergencyStopHealthSnapshot:
        """Return the read-only emergency-stop chain state."""

    def close(self) -> None: ...


@runtime_checkable
class VisionWorkerHealthPort(Protocol):
    def snapshot(self) -> VisionWorkerHealthSnapshot:
        """Return current isolated inference-worker health and identity."""

    def close(self) -> None: ...


__all__ = [
    "CameraHealthPort",
    "EmergencyStopHealthPort",
    "VisionWorkerHealthPort",
]
