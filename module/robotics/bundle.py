"""Composition boundary shared by Mock, Isaac, shadow, and hardware backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from proto import ArmSide

from .clock import Clock
from .control import ControlLeaseManager
from .ports import ArmControlPort, HandControlPort, TactilePort


@runtime_checkable
class RoboticsBundle(Protocol):
    """One same-side robot view owned by a task composition root."""

    side: ArmSide
    clock: Clock
    leases: ControlLeaseManager
    arm: ArmControlPort
    hand: HandControlPort
    tactile: TactilePort

    def task_component_overrides(self) -> Mapping[str, Mapping[str, Any]]:
        """Return backend-specific test/config overrides, normally empty."""

    def verifier_context(self) -> Mapping[str, Any]:
        """Return backend-owned evidence resources for verifier construction."""

    def close(self) -> None: ...


__all__ = ["RoboticsBundle"]
