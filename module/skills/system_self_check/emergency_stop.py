"""Offline emergency-stop health probe."""

from __future__ import annotations

from module.robotics import Clock
from proto import EmergencyStopHealthSnapshot


class MockEmergencyStopHealthPort:
    def __init__(
        self,
        *,
        clock: Clock,
        available: bool = True,
        healthy: bool | None = True,
        stale_by_s: float = 0.0,
    ) -> None:
        if not isinstance(available, bool):
            raise TypeError("available must be a bool")
        if healthy is not None and not isinstance(healthy, bool):
            raise TypeError("healthy must be a bool or None")
        if (
            isinstance(stale_by_s, bool)
            or not isinstance(stale_by_s, (int, float))
            or stale_by_s < 0
        ):
            raise ValueError("stale_by_s must be >= 0")
        self._clock = clock
        self._available = available
        self._healthy = healthy
        self._stale_by_ns = round(float(stale_by_s) * 1_000_000_000)
        self._closed = False

    def snapshot(self) -> EmergencyStopHealthSnapshot:
        if self._closed:
            raise RuntimeError("emergency-stop health port is closed")
        return EmergencyStopHealthSnapshot(
            backend_name="mock_emergency_stop",
            available=self._available,
            healthy=self._healthy,
            simulated=True,
            sample_monotonic_ns=max(
                0, self._clock.monotonic_ns() - self._stale_by_ns
            ),
            evidence_sources=("mock:emergency_stop_state",),
        )

    def close(self) -> None:
        self._closed = True


__all__ = ["MockEmergencyStopHealthPort"]
