"""Injectable monotonic clocks for real-time and zero-wait offline execution."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def monotonic_ns(self) -> int:
        """Return monotonic nanoseconds."""

    def sleep(self, seconds: float) -> None:
        """Wait or deterministically advance by a positive duration."""


class SystemClock:
    """Production-compatible clock wrapper."""

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class ManualClock:
    """Deterministic monotonic clock whose sleep advances without wall time."""

    def __init__(self, *, start_ns: int = 1_000_000_000) -> None:
        if isinstance(start_ns, bool) or not isinstance(start_ns, int) or start_ns < 0:
            raise ValueError("start_ns must be an integer >= 0")
        self._now_ns = start_ns

    def monotonic_ns(self) -> int:
        return self._now_ns

    def sleep(self, seconds: float) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise TypeError("seconds must be a real number")
        if seconds < 0.0:
            raise ValueError("seconds must be >= 0")
        self._now_ns += round(float(seconds) * 1_000_000_000)

    def advance_s(self, seconds: float) -> None:
        self.sleep(seconds)


__all__ = ["Clock", "ManualClock", "SystemClock"]
