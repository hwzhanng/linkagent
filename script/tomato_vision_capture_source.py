"""SDK-neutral lifecycle for a synchronized RGB-D capture adapter.

The future Orbbec implementation belongs behind :class:`SynchronizedRgbdSource`.
This module never imports a vendor SDK, opens hardware, performs D2C alignment,
or invents calibration.  It only validates source lifecycle and publishes each
already synchronized frame through the atomic spool writer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Event
from typing import Protocol, runtime_checkable

from script.tomato_vision_spool import RgbdSpoolFrame, TomatoVisionSpoolWriter


def _identity(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@runtime_checkable
class SynchronizedRgbdSource(Protocol):
    """One explicitly configured, frame-synchronized RGB-D device source."""

    def start(self) -> None:
        """Open and configure the exact device; raise if any gate is unavailable."""

    def read_synchronized_frame(self, *, timeout_s: float) -> RgbdSpoolFrame:
        """Return one D2C-aligned frame or raise on timeout/data loss."""

    def close(self) -> None:
        """Release the source.  This operation must be idempotent."""


@dataclass(frozen=True, slots=True)
class CaptureLoopSummary:
    published_count: int
    first_sample_monotonic_ns: int | None
    last_sample_monotonic_ns: int | None
    stopped_by_event: bool


def run_capture_loop(
    *,
    source: SynchronizedRgbdSource,
    writer: TomatoVisionSpoolWriter,
    expected_device_id: str,
    expected_calibration_id: str,
    stop_event: Event,
    read_timeout_s: float = 1.0,
    max_frames: int | None = None,
) -> CaptureLoopSummary:
    """Publish frames until explicitly stopped or ``max_frames`` is reached.

    Source, serial and calibration mismatches are fatal.  No mock or stale-frame
    fallback is selected here; callers may implement bounded service restart at
    a higher layer after surfacing the failure.
    """

    if not isinstance(source, SynchronizedRgbdSource):
        raise TypeError("source must implement SynchronizedRgbdSource")
    if not isinstance(writer, TomatoVisionSpoolWriter):
        raise TypeError("writer must be TomatoVisionSpoolWriter")
    device_id = _identity(expected_device_id, field_name="expected_device_id")
    calibration_id = _identity(
        expected_calibration_id, field_name="expected_calibration_id"
    )
    if not isinstance(stop_event, Event):
        raise TypeError("stop_event must be threading.Event")
    if (
        isinstance(read_timeout_s, bool)
        or not isinstance(read_timeout_s, (int, float))
        or not math.isfinite(float(read_timeout_s))
        or not 0.001 <= float(read_timeout_s) <= 30.0
    ):
        raise ValueError("read_timeout_s must be finite and in [0.001, 30]")
    if max_frames is not None and (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames < 1
    ):
        raise ValueError("max_frames must be an integer >= 1 or None")

    published = 0
    first_sample_ns: int | None = None
    last_sample_ns: int | None = None
    baseline_intrinsics = None
    baseline_depth_scale: float | None = None
    try:
        source.start()
        while not stop_event.is_set() and (
            max_frames is None or published < max_frames
        ):
            frame = source.read_synchronized_frame(timeout_s=float(read_timeout_s))
            if not isinstance(frame, RgbdSpoolFrame):
                raise TypeError("capture source returned a non-RgbdSpoolFrame value")
            if frame.device_id != device_id:
                raise ValueError("capture source device_id does not match deployment profile")
            if frame.intrinsics.calibration_id != calibration_id:
                raise ValueError(
                    "capture source calibration_id does not match deployment profile"
                )
            if baseline_intrinsics is None:
                baseline_intrinsics = frame.intrinsics
                baseline_depth_scale = frame.depth_scale_m_per_unit
            elif (
                frame.intrinsics != baseline_intrinsics
                or frame.depth_scale_m_per_unit != baseline_depth_scale
            ):
                raise ValueError(
                    "capture source calibration parameters changed without a new profile"
                )
            publication = writer.publish(frame)
            sample_ns = publication.pointer.sample_monotonic_ns
            if first_sample_ns is None:
                first_sample_ns = sample_ns
            last_sample_ns = sample_ns
            published += 1
    finally:
        source.close()
    return CaptureLoopSummary(
        published_count=published,
        first_sample_monotonic_ns=first_sample_ns,
        last_sample_monotonic_ns=last_sample_ns,
        stopped_by_event=stop_event.is_set(),
    )


__all__ = [
    "CaptureLoopSummary",
    "SynchronizedRgbdSource",
    "run_capture_loop",
]
