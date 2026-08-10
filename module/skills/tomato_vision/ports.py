"""Replaceable ports for the RGB-D tomato vision pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from proto import (
    Observation,
    SynchronizedRgbdFrame,
    TomatoGeometryEvidence,
    TomatoInstance2D,
    VlmTargetSelection,
)


@runtime_checkable
class RgbdCapturePort(Protocol):
    def capture(self, observation: Observation) -> SynchronizedRgbdFrame: ...
    def close(self) -> None: ...


@runtime_checkable
class TomatoInstanceSegmentationPort(Protocol):
    def segment(
        self, frame: SynchronizedRgbdFrame
    ) -> tuple[TomatoInstance2D, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class VlmTargetSelectorPort(Protocol):
    def select(
        self,
        frame: SynchronizedRgbdFrame,
        candidates: tuple[TomatoInstance2D, ...],
        *,
        requested_track_id: str | None = None,
    ) -> VlmTargetSelection: ...

    def close(self) -> None: ...


@runtime_checkable
class TomatoGeometryPort(Protocol):
    def estimate(
        self,
        frame: SynchronizedRgbdFrame,
        candidate: TomatoInstance2D,
        selection: VlmTargetSelection,
    ) -> TomatoGeometryEvidence: ...

    def close(self) -> None: ...


__all__ = [
    "RgbdCapturePort",
    "TomatoGeometryPort",
    "TomatoInstanceSegmentationPort",
    "VlmTargetSelectorPort",
]
