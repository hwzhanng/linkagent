"""Deterministic replay adapters for YOLO-seg, VLM choice, and RGB-D evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from module.robotics import Clock
from proto import (
    CameraIntrinsics,
    Observation,
    Pose,
    SynchronizedRgbdFrame,
    TomatoInstance2D,
    VlmTargetSelection,
)


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _sequence(value: object, *, field_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{field_name} must be a sequence")
    if not isinstance(value, Sequence):
        try:
            value = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"{field_name} must be a sequence") from exc
    return tuple(value)


def _vision(observation: Observation) -> Mapping[str, Any]:
    return _mapping(observation.metadata.get("tomato_vision"), field_name="metadata.tomato_vision")


class ReplayRgbdCapturePort:
    """Read a previously synchronized frame; never opens a camera device."""

    def __init__(self, *, clock: Clock) -> None:
        if not isinstance(clock, Clock):
            raise TypeError("clock must implement Clock")
        self._clock = clock
        self._closed = False

    def capture(self, observation: Observation) -> SynchronizedRgbdFrame:
        if self._closed:
            raise RuntimeError("RGB-D replay port is closed")
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        vision = _vision(observation)
        camera = _mapping(vision.get("camera"), field_name="tomato_vision.camera")
        intrinsics = CameraIntrinsics(**dict(_mapping(camera.get("intrinsics"), field_name="camera.intrinsics")))
        pose = Pose(**dict(_mapping(camera.get("camera_pose_in_world"), field_name="camera.camera_pose_in_world")))
        now_ns = self._clock.monotonic_ns()
        rgb_ns = camera.get("rgb_sample_monotonic_ns", now_ns)
        depth_ns = camera.get("depth_sample_monotonic_ns", rgb_ns)
        return SynchronizedRgbdFrame(
            observation=observation,
            intrinsics=intrinsics,
            camera_pose_in_world=pose,
            rgb_sample_monotonic_ns=rgb_ns,
            depth_sample_monotonic_ns=depth_ns,
            depth_aligned_to_rgb=camera.get("depth_aligned_to_rgb", False),
            device_id=camera.get("device_id", "replay-unknown"),
            evidence_sources=tuple(camera.get("evidence_sources", ("replay:rgbd",))),
        )

    def close(self) -> None:
        self._closed = True


class ReplayYoloSegmentationPort:
    """Replay per-instance masks using the same DTO a YOLO worker must return."""

    def __init__(self) -> None:
        self._closed = False

    def segment(self, frame: SynchronizedRgbdFrame) -> tuple[TomatoInstance2D, ...]:
        if self._closed:
            raise RuntimeError("segmentation replay port is closed")
        rows = _sequence(
            _vision(frame.observation).get("instances", ()),
            field_name="tomato_vision.instances",
        )
        candidates: list[TomatoInstance2D] = []
        for index, row in enumerate(rows):
            options = dict(_mapping(row, field_name=f"instances[{index}]"))
            candidates.append(TomatoInstance2D(**options))
        if not candidates:
            raise ValueError("YOLO-seg returned no tomato instances")
        return tuple(candidates)

    def close(self) -> None:
        self._closed = True


class ReplayVlmTargetSelectorPort:
    """Replay a Set-of-Mark VLM decision and validate it against YOLO masks."""

    def __init__(self) -> None:
        self._closed = False

    def select(
        self,
        frame: SynchronizedRgbdFrame,
        candidates: tuple[TomatoInstance2D, ...],
        *,
        requested_track_id: str | None = None,
    ) -> VlmTargetSelection:
        if self._closed:
            raise RuntimeError("VLM selector replay port is closed")
        selection = VlmTargetSelection(
            **dict(
                _mapping(
                    _vision(frame.observation).get("vlm_selection"),
                    field_name="tomato_vision.vlm_selection",
                )
            )
        )
        by_instance = {candidate.instance_id: candidate for candidate in candidates}
        try:
            candidate = by_instance[selection.selected_instance_id]
        except KeyError as exc:
            raise ValueError("VLM selected an ID that YOLO-seg did not produce") from exc
        if requested_track_id is not None and candidate.track_id != requested_track_id:
            raise ValueError(
                "VLM selection changed target identity during pinned re-observation"
            )
        if selection.point_xy is not None:
            containing = tuple(
                item
                for item in candidates
                if item.contains_pixel(selection.point_xy)
            )
            if len(containing) != 1 or containing[0].instance_id != candidate.instance_id:
                raise ValueError(
                    "VLM point must belong to exactly one selected YOLO instance mask"
                )
        return selection

    def close(self) -> None:
        self._closed = True


__all__ = [
    "ReplayRgbdCapturePort",
    "ReplayVlmTargetSelectorPort",
    "ReplayYoloSegmentationPort",
]
