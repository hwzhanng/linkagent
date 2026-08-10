"""Composition root for synchronized RGB-D -> instances -> VLM -> 3D target."""

from __future__ import annotations

from proto import Observation, TomatoVisionResult

from .ports import (
    RgbdCapturePort,
    TomatoGeometryPort,
    TomatoInstanceSegmentationPort,
    VlmTargetSelectorPort,
)


class TomatoVisionPipeline:
    def __init__(
        self,
        *,
        capture: RgbdCapturePort,
        segmentation: TomatoInstanceSegmentationPort,
        selector: VlmTargetSelectorPort,
        geometry: TomatoGeometryPort,
        max_rgb_depth_skew_s: float = 0.035,
        minimum_detector_confidence: float = 0.50,
        minimum_selection_confidence: float = 0.60,
        expected_calibration_id: str | None = None,
        expected_device_id: str | None = None,
    ) -> None:
        if not isinstance(capture, RgbdCapturePort):
            raise TypeError("capture must implement RgbdCapturePort")
        if not isinstance(segmentation, TomatoInstanceSegmentationPort):
            raise TypeError("segmentation must implement TomatoInstanceSegmentationPort")
        if not isinstance(selector, VlmTargetSelectorPort):
            raise TypeError("selector must implement VlmTargetSelectorPort")
        if not isinstance(geometry, TomatoGeometryPort):
            raise TypeError("geometry must implement TomatoGeometryPort")
        if max_rgb_depth_skew_s <= 0.0:
            raise ValueError("max_rgb_depth_skew_s must be > 0")
        for name, value in {
            "minimum_detector_confidence": minimum_detector_confidence,
            "minimum_selection_confidence": minimum_selection_confidence,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.capture = capture
        self.segmentation = segmentation
        self.selector = selector
        self.geometry = geometry
        self.max_rgb_depth_skew_s = float(max_rgb_depth_skew_s)
        self.minimum_detector_confidence = float(minimum_detector_confidence)
        self.minimum_selection_confidence = float(minimum_selection_confidence)
        for name, value in {
            "expected_calibration_id": expected_calibration_id,
            "expected_device_id": expected_device_id,
        }.items():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        self.expected_calibration_id = (
            None if expected_calibration_id is None else expected_calibration_id.strip()
        )
        self.expected_device_id = (
            None if expected_device_id is None else expected_device_id.strip()
        )
        self._closed = False

    def observe(
        self,
        observation: Observation,
        *,
        requested_track_id: str | None = None,
    ) -> TomatoVisionResult:
        if self._closed:
            raise RuntimeError("tomato vision pipeline is closed")
        frame = self.capture.capture(observation)
        if (
            self.expected_calibration_id is not None
            and frame.intrinsics.calibration_id != self.expected_calibration_id
        ):
            raise ValueError("RGB-D frame calibration ID does not match the deployment profile")
        if self.expected_device_id is not None and frame.device_id != self.expected_device_id:
            raise ValueError("RGB-D device ID does not match the deployment profile")
        if not frame.depth_aligned_to_rgb:
            raise ValueError("depth is not aligned to the RGB image")
        if frame.skew_s > self.max_rgb_depth_skew_s:
            raise ValueError("RGB/depth timestamp skew exceeds the configured limit")
        candidates = tuple(
            candidate
            for candidate in self.segmentation.segment(frame)
            if candidate.detector_confidence >= self.minimum_detector_confidence
        )
        if not candidates:
            raise ValueError("no instance candidate passed the confidence gate")
        selection = self.selector.select(
            frame,
            candidates,
            requested_track_id=requested_track_id,
        )
        if selection.confidence < self.minimum_selection_confidence:
            raise ValueError("VLM selection confidence is below the configured limit")
        candidate = next(
            item for item in candidates if item.instance_id == selection.selected_instance_id
        )
        geometry = self.geometry.estimate(frame, candidate, selection)
        return TomatoVisionResult(
            frame=frame,
            candidates=candidates,
            selection=selection,
            geometry=geometry,
            details={
                "authority": {
                    "semantic": "vlm_selection_over_numbered_candidates",
                    "pixels": "validated_instance_support",
                    "metric_geometry": "aligned_depth_and_calibration",
                },
                "requested_track_id": requested_track_id,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for resource in (self.geometry, self.selector, self.segmentation, self.capture):
            try:
                resource.close()
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("tomato vision pipeline close failed", errors)


__all__ = ["TomatoVisionPipeline"]
