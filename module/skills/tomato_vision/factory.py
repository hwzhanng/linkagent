"""Reviewed constructors for the dependency-free replay vision stack."""

from __future__ import annotations

from module.robotics import Clock

from .geometry import RobustMaskedDepthGeometryPort
from .pipeline import TomatoVisionPipeline
from .replay import (
    ReplayRgbdCapturePort,
    ReplayVlmTargetSelectorPort,
    ReplayYoloSegmentationPort,
)


def build_replay_vision_pipeline(
    *,
    clock: Clock,
    max_rgb_depth_skew_s: float = 0.035,
    minimum_detector_confidence: float = 0.50,
    minimum_selection_confidence: float = 0.60,
    expected_calibration_id: str | None = None,
    expected_device_id: str | None = None,
    min_points: int = 6,
    min_valid_depth_ratio: float = 0.60,
    max_fit_residual_m: float = 0.012,
    min_axis_m: float = 0.015,
    max_axis_m: float = 0.060,
    silhouette_axis_floor_ratio: float = 0.0,
    visible_surface_center_correction_ratio: float = 0.0,
    pull_axis_prior_world: tuple[float, ...] | None = None,
    max_pull_axis_deviation_deg: float = 180.0,
) -> TomatoVisionPipeline:
    return TomatoVisionPipeline(
        capture=ReplayRgbdCapturePort(clock=clock),
        segmentation=ReplayYoloSegmentationPort(),
        selector=ReplayVlmTargetSelectorPort(),
        geometry=RobustMaskedDepthGeometryPort(
            min_points=min_points,
            min_valid_depth_ratio=min_valid_depth_ratio,
            max_fit_residual_m=max_fit_residual_m,
            min_axis_m=min_axis_m,
            max_axis_m=max_axis_m,
            silhouette_axis_floor_ratio=silhouette_axis_floor_ratio,
            visible_surface_center_correction_ratio=visible_surface_center_correction_ratio,
            pull_axis_prior_world=pull_axis_prior_world,
            max_pull_axis_deviation_deg=max_pull_axis_deviation_deg,
        ),
        max_rgb_depth_skew_s=max_rgb_depth_skew_s,
        minimum_detector_confidence=minimum_detector_confidence,
        minimum_selection_confidence=minimum_selection_confidence,
        expected_calibration_id=expected_calibration_id,
        expected_device_id=expected_device_id,
    )


__all__ = ["build_replay_vision_pipeline"]
