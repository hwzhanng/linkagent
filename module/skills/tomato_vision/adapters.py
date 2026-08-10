"""Adapters from the vision pipeline to existing perception/re-observation ports."""

from __future__ import annotations

from module.robotics import Clock
from proto import (
    Observation,
    TargetReobserveEvidence,
    TargetReobserveRequest,
    TomatoDetectionResult,
)

from .factory import build_replay_vision_pipeline


class ReplayTomatoVisionPerceptionPort:
    """Publish only the VLM-selected, metric 3D target to the planning layer."""

    def __init__(self, *, clock: Clock, **pipeline_options: object) -> None:
        self._clock = clock
        self._pipeline = build_replay_vision_pipeline(clock=clock, **pipeline_options)

    def detect(self, observation: Observation) -> TomatoDetectionResult:
        result = self._pipeline.observe(observation)
        return TomatoDetectionResult(
            targets=(result.geometry.target,),
            observation_sequence=observation.sequence,
            model="yolo_seg_vlm_rgbd_replay_v1",
            sample_monotonic_ns=result.geometry.target.sample_monotonic_ns,
        )

    def close(self) -> None:
        self._pipeline.close()


class ReplayTomatoVisionObservationPort:
    """Re-observe a pinned track and reject any VLM identity switch."""

    def __init__(self, *, clock: Clock, **pipeline_options: object) -> None:
        self._pipeline = build_replay_vision_pipeline(clock=clock, **pipeline_options)

    def reobserve(self, request: TargetReobserveRequest) -> TargetReobserveEvidence:
        result = self._pipeline.observe(
            request.observation,
            requested_track_id=request.target_id,
        )
        target = result.geometry.target
        return TargetReobserveEvidence(
            target_id=request.target_id,
            observed=target.target_id == request.target_id,
            occlusion_ratio=target.occlusion,
            depth_available=True,
            evidence_sources=result.geometry.evidence_sources,
            target=target,
            details={
                "instance_id": result.geometry.instance_id,
                "candidate_count": len(result.candidates),
                "valid_depth_ratio": result.geometry.valid_depth_ratio,
                "point_count": result.geometry.point_count,
                "inlier_count": result.geometry.inlier_count,
                "fit_residual_m": result.geometry.fit_residual_m,
                "calibration_id": result.geometry.calibration_id,
                "rgb_depth_skew_s": result.frame.skew_s,
                "selector_model": result.selection.selector_model,
            },
        )

    def close(self) -> None:
        self._pipeline.close()


__all__ = [
    "ReplayTomatoVisionObservationPort",
    "ReplayTomatoVisionPerceptionPort",
]
