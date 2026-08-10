"""Loopback-only client adapters for the isolated tomato vision worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from module.robotics import Clock
from proto import (
    Observation,
    TargetReobserveEvidence,
    TargetReobserveRequest,
    TomatoDetectionResult,
    TomatoVisionResult,
)

from .factory import build_replay_vision_pipeline


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


class TomatoVisionWorkerClient:
    """Submit bounded capture references and revalidate every worker response."""

    def __init__(
        self,
        *,
        clock: Clock,
        endpoint: str,
        capture_root: str | Path,
        worker_id: str,
        timeout_s: float = 8.0,
        max_response_bytes: int = 2_000_000,
        expected_calibration_id: str | None = None,
        expected_device_id: str | None = None,
        max_rgb_depth_skew_s: float = 0.035,
        minimum_detector_confidence: float = 0.50,
        minimum_selection_confidence: float = 0.60,
        min_points: int = 6,
        min_valid_depth_ratio: float = 0.60,
        max_fit_residual_m: float = 0.012,
        min_axis_m: float = 0.015,
        max_axis_m: float = 0.060,
        new_frame_wait_timeout_s: float = 0.5,
    ) -> None:
        if not isinstance(clock, Clock):
            raise TypeError("clock must implement Clock")
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in _LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "tomato vision worker endpoint must be credential-free loopback HTTP"
            )
        if parsed.path not in {"", "/"}:
            raise ValueError("tomato vision worker endpoint must not contain a path")
        if parsed.port is None:
            raise ValueError("tomato vision worker endpoint must include an explicit port")
        root = Path(capture_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1024
        ):
            raise ValueError("max_response_bytes must be an integer >= 1024")
        if (
            isinstance(new_frame_wait_timeout_s, bool)
            or not isinstance(new_frame_wait_timeout_s, (int, float))
            or not 0.0 <= new_frame_wait_timeout_s <= 5.0
        ):
            raise ValueError("new_frame_wait_timeout_s must be in [0, 5]")
        if new_frame_wait_timeout_s >= timeout_s:
            raise ValueError("new_frame_wait_timeout_s must be less than timeout_s")
        self._clock = clock
        self._endpoint = endpoint.rstrip("/")
        self._capture_root = root
        self._worker_id = worker_id.strip()
        self._timeout_s = float(timeout_s)
        self._max_response_bytes = max_response_bytes
        self._new_frame_wait_timeout_s = float(new_frame_wait_timeout_s)
        self._validator = build_replay_vision_pipeline(
            clock=clock,
            expected_calibration_id=expected_calibration_id,
            expected_device_id=expected_device_id,
            max_rgb_depth_skew_s=max_rgb_depth_skew_s,
            minimum_detector_confidence=minimum_detector_confidence,
            minimum_selection_confidence=minimum_selection_confidence,
            min_points=min_points,
            min_valid_depth_ratio=min_valid_depth_ratio,
            max_fit_residual_m=max_fit_residual_m,
            min_axis_m=min_axis_m,
            max_axis_m=max_axis_m,
        )
        self._request_index = 0
        self._closed = False

    def health(self) -> Mapping[str, object]:
        """Verify reachability, identity, schema, and backend readiness."""

        if self._closed:
            raise RuntimeError("tomato vision worker client is closed")
        request = Request(
            f"{self._endpoint}/health",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            raise RuntimeError(
                f"tomato vision worker health failed with HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("tomato vision worker is unavailable") from exc
        if len(raw) > self._max_response_bytes:
            raise ValueError("tomato vision worker health response exceeds the limit")
        try:
            payload = _mapping(
                json.loads(raw.decode("utf-8")), field_name="worker health response"
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("tomato vision worker returned invalid health JSON") from exc
        if payload.get("schema_version") != 1:
            raise ValueError("tomato vision worker health schema_version must be 1")
        if payload.get("worker_id") != self._worker_id:
            raise ValueError("tomato vision worker identity mismatch")
        if payload.get("ready") is not True:
            raise RuntimeError("tomato vision worker backend is not ready")
        return dict(payload)

    def observe(
        self,
        observation: Observation,
        *,
        requested_track_id: str | None = None,
        minimum_sample_monotonic_ns: int | None = None,
    ) -> TomatoVisionResult:
        if self._closed:
            raise RuntimeError("tomato vision worker client is closed")
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        request_metadata = _mapping(
            observation.metadata.get("tomato_vision_worker"),
            field_name="metadata.tomato_vision_worker",
        )
        bundle_ref = request_metadata.get("capture_bundle")
        latest_ref = request_metadata.get("capture_latest")
        if (bundle_ref is None) == (latest_ref is None):
            raise ValueError(
                "tomato_vision_worker must contain exactly one of capture_bundle or capture_latest"
            )
        reference_kind = "bundle" if bundle_ref is not None else "latest"
        reference_value = bundle_ref if bundle_ref is not None else latest_ref
        if not isinstance(reference_value, str) or not reference_value.strip():
            raise ValueError(
                f"tomato_vision_worker.capture_{reference_kind} must be a non-empty path"
            )
        if minimum_sample_monotonic_ns is not None and (
            isinstance(minimum_sample_monotonic_ns, bool)
            or not isinstance(minimum_sample_monotonic_ns, int)
            or minimum_sample_monotonic_ns < 0
        ):
            raise ValueError("minimum_sample_monotonic_ns must be an integer >= 0")
        relative = Path(reference_value)
        if relative.is_absolute():
            raise ValueError("capture_bundle must be relative to capture_root")
        bundle_path = (self._capture_root / relative).resolve()
        if not bundle_path.is_relative_to(self._capture_root):
            raise ValueError("capture_bundle escapes capture_root")
        if not bundle_path.is_file():
            raise FileNotFoundError(bundle_path)
        self._request_index += 1
        request_id = (
            f"vision-{observation.sequence}-{self._clock.monotonic_ns()}-"
            f"{self._request_index}"
        )
        body = json.dumps(
            {
                "schema_version": 1,
                "request_id": request_id,
                "observation_sequence": observation.sequence,
                "capture_bundle": relative.as_posix(),
                "capture_reference_kind": reference_kind,
                "requested_track_id": requested_track_id,
                "minimum_sample_monotonic_ns": minimum_sample_monotonic_ns,
                "new_frame_wait_timeout_s": (
                    self._new_frame_wait_timeout_s
                    if minimum_sample_monotonic_ns is not None
                    else 0.0
                ),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            f"{self._endpoint}/v1/process",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            raise RuntimeError(
                f"tomato vision worker rejected request with HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("tomato vision worker is unavailable") from exc
        if len(raw) > self._max_response_bytes:
            raise ValueError("tomato vision worker response exceeds the configured limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("tomato vision worker returned invalid JSON") from exc
        response_mapping = _mapping(payload, field_name="worker response")
        if set(response_mapping) != {
            "schema_version",
            "request_id",
            "worker_id",
            "tomato_vision",
        }:
            raise ValueError("tomato vision worker response fields do not match schema v1")
        if response_mapping.get("schema_version") != 1:
            raise ValueError("tomato vision worker response schema_version must be 1")
        if response_mapping.get("request_id") != request_id:
            raise ValueError("tomato vision worker response request_id mismatch")
        if response_mapping.get("worker_id") != self._worker_id:
            raise ValueError("tomato vision worker identity mismatch")
        vision = _mapping(
            response_mapping.get("tomato_vision"),
            field_name="worker response.tomato_vision",
        )
        validated_observation = Observation(
            sequence=observation.sequence,
            timestamp_s=observation.timestamp_s,
            metadata={"tomato_vision": dict(vision)},
        )
        result = self._validator.observe(
            validated_observation,
            requested_track_id=requested_track_id,
        )
        if (
            minimum_sample_monotonic_ns is not None
            and result.geometry.target.sample_monotonic_ns
            <= minimum_sample_monotonic_ns
        ):
            raise ValueError(
                "vision worker did not return a capture newer than the pinned target sample"
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._validator.close()


class WorkerTomatoVisionPerceptionPort:
    def __init__(self, *, clock: Clock, **client_options: object) -> None:
        self._client = TomatoVisionWorkerClient(clock=clock, **client_options)

    def detect(self, observation: Observation) -> TomatoDetectionResult:
        result = self._client.observe(observation)
        target = result.geometry.target
        return TomatoDetectionResult(
            targets=(target,),
            observation_sequence=observation.sequence,
            model=f"worker:{result.selection.selector_model}",
            sample_monotonic_ns=target.sample_monotonic_ns,
        )

    def close(self) -> None:
        self._client.close()


class WorkerTomatoVisionObservationPort:
    def __init__(self, *, clock: Clock, **client_options: object) -> None:
        self._client = TomatoVisionWorkerClient(clock=clock, **client_options)

    def reobserve(self, request: TargetReobserveRequest) -> TargetReobserveEvidence:
        result = self._client.observe(
            request.observation,
            requested_track_id=request.target_id,
            minimum_sample_monotonic_ns=(
                request.prior_sample_monotonic_ns
                if request.require_new_sample
                else None
            ),
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
                "worker_validated": True,
                "instance_id": result.geometry.instance_id,
                "candidate_count": len(result.candidates),
                "valid_depth_ratio": result.geometry.valid_depth_ratio,
                "point_count": result.geometry.point_count,
                "inlier_count": result.geometry.inlier_count,
                "fit_residual_m": result.geometry.fit_residual_m,
                "calibration_id": result.geometry.calibration_id,
                "selector_model": result.selection.selector_model,
            },
        )

    def close(self) -> None:
        self._client.close()


__all__ = [
    "TomatoVisionWorkerClient",
    "WorkerTomatoVisionObservationPort",
    "WorkerTomatoVisionPerceptionPort",
]
