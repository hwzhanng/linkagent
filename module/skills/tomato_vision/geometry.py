"""Dependency-free mask-depth deprojection and robust tomato geometry fitting."""

from __future__ import annotations

import math
from statistics import median

from module.skills.tomato_pick.geometry import rotate_vector
from proto import (
    Pose,
    SynchronizedRgbdFrame,
    TomatoGeometryEvidence,
    TomatoInstance2D,
    TomatoTarget,
    VlmTargetSelection,
)


Vector3 = tuple[float, float, float]


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _scale(value: Vector3, factor: float) -> Vector3:
    return tuple(item * factor for item in value)  # type: ignore[return-value]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: Vector3) -> float:
    return math.sqrt(_dot(value, value))


def _normalize(value: Vector3, *, field_name: str) -> Vector3:
    length = _norm(value)
    if length <= 1.0e-9:
        raise ValueError(f"{field_name} cannot be a zero vector")
    return _scale(value, 1.0 / length)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty list")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _basis_quaternion(x_axis: Vector3, y_axis: Vector3, z_axis: Vector3) -> tuple[float, float, float, float]:
    """Convert a rotation matrix whose columns are x/y/z to quaternion wxyz."""

    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = (0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        result = ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        result = ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        result = ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)
    length = math.sqrt(sum(value * value for value in result))
    return tuple(value / length for value in result)  # type: ignore[return-value]


class RobustMaskedDepthGeometryPort:
    """Fit a bounded oriented ellipsoid from the selected mask's depth samples."""

    def __init__(
        self,
        *,
        min_points: int = 6,
        min_valid_depth_ratio: float = 0.60,
        max_fit_residual_m: float = 0.012,
        min_axis_m: float = 0.015,
        max_axis_m: float = 0.060,
        silhouette_axis_floor_ratio: float = 0.0,
        visible_surface_center_correction_ratio: float = 0.0,
        pull_axis_prior_world: tuple[float, ...] | None = None,
        max_pull_axis_deviation_deg: float = 180.0,
    ) -> None:
        if isinstance(min_points, bool) or not isinstance(min_points, int) or min_points < 6:
            raise ValueError("min_points must be an integer >= 6")
        if not 0.0 < min_valid_depth_ratio <= 1.0:
            raise ValueError("min_valid_depth_ratio must be in (0, 1]")
        if max_fit_residual_m <= 0.0 or not 0.0 < min_axis_m < max_axis_m:
            raise ValueError("geometry limits must be positive and ordered")
        self._min_points = min_points
        self._min_valid_depth_ratio = float(min_valid_depth_ratio)
        self._max_fit_residual_m = float(max_fit_residual_m)
        self._axis_limits = (float(min_axis_m), float(max_axis_m))
        if not 0.0 <= float(silhouette_axis_floor_ratio) <= 1.0:
            raise ValueError("silhouette_axis_floor_ratio must be in [0, 1]")
        self._silhouette_axis_floor_ratio = float(silhouette_axis_floor_ratio)
        if not 0.0 <= float(visible_surface_center_correction_ratio) <= 1.5:
            raise ValueError(
                "visible_surface_center_correction_ratio must be in [0, 1.5]"
            )
        self._visible_surface_center_correction_ratio = float(
            visible_surface_center_correction_ratio
        )
        self._pull_axis_prior_world = (
            None
            if pull_axis_prior_world is None
            else _normalize(tuple(pull_axis_prior_world), field_name="pull-axis prior")
        )
        if not 0.0 < float(max_pull_axis_deviation_deg) <= 180.0:
            raise ValueError("max_pull_axis_deviation_deg must be in (0, 180]")
        self._max_pull_axis_deviation_rad = math.radians(
            float(max_pull_axis_deviation_deg)
        )
        self._closed = False

    def _regularize_pull_axis(self, raw_world: Vector3) -> Vector3:
        prior = self._pull_axis_prior_world
        if prior is None:
            return raw_world
        cosine = max(-1.0, min(1.0, _dot(prior, raw_world)))
        angle = math.acos(cosine)
        if angle <= self._max_pull_axis_deviation_rad:
            return raw_world
        perpendicular = _sub(raw_world, _scale(prior, cosine))
        perpendicular = _normalize(perpendicular, field_name="pull-axis deviation")
        limit = self._max_pull_axis_deviation_rad
        return _normalize(
            _add(_scale(prior, math.cos(limit)), _scale(perpendicular, math.sin(limit))),
            field_name="regularized pull axis",
        )

    def estimate(
        self,
        frame: SynchronizedRgbdFrame,
        candidate: TomatoInstance2D,
        selection: VlmTargetSelection,
    ) -> TomatoGeometryEvidence:
        if self._closed:
            raise RuntimeError("geometry port is closed")
        if selection.selected_instance_id != candidate.instance_id:
            raise ValueError("geometry candidate must match the VLM selection")
        if candidate.stem_point_camera_m is None:
            raise ValueError("stem evidence is required to define the away-from-plant axis")
        intrinsics = frame.intrinsics
        mask = frozenset(candidate.mask_pixels_xy)
        points_camera: list[Vector3] = []
        for u, v, depth_m in candidate.depth_samples_uvm:
            pixel = (int(round(u)), int(round(v)))
            if pixel not in mask:
                continue
            if not 0.0 <= u < intrinsics.width_px or not 0.0 <= v < intrinsics.height_px:
                continue
            points_camera.append(
                (
                    (u - intrinsics.cx_px) * depth_m / intrinsics.fx_px,
                    (v - intrinsics.cy_px) * depth_m / intrinsics.fy_px,
                    depth_m,
                )
            )
        if candidate.mask_area_px is None:
            # Backward-compatible replay bundles predate dense mask coverage
            # counters. Real workers must publish both counters so unsampled or
            # invalid depth cannot be hidden by returning only valid samples.
            valid_ratio = len(points_camera) / max(1, len(candidate.depth_samples_uvm))
        else:
            valid_ratio = candidate.valid_depth_pixel_count / candidate.mask_area_px
        if valid_ratio < self._min_valid_depth_ratio:
            raise ValueError("selected mask has insufficient valid aligned depth")
        if len(points_camera) < self._min_points:
            raise ValueError("selected mask has too few 3D points")

        initial_center = tuple(median(point[index] for point in points_camera) for index in range(3))
        distances = [_norm(_sub(point, initial_center)) for point in points_camera]
        distance_median = median(distances)
        mad = median(abs(value - distance_median) for value in distances)
        threshold = max(0.010, 3.5 * mad)
        inliers = [
            point
            for point, distance in zip(points_camera, distances)
            if abs(distance - distance_median) <= threshold
        ]
        if len(inliers) < self._min_points:
            raise ValueError("robust depth filtering left too few inliers")
        center_camera: Vector3 = tuple(
            median(point[index] for point in inliers) for index in range(3)
        )  # type: ignore[assignment]

        pull_camera = _normalize(
            _sub(center_camera, candidate.stem_point_camera_m),
            field_name="stem-to-fruit pull axis",
        )
        camera_ray = _normalize(center_camera, field_name="camera-to-fruit ray")
        approach_camera = _sub(camera_ray, _scale(pull_camera, _dot(camera_ray, pull_camera)))
        if _norm(approach_camera) <= 1.0e-6:
            fallback = (1.0, 0.0, 0.0) if abs(pull_camera[0]) < 0.8 else (0.0, 1.0, 0.0)
            approach_camera = _sub(fallback, _scale(pull_camera, _dot(fallback, pull_camera)))
        x_camera = _normalize(approach_camera, field_name="approach axis")
        y_camera = _normalize(_cross(pull_camera, x_camera), field_name="lateral axis")
        x_camera = _normalize(_cross(y_camera, pull_camera), field_name="approach axis")

        projections = [
            (
                _dot(_sub(point, center_camera), x_camera),
                _dot(_sub(point, center_camera), y_camera),
                _dot(_sub(point, center_camera), pull_camera),
            )
            for point in inliers
        ]
        axes = tuple(
            _quantile([abs(value[index]) for value in projections], 0.95)
            for index in range(3)
        )
        if self._silhouette_axis_floor_ratio > 0.0:
            x1, y1, x2, y2 = candidate.bbox_xyxy_px
            silhouette_radii = (
                0.5 * (x2 - x1) * center_camera[2] / intrinsics.fx_px,
                0.5 * (y2 - y1) * center_camera[2] / intrinsics.fy_px,
            )
            # Box-local depth points cover only the visible hemisphere. The
            # calibrated fill ratio turns the smaller metric silhouette radius
            # into a conservative lower bound for an occluded ellipsoid axis.
            silhouette_floor = (
                self._silhouette_axis_floor_ratio * min(silhouette_radii)
            )
            axes = tuple(max(value, silhouette_floor) for value in axes)
        lower, upper = self._axis_limits
        if any(value < lower or value > upper for value in axes):
            raise ValueError("fitted fruit axes are outside the calibrated geometry limits")
        surface_errors = []
        mean_axis = sum(axes) / 3.0
        for point in projections:
            normalized_radius = math.sqrt(
                sum((value / axis) ** 2 for value, axis in zip(point, axes))
            )
            surface_errors.append(abs(normalized_radius - 1.0) * mean_axis)
        residual = median(surface_errors)
        if residual > self._max_fit_residual_m:
            raise ValueError("ellipsoid fit residual exceeds the calibrated limit")

        metric_center_camera = _add(
            center_camera,
            _scale(
                camera_ray,
                self._visible_surface_center_correction_ratio * mean_axis,
            ),
        )
        camera_q = frame.camera_pose_in_world.quaternion_wxyz
        camera_t = frame.camera_pose_in_world.position_m
        center_world = _add(camera_t, rotate_vector(camera_q, metric_center_camera))
        x_world = _normalize(rotate_vector(camera_q, x_camera), field_name="world approach axis")
        y_world = _normalize(rotate_vector(camera_q, y_camera), field_name="world lateral axis")
        pull_world = self._regularize_pull_axis(
            _normalize(rotate_vector(camera_q, pull_camera), field_name="world pull axis")
        )
        # Rebuild the other axes after any deployment-profile regularization.
        x_world = _sub(x_world, _scale(pull_world, _dot(x_world, pull_world)))
        if _norm(x_world) <= 1.0e-6:
            fallback = (1.0, 0.0, 0.0) if abs(pull_world[0]) < 0.8 else (0.0, 1.0, 0.0)
            x_world = _sub(fallback, _scale(pull_world, _dot(fallback, pull_world)))
        x_world = _normalize(x_world, field_name="regularized world approach axis")
        y_world = _normalize(_cross(pull_world, x_world), field_name="regularized world lateral axis")
        x_world = _normalize(_cross(y_world, pull_world), field_name="regularized world approach axis")
        grasp_q = _basis_quaternion(x_world, y_world, pull_world)
        quality = (
            candidate.detector_confidence
            * selection.confidence
            * valid_ratio
            * (1.0 - candidate.occlusion_ratio)
        )
        target = TomatoTarget(
            target_id=candidate.track_id,
            center_pose=Pose(center_world, grasp_q, "world"),
            semi_axes_m=axes,
            ripeness_score=candidate.ripeness_score,
            pick_confidence=max(0.0, min(1.0, quality)),
            occlusion=candidate.occlusion_ratio,
            stem_direction_xyz=pull_world,
            observation_sequence=frame.observation.sequence,
            sample_monotonic_ns=max(
                frame.rgb_sample_monotonic_ns, frame.depth_sample_monotonic_ns
            ),
        )
        return TomatoGeometryEvidence(
            target=target,
            instance_id=candidate.instance_id,
            calibration_id=intrinsics.calibration_id,
            valid_depth_ratio=valid_ratio,
            point_count=len(points_camera),
            inlier_count=len(inliers),
            fit_residual_m=residual,
            pull_axis_away_from_plant_world=pull_world,
            evidence_sources=(
                *frame.evidence_sources,
                f"instance-support:{candidate.instance_id}",
                f"vlm-selection:{selection.selector_model}",
                f"calibration:{intrinsics.calibration_id}",
                "geometry:metric-silhouette-plus-aligned-depth",
                "geometry:visible-surface-center-correction",
            ),
        )

    def close(self) -> None:
        self._closed = True


__all__ = ["RobustMaskedDepthGeometryPort"]
