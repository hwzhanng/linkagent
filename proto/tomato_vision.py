"""Frozen DTOs for the RGB-D tomato-selection and geometry boundary.

The contracts deliberately keep images, masks, and dense point clouds out of
the Agent/PPO interface.  They describe the evidence needed to decide whether
a compact :class:`TomatoTarget` is safe to publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ._base_proto import Observation, _finite_float, _finite_float_tuple, _freeze_mapping
from .manipulation import Pose
from .tomato_pick import TomatoTarget


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0")
    return value


def _positive(value: object, *, field_name: str) -> float:
    result = _finite_float(value, field_name=field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return result


def _unit_interval(value: object, *, field_name: str) -> float:
    result = _finite_float(value, field_name=field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return result


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of strings") from exc
    return tuple(_non_empty(item, field_name=f"{field_name} item") for item in items)


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Aligned depth-camera pinhole calibration."""

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    calibration_id: str

    def __post_init__(self) -> None:
        for name in ("width_px", "height_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be an integer > 0")
        for name in ("fx_px", "fy_px"):
            object.__setattr__(self, name, _positive(getattr(self, name), field_name=name))
        for name in ("cx_px", "cy_px"):
            object.__setattr__(
                self, name, _finite_float(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "calibration_id",
            _non_empty(self.calibration_id, field_name="calibration_id"),
        )


@dataclass(frozen=True, slots=True)
class SynchronizedRgbdFrame:
    """Evidence that RGB, aligned depth, and extrinsics describe one frame."""

    observation: Observation
    intrinsics: CameraIntrinsics
    camera_pose_in_world: Pose
    rgb_sample_monotonic_ns: int
    depth_sample_monotonic_ns: int
    depth_aligned_to_rgb: bool
    device_id: str
    evidence_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation")
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics must be CameraIntrinsics")
        if not isinstance(self.camera_pose_in_world, Pose):
            raise TypeError("camera_pose_in_world must be a Pose")
        if self.camera_pose_in_world.frame_id != "world":
            raise ValueError("camera_pose_in_world.frame_id must be 'world'")
        for name in ("rgb_sample_monotonic_ns", "depth_sample_monotonic_ns"):
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), field_name=name),
            )
        if not isinstance(self.depth_aligned_to_rgb, bool):
            raise TypeError("depth_aligned_to_rgb must be a bool")
        object.__setattr__(self, "device_id", _non_empty(self.device_id, field_name="device_id"))
        sources = _string_tuple(self.evidence_sources, field_name="evidence_sources")
        if not sources:
            raise ValueError("evidence_sources must not be empty")
        object.__setattr__(self, "evidence_sources", sources)

    @property
    def skew_s(self) -> float:
        return abs(self.rgb_sample_monotonic_ns - self.depth_sample_monotonic_ns) / 1.0e9


@dataclass(frozen=True, slots=True)
class TomatoInstance2D:
    """One YOLO-seg candidate; its instance ID is the 2D geometry authority."""

    instance_id: str
    track_id: str
    bbox_xyxy_px: tuple[float, ...]
    mask_pixels_xy: tuple[tuple[int, int], ...]
    detector_confidence: float
    ripeness_score: float
    occlusion_ratio: float
    depth_samples_uvm: tuple[tuple[float, float, float], ...]
    stem_point_camera_m: tuple[float, float, float] | None = None
    mask_area_px: int | None = None
    valid_depth_pixel_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instance_id", _non_empty(self.instance_id, field_name="instance_id"))
        object.__setattr__(self, "track_id", _non_empty(self.track_id, field_name="track_id"))
        bbox = _finite_float_tuple(self.bbox_xyxy_px, field_name="bbox_xyxy_px", length=4)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox_xyxy_px must have positive width and height")
        object.__setattr__(self, "bbox_xyxy_px", bbox)
        mask: list[tuple[int, int]] = []
        for index, point in enumerate(self.mask_pixels_xy):
            if len(tuple(point)) != 2:
                raise ValueError(f"mask_pixels_xy[{index}] must contain xy")
            x, y = tuple(point)
            if isinstance(x, bool) or not isinstance(x, int) or x < 0:
                raise ValueError("mask pixel x must be an integer >= 0")
            if isinstance(y, bool) or not isinstance(y, int) or y < 0:
                raise ValueError("mask pixel y must be an integer >= 0")
            mask.append((x, y))
        if not mask:
            raise ValueError("mask_pixels_xy must not be empty")
        object.__setattr__(self, "mask_pixels_xy", tuple(mask))
        for name in ("detector_confidence", "ripeness_score", "occlusion_ratio"):
            object.__setattr__(
                self, name, _unit_interval(getattr(self, name), field_name=name)
            )
        samples: list[tuple[float, float, float]] = []
        for index, sample in enumerate(self.depth_samples_uvm):
            prepared = _finite_float_tuple(
                sample, field_name=f"depth_samples_uvm[{index}]", length=3
            )
            if prepared[2] <= 0.0:
                raise ValueError("depth sample metres must be > 0")
            samples.append(prepared)
        object.__setattr__(self, "depth_samples_uvm", tuple(samples))
        if self.stem_point_camera_m is not None:
            object.__setattr__(
                self,
                "stem_point_camera_m",
                _finite_float_tuple(
                    self.stem_point_camera_m,
                    field_name="stem_point_camera_m",
                    length=3,
                ),
            )
        if (self.mask_area_px is None) != (self.valid_depth_pixel_count is None):
            raise ValueError(
                "mask_area_px and valid_depth_pixel_count must be provided together"
            )
        if self.mask_area_px is not None:
            mask_area = _non_negative_int(self.mask_area_px, field_name="mask_area_px")
            valid_depth = _non_negative_int(
                self.valid_depth_pixel_count,
                field_name="valid_depth_pixel_count",
            )
            if mask_area <= 0:
                raise ValueError("mask_area_px must be > 0")
            if valid_depth > mask_area:
                raise ValueError(
                    "valid_depth_pixel_count must not exceed mask_area_px"
                )
            object.__setattr__(self, "mask_area_px", mask_area)
            object.__setattr__(self, "valid_depth_pixel_count", valid_depth)

    def contains_pixel(self, point_xy: tuple[int, int]) -> bool:
        return point_xy in frozenset(self.mask_pixels_xy)


@dataclass(frozen=True, slots=True)
class VlmTargetSelection:
    """Semantic choice over numbered instances, never free-form geometry."""

    selected_instance_id: str
    confidence: float
    reason: str
    point_xy: tuple[int, int] | None = None
    selector_model: str = "replay_vlm_selector_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_instance_id",
            _non_empty(self.selected_instance_id, field_name="selected_instance_id"),
        )
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, field_name="confidence"))
        object.__setattr__(self, "reason", _non_empty(self.reason, field_name="reason"))
        object.__setattr__(self, "selector_model", _non_empty(self.selector_model, field_name="selector_model"))
        if self.point_xy is not None:
            point = tuple(self.point_xy)
            if len(point) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in point
            ):
                raise ValueError("point_xy must contain two integers >= 0")
            object.__setattr__(self, "point_xy", point)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TomatoGeometryEvidence:
    """Quality report for mask-depth deprojection and robust 3D fitting."""

    target: TomatoTarget
    instance_id: str
    calibration_id: str
    valid_depth_ratio: float
    point_count: int
    inlier_count: int
    fit_residual_m: float
    pull_axis_away_from_plant_world: tuple[float, float, float]
    evidence_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target, TomatoTarget):
            raise TypeError("target must be a TomatoTarget")
        object.__setattr__(self, "instance_id", _non_empty(self.instance_id, field_name="instance_id"))
        object.__setattr__(self, "calibration_id", _non_empty(self.calibration_id, field_name="calibration_id"))
        object.__setattr__(self, "valid_depth_ratio", _unit_interval(self.valid_depth_ratio, field_name="valid_depth_ratio"))
        for name in ("point_count", "inlier_count"):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), field_name=name))
        if self.inlier_count > self.point_count:
            raise ValueError("inlier_count must not exceed point_count")
        residual = _finite_float(self.fit_residual_m, field_name="fit_residual_m")
        if residual < 0.0:
            raise ValueError("fit_residual_m must be >= 0")
        object.__setattr__(self, "fit_residual_m", residual)
        axis = _finite_float_tuple(
            self.pull_axis_away_from_plant_world,
            field_name="pull_axis_away_from_plant_world",
            length=3,
        )
        object.__setattr__(self, "pull_axis_away_from_plant_world", axis)
        object.__setattr__(self, "evidence_sources", _string_tuple(self.evidence_sources, field_name="evidence_sources"))


@dataclass(frozen=True, slots=True)
class TomatoVisionResult:
    frame: SynchronizedRgbdFrame
    candidates: tuple[TomatoInstance2D, ...]
    selection: VlmTargetSelection
    geometry: TomatoGeometryEvidence
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, SynchronizedRgbdFrame):
            raise TypeError("frame must be a SynchronizedRgbdFrame")
        candidates = tuple(self.candidates)
        if not candidates or not all(isinstance(item, TomatoInstance2D) for item in candidates):
            raise TypeError("candidates must contain TomatoInstance2D values")
        ids = [item.instance_id for item in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate instance IDs must be unique")
        if not isinstance(self.selection, VlmTargetSelection):
            raise TypeError("selection must be VlmTargetSelection")
        if self.selection.selected_instance_id not in ids:
            raise ValueError("selection must reference one candidate instance")
        if not isinstance(self.geometry, TomatoGeometryEvidence):
            raise TypeError("geometry must be TomatoGeometryEvidence")
        if self.geometry.instance_id != self.selection.selected_instance_id:
            raise ValueError("geometry and selection instance IDs must agree")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "details", _freeze_mapping(self.details, field_name="details"))


__all__ = [
    "CameraIntrinsics",
    "SynchronizedRgbdFrame",
    "TomatoGeometryEvidence",
    "TomatoInstance2D",
    "TomatoVisionResult",
    "VlmTargetSelection",
]
