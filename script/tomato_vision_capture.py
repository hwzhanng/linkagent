"""Validated sidecar contract for one raw, synchronized RGB-D capture.

This module only validates files and calibration metadata.  It deliberately
does not import or open a camera SDK; a future Orbbec adapter must explicitly
write this bundle before the isolated vision worker may consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from proto import CameraIntrinsics, Pose
from script.utils import load_mapping_file


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0")
    return value


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number > 0")
    result = float(value)
    if not 0.0 < result < float("inf"):
        raise ValueError(f"{field_name} must be finite and > 0")
    return result


def _confined_file(reference: object, *, base: Path, root: Path, field_name: str) -> Path:
    text = _non_empty(reference, field_name=field_name)
    relative = Path(text)
    if relative.is_absolute():
        raise ValueError(f"{field_name} must be relative to the capture bundle")
    path = (base / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field_name} escapes capture_root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass(frozen=True, slots=True)
class RawRgbdCaptureBundle:
    """A raw RGB image, D2C-aligned depth image, and immutable calibration."""

    bundle_path: Path
    capture_id: str
    rgb_path: Path
    aligned_depth_path: Path
    depth_scale_m_per_unit: float
    intrinsics: CameraIntrinsics
    camera_pose_in_world: Pose
    rgb_sample_monotonic_ns: int
    depth_sample_monotonic_ns: int
    device_id: str
    evidence_sources: tuple[str, ...]

    @property
    def skew_s(self) -> float:
        return abs(
            self.rgb_sample_monotonic_ns - self.depth_sample_monotonic_ns
        ) / 1.0e9

    @classmethod
    def load(
        cls,
        bundle_path: str | Path,
        *,
        capture_root: str | Path,
    ) -> "RawRgbdCaptureBundle":
        root = Path(capture_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        path = Path(bundle_path).expanduser().resolve()
        if not path.is_relative_to(root):
            raise ValueError("capture bundle escapes capture_root")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > 2_000_000:
            raise ValueError("raw capture sidecar exceeds 2000000 bytes")
        payload = load_mapping_file(path)
        allowed = {
            "schema_version",
            "capture_id",
            "rgb_path",
            "aligned_depth_path",
            "depth_scale_m_per_unit",
            "camera",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "unknown raw capture fields: " + ", ".join(sorted(unknown))
            )
        if payload.get("schema_version") != 1:
            raise ValueError("raw capture schema_version must be 1")
        camera = _mapping(payload.get("camera"), field_name="camera")
        camera_allowed = {
            "intrinsics",
            "camera_pose_in_world",
            "rgb_sample_monotonic_ns",
            "depth_sample_monotonic_ns",
            "depth_aligned_to_rgb",
            "device_id",
            "evidence_sources",
        }
        camera_unknown = set(camera) - camera_allowed
        if camera_unknown:
            raise ValueError(
                "unknown raw camera fields: " + ", ".join(sorted(camera_unknown))
            )
        if camera.get("depth_aligned_to_rgb") is not True:
            raise ValueError("raw depth must already be aligned to the RGB image")
        sources = camera.get("evidence_sources")
        if isinstance(sources, (str, bytes, bytearray, Mapping)):
            raise TypeError("camera.evidence_sources must be a sequence")
        try:
            prepared_sources = tuple(
                _non_empty(item, field_name="camera.evidence_sources item")
                for item in sources  # type: ignore[union-attr]
            )
        except TypeError as exc:
            raise TypeError("camera.evidence_sources must be a sequence") from exc
        if not prepared_sources:
            raise ValueError("camera.evidence_sources must not be empty")
        intrinsics = CameraIntrinsics(
            **dict(_mapping(camera.get("intrinsics"), field_name="camera.intrinsics"))
        )
        pose = Pose(
            **dict(
                _mapping(
                    camera.get("camera_pose_in_world"),
                    field_name="camera.camera_pose_in_world",
                )
            )
        )
        if pose.frame_id != "world":
            raise ValueError("camera_pose_in_world.frame_id must be 'world'")
        return cls(
            bundle_path=path,
            capture_id=_non_empty(payload.get("capture_id"), field_name="capture_id"),
            rgb_path=_confined_file(
                payload.get("rgb_path"),
                base=path.parent,
                root=root,
                field_name="rgb_path",
            ),
            aligned_depth_path=_confined_file(
                payload.get("aligned_depth_path"),
                base=path.parent,
                root=root,
                field_name="aligned_depth_path",
            ),
            depth_scale_m_per_unit=_positive_float(
                payload.get("depth_scale_m_per_unit"),
                field_name="depth_scale_m_per_unit",
            ),
            intrinsics=intrinsics,
            camera_pose_in_world=pose,
            rgb_sample_monotonic_ns=_non_negative_int(
                camera.get("rgb_sample_monotonic_ns"),
                field_name="camera.rgb_sample_monotonic_ns",
            ),
            depth_sample_monotonic_ns=_non_negative_int(
                camera.get("depth_sample_monotonic_ns"),
                field_name="camera.depth_sample_monotonic_ns",
            ),
            device_id=_non_empty(camera.get("device_id"), field_name="camera.device_id"),
            evidence_sources=prepared_sources,
        )

    def camera_payload(self) -> dict[str, object]:
        return {
            "intrinsics": {
                "width_px": self.intrinsics.width_px,
                "height_px": self.intrinsics.height_px,
                "fx_px": self.intrinsics.fx_px,
                "fy_px": self.intrinsics.fy_px,
                "cx_px": self.intrinsics.cx_px,
                "cy_px": self.intrinsics.cy_px,
                "calibration_id": self.intrinsics.calibration_id,
            },
            "camera_pose_in_world": self.camera_pose_in_world.as_dict(),
            "rgb_sample_monotonic_ns": self.rgb_sample_monotonic_ns,
            "depth_sample_monotonic_ns": self.depth_sample_monotonic_ns,
            "depth_aligned_to_rgb": True,
            "device_id": self.device_id,
            "evidence_sources": list(self.evidence_sources),
        }


__all__ = ["RawRgbdCaptureBundle"]
