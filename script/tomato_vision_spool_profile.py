"""Strict configuration contract for an offline RGB-D recorded-shadow spool."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from script.utils import load_mapping_file


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _identity(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field_name} must be finite and > 0")
    return float(value)


@dataclass(frozen=True, slots=True)
class RecordedShadowAuditProfile:
    min_capture_count: int
    pixel_check: bool
    min_frame_valid_depth_ratio: float
    min_depth_m: float
    max_depth_m: float


@dataclass(frozen=True, slots=True)
class TomatoVisionSpoolProfile:
    profile_id: str
    capture_root: Path
    expected_device_id: str
    expected_calibration_id: str
    max_rgb_depth_skew_s: float
    max_image_bytes: int
    reserve_free_bytes: int
    audit: RecordedShadowAuditProfile

    @classmethod
    def load(cls, path: str | Path) -> "TomatoVisionSpoolProfile":
        profile_path = Path(path).expanduser().resolve()
        payload = load_mapping_file(profile_path)
        allowed = {
            "schema_version",
            "profile_id",
            "deployment_mode",
            "hardware_approved",
            "capture_root",
            "expected_device_id",
            "expected_calibration_id",
            "max_rgb_depth_skew_s",
            "max_image_bytes",
            "reserve_free_bytes",
            "audit",
        }
        if set(payload) != allowed:
            raise ValueError("spool profile fields do not match schema v1")
        if payload.get("schema_version") != 1:
            raise ValueError("spool profile schema_version must be 1")
        if payload.get("deployment_mode") != "recorded_shadow":
            raise ValueError("spool profile deployment_mode must be recorded_shadow")
        if payload.get("hardware_approved") is not False:
            raise ValueError("recorded-shadow spool profile must not approve hardware")
        capture_ref = _identity(payload.get("capture_root"), field_name="capture_root")
        capture_path = Path(capture_ref).expanduser()
        if not capture_path.is_absolute():
            capture_path = profile_path.parent / capture_path
        capture_root = capture_path.resolve()
        image_limit = payload.get("max_image_bytes")
        reserve = payload.get("reserve_free_bytes")
        if isinstance(image_limit, bool) or not isinstance(image_limit, int) or image_limit < 1024:
            raise ValueError("max_image_bytes must be an integer >= 1024")
        if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
            raise ValueError("reserve_free_bytes must be an integer >= 0")
        skew = _positive(
            payload.get("max_rgb_depth_skew_s"),
            field_name="max_rgb_depth_skew_s",
        )
        if skew > 1.0:
            raise ValueError("max_rgb_depth_skew_s must be <= 1")

        audit_payload = _mapping(payload.get("audit"), field_name="audit")
        audit_fields = {
            "min_capture_count",
            "pixel_check",
            "min_frame_valid_depth_ratio",
            "min_depth_m",
            "max_depth_m",
        }
        if set(audit_payload) != audit_fields:
            raise ValueError("spool profile audit fields do not match schema v1")
        count = audit_payload.get("min_capture_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("audit.min_capture_count must be an integer >= 1")
        pixel_check = audit_payload.get("pixel_check")
        if not isinstance(pixel_check, bool):
            raise TypeError("audit.pixel_check must be a bool")
        ratio = audit_payload.get("min_frame_valid_depth_ratio")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or not 0.0 <= float(ratio) <= 1.0
        ):
            raise ValueError("audit.min_frame_valid_depth_ratio must be in [0, 1]")
        min_depth = _positive(audit_payload.get("min_depth_m"), field_name="audit.min_depth_m")
        max_depth = _positive(audit_payload.get("max_depth_m"), field_name="audit.max_depth_m")
        if min_depth >= max_depth:
            raise ValueError("audit.min_depth_m must be less than audit.max_depth_m")
        return cls(
            profile_id=_identity(payload.get("profile_id"), field_name="profile_id"),
            capture_root=capture_root,
            expected_device_id=_identity(
                payload.get("expected_device_id"), field_name="expected_device_id"
            ),
            expected_calibration_id=_identity(
                payload.get("expected_calibration_id"),
                field_name="expected_calibration_id",
            ),
            max_rgb_depth_skew_s=skew,
            max_image_bytes=image_limit,
            reserve_free_bytes=reserve,
            audit=RecordedShadowAuditProfile(
                min_capture_count=count,
                pixel_check=pixel_check,
                min_frame_valid_depth_ratio=float(ratio),
                min_depth_m=min_depth,
                max_depth_m=max_depth,
            ),
        )


__all__ = ["RecordedShadowAuditProfile", "TomatoVisionSpoolProfile"]
