"""Build a fail-closed recorded-shadow quality manifest from an RGB-D spool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

from script.tomato_vision_capture import RawRgbdCaptureBundle
from script.tomato_vision_spool import TomatoVisionSpoolReader, inspect_png
from script.tomato_vision_spool_profile import TomatoVisionSpoolProfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _issue(code: str, message: str, *, bundle: Path | None = None) -> dict[str, object]:
    result: dict[str, object] = {"code": code, "message": message}
    if bundle is not None:
        result["bundle"] = str(bundle)
    return result


def _pixel_depth_ratio(
    bundle: RawRgbdCaptureBundle,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> float:
    try:
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pixel depth audit requires numpy and Pillow") from exc
    try:
        with Image.open(bundle.aligned_depth_path) as image:
            depth = np.asarray(image).copy()
    except OSError as exc:
        raise ValueError("aligned depth PNG cannot be decoded") from exc
    expected_shape = (bundle.intrinsics.height_px, bundle.intrinsics.width_px)
    if depth.ndim != 2 or depth.shape != expected_shape:
        raise ValueError("decoded depth shape does not match intrinsics")
    if depth.dtype.kind not in {"u", "i"}:
        raise ValueError("decoded depth must use integer sensor units")
    depth_m = depth.astype(np.float64) * bundle.depth_scale_m_per_unit
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    return float(valid.sum() / max(1, valid.size))


def audit_spool(
    capture_root: str | Path,
    *,
    expected_calibration_id: str | None = None,
    expected_device_id: str | None = None,
    min_capture_count: int = 1,
    max_rgb_depth_skew_s: float = 0.035,
    pixel_check: bool = False,
    min_frame_valid_depth_ratio: float = 0.10,
    min_depth_m: float = 0.10,
    max_depth_m: float = 3.0,
) -> dict[str, object]:
    if isinstance(min_capture_count, bool) or not isinstance(min_capture_count, int) or min_capture_count < 1:
        raise ValueError("min_capture_count must be an integer >= 1")
    for name, value in {
        "max_rgb_depth_skew_s": max_rgb_depth_skew_s,
        "min_depth_m": min_depth_m,
        "max_depth_m": max_depth_m,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")
    if min_depth_m >= max_depth_m:
        raise ValueError("min_depth_m must be less than max_depth_m")
    if not 0.0 <= min_frame_valid_depth_ratio <= 1.0:
        raise ValueError("min_frame_valid_depth_ratio must be in [0, 1]")
    if not isinstance(pixel_check, bool):
        raise TypeError("pixel_check must be a bool")

    root = Path(capture_root).expanduser().resolve()
    reader = TomatoVisionSpoolReader(root)
    issues: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    seen_capture_ids: set[str] = set()
    previous_sample_ns: int | None = None
    baseline_calibration = expected_calibration_id
    baseline_device = expected_device_id
    baseline_intrinsics: object | None = None
    baseline_depth_scale: float | None = None

    staging = tuple(
        path
        for path in reader.captures_path.iterdir()
        if path.name.startswith(".staging-") and path.is_dir()
    )
    if staging:
        issues.append(
            _issue(
                "staging_present",
                f"{len(staging)} incomplete staging directories require recovery",
            )
        )

    for path in reader.capture_paths():
        relative = path.relative_to(root)
        try:
            bundle = RawRgbdCaptureBundle.load(path, capture_root=root)
            rgb_header = inspect_png(
                bundle.rgb_path.read_bytes(), field_name="recorded RGB"
            )
            depth_header = inspect_png(
                bundle.aligned_depth_path.read_bytes(),
                field_name="recorded aligned depth",
            )
            if rgb_header.bit_depth != 8 or rgb_header.colour_type not in {2, 6}:
                raise ValueError("recorded RGB must be 8-bit RGB/RGBA PNG")
            if depth_header.bit_depth != 16 or depth_header.colour_type != 0:
                raise ValueError("recorded aligned depth must be 16-bit grayscale PNG")
            expected_dimensions = (
                bundle.intrinsics.width_px,
                bundle.intrinsics.height_px,
            )
            if (rgb_header.width_px, rgb_header.height_px) != expected_dimensions or (
                depth_header.width_px,
                depth_header.height_px,
            ) != expected_dimensions:
                raise ValueError("recorded image dimensions do not match intrinsics")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            issues.append(_issue("bundle_invalid", str(exc), bundle=relative))
            continue

        sample_ns = max(
            bundle.rgb_sample_monotonic_ns, bundle.depth_sample_monotonic_ns
        )
        directory_prefix = path.parent.name.split("-", 1)[0]
        if not directory_prefix.isdigit() or int(directory_prefix) != sample_ns:
            issues.append(
                _issue(
                    "capture_layout",
                    "capture directory timestamp prefix does not match sidecar",
                    bundle=relative,
                )
            )
        if previous_sample_ns is not None and sample_ns <= previous_sample_ns:
            issues.append(
                _issue(
                    "timestamp_order",
                    "capture sample timestamps are not strictly increasing",
                    bundle=relative,
                )
            )
        previous_sample_ns = sample_ns
        if bundle.capture_id in seen_capture_ids:
            issues.append(
                _issue(
                    "duplicate_capture_id",
                    f"duplicate capture_id {bundle.capture_id!r}",
                    bundle=relative,
                )
            )
        seen_capture_ids.add(bundle.capture_id)
        if baseline_calibration is None:
            baseline_calibration = bundle.intrinsics.calibration_id
        if baseline_device is None:
            baseline_device = bundle.device_id
        if bundle.intrinsics.calibration_id != baseline_calibration:
            issues.append(
                _issue(
                    "calibration_mismatch",
                    "capture calibration_id differs from dataset profile",
                    bundle=relative,
                )
            )
        if baseline_intrinsics is None:
            baseline_intrinsics = bundle.intrinsics
            baseline_depth_scale = bundle.depth_scale_m_per_unit
        elif bundle.intrinsics != baseline_intrinsics:
            issues.append(
                _issue(
                    "intrinsics_mismatch",
                    "capture intrinsics changed within one calibration profile",
                    bundle=relative,
                )
            )
        if (
            baseline_depth_scale is not None
            and bundle.depth_scale_m_per_unit != baseline_depth_scale
        ):
            issues.append(
                _issue(
                    "depth_scale_mismatch",
                    "capture depth scale changed within one calibration profile",
                    bundle=relative,
                )
            )
        if bundle.device_id != baseline_device:
            issues.append(
                _issue(
                    "device_mismatch",
                    "capture device_id differs from dataset profile",
                    bundle=relative,
                )
            )
        if bundle.skew_s > max_rgb_depth_skew_s:
            issues.append(
                _issue(
                    "rgb_depth_skew",
                    "capture RGB/depth skew exceeds dataset gate",
                    bundle=relative,
                )
            )
        valid_depth_ratio: float | None = None
        if pixel_check:
            try:
                valid_depth_ratio = _pixel_depth_ratio(
                    bundle,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
                if valid_depth_ratio < min_frame_valid_depth_ratio:
                    issues.append(
                        _issue(
                            "depth_coverage",
                            "frame valid-depth ratio is below the dataset gate",
                            bundle=relative,
                        )
                    )
            except (RuntimeError, ValueError) as exc:
                issues.append(_issue("depth_decode", str(exc), bundle=relative))
        rows.append(
            {
                "capture_id": bundle.capture_id,
                "capture_bundle": relative.as_posix(),
                "sample_monotonic_ns": sample_ns,
                "rgb_sample_monotonic_ns": bundle.rgb_sample_monotonic_ns,
                "depth_sample_monotonic_ns": bundle.depth_sample_monotonic_ns,
                "rgb_depth_skew_s": bundle.skew_s,
                "calibration_id": bundle.intrinsics.calibration_id,
                "device_id": bundle.device_id,
                "width_px": bundle.intrinsics.width_px,
                "height_px": bundle.intrinsics.height_px,
                "fx_px": bundle.intrinsics.fx_px,
                "fy_px": bundle.intrinsics.fy_px,
                "cx_px": bundle.intrinsics.cx_px,
                "cy_px": bundle.intrinsics.cy_px,
                "depth_scale_m_per_unit": bundle.depth_scale_m_per_unit,
                "camera_pose_in_world": bundle.camera_pose_in_world.as_dict(),
                "rgb_sha256": _sha256(bundle.rgb_path),
                "aligned_depth_sha256": _sha256(bundle.aligned_depth_path),
                "sidecar_sha256": _sha256(path),
                "valid_depth_ratio": valid_depth_ratio,
            }
        )

    if len(rows) < min_capture_count:
        issues.append(
            _issue(
                "capture_count",
                f"dataset has {len(rows)} valid captures; requires {min_capture_count}",
            )
        )
    try:
        latest = reader.resolve_latest()
        if rows and latest.bundle_path.relative_to(root).as_posix() != rows[-1]["capture_bundle"]:
            issues.append(
                _issue(
                    "latest_mismatch",
                    "latest pointer does not reference the newest capture",
                )
            )
    except (OSError, TypeError, ValueError) as exc:
        issues.append(_issue("latest_invalid", str(exc)))

    return {
        "schema_version": 1,
        "audit_type": "tomato_rgbd_recorded_shadow_v1",
        "generated_unix_ns": time.time_ns(),
        "capture_root": str(root),
        "gates": {
            "expected_calibration_id": expected_calibration_id,
            "expected_device_id": expected_device_id,
            "min_capture_count": min_capture_count,
            "max_rgb_depth_skew_s": max_rgb_depth_skew_s,
            "pixel_check": pixel_check,
            "min_frame_valid_depth_ratio": min_frame_valid_depth_ratio,
            "min_depth_m": min_depth_m,
            "max_depth_m": max_depth_m,
        },
        "summary": {
            "passed": not issues,
            "capture_count": len(rows),
            "issue_count": len(issues),
            "calibration_id": baseline_calibration,
            "device_id": baseline_device,
        },
        "captures": rows,
        "issues": issues,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit an RGB-D recorded-shadow spool")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-root", type=Path)
    source.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-calibration-id")
    parser.add_argument("--expected-device-id")
    parser.add_argument("--min-capture-count", type=int)
    parser.add_argument("--max-rgb-depth-skew-s", type=float)
    parser.add_argument(
        "--pixel-check", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--min-frame-valid-depth-ratio", type=float)
    parser.add_argument("--min-depth-m", type=float)
    parser.add_argument("--max-depth-m", type=float)
    args = parser.parse_args(argv)
    try:
        profile = (
            None if args.profile is None else TomatoVisionSpoolProfile.load(args.profile)
        )
        profile_overrides = {
            "expected_calibration_id": args.expected_calibration_id,
            "expected_device_id": args.expected_device_id,
            "min_capture_count": args.min_capture_count,
            "max_rgb_depth_skew_s": args.max_rgb_depth_skew_s,
            "pixel_check": args.pixel_check,
            "min_frame_valid_depth_ratio": args.min_frame_valid_depth_ratio,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
        }
        if profile is not None and any(
            value is not None for value in profile_overrides.values()
        ):
            raise ValueError("quality gates cannot override an authoritative profile")
        profile_audit = None if profile is None else profile.audit
        report = audit_spool(
            args.capture_root if profile is None else profile.capture_root,
            expected_calibration_id=(
                args.expected_calibration_id
                if args.expected_calibration_id is not None
                else None if profile is None else profile.expected_calibration_id
            ),
            expected_device_id=(
                args.expected_device_id
                if args.expected_device_id is not None
                else None if profile is None else profile.expected_device_id
            ),
            min_capture_count=(
                args.min_capture_count
                if args.min_capture_count is not None
                else 1 if profile_audit is None else profile_audit.min_capture_count
            ),
            max_rgb_depth_skew_s=(
                args.max_rgb_depth_skew_s
                if args.max_rgb_depth_skew_s is not None
                else 0.035 if profile is None else profile.max_rgb_depth_skew_s
            ),
            pixel_check=(
                args.pixel_check
                if args.pixel_check is not None
                else False if profile_audit is None else profile_audit.pixel_check
            ),
            min_frame_valid_depth_ratio=(
                args.min_frame_valid_depth_ratio
                if args.min_frame_valid_depth_ratio is not None
                else 0.10
                if profile_audit is None
                else profile_audit.min_frame_valid_depth_ratio
            ),
            min_depth_m=(
                args.min_depth_m
                if args.min_depth_m is not None
                else 0.10 if profile_audit is None else profile_audit.min_depth_m
            ),
            max_depth_m=(
                args.max_depth_m
                if args.max_depth_m is not None
                else 3.0 if profile_audit is None else profile_audit.max_depth_m
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "audit_type": "tomato_rgbd_recorded_shadow_v1",
            "summary": {"passed": False, "capture_count": 0, "issue_count": 1},
            "captures": [],
            "issues": [{"code": "audit_error", "message": str(exc)}],
        }
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["summary"]["passed"] else 2  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_spool", "main"]
