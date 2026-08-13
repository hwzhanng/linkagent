"""Import one validated recorded RGB-D bundle into the atomic spool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from script.tomato_vision_capture import RawRgbdCaptureBundle
from script.tomato_vision_spool import RgbdSpoolFrame, TomatoVisionSpoolWriter
from script.tomato_vision_spool_profile import TomatoVisionSpoolProfile


def import_capture(
    *,
    source_capture_root: str | Path,
    source_bundle: str | Path,
    destination_spool_root: str | Path,
    expected_device_id: str | None = None,
    expected_calibration_id: str | None = None,
    max_rgb_depth_skew_s: float = 0.035,
    max_image_bytes: int = 128_000_000,
    reserve_free_bytes: int = 0,
) -> dict[str, object]:
    source_root = Path(source_capture_root).expanduser().resolve()
    reference = Path(source_bundle)
    if reference.is_absolute() or any(part in {"", ".", ".."} for part in reference.parts):
        raise ValueError("source_bundle must be a confined relative path")
    bundle = RawRgbdCaptureBundle.load(
        source_root / reference,
        capture_root=source_root,
    )
    if expected_device_id is not None and bundle.device_id != expected_device_id:
        raise ValueError("recorded capture device_id does not match import profile")
    if (
        expected_calibration_id is not None
        and bundle.intrinsics.calibration_id != expected_calibration_id
    ):
        raise ValueError("recorded capture calibration_id does not match import profile")
    frame = RgbdSpoolFrame(
        capture_id=bundle.capture_id,
        rgb_png=bundle.rgb_path.read_bytes(),
        aligned_depth_png=bundle.aligned_depth_path.read_bytes(),
        depth_scale_m_per_unit=bundle.depth_scale_m_per_unit,
        intrinsics=bundle.intrinsics,
        camera_pose_in_world=bundle.camera_pose_in_world,
        rgb_sample_monotonic_ns=bundle.rgb_sample_monotonic_ns,
        depth_sample_monotonic_ns=bundle.depth_sample_monotonic_ns,
        device_id=bundle.device_id,
        evidence_sources=bundle.evidence_sources,
    )
    publication = TomatoVisionSpoolWriter(
        destination_spool_root,
        max_rgb_depth_skew_s=max_rgb_depth_skew_s,
        max_image_bytes=max_image_bytes,
        reserve_free_bytes=reserve_free_bytes,
    ).publish(frame)
    return {
        "schema_version": 1,
        "capture_id": publication.pointer.capture_id,
        "generation": publication.pointer.generation,
        "capture_bundle": publication.pointer.capture_bundle,
        "sample_monotonic_ns": publication.pointer.sample_monotonic_ns,
        "calibration_id": publication.pointer.calibration_id,
        "device_id": publication.pointer.device_id,
        "sidecar_sha256": publication.pointer.sidecar_sha256,
        "rgb_sha256": publication.rgb_sha256,
        "aligned_depth_sha256": publication.aligned_depth_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a validated recorded RGB-D bundle into a spool"
    )
    parser.add_argument("--source-capture-root", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--destination-spool-root", type=Path)
    destination.add_argument("--destination-profile", type=Path)
    parser.add_argument("--expected-device-id")
    parser.add_argument("--expected-calibration-id")
    parser.add_argument("--max-rgb-depth-skew-s", type=float)
    parser.add_argument("--max-image-bytes", type=int)
    parser.add_argument("--reserve-free-bytes", type=int)
    args = parser.parse_args(argv)
    try:
        profile = (
            None
            if args.destination_profile is None
            else TomatoVisionSpoolProfile.load(args.destination_profile)
        )
        profile_overrides = {
            "expected_device_id": args.expected_device_id,
            "expected_calibration_id": args.expected_calibration_id,
            "max_rgb_depth_skew_s": args.max_rgb_depth_skew_s,
            "max_image_bytes": args.max_image_bytes,
            "reserve_free_bytes": args.reserve_free_bytes,
        }
        if profile is not None and any(
            value is not None for value in profile_overrides.values()
        ):
            raise ValueError("import gates cannot override an authoritative profile")
        result = import_capture(
            source_capture_root=args.source_capture_root,
            source_bundle=args.source_bundle,
            destination_spool_root=(
                args.destination_spool_root if profile is None else profile.capture_root
            ),
            expected_device_id=(
                args.expected_device_id
                if args.expected_device_id is not None
                else None if profile is None else profile.expected_device_id
            ),
            expected_calibration_id=(
                args.expected_calibration_id
                if args.expected_calibration_id is not None
                else None if profile is None else profile.expected_calibration_id
            ),
            max_rgb_depth_skew_s=(
                (
                    0.035
                    if args.max_rgb_depth_skew_s is None
                    else args.max_rgb_depth_skew_s
                )
                if profile is None
                else profile.max_rgb_depth_skew_s
            ),
            max_image_bytes=(
                (128_000_000 if args.max_image_bytes is None else args.max_image_bytes)
                if profile is None
                else profile.max_image_bytes
            ),
            reserve_free_bytes=(
                (0 if args.reserve_free_bytes is None else args.reserve_free_bytes)
                if profile is None
                else profile.reserve_free_bytes
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["import_capture", "main"]
