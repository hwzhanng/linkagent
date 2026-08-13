"""Republish an audited RGB-D recording through the live ``latest.json`` path."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

from script.audit_tomato_vision_spool import audit_spool
from script.import_tomato_rgbd_capture import import_capture
from script.tomato_vision_spool_profile import TomatoVisionSpoolProfile


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


def replay_recording(
    *,
    source_spool_root: str | Path,
    destination_profile: str | Path,
    publish_interval_s: float = 0.0,
    max_frames: int | None = None,
) -> dict[str, object]:
    if (
        isinstance(publish_interval_s, bool)
        or not isinstance(publish_interval_s, (int, float))
        or not math.isfinite(float(publish_interval_s))
        or not 0.0 <= float(publish_interval_s) <= 60.0
    ):
        raise ValueError("publish_interval_s must be finite and in [0, 60]")
    if max_frames is not None and (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames < 1
    ):
        raise ValueError("max_frames must be an integer >= 1 or None")
    profile = TomatoVisionSpoolProfile.load(destination_profile)
    source_root = Path(source_spool_root).expanduser().resolve()
    if source_root == profile.capture_root:
        raise ValueError("source and destination spool roots must be different")
    audit = audit_spool(
        source_root,
        expected_calibration_id=profile.expected_calibration_id,
        expected_device_id=profile.expected_device_id,
        min_capture_count=profile.audit.min_capture_count,
        max_rgb_depth_skew_s=profile.max_rgb_depth_skew_s,
        pixel_check=profile.audit.pixel_check,
        min_frame_valid_depth_ratio=profile.audit.min_frame_valid_depth_ratio,
        min_depth_m=profile.audit.min_depth_m,
        max_depth_m=profile.audit.max_depth_m,
    )
    summary = audit["summary"]
    if not isinstance(summary, Mapping) or summary.get("passed") is not True:
        raise ValueError("source recording failed the destination profile audit")
    rows = audit["captures"]
    if not isinstance(rows, list):
        raise TypeError("source audit captures must be a list")
    selected = rows if max_frames is None else rows[:max_frames]
    publications: list[dict[str, object]] = []
    started_ns = time.time_ns()
    for index, row in enumerate(selected):
        if not isinstance(row, Mapping):
            raise TypeError("source audit capture row must be a mapping")
        reference = row.get("capture_bundle")
        if not isinstance(reference, str) or not reference:
            raise ValueError("source audit capture_bundle must be a non-empty string")
        publications.append(
            import_capture(
                source_capture_root=source_root,
                source_bundle=reference,
                destination_spool_root=profile.capture_root,
                expected_device_id=profile.expected_device_id,
                expected_calibration_id=profile.expected_calibration_id,
                max_rgb_depth_skew_s=profile.max_rgb_depth_skew_s,
                max_image_bytes=profile.max_image_bytes,
                reserve_free_bytes=profile.reserve_free_bytes,
            )
        )
        if index + 1 < len(selected) and publish_interval_s > 0.0:
            time.sleep(float(publish_interval_s))
    return {
        "schema_version": 1,
        "replay_type": "tomato_rgbd_latest_spool_v1",
        "profile_id": profile.profile_id,
        "source_spool_root": str(source_root),
        "destination_spool_root": str(profile.capture_root),
        "started_unix_ns": started_ns,
        "completed_unix_ns": time.time_ns(),
        "publish_interval_s": float(publish_interval_s),
        "source_audit_summary": dict(summary),
        "published_count": len(publications),
        "publications": publications,
        "claim_boundary": (
            "publication replay only; this manifest does not prove every frame was "
            "processed by YOLO, VLM, geometry, or hardware"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Republish an audited recording through a latest-frame spool"
    )
    parser.add_argument("--source-spool-root", type=Path, required=True)
    parser.add_argument("--destination-profile", type=Path, required=True)
    parser.add_argument("--publish-interval-s", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = replay_recording(
            source_spool_root=args.source_spool_root,
            destination_profile=args.destination_profile,
            publish_interval_s=args.publish_interval_s,
            max_frames=args.max_frames,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"schema_version": 1, "status": "failed", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "replay_recording"]
