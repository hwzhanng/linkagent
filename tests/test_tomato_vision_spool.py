"""Atomic RGB-D spool publication, recovery, and freshness tests."""

from __future__ import annotations

import binascii
from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
import zlib

from proto import CameraIntrinsics, Pose
from script.audit_tomato_vision_spool import audit_spool, main as audit_main
from script.import_tomato_rgbd_capture import import_capture, main as import_main
from script.tomato_vision_capture_source import run_capture_loop
from script.tomato_vision_spool import (
    RgbdSpoolFrame,
    TomatoVisionSpoolReader,
    TomatoVisionSpoolWriter,
    inspect_png,
)
from script.tomato_vision_spool_profile import TomatoVisionSpoolProfile
from script.replay_tomato_rgbd_recording import replay_recording


ROOT = Path(__file__).resolve().parents[1]


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _png(
    width: int,
    height: int,
    *,
    bit_depth: int,
    colour_type: int,
    value: int = 1,
) -> bytes:
    if colour_type == 2 and bit_depth == 8:
        pixel = bytes((value & 0xFF, 20, 30))
    elif colour_type == 0 and bit_depth == 16:
        pixel = struct.pack(">H", value & 0xFFFF)
    elif colour_type == 0 and bit_depth == 8:
        pixel = bytes((value & 0xFF,))
    else:
        raise AssertionError("unsupported test PNG shape")
    rows = b"".join(b"\x00" + pixel * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, colour_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(rows))
        + _chunk(b"IEND", b"")
    )


def _frame(sample_ns: int, *, capture_id: str | None = None) -> RgbdSpoolFrame:
    return RgbdSpoolFrame(
        capture_id=capture_id or f"capture-{sample_ns}",
        rgb_png=_png(4, 3, bit_depth=8, colour_type=2, value=sample_ns),
        aligned_depth_png=_png(
            4, 3, bit_depth=16, colour_type=0, value=800 + sample_ns % 20
        ),
        depth_scale_m_per_unit=0.001,
        intrinsics=CameraIntrinsics(
            width_px=4,
            height_px=3,
            fx_px=100.0,
            fy_px=100.0,
            cx_px=1.5,
            cy_px=1.0,
            calibration_id="calibration-cell-a",
        ),
        camera_pose_in_world=Pose(
            position_m=(0.1, 0.2, 0.3),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            frame_id="world",
        ),
        rgb_sample_monotonic_ns=sample_ns,
        depth_sample_monotonic_ns=sample_ns + 1,
        device_id="orbbec-test-001",
        evidence_sources=("orbbec:frame-sync", "orbbec:d2c-depth"),
    )


class _FakeSynchronizedSource:
    def __init__(
        self, frames: list[RgbdSpoolFrame], *, start_error: BaseException | None = None
    ) -> None:
        self.frames = list(frames)
        self.start_error = start_error
        self.started = False
        self.closed = False
        self.timeouts: list[float] = []

    def start(self) -> None:
        self.started = True
        if self.start_error is not None:
            raise self.start_error

    def read_synchronized_frame(self, *, timeout_s: float) -> RgbdSpoolFrame:
        if not self.started or self.closed:
            raise RuntimeError("fake source is not active")
        self.timeouts.append(timeout_s)
        if not self.frames:
            raise TimeoutError("fake source exhausted")
        return self.frames.pop(0)

    def close(self) -> None:
        self.closed = True


class TomatoVisionSpoolTests(unittest.TestCase):
    def test_publish_is_resolvable_and_latest_generation_advances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TomatoVisionSpoolWriter(root)
            first = writer.publish(_frame(1_000))
            second = writer.publish(_frame(2_000))
            reader = TomatoVisionSpoolReader(root)
            resolved = reader.resolve_latest()
            self.assertEqual(first.pointer.generation, 1)
            self.assertEqual(second.pointer.generation, 2)
            self.assertEqual(resolved.pointer.capture_id, "capture-2000")
            self.assertEqual(resolved.bundle.capture_id, "capture-2000")
            self.assertEqual(resolved.bundle.intrinsics.calibration_id, "calibration-cell-a")
            self.assertTrue(first.bundle_path.is_file())
            self.assertEqual(len(first.rgb_sha256), 64)
            self.assertEqual(len(first.aligned_depth_sha256), 64)

    def test_reader_waits_for_a_strictly_newer_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TomatoVisionSpoolWriter(root)
            first = writer.publish(_frame(1_000))
            reader = TomatoVisionSpoolReader(root)
            result: list[object] = []

            def wait_for_frame() -> None:
                try:
                    result.append(
                        reader.resolve_latest(
                            minimum_sample_monotonic_ns=first.pointer.sample_monotonic_ns,
                            timeout_s=0.5,
                            poll_interval_s=0.005,
                        )
                    )
                except BaseException as exc:
                    result.append(exc)

            thread = threading.Thread(target=wait_for_frame)
            thread.start()
            time.sleep(0.03)
            writer.publish(_frame(2_000))
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(result), 1)
            self.assertFalse(isinstance(result[0], BaseException), result)
            self.assertEqual(result[0].pointer.capture_id, "capture-2000")  # type: ignore[union-attr]

    def test_reader_times_out_instead_of_reusing_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication = TomatoVisionSpoolWriter(root).publish(_frame(1_000))
            with self.assertRaisesRegex(TimeoutError, "no newer RGB-D"):
                TomatoVisionSpoolReader(root).resolve_latest(
                    minimum_sample_monotonic_ns=publication.pointer.sample_monotonic_ns,
                    timeout_s=0.01,
                    poll_interval_s=0.002,
                )

    def test_timestamp_rollback_is_rejected_without_changing_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TomatoVisionSpoolWriter(root)
            publication = writer.publish(_frame(2_000))
            with self.assertRaisesRegex(ValueError, "must advance"):
                writer.publish(_frame(1_000))
            pointer = TomatoVisionSpoolReader(root).read_pointer()
            self.assertEqual(pointer.capture_id, publication.pointer.capture_id)
            self.assertEqual(pointer.generation, 1)

    def test_latest_sidecar_tampering_fails_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication = TomatoVisionSpoolWriter(root).publish(_frame(1_000))
            payload = json.loads(publication.bundle_path.read_text(encoding="utf-8"))
            payload["capture_id"] = "tampered"
            publication.bundle_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                TomatoVisionSpoolReader(root).resolve_latest()
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                TomatoVisionSpoolWriter(root).publish(_frame(2_000))

    def test_sdk_neutral_capture_loop_publishes_and_closes_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _FakeSynchronizedSource([_frame(1_000), _frame(2_000)])
            summary = run_capture_loop(
                source=source,
                writer=TomatoVisionSpoolWriter(root),
                expected_device_id="orbbec-test-001",
                expected_calibration_id="calibration-cell-a",
                stop_event=threading.Event(),
                read_timeout_s=0.25,
                max_frames=2,
            )
            self.assertTrue(source.started)
            self.assertTrue(source.closed)
            self.assertEqual(source.timeouts, [0.25, 0.25])
            self.assertEqual(summary.published_count, 2)
            self.assertEqual(summary.first_sample_monotonic_ns, 1_001)
            self.assertEqual(summary.last_sample_monotonic_ns, 2_001)
            self.assertFalse(summary.stopped_by_event)
            self.assertEqual(
                TomatoVisionSpoolReader(root).resolve_latest().pointer.capture_id,
                "capture-2000",
            )

    def test_capture_loop_identity_drift_fails_and_still_closes_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = replace(_frame(2_000), device_id="unexpected-device")
            source = _FakeSynchronizedSource([_frame(1_000), wrong])
            with self.assertRaisesRegex(ValueError, "device_id"):
                run_capture_loop(
                    source=source,
                    writer=TomatoVisionSpoolWriter(root),
                    expected_device_id="orbbec-test-001",
                    expected_calibration_id="calibration-cell-a",
                    stop_event=threading.Event(),
                    max_frames=2,
                )
            self.assertTrue(source.closed)
            reader = TomatoVisionSpoolReader(root)
            self.assertEqual(reader.read_pointer().generation, 1)
            self.assertEqual(len(reader.capture_paths()), 1)

    def test_capture_loop_closes_on_start_failure_and_calibration_value_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failed_start = _FakeSynchronizedSource(
                [], start_error=RuntimeError("camera start failed")
            )
            with self.assertRaisesRegex(RuntimeError, "camera start failed"):
                run_capture_loop(
                    source=failed_start,
                    writer=TomatoVisionSpoolWriter(Path(directory) / "failed"),
                    expected_device_id="orbbec-test-001",
                    expected_calibration_id="calibration-cell-a",
                    stop_event=threading.Event(),
                    max_frames=1,
                )
            self.assertTrue(failed_start.closed)

            changed_intrinsics = replace(
                _frame(2_000).intrinsics,
                fx_px=101.0,
            )
            drift_source = _FakeSynchronizedSource(
                [
                    _frame(1_000),
                    replace(_frame(2_000), intrinsics=changed_intrinsics),
                ]
            )
            with self.assertRaisesRegex(ValueError, "calibration parameters changed"):
                run_capture_loop(
                    source=drift_source,
                    writer=TomatoVisionSpoolWriter(Path(directory) / "drift"),
                    expected_device_id="orbbec-test-001",
                    expected_calibration_id="calibration-cell-a",
                    stop_event=threading.Event(),
                    max_frames=2,
                )
            self.assertTrue(drift_source.closed)

    def test_recorded_bundle_import_preserves_identity_hashes_and_cli_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "source"
            destination = base / "destination"
            publication = TomatoVisionSpoolWriter(source_root).publish(_frame(1_000))
            reference = publication.bundle_path.relative_to(source_root)
            result = import_capture(
                source_capture_root=source_root,
                source_bundle=reference,
                destination_spool_root=destination,
                expected_device_id="orbbec-test-001",
                expected_calibration_id="calibration-cell-a",
            )
            self.assertEqual(result["capture_id"], "capture-1000")
            imported = TomatoVisionSpoolReader(destination).resolve_latest()
            self.assertEqual(imported.bundle.capture_id, "capture-1000")
            self.assertEqual(
                imported.bundle.rgb_path.read_bytes(),
                publication.bundle_path.with_name("rgb.png").read_bytes(),
            )
            rejected_destination = base / "rejected"
            self.assertEqual(
                import_main(
                    [
                        "--source-capture-root",
                        str(source_root),
                        "--source-bundle",
                        str(reference),
                        "--destination-spool-root",
                        str(rejected_destination),
                        "--expected-device-id",
                        "wrong-device",
                    ]
                ),
                2,
            )
            self.assertFalse(rejected_destination.exists())

    def test_recorded_shadow_profile_drives_import_and_audit(self) -> None:
        shipped = TomatoVisionSpoolProfile.load(
            ROOT / "config" / "tomato_vision_spool_recorded_shadow.yaml"
        )
        self.assertEqual(
            shipped.capture_root,
            ROOT / "artifacts" / "vision_spool" / "recorded-shadow",
        )
        self.assertEqual(shipped.audit.min_capture_count, 100)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_root = base / "source"
            source_publication = TomatoVisionSpoolWriter(source_root).publish(
                _frame(1_000)
            )
            profile_path = base / "profile.json"
            destination = base / "recorded-shadow"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile_id": "test-recorded-shadow",
                        "deployment_mode": "recorded_shadow",
                        "hardware_approved": False,
                        "capture_root": "recorded-shadow",
                        "expected_device_id": "orbbec-test-001",
                        "expected_calibration_id": "calibration-cell-a",
                        "max_rgb_depth_skew_s": 0.035,
                        "max_image_bytes": 1000000,
                        "reserve_free_bytes": 0,
                        "audit": {
                            "min_capture_count": 1,
                            "pixel_check": True,
                            "min_frame_valid_depth_ratio": 1.0,
                            "min_depth_m": 0.1,
                            "max_depth_m": 3.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                import_main(
                    [
                        "--source-capture-root",
                        str(source_root),
                        "--source-bundle",
                        str(source_publication.bundle_path.relative_to(source_root)),
                        "--destination-profile",
                        str(profile_path),
                    ]
                ),
                0,
            )
            self.assertTrue(destination.is_dir())
            self.assertEqual(audit_main(["--profile", str(profile_path)]), 0)

    def test_audited_recording_replay_republishes_ordered_latest_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            writer = TomatoVisionSpoolWriter(source)
            writer.publish(_frame(1_000))
            writer.publish(_frame(2_000))
            destination = base / "replay-destination"
            profile_path = base / "replay-profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile_id": "test-replay",
                        "deployment_mode": "recorded_shadow",
                        "hardware_approved": False,
                        "capture_root": str(destination),
                        "expected_device_id": "orbbec-test-001",
                        "expected_calibration_id": "calibration-cell-a",
                        "max_rgb_depth_skew_s": 0.035,
                        "max_image_bytes": 1000000,
                        "reserve_free_bytes": 0,
                        "audit": {
                            "min_capture_count": 2,
                            "pixel_check": True,
                            "min_frame_valid_depth_ratio": 1.0,
                            "min_depth_m": 0.1,
                            "max_depth_m": 3.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = replay_recording(
                source_spool_root=source,
                destination_profile=profile_path,
            )
            self.assertEqual(report["published_count"], 2)
            latest = TomatoVisionSpoolReader(destination).resolve_latest()
            self.assertEqual(latest.pointer.generation, 2)
            self.assertEqual(latest.pointer.capture_id, "capture-2000")
            self.assertIn("does not prove", report["claim_boundary"])

    def test_png_format_dimensions_skew_and_size_fail_closed(self) -> None:
        frame = _frame(1_000)
        with self.assertRaisesRegex(ValueError, "16-bit grayscale"):
            replace(
                frame,
                aligned_depth_png=_png(4, 3, bit_depth=8, colour_type=0),
            )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            replace(
                frame,
                aligned_depth_png=_png(5, 3, bit_depth=16, colour_type=0),
            )
        with self.assertRaisesRegex(ValueError, "IEND"):
            inspect_png(frame.rgb_png[:-12], field_name="truncated")
        with tempfile.TemporaryDirectory() as directory:
            writer = TomatoVisionSpoolWriter(directory, max_rgb_depth_skew_s=0.001)
            with self.assertRaisesRegex(ValueError, "timestamp skew"):
                writer.publish(
                    replace(frame, depth_sample_monotonic_ns=2_000_000)
                )
            tiny_limit = TomatoVisionSpoolWriter(directory, max_image_bytes=1024)
            oversized_rgb = (
                frame.rgb_png[:-12]
                + _chunk(b"tEXt", b"x" * 2000)
                + frame.rgb_png[-12:]
            )
            with self.assertRaisesRegex(ValueError, "exceeds max_image_bytes"):
                tiny_limit.publish(replace(frame, rgb_png=oversized_rgb))

    def test_concurrent_same_timestamp_has_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TomatoVisionSpoolWriter(root)
            barrier = threading.Barrier(3)
            outcomes: list[object] = []

            def publish(capture_id: str) -> None:
                barrier.wait()
                try:
                    outcomes.append(
                        writer.publish(_frame(1_000, capture_id=capture_id))
                    )
                except BaseException as exc:
                    outcomes.append(exc)

            threads = [
                threading.Thread(target=publish, args=(f"concurrent-{index}",))
                for index in range(2)
            ]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=2.0)
            successes = [item for item in outcomes if not isinstance(item, BaseException)]
            failures = [item for item in outcomes if isinstance(item, BaseException)]
            self.assertEqual(len(successes), 1, outcomes)
            self.assertEqual(len(failures), 1, outcomes)
            self.assertIn("must advance", str(failures[0]))

    def test_staging_recovery_and_prune_are_dry_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TomatoVisionSpoolWriter(root)
            writer.publish(_frame(1_000))
            writer.publish(_frame(2_000))
            writer.publish(_frame(3_000))
            staging = root / "captures" / ".staging-crashed"
            staging.mkdir()
            old = time.time() - 120.0
            os.utime(staging, (old, old))
            self.assertEqual(writer.recover_staging(older_than_s=60.0), (staging,))
            self.assertTrue(staging.exists())
            writer.recover_staging(older_than_s=60.0, dry_run=False)
            self.assertFalse(staging.exists())
            removable = writer.prune(keep_last=1)
            self.assertEqual(len(removable), 2)
            self.assertTrue(all(path.exists() for path in removable))
            writer.prune(keep_last=1, dry_run=False)
            self.assertTrue(all(not path.exists() for path in removable))
            self.assertEqual(
                TomatoVisionSpoolReader(root).resolve_latest().pointer.capture_id,
                "capture-3000",
            )

    def test_symlink_and_relative_path_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "spool"
            outside = parent / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "captures").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes capture_root"):
                TomatoVisionSpoolWriter(root)
            with self.assertRaisesRegex(ValueError, "confined relative"):
                TomatoVisionSpoolReader(root, latest_reference="../latest.json")

    def test_recorded_shadow_audit_checks_pixels_hashes_identity_and_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TomatoVisionSpoolWriter(root)
            writer.publish(_frame(1_000))
            writer.publish(_frame(2_000))
            report = audit_spool(
                root,
                expected_calibration_id="calibration-cell-a",
                expected_device_id="orbbec-test-001",
                min_capture_count=2,
                pixel_check=True,
                min_frame_valid_depth_ratio=1.0,
            )
            self.assertTrue(report["summary"]["passed"])
            self.assertEqual(report["summary"]["capture_count"], 2)
            self.assertEqual(
                {row["valid_depth_ratio"] for row in report["captures"]}, {1.0}
            )
            output = root / "audit.json"
            self.assertEqual(
                audit_main(
                    [
                        "--capture-root",
                        str(root),
                        "--output",
                        str(output),
                        "--expected-calibration-id",
                        "calibration-cell-a",
                        "--expected-device-id",
                        "orbbec-test-001",
                        "--min-capture-count",
                        "2",
                    ]
                ),
                0,
            )
            self.assertTrue(output.is_file())

    def test_recorded_shadow_audit_fails_on_profile_drift_and_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TomatoVisionSpoolWriter(root)
            writer.publish(_frame(1_000))
            changed = replace(
                _frame(2_000),
                intrinsics=CameraIntrinsics(
                    width_px=4,
                    height_px=3,
                    fx_px=100.0,
                    fy_px=100.0,
                    cx_px=1.5,
                    cy_px=1.0,
                    calibration_id="wrong-calibration",
                ),
            )
            writer.publish(changed)
            changed_values = replace(
                _frame(3_000),
                intrinsics=replace(_frame(3_000).intrinsics, fx_px=101.0),
                depth_scale_m_per_unit=0.002,
            )
            writer.publish(changed_values)
            (root / "captures" / ".staging-interrupted").mkdir()
            report = audit_spool(root, min_capture_count=3)
            self.assertFalse(report["summary"]["passed"])
            codes = {issue["code"] for issue in report["issues"]}
            self.assertIn("calibration_mismatch", codes)
            self.assertIn("intrinsics_mismatch", codes)
            self.assertIn("depth_scale_mismatch", codes)
            self.assertIn("staging_present", codes)


if __name__ == "__main__":
    unittest.main()
