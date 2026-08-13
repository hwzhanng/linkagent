"""Loopback worker, capture-bundle, and fail-closed inference tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from module.robotics import ManualClock
from module.skills.system_self_check import LoopbackVisionWorkerHealthPort
from module.skills.tomato_vision import TomatoVisionWorkerClient
from proto import CameraIntrinsics, Observation, Pose
from script.generate_tomato_vision_sim_capture import generate_capture
from script.tomato_vision_capture import RawRgbdCaptureBundle
from script.tomato_vision_spool import RgbdSpoolFrame, TomatoVisionSpoolWriter
from script.tomato_vision_worker import (
    VisionWorkerServer,
    build_worker_from_config,
    main as worker_main,
)
from script.tomato_vision_worker_ark import (
    ArkResponsesTomatoBoxDetector,
    ArkVlmBoxDepthBackend,
)
from script.tomato_vision_worker_inference import (
    HttpJsonVlmSelector,
    UltralyticsVlmBackend,
)
from tests.test_tomato_vision_spool import _png


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = ROOT / "artifacts" / "vision_worker" / "replay"


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self._raw[:limit]


class _FakeArkDetector:
    model_id = "ark-test-model"
    configured = True
    last_response_id = "resp-test-001"
    last_resolved_model = "ark-test-model-resolved"

    def detect(
        self,
        *,
        rgb_png: bytes,
        width: int,
        height: int,
        requested_track_id: str | None,
    ) -> list[dict[str, object]]:
        del rgb_png, requested_track_id
        if (width, height) != (640, 480):
            raise AssertionError("unexpected fake capture dimensions")
        return [
            {
                "bbox": (365.0, 210.0, 425.0, 270.0),
                "confidence": 0.96,
                "ripeness": 0.91,
                "occlusion": 0.08,
                "center_point": (395, 240),
                "stem_point": (395, 218),
                "reason": "ripe visible tomato",
                "label": "tomato",
            }
        ]


def _observation(bundle: str = "tomato_001.json") -> Observation:
    return Observation(
        sequence=7,
        timestamp_s=1.0,
        metadata={"tomato_vision_worker": {"capture_bundle": bundle}},
    )


class _RawSpoolReplayBackend:
    backend_id = "raw_spool_replay_test_v1"

    def process(
        self,
        *,
        bundle_path: Path,
        observation_sequence: int,
        requested_track_id: str | None,
    ) -> dict[str, object]:
        del observation_sequence
        raw = RawRgbdCaptureBundle.load(
            bundle_path, capture_root=bundle_path.parents[2]
        )
        payload = json.loads(
            (CAPTURE_ROOT / "tomato_001.json").read_text(encoding="utf-8")
        )["tomato_vision"]
        payload["camera"] = raw.camera_payload()
        if requested_track_id not in {None, "tomato-1"}:
            raise ValueError("test replay target track is not visible")
        return payload

    def health(self) -> dict[str, object]:
        return {"ready": True, "backend_id": self.backend_id, "simulated": True}

    def close(self) -> None:
        return None


def _raw_spool_frame(sample_ns: int) -> RgbdSpoolFrame:
    return RgbdSpoolFrame(
        capture_id=f"raw-{sample_ns}",
        rgb_png=_png(640, 480, bit_depth=8, colour_type=2, value=20),
        aligned_depth_png=_png(
            640, 480, bit_depth=16, colour_type=0, value=800
        ),
        depth_scale_m_per_unit=0.001,
        intrinsics=CameraIntrinsics(
            width_px=640,
            height_px=480,
            fx_px=600.0,
            fy_px=600.0,
            cx_px=320.0,
            cy_px=240.0,
            calibration_id="orbbec_gemini_pro_mock_calibration_v1",
        ),
        camera_pose_in_world=Pose(
            position_m=(0.35, -0.25, -0.08),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            frame_id="world",
        ),
        rgb_sample_monotonic_ns=sample_ns,
        depth_sample_monotonic_ns=sample_ns + 1,
        device_id="orbbec-replay-001",
        evidence_sources=("test:frame-sync", "test:d2c-depth"),
    )


class TomatoVisionWorkerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = build_worker_from_config(
            {
                "schema_version": 1,
                "worker_id": "worker-test-01",
                "host": "127.0.0.1",
                "port": 0,
                "capture_root": "../artifacts/vision_worker/replay",
                "max_request_bytes": 64000,
                "backend": {"type": "replay"},
            },
            config_dir=ROOT / "config",
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2.0)
        self.server.close_resources()

    def _client(self, *, worker_id: str = "worker-test-01") -> TomatoVisionWorkerClient:
        return TomatoVisionWorkerClient(
            clock=ManualClock(),
            endpoint=self.endpoint,
            capture_root=CAPTURE_ROOT,
            worker_id=worker_id,
            expected_calibration_id="orbbec_gemini_pro_mock_calibration_v1",
            expected_device_id="orbbec-replay-001",
        )

    def test_health_and_http_result_are_identity_checked_and_revalidated(self) -> None:
        client = self._client()
        try:
            health = client.health()
            self.assertTrue(health["ready"])
            self.assertEqual(health["backend_id"], "replay_v1")
            result = client.observe(_observation())
            self.assertEqual(result.geometry.target.target_id, "tomato-1")
            self.assertEqual(result.geometry.instance_id, "mask-1")
            self.assertEqual(result.frame.device_id, "orbbec-replay-001")
        finally:
            client.close()

    def test_system_self_check_health_port_uses_the_live_worker_endpoint(self) -> None:
        port = LoopbackVisionWorkerHealthPort(
            clock=ManualClock(),
            endpoint=self.endpoint,
            capture_root=CAPTURE_ROOT,
            worker_id="worker-test-01",
        )
        try:
            snapshot = port.snapshot()
            self.assertTrue(snapshot.ready)
            self.assertTrue(snapshot.simulated)
            self.assertEqual(snapshot.backend_id, "replay_v1")
        finally:
            port.close()

    def test_client_rejects_capture_path_traversal_before_network(self) -> None:
        client = self._client()
        try:
            with self.assertRaisesRegex(ValueError, "escapes capture_root"):
                client.observe(_observation("../outside.json"))
        finally:
            client.close()

    def test_server_rejects_unknown_fields_and_path_traversal(self) -> None:
        for payload in (
            {
                "schema_version": 1,
                "request_id": "bad-field",
                "observation_sequence": 1,
                "capture_bundle": "tomato_001.json",
                "capture_reference_kind": "bundle",
                "requested_track_id": None,
                "minimum_sample_monotonic_ns": None,
                "new_frame_wait_timeout_s": 0.0,
                "extra": True,
            },
            {
                "schema_version": 1,
                "request_id": "bad-path",
                "observation_sequence": 1,
                "capture_bundle": "../outside.json",
                "capture_reference_kind": "bundle",
                "requested_track_id": None,
                "minimum_sample_monotonic_ns": None,
                "new_frame_wait_timeout_s": 0.0,
            },
        ):
            with self.subTest(payload=payload):
                request = Request(
                    f"{self.endpoint}/v1/process",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as raised:
                    urlopen(request, timeout=2.0)
                self.assertEqual(raised.exception.code, 422)

    def test_worker_identity_and_pinned_track_mismatch_fail_closed(self) -> None:
        wrong_identity = self._client(worker_id="some-other-worker")
        try:
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                wrong_identity.health()
        finally:
            wrong_identity.close()
        client = self._client()
        try:
            with self.assertRaisesRegex(RuntimeError, "HTTP 422"):
                client.observe(_observation(), requested_track_id="tomato-2")
        finally:
            client.close()

    def test_worker_bounds_capture_sidecar_size(self) -> None:
        self.server.max_bundle_bytes = 1024
        client = self._client()
        try:
            with self.assertRaisesRegex(RuntimeError, "HTTP 422"):
                client.observe(_observation())
        finally:
            client.close()

    def test_check_mode_validates_without_rebinding_the_configured_live_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "worker.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "worker_id": "worker-test-01",
                        "host": "127.0.0.1",
                        "port": self.server.server_address[1],
                        "capture_root": str(CAPTURE_ROOT),
                        "backend": {"type": "replay"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                worker_main(["--config", str(config_path), "--check"]), 0
            )


class TomatoVisionCaptureContractTests(unittest.TestCase):
    def test_synthetic_capture_generator_publishes_aligned_rgbd_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication = generate_capture(
                root,
                sample_monotonic_ns=999_000_000,
            )
            capture = RawRgbdCaptureBundle.load(
                publication.bundle_path,
                capture_root=root,
            )
            self.assertEqual(capture.rgb_sample_monotonic_ns, 999_000_000)
            self.assertEqual(capture.depth_sample_monotonic_ns, 1_000_000_000)
            self.assertEqual(
                capture.intrinsics.calibration_id,
                "synthetic-orbbec-aligned-rgbd-v1",
            )
            self.assertEqual(capture.intrinsics.width_px, 640)
            self.assertEqual(capture.intrinsics.height_px, 480)

    def _write_bundle(self, root: Path, *, rgb_path: str = "rgb.png") -> Path:
        (root / "rgb.png").write_bytes(b"rgb-placeholder")
        (root / "depth.png").write_bytes(b"depth-placeholder")
        path = root / "capture.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "capture_id": "capture-001",
                    "rgb_path": rgb_path,
                    "aligned_depth_path": "depth.png",
                    "depth_scale_m_per_unit": 0.001,
                    "camera": {
                        "intrinsics": {
                            "width_px": 640,
                            "height_px": 480,
                            "fx_px": 600.0,
                            "fy_px": 600.0,
                            "cx_px": 320.0,
                            "cy_px": 240.0,
                            "calibration_id": "calibration-001",
                        },
                        "camera_pose_in_world": {
                            "position_m": [0.0, 0.0, 0.0],
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "frame_id": "world",
                        },
                        "rgb_sample_monotonic_ns": 1000000000,
                        "depth_sample_monotonic_ns": 1001000000,
                        "depth_aligned_to_rgb": True,
                        "device_id": "orbbec-test-001",
                        "evidence_sources": ["orbbec:rgb", "orbbec:d2c-depth"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_raw_bundle_carries_calibration_sync_and_d2c_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = RawRgbdCaptureBundle.load(
                self._write_bundle(root), capture_root=root
            )
            self.assertEqual(capture.capture_id, "capture-001")
            self.assertEqual(capture.depth_scale_m_per_unit, 0.001)
            self.assertEqual(capture.intrinsics.calibration_id, "calibration-001")
            self.assertTrue(capture.camera_payload()["depth_aligned_to_rgb"])

    def test_raw_bundle_image_references_cannot_escape_capture_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "captures"
            root.mkdir()
            (parent / "outside.png").write_bytes(b"outside")
            path = self._write_bundle(root, rgb_path="../outside.png")
            with self.assertRaisesRegex(ValueError, "escapes capture_root"):
                RawRgbdCaptureBundle.load(path, capture_root=root)

    def test_real_backend_needs_explicit_local_weights_before_optional_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captures = root / "captures"
            captures.mkdir()
            config = {
                "schema_version": 1,
                "worker_id": "real-worker-test",
                "host": "127.0.0.1",
                "port": 0,
                "capture_root": "captures",
                "max_request_bytes": 64000,
                "backend": {
                    "type": "ultralytics_vlm",
                    "model_path": "missing.pt",
                    "vlm": {
                        "type": "http_json",
                        "endpoint": "http://127.0.0.1:9999/v1/select",
                        "model_id": "test-vlm",
                    },
                },
            }
            with self.assertRaisesRegex(FileNotFoundError, "local YOLO"):
                build_worker_from_config(config, config_dir=root)

    def test_plain_http_vlm_is_loopback_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            HttpJsonVlmSelector(
                endpoint="http://example.com/v1/select", model_id="test-vlm"
            )

    def test_ambiguous_iou_assignment_does_not_reuse_an_old_track(self) -> None:
        backend = object.__new__(UltralyticsVlmBackend)
        backend._previous_tracks = {  # type: ignore[attr-defined]
            "tomato-000001": (0.0, 0.0, 10.0, 10.0),
            "tomato-000002": (2.0, 0.0, 12.0, 10.0),
        }
        backend._next_track_id = 3  # type: ignore[attr-defined]
        backend._track_iou = 0.3  # type: ignore[attr-defined]
        backend._track_iou_ambiguity_margin = 0.05  # type: ignore[attr-defined]
        detections: list[dict[str, object]] = [
            {"bbox": (1.0, 0.0, 11.0, 10.0)}
        ]
        backend._assign_tracks(detections)
        self.assertEqual(detections[0]["track_id"], "tomato-000003")


class ArkVlmBoxBackendTests(unittest.TestCase):
    def test_ark_responses_client_sends_image_and_strict_schema(self) -> None:
        response = {
            "id": "resp-live-shape-test",
            "model": "doubao-resolved-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "boxes": [
                                        {
                                            "bbox_xyxy_1000": [150, 150, 800, 800],
                                            "confidence": 0.94,
                                            "ripeness_score": 0.90,
                                            "occlusion_ratio": 0.10,
                                            "center_point_xy_1000": [470, 470],
                                            "stem_point_xy_1000": [470, 190],
                                            "reason": "ripe visible tomato",
                                        }
                                    ]
                                }
                            ),
                        }
                    ],
                }
            ],
        }
        captured: list[dict[str, object]] = []

        def fake_urlopen(request: Request, *, timeout: float) -> _FakeHttpResponse:
            self.assertEqual(timeout, 3.0)
            self.assertEqual(request.full_url, "http://127.0.0.1:9999/responses")
            payload = json.loads(request.data.decode("utf-8"))
            captured.append(payload)
            return _FakeHttpResponse(response)

        detector = ArkResponsesTomatoBoxDetector(
            base_url="http://127.0.0.1:9999",
            model_id="doubao-test",
            api_key_env="ARK_TEST_KEY",
            timeout_s=3.0,
        )
        with patch.dict(os.environ, {"ARK_TEST_KEY": "test-only"}, clear=False):
            with patch("script.tomato_vision_worker_ark.urlopen", fake_urlopen):
                boxes = detector.detect(
                    rgb_png=_png(64, 64, bit_depth=8, colour_type=2, value=200),
                    width=64,
                    height=64,
                    requested_track_id=None,
                )

        self.assertEqual(len(captured), 1)
        request_payload = captured[0]
        self.assertEqual(request_payload["model"], "doubao-test")
        content = request_payload["input"][0]["content"]  # type: ignore[index]
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertTrue(request_payload["text"]["format"]["strict"])  # type: ignore[index]
        self.assertEqual(request_payload["thinking"], {"type": "disabled"})
        self.assertEqual(boxes[0]["bbox"], (9.6, 9.6, 51.2, 51.2))
        self.assertEqual(detector.last_response_id, "resp-live-shape-test")
        self.assertEqual(detector.last_resolved_model, "doubao-resolved-test")

    def test_plain_http_ark_is_loopback_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            ArkResponsesTomatoBoxDetector(
                base_url="http://example.com/api/coding/v3",
                model_id="doubao-test",
            )

    def test_ark_box_backend_builds_depth_supported_instances_and_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication = TomatoVisionSpoolWriter(root).publish(
                _raw_spool_frame(2_000_000_000)
            )
            backend = ArkVlmBoxDepthBackend(
                capture_root=root,
                config_dir=root,
                ark_vlm={
                    "type": "ark_responses",
                    "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
                    "model_id": "doubao-test",
                    "api_key_env": "ARK_TEST_KEY",
                },
            )
            backend._detector = _FakeArkDetector()  # type: ignore[assignment]
            try:
                first = backend.process(
                    bundle_path=publication.bundle_path,
                    observation_sequence=7,
                    requested_track_id=None,
                )
                second = backend.process(
                    bundle_path=publication.bundle_path,
                    observation_sequence=8,
                    requested_track_id="tomato-000001",
                )
            finally:
                backend.close()

            instance = first["instances"][0]  # type: ignore[index]
            self.assertEqual(instance["track_id"], "tomato-000001")
            self.assertGreater(instance["mask_area_px"], 6)
            self.assertEqual(
                instance["mask_area_px"], instance["valid_depth_pixel_count"]
            )
            self.assertEqual(
                second["vlm_selection"]["selected_instance_id"],  # type: ignore[index]
                "ark-box-depth-1",
            )
            evidence = first["camera"]["evidence_sources"]  # type: ignore[index]
            self.assertIn("ark-response-id:resp-test-001", evidence)
            self.assertIn(
                "instance-authority:ark-vlm-box+aligned-depth-pseudomask",
                evidence,
            )


class TomatoVisionLatestSpoolProtocolTests(unittest.TestCase):
    def test_worker_waits_for_new_latest_capture_and_never_reuses_old_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = TomatoVisionSpoolWriter(root)
            first_publication = writer.publish(_raw_spool_frame(1_000_000_000))
            server = VisionWorkerServer(
                ("127.0.0.1", 0),
                worker_id="latest-worker-test",
                capture_root=root,
                backend=_RawSpoolReplayBackend(),
                max_request_bytes=64_000,
                max_bundle_bytes=2_000_000,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = TomatoVisionWorkerClient(
                clock=ManualClock(start_ns=1_000_000_000),
                endpoint=f"http://127.0.0.1:{server.server_address[1]}",
                capture_root=root,
                worker_id="latest-worker-test",
                expected_calibration_id="orbbec_gemini_pro_mock_calibration_v1",
                expected_device_id="orbbec-replay-001",
                new_frame_wait_timeout_s=0.20,
            )
            observation = Observation(
                sequence=11,
                metadata={
                    "tomato_vision_worker": {"capture_latest": "latest.json"}
                },
            )
            try:
                first = client.observe(observation)
                self.assertEqual(
                    first.geometry.target.sample_monotonic_ns,
                    first_publication.pointer.sample_monotonic_ns,
                )
                result: list[object] = []

                def reobserve() -> None:
                    try:
                        result.append(
                            client.observe(
                                observation,
                                requested_track_id="tomato-1",
                                minimum_sample_monotonic_ns=first.geometry.target.sample_monotonic_ns,
                            )
                        )
                    except BaseException as exc:
                        result.append(exc)

                reobserve_thread = threading.Thread(target=reobserve)
                reobserve_thread.start()
                time.sleep(0.03)
                second_publication = writer.publish(
                    _raw_spool_frame(1_100_000_000)
                )
                reobserve_thread.join(timeout=1.0)
                self.assertFalse(reobserve_thread.is_alive())
                self.assertEqual(len(result), 1)
                self.assertFalse(isinstance(result[0], BaseException), result)
                self.assertEqual(
                    result[0].geometry.target.sample_monotonic_ns,  # type: ignore[union-attr]
                    second_publication.pointer.sample_monotonic_ns,
                )
                with self.assertRaisesRegex(RuntimeError, "HTTP 409"):
                    client.observe(
                        observation,
                        requested_track_id="tomato-1",
                        minimum_sample_monotonic_ns=second_publication.pointer.sample_monotonic_ns,
                    )
            finally:
                client.close()
                server.shutdown()
                thread.join(timeout=2.0)
                server.close_resources()


class TomatoVisionWorkerProcessTests(unittest.TestCase):
    def test_shipped_replay_backend_crosses_a_real_process_boundary(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "worker.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "worker_id": "worker-process-01",
                        "host": "127.0.0.1",
                        "port": port,
                        "capture_root": str(CAPTURE_ROOT),
                        "max_request_bytes": 64000,
                        "backend": {"type": "replay"},
                    }
                ),
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "script.tomato_vision_worker",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            endpoint = f"http://127.0.0.1:{port}"
            client = TomatoVisionWorkerClient(
                clock=ManualClock(),
                endpoint=endpoint,
                capture_root=CAPTURE_ROOT,
                worker_id="worker-process-01",
            )
            try:
                deadline = time.monotonic() + 5.0
                while True:
                    try:
                        client.health()
                        break
                    except RuntimeError:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            self.fail("vision worker subprocess did not become ready")
                        time.sleep(0.05)
                result = client.observe(_observation())
                self.assertEqual(result.geometry.target.target_id, "tomato-1")
            finally:
                client.close()
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.0)


if __name__ == "__main__":
    unittest.main()
