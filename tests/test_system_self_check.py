"""Admission, Orbbec SDK, and no-motion tests for system_self_check."""

from __future__ import annotations

from pathlib import Path
import unittest

from module.agent import RecoverableToolError
from module.robotics import ManualClock
from module.runtime import RuntimeContext
from module.skills.system_self_check import OrbbecViewerCameraHealthPort
from module.tools.env import MockEnv
from proto import HarvestCycleState
from task import build_task
from tests.test_harvest_cycle_task import _observation, _task_config, _tool_map


ROOT = Path(__file__).resolve().parents[1]
ORBBEC_ROOT = ROOT / "botSDK" / "orbbec-gemini-pro-viewer"


class SystemSelfCheckTests(unittest.TestCase):
    def _build(self, config: dict[str, object]):
        task = build_task(config)
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        return task, env, context, _tool_map(task, context)

    def test_healthy_check_is_read_only_and_admits_detection(self) -> None:
        config = _task_config(
            camera_health={
                "type": "mock",
                "sdk_root": str(ORBBEC_ROOT),
            }
        )
        task, env, _, tools = self._build(config)
        world = task.service.pick_service.robotics.verifier_context()["mock_world"]
        before = (
            world.ee_pose,
            world.hand_synergy,
            world.arm_sequence,
            world.hand_sequence,
            world.tactile_sequence,
        )
        try:
            result = tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            self.assertTrue(result["success"])
            self.assertTrue(result["output"]["admission_allowed"])
            checks = {
                item["name"]: item["state"] for item in result["output"]["checks"]
            }
            self.assertTrue(checks)
            self.assertEqual({"pass"}, set(checks.values()))
            self.assertEqual(
                before,
                (
                    world.ee_pose,
                    world.hand_synergy,
                    world.arm_sequence,
                    world.hand_sequence,
                    world.tactile_sequence,
                ),
            )
            detected = tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
            self.assertEqual(detected["workflow_state"], "select")
        finally:
            task.close()
            env.close()

    def test_detection_is_blocked_before_admission(self) -> None:
        task, env, _, tools = self._build(_task_config())
        try:
            with self.assertRaisesRegex(
                RecoverableToolError, "call check_harvest_system"
            ):
                tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
            self.assertIs(task.service.state, HarvestCycleState.INIT)
        finally:
            task.close()
            env.close()

    def test_camera_depth_and_calibration_fail_closed(self) -> None:
        cases = (
            (
                {
                    "type": "mock",
                    "sdk_root": "/tmp/linkagent-missing-orbbec-sdk",
                },
                "camera_sdk",
            ),
            (
                {"type": "mock", "depth_live": False},
                "camera_depth",
            ),
            (
                {"type": "mock", "calibration_id": "wrong-calibration"},
                "camera_calibration",
            ),
            (
                {"type": "mock", "stale_by_s": 2.0},
                "camera_connection",
            ),
        )
        for camera_health, expected_check in cases:
            with self.subTest(expected_check=expected_check):
                task, env, _, tools = self._build(
                    _task_config(camera_health=camera_health)
                )
                try:
                    with self.assertRaises(RecoverableToolError):
                        tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
                    result = task.service.atomic_results["system-self-check-1"]
                    self.assertIn(expected_check, result.output["failed_check_names"])
                    self.assertIs(task.service.state, HarvestCycleState.INIT)
                finally:
                    task.close()
                    env.close()

    def test_unready_vision_worker_blocks_detection_admission(self) -> None:
        task, env, _, tools = self._build(
            _task_config(vision_worker_health={"type": "mock", "ready": False})
        )
        try:
            with self.assertRaises(RecoverableToolError):
                tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            result = task.service.atomic_results["system-self-check-1"]
            self.assertIn("vision_worker", result.output["failed_check_names"])
            self.assertIs(task.service.state, HarvestCycleState.INIT)
        finally:
            task.close()
            env.close()

    def test_robot_hand_and_tactile_faults_fail_closed(self) -> None:
        cases = (
            ("arm_state_stale", "arm_state"),
            ("device_fault", "hand_state"),
            ("tactile_stale", "tactile_state"),
        )
        for outcome, expected_check in cases:
            with self.subTest(outcome=outcome):
                task, env, _, tools = self._build(_task_config(outcome=outcome))
                try:
                    with self.assertRaises(RecoverableToolError):
                        tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
                    result = task.service.atomic_results["system-self-check-1"]
                    self.assertIn(expected_check, result.output["failed_check_names"])
                finally:
                    task.close()
                    env.close()

    def test_unknown_estop_and_busy_control_lease_fail_closed(self) -> None:
        task, env, _, tools = self._build(
            _task_config(
                emergency_stop_health={
                    "type": "mock",
                    "available": True,
                    "healthy": None,
                }
            )
        )
        try:
            with self.assertRaises(RecoverableToolError):
                tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            result = task.service.atomic_results["system-self-check-1"]
            self.assertIn("emergency_stop", result.output["failed_check_names"])
        finally:
            task.close()
            env.close()

        task, env, _, tools = self._build(_task_config())
        bundle = task.service.pick_service.robotics
        lease = bundle.leases.acquire(
            owner_id="external-controller",
            side=bundle.side,
            ttl_s=1.0,
        )
        try:
            with self.assertRaises(RecoverableToolError):
                tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            result = task.service.atomic_results["system-self-check-1"]
            self.assertIn("control_lease", result.output["failed_check_names"])
        finally:
            bundle.leases.release(lease)
            task.close()
            env.close()

    def test_hardware_mode_rejects_simulated_health_evidence(self) -> None:
        config = _task_config()
        config["atomic_skills"]["system_self_check"] = {  # type: ignore[index]
            "type": "system_self_check",
            "execution_mode": "hardware",
        }
        config["atomic_skills"]["target_reobserve"][  # type: ignore[index]
            "require_new_sample"
        ] = True
        task, env, _, tools = self._build(config)
        try:
            with self.assertRaises(RecoverableToolError):
                tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            result = task.service.atomic_results["system-self-check-1"]
            self.assertIn("execution_mode", result.output["failed_check_names"])
        finally:
            task.close()
            env.close()

    def test_shadow_or_hardware_config_cannot_disable_fresh_reobservation(self) -> None:
        config = _task_config()
        config["atomic_skills"]["system_self_check"] = {  # type: ignore[index]
            "type": "system_self_check",
            "execution_mode": "shadow",
        }
        with self.assertRaisesRegex(ValueError, "require_new_sample=true"):
            build_task(config)

    def test_policy_mapper_mismatch_fails_closed(self) -> None:
        config = _task_config()
        config["atomic_skills"]["system_self_check"] = {  # type: ignore[index]
            "type": "system_self_check",
            "execution_mode": "offline_mock",
            "expected_hand_mapper_id": "wrong-mapper",
        }
        task, env, _, tools = self._build(config)
        try:
            with self.assertRaises(RecoverableToolError):
                tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            result = task.service.atomic_results["system-self-check-1"]
            self.assertIn("policy_manifest", result.output["failed_check_names"])
        finally:
            task.close()
            env.close()

    def test_orbbec_adapter_offline_probe_only_checks_sdk_layout(self) -> None:
        port = OrbbecViewerCameraHealthPort(
            clock=ManualClock(),
            sdk_root=ORBBEC_ROOT,
            live_probe=False,
        )
        try:
            snapshot = port.snapshot()
            self.assertTrue(snapshot.sdk_available)
            self.assertFalse(snapshot.device_connected)
            self.assertFalse(snapshot.rgb_live)
            self.assertFalse(snapshot.depth_live)
            self.assertFalse(snapshot.simulated)
            self.assertFalse(snapshot.details["live_probe"])
            self.assertIn("no device access attempted", snapshot.details["reason"])
        finally:
            port.close()

    def test_orbbec_status_parser_accepts_viewer_schema_without_hardware(self) -> None:
        clock = ManualClock()
        snapshot = OrbbecViewerCameraHealthPort.parse_status(
            {
                "state": "connected",
                "device": {
                    "serial": "recorded-serial",
                    "firmware": "recorded-firmware",
                    "sdk_version": "recorded-sdk",
                },
                "streams": {
                    "rgb": {"live": True, "source": "/dev/videoX", "error": None},
                    "depth": {
                        "live": True,
                        "source": "Orbbec SDK / PID 0614",
                        "error": None,
                    },
                },
            },
            clock=clock,
            sdk_available=True,
            calibration_id="recorded-calibration",
            evidence_sources=("recorded:/api/status",),
        )
        self.assertTrue(snapshot.device_connected)
        self.assertTrue(snapshot.rgb_live)
        self.assertTrue(snapshot.depth_live)
        self.assertEqual(snapshot.device_id, "recorded-serial")


if __name__ == "__main__":
    unittest.main()
