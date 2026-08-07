"""Offline C4-O tests for dual-arm/dual-G20 adapter translation."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from module.robotics import (
    BimanualDeploymentConfig,
    BimanualRobotInfra,
    ControlLeaseManager,
    HandControlPort,
    HandSynergyCommand,
    JointMoveCommand,
    ManualClock,
    MockHandSynergyMapper,
    PoseMoveCommand,
    TactilePort,
    ArmControlPort,
)
from module.robotics.adapters import (
    ArmSdkState,
    G20FingerSample,
    G20HandTactileAdapter,
    G20SdkState,
    G20TactileSample,
    LinkBotArmAdapter,
    SdkPoseRpy,
)
from module.robotics.adapters.linkbot_g20 import (
    quaternion_wxyz_to_rpy,
    rpy_to_quaternion_wxyz,
)
from proto import (
    ArmSide,
    CommandResult,
    ExecutionHandle,
    ExecutionState,
    HandSynergy,
    Pose,
)


ROOT = Path(__file__).resolve().parents[1]


def _terminal(command_id: str, *, now: int, message: str = "ok") -> CommandResult:
    return CommandResult(
        command_id,
        ExecutionState.SUCCEEDED,
        message,
        started_monotonic_ns=now,
        finished_monotonic_ns=now,
    )


class FakeArmGateway:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.open_calls: list[bool] = []
        self.operations: list[tuple[str, object]] = []
        self.closed = 0
        self.positions = {ArmSide.LEFT: (0.0,) * 7, ArmSide.RIGHT: (0.1,) * 7}
        self.results: dict[str, CommandResult] = {}
        self.fail_emergency_side: ArmSide | None = None

    def open(self, *, read_only: bool) -> None:
        self.open_calls.append(read_only)

    def read_state(self, side: ArmSide) -> ArmSdkState:
        now = self.clock.monotonic_ns()
        return ArmSdkState(
            side,
            self.positions[side],
            SdkPoseRpy((0.4, -0.2 if side is ArmSide.RIGHT else 0.2, 0.7), (0.1, -0.2, 0.3), "world"),
            source_sequence=len(self.operations),
            sample_monotonic_ns=now,
            received_monotonic_ns=now,
            joint_velocities_rad_s=None,
        )

    def fk(self, side: ArmSide, joints_rad: tuple[float, ...]):
        self.operations.append(("fk", side))
        return True, SdkPoseRpy((0.4, 0.0, 0.7), (0.0, 0.0, 0.0), "world"), None

    def ik(self, side: ArmSide, pose: SdkPoseRpy, seed_rad: tuple[float, ...]):
        self.operations.append(("ik", pose))
        return True, seed_rad, None

    def _submit(self, name: str, *, owner_id: str, command_id: str, side: ArmSide, **kwargs):
        now = self.clock.monotonic_ns()
        self.operations.append((name, {"side": side, **kwargs}))
        self.results[command_id] = _terminal(command_id, now=now)
        return ExecutionHandle(command_id, owner_id, ExecutionState.SUCCEEDED, now)

    def submit_move_j(self, **kwargs): return self._submit("move_j", **kwargs)
    def submit_move_jp(self, **kwargs): return self._submit("move_jp", **kwargs)
    def submit_move_l(self, **kwargs): return self._submit("move_l", **kwargs)
    def submit_follow_p(self, **kwargs): return self._submit("follow_p", **kwargs)

    def get_result(self, command_id: str) -> CommandResult:
        return self.results[command_id]

    def cancel(self, *, owner_id: str, command_id: str) -> CommandResult:
        now = self.clock.monotonic_ns()
        result = CommandResult(command_id, ExecutionState.CANCELLED, "cancelled", finished_monotonic_ns=now)
        self.results[command_id] = result
        return result

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        now = self.clock.monotonic_ns()
        self.operations.append(("emergency_stop", side))
        if side is self.fail_emergency_side:
            raise RuntimeError("injected arm stop failure")
        return _terminal(f"arm-estop-{side.value}", now=now, message=reason)

    def close(self) -> None:
        self.closed += 1


class FakeG20Gateway:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.open_calls: list[tuple[ArmSide, bool]] = []
        self.operations: list[tuple[str, object]] = []
        self.closed = 0
        self.logical = {ArmSide.LEFT: (255,) * 20, ArmSide.RIGHT: (255,) * 20}
        self.results: dict[str, CommandResult] = {}
        self.fail_open = False

    def open(self, side: ArmSide, *, read_only: bool) -> None:
        self.open_calls.append((side, read_only))
        if self.fail_open:
            raise RuntimeError("injected hand open failure")

    def read_state(self, side: ArmSide) -> G20SdkState:
        now = self.clock.monotonic_ns()
        return G20SdkState(side, self.logical[side], (31.0,) * 20, (0,) * 20, 1, now, now)

    def read_tactile(self, side: ArmSide) -> G20TactileSample:
        now = self.clock.monotonic_ns()
        matrix = tuple((0,) * 6 for _ in range(12))
        fingers = tuple(
            G20FingerSample(0.0, 0.0, float(index), matrix, 0.0, now, now, True, True)
            for index in range(5)
        )
        return G20TactileSample(side, fingers, f"{side.value}_g20_calibration", "g20_features_v1", now, 1)

    def submit_logical(self, *, owner_id: str, command_id: str, side: ArmSide, logical_raw: tuple[int, ...], timeout_s: float):
        now = self.clock.monotonic_ns()
        self.operations.append(("logical", (side, logical_raw, timeout_s)))
        self.logical[side] = logical_raw
        self.results[command_id] = _terminal(command_id, now=now)
        return ExecutionHandle(command_id, owner_id, ExecutionState.SUCCEEDED, now)

    def hold(self, *, owner_id: str, side: ArmSide) -> CommandResult:
        return _terminal(f"hold-{side.value}", now=self.clock.monotonic_ns())

    def stop(self, *, owner_id: str, side: ArmSide, reason: str) -> CommandResult:
        return _terminal(f"stop-{side.value}", now=self.clock.monotonic_ns(), message=reason)

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        self.operations.append(("emergency_stop", side))
        return _terminal(f"hand-estop-{side.value}", now=self.clock.monotonic_ns(), message=reason)

    def get_result(self, command_id: str) -> CommandResult:
        return self.results[command_id]

    def close(self) -> None:
        self.closed += 1


def _infra():
    clock = ManualClock()
    leases = ControlLeaseManager(clock)
    arm_gateway = FakeArmGateway(clock)
    left_gateway = FakeG20Gateway(clock)
    right_gateway = FakeG20Gateway(clock)
    arm = LinkBotArmAdapter(arm_gateway, leases, clock)
    left_hand = G20HandTactileAdapter(
        left_gateway, leases, MockHandSynergyMapper(), clock, side=ArmSide.LEFT
    )
    right_hand = G20HandTactileAdapter(
        right_gateway, leases, MockHandSynergyMapper(), clock, side=ArmSide.RIGHT
    )
    infra = BimanualRobotInfra(
        arm=arm,
        left_hand_tactile=left_hand,
        right_hand_tactile=right_hand,
        leases=leases,
        clock=clock,
    )
    return infra, arm_gateway, {
        ArmSide.LEFT: left_gateway,
        ArmSide.RIGHT: right_gateway,
    }


class DeploymentConfigTests(unittest.TestCase):
    def test_offline_template_has_four_slots_but_refuses_hardware_use(self) -> None:
        data = json.loads((ROOT / "config/robot_bimanual_offline.yaml").read_text(encoding="utf-8"))
        config = BimanualDeploymentConfig.from_mapping(data)
        self.assertFalse(config.hardware_ready)
        self.assertEqual(
            set(config.missing_assignments()),
            {
                "hands.right.validated",
                "topology_validated",
            },
        )
        self.assertEqual(config.arms[0].can_interface, "can0")
        self.assertEqual(config.arms[1].can_interface, "can1")
        self.assertEqual(config.hands[0].can_id, 0x28)
        self.assertEqual(config.hands[1].can_id, 0x27)
        with self.assertRaisesRegex(RuntimeError, "hardware deployment is incomplete"):
            config.require_hardware_ready()
        data["hands"]["right"]["validated"] = True
        data["topology_validated"] = True
        ready = BimanualDeploymentConfig.from_mapping(data)
        self.assertTrue(ready.hardware_ready)
        ready.require_hardware_ready()

    def test_sdk_manifest_points_only_to_present_canonical_sources(self) -> None:
        manifest = json.loads(
            (ROOT / "botSDK/INTEGRATION_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["device_shape"]["arms"], 2)
        self.assertEqual(manifest["device_shape"]["joints_per_arm"], 7)
        self.assertEqual(manifest["device_shape"]["hands"], 2)
        for relative in manifest["canonical"].values():
            self.assertTrue((ROOT / "botSDK" / relative).exists(), relative)
        for relative in manifest["evidence"].values():
            self.assertTrue((ROOT / "botSDK" / relative).is_file(), relative)
        self.assertFalse(
            (ROOT / "botSDK/lbot_arm_api-master/isaaclab_linkbot").exists()
        )
        for binding in manifest["bindings"].values():
            source = ROOT / "botSDK" / binding["file"]
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            classes = {
                node.name: node
                for node in tree.body
                if isinstance(node, ast.ClassDef)
            }
            self.assertIn(binding["class"], classes)
            methods = {
                node.name
                for node in classes[binding["class"]].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertEqual(set(binding["methods"]) - methods, set())


class AdapterTranslationTests(unittest.TestCase):
    def test_construction_is_side_effect_free_and_read_only_blocks_writes(self) -> None:
        infra, arm_gateway, hand_gateways = _infra()
        self.assertEqual(arm_gateway.open_calls, [])
        self.assertEqual(hand_gateways[ArmSide.LEFT].open_calls, [])
        self.assertEqual(hand_gateways[ArmSide.RIGHT].open_calls, [])
        self.assertIsInstance(infra.arm, ArmControlPort)
        self.assertIsInstance(infra.hand_for(ArmSide.LEFT), HandControlPort)
        self.assertIsInstance(infra.tactile_for(ArmSide.RIGHT), TactilePort)
        infra.open(read_only=True)
        snapshot = infra.snapshot()
        self.assertEqual(snapshot.left_arm.side, ArmSide.LEFT)
        self.assertEqual(snapshot.right_arm.side, ArmSide.RIGHT)
        self.assertEqual(len(snapshot.left_hand.logical_raw or ()), 20)
        self.assertEqual(len(snapshot.left_hand.temperatures_c), 20)
        self.assertEqual(len(snapshot.right_tactile.fingers), 5)
        with self.assertRaisesRegex(RuntimeError, "cannot grant control leases"):
            infra.acquire(owner_id="policy", side=ArmSide.RIGHT, ttl_s=1.0)
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            infra.arm.emergency_stop(ArmSide.RIGHT, "test")
        infra.close()
        infra.close()
        self.assertEqual(arm_gateway.closed, 1)
        self.assertEqual(hand_gateways[ArmSide.LEFT].closed, 1)
        self.assertEqual(hand_gateways[ArmSide.RIGHT].closed, 1)

    def test_pose_conversion_and_missing_velocity_estimation(self) -> None:
        infra, arm_gateway, _ = _infra()
        infra.open(read_only=True)
        first = infra.arm.get_state(ArmSide.RIGHT)
        infra.clock.advance_s(0.5)
        arm_gateway.positions[ArmSide.RIGHT] = (0.2,) * 7
        second = infra.arm.get_state(ArmSide.RIGHT)
        self.assertEqual(first.velocity_source, "estimated")
        self.assertTrue(all(abs(value - 0.2) < 1e-9 for value in second.joint_velocities_rad_s))
        rpy = (0.2, -0.3, 0.4)
        restored = quaternion_wxyz_to_rpy(rpy_to_quaternion_wxyz(rpy))
        for actual, expected in zip(restored, rpy):
            self.assertAlmostEqual(actual, expected, places=8)
        infra.close()

    def test_partial_open_failure_rolls_back_every_attempted_device(self) -> None:
        infra, arm_gateway, hand_gateways = _infra()
        hand_gateways[ArmSide.RIGHT].fail_open = True
        with self.assertRaisesRegex(RuntimeError, "injected hand open failure"):
            infra.open(read_only=True)
        self.assertEqual(arm_gateway.closed, 1)
        self.assertEqual(hand_gateways[ArmSide.LEFT].closed, 1)
        self.assertEqual(hand_gateways[ArmSide.RIGHT].closed, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            infra.open(read_only=True)

    def test_writable_commands_route_once_and_preserve_g20_reserved_channels(self) -> None:
        infra, arm_gateway, hand_gateways = _infra()
        infra.open(read_only=False)
        lease = infra.acquire(owner_id="pick-exec", side=ArmSide.RIGHT, ttl_s=2.0)
        move = JointMoveCommand("arm-1", ArmSide.RIGHT, (0.2,) * 7, 0.2, 0.2, 1.0)
        first = infra.arm.move_joint(lease, move)
        second = infra.arm.move_joint(lease, move)
        self.assertEqual(first, second)
        self.assertEqual([name for name, _ in arm_gateway.operations].count("move_j"), 1)
        pose = Pose((0.4, -0.2, 0.7), rpy_to_quaternion_wxyz((0.1, 0.2, -0.3)), "world")
        infra.arm.move_pose(lease, PoseMoveCommand("arm-2", ArmSide.RIGHT, pose, 0.1, 0.1, 1.0))
        self.assertIn("move_jp", [name for name, _ in arm_gateway.operations])
        hand_command = HandSynergyCommand("hand-1", ArmSide.RIGHT, HandSynergy(0.5, 0.2, -0.1), 1.0)
        infra.hand_for(ArmSide.RIGHT).command_synergy(lease, hand_command)
        logical = hand_gateways[ArmSide.RIGHT].logical[ArmSide.RIGHT]
        self.assertEqual(len(logical), 20)
        self.assertEqual(logical[11:15], (255, 255, 255, 255))
        with self.assertRaisesRegex(ValueError, "reused"):
            infra.arm.move_joint(lease, JointMoveCommand("arm-1", ArmSide.RIGHT, (0.3,) * 7, 0.2, 0.2, 1.0))
        infra.close()

    def test_whole_system_stop_covers_two_arms_and_two_hands(self) -> None:
        infra, arm_gateway, hand_gateways = _infra()
        infra.open(read_only=False)
        left = infra.acquire(owner_id="left-owner", side=ArmSide.LEFT, ttl_s=2.0)
        right = infra.acquire(owner_id="right-owner", side=ArmSide.RIGHT, ttl_s=2.0)
        report = infra.emergency_stop_all("operator")
        self.assertEqual(len(report.arm_results), 2)
        self.assertEqual(len(report.hand_results), 2)
        self.assertIsNone(infra.leases.active(ArmSide.LEFT))
        self.assertIsNone(infra.leases.active(ArmSide.RIGHT))
        self.assertNotEqual(left.lease_id, right.lease_id)
        self.assertEqual([name for name, _ in arm_gateway.operations].count("emergency_stop"), 2)
        self.assertEqual(
            [name for gateway in hand_gateways.values() for name, _ in gateway.operations].count("emergency_stop"),
            2,
        )
        infra.close()

    def test_whole_system_stop_continues_after_one_channel_fails(self) -> None:
        infra, arm_gateway, hand_gateways = _infra()
        infra.open(read_only=False)
        infra.acquire(owner_id="left-owner", side=ArmSide.LEFT, ttl_s=2.0)
        infra.acquire(owner_id="right-owner", side=ArmSide.RIGHT, ttl_s=2.0)
        arm_gateway.fail_emergency_side = ArmSide.LEFT
        with self.assertRaisesRegex(BaseExceptionGroup, "emergency stop incomplete"):
            infra.emergency_stop_all("operator")
        self.assertEqual([name for name, _ in arm_gateway.operations].count("emergency_stop"), 2)
        self.assertEqual(
            [name for gateway in hand_gateways.values() for name, _ in gateway.operations].count("emergency_stop"),
            2,
        )
        self.assertIsNone(infra.leases.active(ArmSide.LEFT))
        self.assertIsNone(infra.leases.active(ArmSide.RIGHT))
        infra.close()


if __name__ == "__main__":
    unittest.main()
