"""Agent→Skill→Port→SDK-adapter offline closed-loop gates."""

from __future__ import annotations

from pathlib import Path
import unittest

from module.robotics import (
    ControlBusyError,
    ControlLeaseManager,
    FakeArmSdkGateway,
    FakeG20SdkGateway,
    FakeGatewayFaults,
    InvalidControlLeaseError,
    JointMoveCommand,
    LinkBotG20ShadowBundle,
    ManualClock,
    MockHandSynergyMapper,
    MockPickScenario,
    MockPickWorld,
    build_fake_gateway_robotics_bundle,
)
from proto import ArmSide, ExecutionState, Observation
from script.run import build_runtime


def _observation() -> Observation:
    return Observation(
        sequence=0,
        metadata={
            "tomatoes": [{
                "target_id": "tomato-1",
                "center_pose": {
                    "position_m": [0.45, -0.25, 0.72],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "frame_id": "world",
                },
                "semi_axes_m": [0.035, 0.032, 0.038],
                "ripeness_score": 0.92,
                "pick_confidence": 0.96,
                "occlusion": 0.08,
                "stem_direction_xyz": [0.0, 0.0, -1.0],
            }]
        },
    )


class FakeGatewayChainTests(unittest.TestCase):
    def test_full_agent_workflow_uses_fake_sdk_gateway_backend(self) -> None:
        root = {
            "models_file": "_model.yaml",
            "perception_file": "_perception.yaml",
            "agent": {"model": "offline_agri_pick", "max_iterations": 8, "max_tool_calls": 5, "max_tool_calls_per_turn": 1},
            "env": {"type": "mock", "observations": [_observation()]},
            "task": {
                "type": "agri_pick",
                "default_arm_side": "right",
                "max_skill_duration_s": 4.0,
                "robotics": {"type": "fake_gateway"},
                "skill": {"type": "mock"},
                "perception": {"type": "mock"},
                "planner": {"type": "mock"},
                "safety": {"type": "mock"},
                "verifier": {"type": "mock"},
            },
            "runtime": {"max_steps": 1},
        }
        report = build_runtime(
            root, config_dir=Path(__file__).resolve().parents[1] / "config"
        ).run(seed=7)
        self.assertTrue(report.completed)

    def test_lease_expiry_cancel_idempotency_and_timeout_fault(self) -> None:
        bundle = build_fake_gateway_robotics_bundle(side=ArmSide.RIGHT)
        try:
            lease = bundle.leases.acquire(owner_id="owner", side=ArmSide.RIGHT, ttl_s=1.0)
            command = JointMoveCommand("same-command", ArmSide.RIGHT, (0.1,) * 7, 0.2, 0.2, 0.5)
            first = bundle.arm.move_joint(lease, command)
            second = bundle.arm.move_joint(lease, command)
            self.assertEqual(first, second)
            self.assertEqual(bundle.arm_gateway.operations.count("move_j"), 1)
            cancelled = bundle.arm.cancel(lease, command.command_id)
            self.assertIs(cancelled.state, ExecutionState.CANCELLED)
            bundle.clock.advance_s(2.0)
            with self.assertRaises(InvalidControlLeaseError):
                bundle.arm.move_joint(lease, JointMoveCommand("expired", ArmSide.RIGHT, (0.2,) * 7, 0.2, 0.2, 0.5))
        finally:
            bundle.close()

        timed = build_fake_gateway_robotics_bundle(
            side=ArmSide.RIGHT,
            gateway_faults=FakeGatewayFaults(command_timeout=True),
        )
        try:
            lease = timed.leases.acquire(owner_id="owner", side=ArmSide.RIGHT, ttl_s=2.0)
            handle = timed.arm.move_joint(
                lease,
                JointMoveCommand("timeout", ArmSide.RIGHT, (0.1,) * 7, 0.2, 0.2, 0.25),
            )
            self.assertIs(handle.state, ExecutionState.FAILED)
            self.assertEqual(timed.arm.get_result("timeout").error_code, "timeout")
        finally:
            timed.close()

    def test_shadow_lifecycle_is_read_only(self) -> None:
        clock = ManualClock()
        mapper = MockHandSynergyMapper()
        world = MockPickWorld(
            side=ArmSide.RIGHT,
            scenario=MockPickScenario(),
            clock=clock,
            mapper=mapper,
        )
        shadow = LinkBotG20ShadowBundle(
            side=ArmSide.RIGHT,
            arm_gateway=FakeArmSdkGateway(world, FakeGatewayFaults()),
            hand_gateway=FakeG20SdkGateway(world, mapper, FakeGatewayFaults()),
            clock=clock,
        )
        self.assertEqual(shadow.lifecycle_state, "configured")
        shadow.open()
        self.assertEqual(shadow.lifecycle_state, "open_read_only")
        self.assertEqual(shadow.arm.get_state(ArmSide.RIGHT).side, ArmSide.RIGHT)
        with self.assertRaises(ControlBusyError):
            shadow.leases.acquire(owner_id="policy", side=ArmSide.RIGHT, ttl_s=1.0)
        shadow.close()
        self.assertEqual(shadow.lifecycle_state, "closed")


if __name__ == "__main__":
    unittest.main()
