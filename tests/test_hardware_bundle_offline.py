from __future__ import annotations

import json
from pathlib import Path
import unittest

from module.robotics import (
    HandJointPositionCommand,
    LinearMoveCommand,
    LinkBotG20HardwareBundle,
    build_robotics_bundle,
)
from module.robotics.adapters import BotSdkG20Gateway, BotSdkLinkBotGateway
from module.robotics.clock import ManualClock
from module.robotics.g20_calibrated import CalibratedG20Mapper
from proto import ArmSide, ExecutionState, HandSynergy, Pose


ROOT = Path(__file__).resolve().parents[1]


class _ArmRuntime:
    def __init__(self, *_args, **_kwargs) -> None:
        self.enable_calls: list[tuple[bool, str | None]] = []
        self.closed = False

    def start(self) -> None:
        pass

    def set_enabled(self, enabled: bool, confirmation: str | None) -> None:
        self.enable_calls.append((enabled, confirmation))

    def close(self) -> None:
        self.closed = True


class _ArmController:
    def __init__(self, _runtime, *, config_path) -> None:
        self.config_path = config_path
        self.enabled_calls: list[bool] = []
        self.linear_calls: list[tuple[object, ...]] = []

    def set_enabled_state(self, enabled: bool) -> None:
        self.enabled_calls.append(enabled)

    def movel(self, pose, speed, acceleration, block, confirmation) -> bool:
        self.linear_calls.append(
            (pose, speed, acceleration, block, confirmation)
        )
        return True


class _HandDriver:
    def __init__(self) -> None:
        self.closed = False

    def close_can_interface(self) -> None:
        self.closed = True


class _HandApi:
    def __init__(self, **_kwargs) -> None:
        self.hand = _HandDriver()
        self.speed_calls: list[list[int]] = []
        self.torque_calls: list[list[int]] = []
        self.position_calls: list[list[int]] = []

    def set_speed(self, value: list[int]) -> None:
        self.speed_calls.append(value)

    def set_torque(self, value: list[int]) -> None:
        self.torque_calls.append(value)

    def finger_move(self, value: list[int]) -> None:
        self.position_calls.append(value)


class HardwareBundleOfflineTests(unittest.TestCase):
    def test_registered_pending_profile_constructs_without_opening_can(self) -> None:
        profile = json.loads(
            (ROOT / "config/robotics/linkbot_g20_hardware_pending.json").read_text(
                encoding="utf-8"
            )
        )
        bundle = build_robotics_bundle(profile["robotics"], side=ArmSide.LEFT)
        self.assertIsInstance(bundle, LinkBotG20HardwareBundle)
        self.assertEqual("configured", bundle.lifecycle_state)
        self.assertEqual({}, bundle.arm_gateway._runtimes)
        self.assertIsNone(bundle.hand_gateway._api)
        bundle.close()

    def test_open_enable_and_close_follow_explicit_sdk_lifecycle(self) -> None:
        clock = ManualClock()
        arm_gateway = BotSdkLinkBotGateway(
            endpoints={
                ArmSide.RIGHT: {
                    "can_interface": "can1",
                    "profile_path": "right.json",
                    "physical_confirmation": "RIGHT_ARM_AREA_CLEAR",
                }
            },
            clock=clock,
            hardware_write_authorized=True,
            runtime_factory=_ArmRuntime,
            controller_factory=_ArmController,
        )
        hand_gateway = BotSdkG20Gateway(
            side=ArmSide.RIGHT,
            can_interface="can1",
            clock=clock,
            hardware_write_authorized=True,
            api_factory=_HandApi,
        )
        mapper = CalibratedG20Mapper.from_file(
            ROOT / "config/robotics/g20_motion_calibration_20260726.json"
        )
        bundle = LinkBotG20HardwareBundle(
            side=ArmSide.RIGHT,
            arm_gateway=arm_gateway,
            hand_gateway=hand_gateway,
            mapper=mapper,
            clock=clock,
            physical_confirmation="RIGHT_ARM_AREA_CLEAR",
        )
        bundle.open(enable_arm=True)
        runtime = arm_gateway._runtimes[ArmSide.RIGHT]
        controller = arm_gateway._controllers[ArmSide.RIGHT]
        self.assertEqual([(True, "RIGHT_ARM_AREA_CLEAR")], runtime.enable_calls)
        self.assertEqual([True], controller.enabled_calls)
        bundle.close()
        self.assertEqual(
            [(True, "RIGHT_ARM_AREA_CLEAR"), (False, None)],
            runtime.enable_calls,
        )
        self.assertEqual([True, False], controller.enabled_calls)
        self.assertTrue(runtime.closed)
        self.assertTrue(hand_gateway._api.hand.closed)

    def test_ports_translate_motion_and_calibrated_hand_command_end_to_end(self) -> None:
        clock = ManualClock()
        arm_gateway = BotSdkLinkBotGateway(
            endpoints={
                ArmSide.LEFT: {
                    "can_interface": "can0",
                    "profile_path": "left.json",
                    "physical_confirmation": "LEFT_ARM_AREA_CLEAR",
                    "allow_can0": True,
                }
            },
            clock=clock,
            hardware_write_authorized=True,
            runtime_factory=_ArmRuntime,
            controller_factory=_ArmController,
        )
        hand_gateway = BotSdkG20Gateway(
            side=ArmSide.LEFT,
            can_interface="can0",
            clock=clock,
            hardware_write_authorized=True,
            api_factory=_HandApi,
        )
        mapper = CalibratedG20Mapper.from_file(
            ROOT / "config/robotics/g20_motion_calibration_20260726.json"
        )
        bundle = LinkBotG20HardwareBundle(
            side=ArmSide.LEFT,
            arm_gateway=arm_gateway,
            hand_gateway=hand_gateway,
            mapper=mapper,
            clock=clock,
            physical_confirmation="LEFT_ARM_AREA_CLEAR",
        )
        bundle.open(enable_arm=True)
        lease = bundle.leases.acquire(owner_id="offline-route", side=ArmSide.LEFT, ttl_s=5.0)
        try:
            arm_handle = bundle.arm.move_linear(
                lease,
                LinearMoveCommand(
                    command_id="linear",
                    side=ArmSide.LEFT,
                    target_pose=Pose(
                        (0.4, 0.2, 0.7),
                        (1.0, 0.0, 0.0, 0.0),
                        "world",
                    ),
                    speed=0.05,
                    acceleration=0.10,
                    timeout_s=4.0,
                ),
            )
            open_joints = mapper.to_sim_independent(
                ArmSide.LEFT, HandSynergy(0.0, 0.0, 0.0)
            )
            hand_handle = bundle.hand.command_joint_positions(
                lease,
                HandJointPositionCommand(
                    command_id="hand-open",
                    side=ArmSide.LEFT,
                    joint_positions_rad=open_joints,
                    timeout_s=2.0,
                ),
            )
        finally:
            bundle.leases.release(lease)
        self.assertIs(arm_handle.state, ExecutionState.SUCCEEDED)
        self.assertIs(hand_handle.state, ExecutionState.SUCCEEDED)
        controller = arm_gateway._controllers[ArmSide.LEFT]
        self.assertEqual("LEFT_ARM_AREA_CLEAR", controller.linear_calls[0][-1])
        logical = hand_gateway._api.position_calls[0]
        self.assertEqual(20, len(logical))
        self.assertEqual([255, 255, 255, 255], logical[11:15])
        bundle.close()


if __name__ == "__main__":
    unittest.main()
