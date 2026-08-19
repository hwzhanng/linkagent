from __future__ import annotations

import unittest

from module.robotics import ManualClock
from module.robotics.adapters import (
    ArmSdkGateway,
    BotSdkG20Gateway,
    BotSdkLinkBotGateway,
    G20SdkGateway,
    SdkPoseRpy,
)
from proto import ArmSide, ExecutionState


class _ArmRuntime:
    instances: list["_ArmRuntime"] = []

    def __init__(self, interface, *, arm, allow_tx, allow_can0, poll_telemetry):
        self.interface = interface
        self.arm = arm
        self.allow_tx = allow_tx
        self.allow_can0 = allow_can0
        self.poll_telemetry = poll_telemetry
        self.started = False
        self.enabled_calls: list[tuple[bool, str | None]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def set_enabled(self, enabled, confirmation):
        self.enabled_calls.append((enabled, confirmation))

    def close(self):
        self.closed = True


class _ArmController:
    def __init__(self, runtime, *, config_path):
        self.runtime = runtime
        self.config_path = config_path
        self.enabled_states: list[bool] = []
        self.motion_calls: list[tuple[object, ...]] = []

    def set_enabled_state(self, enabled):
        self.enabled_states.append(enabled)

    def movej(self, joints, speed, acceleration, block, confirmation):
        self.motion_calls.append(
            ("movej", tuple(joints), speed, acceleration, block, confirmation)
        )
        return True

    def movejp(self, pose, speed, acceleration, block, confirmation):
        self.motion_calls.append(
            ("movejp", pose, speed, acceleration, block, confirmation)
        )
        return True

    def movel(self, pose, speed, acceleration, block, confirmation):
        self.motion_calls.append(
            ("movel", pose, speed, acceleration, block, confirmation)
        )
        return True

    def followp(self, pose, confirmation):
        self.motion_calls.append(("followp", pose, confirmation))
        return True


class _HandDriver:
    def __init__(self):
        self.closed = False

    def close_can_interface(self):
        self.closed = True


class _HandApi:
    instances: list["_HandApi"] = []

    def __init__(self, *, hand_type, hand_joint, can):
        self.hand_type = hand_type
        self.hand_joint = hand_joint
        self.can = can
        self.speed_calls: list[list[int]] = []
        self.torque_calls: list[list[int]] = []
        self.position_calls: list[list[int]] = []
        self.hand = _HandDriver()
        self.__class__.instances.append(self)

    def set_speed(self, values):
        self.speed_calls.append(list(values))

    def set_torque(self, values):
        self.torque_calls.append(list(values))

    def finger_move(self, values):
        self.position_calls.append(list(values))


class VendorGatewayOfflineTests(unittest.TestCase):
    def setUp(self) -> None:
        _ArmRuntime.instances.clear()
        _HandApi.instances.clear()

    def test_linkbot_construction_is_side_effect_free_and_write_fails_closed(self) -> None:
        calls: list[str] = []

        def runtime_factory(*args, **kwargs):
            calls.append("runtime")
            raise AssertionError((args, kwargs))

        gateway = BotSdkLinkBotGateway(
            endpoints={
                ArmSide.LEFT: {"can_interface": "offline-left"},
                ArmSide.RIGHT: {"can_interface": "offline-right"},
            },
            clock=ManualClock(),
            runtime_factory=runtime_factory,
            controller_factory=lambda *args, **kwargs: None,
        )
        self.assertIsInstance(gateway, ArmSdkGateway)
        self.assertEqual(calls, [])
        with self.assertRaises(PermissionError):
            gateway.open(read_only=False)
        self.assertEqual(calls, [])

    def test_g20_construction_is_side_effect_free_and_write_fails_closed(self) -> None:
        calls: list[str] = []

        def api_factory(*args, **kwargs):
            calls.append("api")
            raise AssertionError((args, kwargs))

        gateway = BotSdkG20Gateway(
            side=ArmSide.RIGHT,
            can_interface="offline-right",
            clock=ManualClock(),
            api_factory=api_factory,
        )
        self.assertIsInstance(gateway, G20SdkGateway)
        self.assertEqual(calls, [])
        with self.assertRaises(PermissionError):
            gateway.open(ArmSide.RIGHT, read_only=False)
        self.assertEqual(calls, [])

    def test_single_arm_write_mode_uses_validated_runtime_and_explicit_enable(self) -> None:
        gateway = BotSdkLinkBotGateway(
            endpoints={
                ArmSide.RIGHT: {
                    "can_interface": "offline-right",
                    "profile_path": "right.json",
                    "physical_confirmation": "RIGHT_ARM_AREA_CLEAR",
                }
            },
            clock=ManualClock(),
            hardware_write_authorized=True,
            runtime_factory=_ArmRuntime,
            controller_factory=_ArmController,
        )

        gateway.open(read_only=False)
        gateway.set_enabled(
            ArmSide.RIGHT,
            True,
            "RIGHT_ARM_AREA_CLEAR",
        )

        self.assertEqual(len(_ArmRuntime.instances), 1)
        runtime = _ArmRuntime.instances[0]
        self.assertTrue(runtime.started)
        self.assertTrue(runtime.allow_tx)
        self.assertTrue(runtime.poll_telemetry)
        self.assertEqual(runtime.enabled_calls, [(True, "RIGHT_ARM_AREA_CLEAR")])
        self.assertEqual(
            gateway._controllers[ArmSide.RIGHT].enabled_states,  # type: ignore[attr-defined]
            [True],
        )
        with self.assertRaisesRegex(ValueError, "left arm is not configured"):
            gateway.read_state(ArmSide.LEFT)
        gateway.close()
        self.assertTrue(runtime.closed)

    def test_g20_write_open_applies_connected_hand_motion_settings(self) -> None:
        gateway = BotSdkG20Gateway(
            side=ArmSide.RIGHT,
            can_interface="offline-right",
            clock=ManualClock(),
            hardware_write_authorized=True,
            api_factory=_HandApi,
        )

        gateway.open(ArmSide.RIGHT, read_only=False)

        api = _HandApi.instances[0]
        self.assertEqual(api.hand_type, "right")
        self.assertEqual(api.hand_joint, "G20")
        self.assertEqual(api.speed_calls, [[204] * 5])
        self.assertEqual(api.torque_calls, [[153] * 5])
        gateway.close()
        self.assertTrue(api.hand.closed)

    def test_arm_commands_translate_to_the_four_validated_controller_calls(self) -> None:
        gateway = BotSdkLinkBotGateway(
            endpoints={
                ArmSide.RIGHT: {
                    "can_interface": "offline-right",
                    "profile_path": "right.json",
                    "physical_confirmation": "RIGHT_ARM_AREA_CLEAR",
                }
            },
            clock=ManualClock(),
            hardware_write_authorized=True,
            runtime_factory=_ArmRuntime,
            controller_factory=_ArmController,
        )
        gateway.open(read_only=False)
        pose = SdkPoseRpy((0.4, -0.2, 0.7), (0.1, 0.2, 0.3), "world")
        handles = (
            gateway.submit_move_j(
                owner_id="owner",
                command_id="movej",
                side=ArmSide.RIGHT,
                joints_rad=(0.0,) * 7,
                speed=0.2,
                acceleration=0.3,
                timeout_s=4.0,
            ),
            gateway.submit_move_jp(
                owner_id="owner",
                command_id="movejp",
                side=ArmSide.RIGHT,
                pose=pose,
                speed=0.2,
                acceleration=0.3,
                timeout_s=4.0,
            ),
            gateway.submit_move_l(
                owner_id="owner",
                command_id="movel",
                side=ArmSide.RIGHT,
                pose=pose,
                speed=0.1,
                acceleration=0.2,
                timeout_s=4.0,
            ),
            gateway.submit_follow_p(
                owner_id="owner",
                command_id="followp",
                side=ArmSide.RIGHT,
                pose=pose,
                timeout_s=4.0,
            ),
        )
        self.assertTrue(all(item.state is ExecutionState.SUCCEEDED for item in handles))
        calls = gateway._controllers[ArmSide.RIGHT].motion_calls
        self.assertEqual(["movej", "movejp", "movel", "followp"], [row[0] for row in calls])
        self.assertEqual("RIGHT_ARM_AREA_CLEAR", calls[0][-1])
        self.assertTrue(calls[0][-2])
        self.assertEqual("RIGHT_ARM_AREA_CLEAR", calls[1][-1])
        self.assertTrue(calls[1][-2])
        self.assertEqual("RIGHT_ARM_AREA_CLEAR", calls[2][-1])
        self.assertTrue(calls[2][-2])
        self.assertEqual("RIGHT_ARM_AREA_CLEAR", calls[3][-1])
        gateway.close()

    def test_g20_command_uses_the_sdk_twenty_value_finger_move(self) -> None:
        gateway = BotSdkG20Gateway(
            side=ArmSide.LEFT,
            can_interface="offline-left",
            clock=ManualClock(),
            hardware_write_authorized=True,
            api_factory=_HandApi,
        )
        gateway.open(ArmSide.LEFT, read_only=False)
        logical = tuple(range(20))
        handle = gateway.submit_logical(
            owner_id="owner",
            command_id="hand",
            side=ArmSide.LEFT,
            logical_raw=logical,
            timeout_s=2.0,
        )
        self.assertIs(handle.state, ExecutionState.SUCCEEDED)
        self.assertEqual([list(logical)], _HandApi.instances[0].position_calls)
        gateway.close()


if __name__ == "__main__":
    unittest.main()
