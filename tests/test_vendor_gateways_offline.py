from __future__ import annotations

import unittest

from module.robotics import ManualClock
from module.robotics.adapters import (
    ArmSdkGateway,
    BotSdkG20Gateway,
    BotSdkLinkBotGateway,
    G20SdkGateway,
)
from proto import ArmSide


class VendorGatewayOfflineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
