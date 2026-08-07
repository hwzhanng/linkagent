from __future__ import annotations

import struct
import unittest

from tools.local_arm_can.protocol import (
    ArmState,
    CanFrame,
    HandshakeReply,
    JointTelemetry,
    ProtocolError,
    RIGHT_BASE_ID,
    TemperatureTelemetry,
    control_mode_frame,
    decode_frame,
    disable_frame,
    enable_frame,
    handshake_frame,
    parse_candump_line,
    position_frame,
    state_query_frame,
)


class ProtocolTests(unittest.TestCase):
    def test_parse_real_right_arm_candump_line(self) -> None:
        frame = parse_candump_line("(1783109483.790959) can0 033#9680307FA7E80000 R")
        self.assertEqual(frame.can_id, 0x33)
        self.assertEqual(frame.data, bytes.fromhex("9680307FA7E80000"))
        self.assertEqual(frame.direction, "R")

    def test_decode_classic_joint_state(self) -> None:
        frame = CanFrame(0x33, bytes.fromhex("9680307FA7E80000"), timestamp=1.0)
        event = decode_frame(frame)
        self.assertIsInstance(event, JointTelemetry)
        assert isinstance(event, JointTelemetry)
        self.assertEqual(event.joint_index, 0)
        self.assertAlmostEqual(event.position, 0.0148012512, places=5)
        self.assertAlmostEqual(event.velocity, -0.0268620269, places=5)
        self.assertAlmostEqual(event.effort, -0.9181929182, places=5)
        self.assertEqual(event.error, 0)

    def test_decode_real_temperature_response(self) -> None:
        frame = CanFrame(0x35, bytes.fromhex("5F34050C42"), timestamp=1.0)
        event = decode_frame(frame)
        self.assertIsInstance(event, TemperatureTelemetry)
        assert isinstance(event, TemperatureTelemetry)
        self.assertEqual(event.joint_index, 2)
        self.assertAlmostEqual(event.temperature, struct.unpack("<f", bytes.fromhex("34050C42"))[0])

    def test_decode_motorevo_state(self) -> None:
        frame = CanFrame(0x38, bytes.fromhex("01811C7FF7FE0023"), timestamp=1.0)
        event = decode_frame(frame)
        self.assertIsInstance(event, JointTelemetry)
        assert isinstance(event, JointTelemetry)
        self.assertEqual(event.joint_index, 5)
        self.assertEqual(event.motor_type, "motorevo")
        self.assertEqual(event.error, 0x23)

    def test_motorevo_query_is_not_misdecoded_as_state(self) -> None:
        self.assertIsNone(decode_frame(CanFrame(0x38, bytes.fromhex("FFFFFFFFFFFFFFFA"))))

    def test_recovered_special_frames(self) -> None:
        self.assertEqual(enable_frame(0x33).data, bytes.fromhex("2A01"))
        self.assertEqual(disable_frame(0x33).data, bytes.fromhex("2A00"))
        self.assertEqual(enable_frame(0x38).data, bytes.fromhex("FFFFFFFFFFFFFFFC"))
        self.assertEqual(disable_frame(0x38).data, bytes.fromhex("FFFFFFFFFFFFFFFD"))
        self.assertEqual(state_query_frame(0x38).data, bytes.fromhex("FFFFFFFFFFFFFFFA"))
        self.assertEqual(handshake_frame(0x38).can_id, 0x638)
        self.assertEqual(handshake_frame(0x38).data, bytes.fromhex("670A000000000476"))
        self.assertEqual(control_mode_frame(0x33).data, bytes.fromhex("0704"))
        self.assertEqual(control_mode_frame(0x38).data, bytes.fromhex("670B010000001576"))

    def test_classic_position_is_little_endian_float(self) -> None:
        self.assertEqual(position_frame(0x33, 1.25).data, b"\x0a" + struct.pack("<f", 1.25))

    def test_motorevo_position_requires_explicit_parameters(self) -> None:
        with self.assertRaises(ProtocolError):
            position_frame(0x38, 0.0)
        frame = position_frame(0x38, 0.0, evo_parameters=[0, 0, 0, 0, 0, 0])
        self.assertEqual(frame.dlc, 8)
        self.assertEqual(frame.data[:2], bytes.fromhex("7FFF"))

    def test_arm_health_tracks_missing_right_joints(self) -> None:
        state = ArmState(RIGHT_BASE_ID, stale_after=0.25)
        now = 10.0
        for can_id, payload in (
            (0x33, "9680307FA7E80000"),
            (0x34, "967FF18047E20000"),
            (0x35, "967FFE8038070000"),
            (0x36, "967FD77FD7F80000"),
        ):
            event = decode_frame(CanFrame(can_id, bytes.fromhex(payload), timestamp=1.0))
            assert event is not None
            state.update(event, now_monotonic=now)
        snapshot = state.snapshot(now_monotonic=now + 0.1)
        self.assertEqual(snapshot["healthy_joint_count"], 4)
        self.assertFalse(snapshot["joints"][4]["healthy"])
        self.assertFalse(snapshot["joints"][5]["healthy"])
        self.assertFalse(snapshot["joints"][6]["healthy"])


if __name__ == "__main__":
    unittest.main()
