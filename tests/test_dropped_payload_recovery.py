"""Contract tests for dropped-fruit localization and deterministic routing."""

from __future__ import annotations

import unittest

from module.robotics import ManualClock
from module.skills.core import AtomicSkillFailureCode
from module.skills.harvest_cycle import (
    LocalizeDroppedPayloadSkill,
    MockDroppedPayloadObservationPort,
    build_dropped_payload_observation_port,
)
from proto import (
    ArmSide,
    DroppedPayloadLocalizationRequest,
    HarvestContainer,
    HarvestPayload,
    Pose,
)


def _payload() -> HarvestPayload:
    return HarvestPayload(
        payload_id="payload-drop-test",
        target_id="tomato-drop-test",
        pick_execution_id="pick-drop-test",
        arm_side=ArmSide.RIGHT,
        detached=True,
        retained=True,
        acquired_monotonic_ns=900_000_000,
    )


def _container() -> HarvestContainer:
    return HarvestContainer(
        container_id="bin-drop-test",
        placement_pose=Pose(
            (0.30, -0.25, 0.25), (1.0, 0.0, 0.0, 0.0), "world"
        ),
        available=True,
        capacity_remaining=5,
    )


def _request() -> DroppedPayloadLocalizationRequest:
    return DroppedPayloadLocalizationRequest(
        execution_id="localize-drop-test",
        payload=_payload(),
        container=_container(),
        max_observation_age_s=1.0,
    )


def _skill(clock: ManualClock, port: MockDroppedPayloadObservationPort):
    return LocalizeDroppedPayloadSkill(
        observation_port=port,
        clock=clock,
        recovery_workspace_xyzxyz_m=(0.0, -0.50, 0.0, 0.80, 0.20, 0.40),
        minimum_confidence=0.60,
    )


class DroppedPayloadRecoveryTests(unittest.TestCase):
    def test_container_drop_routes_to_placement_verification_without_motion(self) -> None:
        clock = ManualClock()
        skill = _skill(
            clock,
            MockDroppedPayloadObservationPort(
                clock=clock,
                center_position_world_m=(0.30, -0.25, 0.20),
                inside_container=True,
            ),
        )
        result = skill.run(_request())
        self.assertTrue(result.success)
        self.assertEqual(result.output["disposition"], "in_container")
        self.assertEqual(result.output["next_action"], "verify_recovered_placement")
        self.assertFalse(result.output["motion_authorized"])
        self.assertFalse(result.output["frozen_hanging_pick_ppo_reuse_allowed"])

    def test_known_recovery_workspace_requires_a_dedicated_policy(self) -> None:
        clock = ManualClock()
        skill = _skill(
            clock,
            MockDroppedPayloadObservationPort(
                clock=clock,
                center_position_world_m=(0.45, -0.20, 0.08),
            ),
        )
        result = skill.run(_request())
        self.assertTrue(result.success)
        self.assertEqual(
            result.output["disposition"], "dedicated_recovery_required"
        )
        self.assertTrue(result.output["inside_recovery_workspace"])
        self.assertFalse(result.output["motion_authorized"])

    def test_outside_workspace_routes_to_manual_intervention(self) -> None:
        clock = ManualClock()
        skill = _skill(
            clock,
            MockDroppedPayloadObservationPort(
                clock=clock,
                center_position_world_m=(1.40, -0.20, 0.08),
            ),
        )
        result = skill.run(_request())
        self.assertTrue(result.success)
        self.assertEqual(
            result.output["disposition"], "manual_intervention_required"
        )
        self.assertEqual(result.output["next_action"], "manual_intervention")

    def test_missing_payload_remains_a_recoverable_perception_failure(self) -> None:
        clock = ManualClock()
        skill = _skill(
            clock,
            MockDroppedPayloadObservationPort(clock=clock, observed=False),
        )
        result = skill.run(_request())
        self.assertFalse(result.success)
        self.assertIs(
            result.failure_code, AtomicSkillFailureCode.PERCEPTION_UNAVAILABLE
        )
        self.assertFalse(result.output["motion_authorized"])

    def test_stale_localization_does_not_authorize_recovery(self) -> None:
        clock = ManualClock(start_ns=5_000_000_000)
        skill = _skill(
            clock,
            MockDroppedPayloadObservationPort(clock=clock, sample_age_s=2.0),
        )
        result = skill.run(_request())
        self.assertFalse(result.success)
        self.assertIs(result.failure_code, AtomicSkillFailureCode.TARGET_STALE)
        self.assertFalse(result.output["motion_authorized"])

    def test_mock_port_is_explicitly_buildable(self) -> None:
        clock = ManualClock()
        port = build_dropped_payload_observation_port(
            {
                "type": "mock",
                "center_position_world_m": [0.4, -0.2, 0.1],
            },
            clock=clock,
        )
        self.assertIsInstance(port, MockDroppedPayloadObservationPort)
        port.close()


if __name__ == "__main__":
    unittest.main()
