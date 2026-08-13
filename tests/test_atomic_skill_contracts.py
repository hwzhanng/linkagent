"""Contract tests for generic atomic-skill lifecycle and bounded traces."""

from __future__ import annotations

import unittest

from module.robotics import ManualClock
from module.skills import (
    ATOMIC_SKILL_REGISTRY,
    AtomicSkillFailureCode,
    AtomicSkillResult,
    AtomicSkillStatus,
    build_atomic_skill,
)
from module.skills.harvest_cycle import MockTargetObservationPort
from proto import Observation, TargetReobserveRequest


class AtomicSkillContractTests(unittest.TestCase):
    def test_registry_is_explicit_and_descriptors_are_not_hardware_approved(self) -> None:
        self.assertEqual(
            {
                "system_self_check",
                "active_view_reposition",
                "select_harvest_target",
                "move_to_pick_ready",
                "localize_dropped_payload",
                "pregrasp_clearance_check",
                "reset_for_next_target",
                "target_reobserve",
                "transport_payload",
                "safe_retract",
                "payload_retreat",
                "payload_place",
            },
            set(ATOMIC_SKILL_REGISTRY),
        )
        skill = build_atomic_skill(
            {"type": "target_reobserve"},
            observation_port=MockTargetObservationPort(clock=ManualClock()),
            clock=ManualClock(),
        )
        try:
            self.assertEqual(skill.descriptor.skill_id, "target_reobserve")
            self.assertFalse(skill.descriptor.hardware_approved)
            self.assertFalse(skill.descriptor.effectful)
        finally:
            skill.close()

    def test_trace_summarizes_binary_sensor_payload_and_is_idempotent(self) -> None:
        skill = build_atomic_skill(
            {"type": "target_reobserve"},
            observation_port=MockTargetObservationPort(clock=ManualClock()),
            clock=ManualClock(),
        )
        request = TargetReobserveRequest(
            execution_id="reobserve-1",
            target_id="tomato-1",
            observation=Observation(
                sequence=7,
                rgb=b"opaque-rgb-payload",
                depth_m=bytearray(b"opaque-depth-payload"),
                metadata={
                    "tomatoes": [
                        {
                            "target_id": "tomato-1",
                            "occlusion": 0.12,
                        }
                    ]
                },
            ),
        )
        try:
            first = skill.run(request)
            second = skill.run(request)
            self.assertIs(first, second)
            self.assertTrue(first.success)
            self.assertEqual([0, 1], [event.sequence for event in first.trace])
            started = first.trace[0].input_summary
            self.assertEqual(dict(started["rgb"]), {"type": "binary", "size": 18})
            self.assertEqual(dict(started["depth_m"]), {"type": "binary", "size": 20})
            self.assertNotIn(b"opaque", repr(first.as_dict()).encode())
        finally:
            skill.close()

    def test_cancel_is_terminal_and_close_is_idempotent(self) -> None:
        skill = build_atomic_skill(
            {"type": "target_reobserve"},
            observation_port=MockTargetObservationPort(clock=ManualClock()),
            clock=ManualClock(),
        )
        cancelled = skill.cancel("reobserve-cancelled", reason="operator stop")
        repeated = skill.cancel("reobserve-cancelled", reason="different reason")
        self.assertIs(cancelled, repeated)
        self.assertIs(cancelled.status, AtomicSkillStatus.CANCELLED)
        self.assertIs(cancelled.failure_code, AtomicSkillFailureCode.CANCELLED)
        skill.close()
        skill.close()

    def test_result_rejects_inconsistent_success_and_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "success must agree"):
            AtomicSkillResult(
                execution_id="bad-1",
                skill_id="bad",
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                success=True,
                duration_s=0.0,
                failure_code=AtomicSkillFailureCode.PRECONDITION_FAILED,
            )
        with self.assertRaisesRegex(ValueError, "requires failure_code"):
            AtomicSkillResult(
                execution_id="bad-2",
                skill_id="bad",
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                success=False,
                duration_s=0.0,
            )

    def test_missing_target_has_stable_failure_code_and_evidence_trace(self) -> None:
        skill = build_atomic_skill(
            {"type": "target_reobserve"},
            observation_port=MockTargetObservationPort(
                clock=ManualClock(), force_missing=True
            ),
            clock=ManualClock(),
        )
        try:
            result = skill.run(
                TargetReobserveRequest(
                    execution_id="reobserve-missing",
                    target_id="tomato-1",
                    observation=Observation(sequence=1, metadata={"tomatoes": []}),
                )
            )
            self.assertFalse(result.success)
            self.assertIs(result.failure_code, AtomicSkillFailureCode.TARGET_STALE)
            self.assertEqual(result.trace[-1].evidence_refs, ("mock.rgb_metadata",))
        finally:
            skill.close()

    def test_visibility_and_depth_admission_gates_fail_closed(self) -> None:
        occluded = build_atomic_skill(
            {"type": "target_reobserve", "max_occlusion_ratio": 0.5},
            observation_port=MockTargetObservationPort(
                clock=ManualClock(), force_occlusion_ratio=0.9
            ),
            clock=ManualClock(),
        )
        depth_required = build_atomic_skill(
            {"type": "target_reobserve", "require_depth": True},
            observation_port=MockTargetObservationPort(clock=ManualClock()),
            clock=ManualClock(),
        )
        observation = Observation(
            sequence=2,
            metadata={"tomatoes": [{"target_id": "tomato-1", "occlusion": 0.1}]},
        )
        try:
            occlusion_result = occluded.run(
                TargetReobserveRequest("occluded", "tomato-1", observation)
            )
            depth_result = depth_required.run(
                TargetReobserveRequest("no-depth", "tomato-1", observation)
            )
            self.assertIs(
                occlusion_result.failure_code,
                AtomicSkillFailureCode.TARGET_OCCLUDED,
            )
            self.assertIs(
                depth_result.failure_code,
                AtomicSkillFailureCode.DEPTH_UNAVAILABLE,
            )
        finally:
            occluded.close()
            depth_required.close()


if __name__ == "__main__":
    unittest.main()
