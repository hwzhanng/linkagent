"""Deterministic, fail-closed tests for harvest target selection."""

from __future__ import annotations

import unittest

from module.robotics import ManualClock
from module.skills import AtomicSkillFailureCode
from module.skills.harvest_cycle import SelectHarvestTargetSkill
from proto import (
    ArmSide,
    HarvestTargetCandidateScore,
    HarvestTargetNextAction,
    HarvestTargetSelection,
    Pose,
    SelectHarvestTargetRequest,
    TomatoDetectionResult,
    TomatoTarget,
)


NOW_NS = 2_000_000_000


def _target(
    target_id: str,
    *,
    position_m: tuple[float, float, float] = (0.45, -0.25, 0.72),
    ripeness: float = 0.90,
    confidence: float = 0.90,
    occlusion: float = 0.10,
    semi_axes_m: tuple[float, float, float] = (0.035, 0.032, 0.038),
    sample_monotonic_ns: int = NOW_NS,
) -> TomatoTarget:
    return TomatoTarget(
        target_id=target_id,
        center_pose=Pose(position_m, (1.0, 0.0, 0.0, 0.0), "world"),
        semi_axes_m=semi_axes_m,
        ripeness_score=ripeness,
        pick_confidence=confidence,
        occlusion=occlusion,
        stem_direction_xyz=(0.0, 0.0, -1.0),
        observation_sequence=12,
        sample_monotonic_ns=sample_monotonic_ns,
    )


def _detection(
    *targets: TomatoTarget,
    sample_monotonic_ns: int = NOW_NS,
) -> TomatoDetectionResult:
    return TomatoDetectionResult(
        targets=targets,
        observation_sequence=12,
        model="test-yolo-vlm-rgbd-v1",
        sample_monotonic_ns=sample_monotonic_ns,
    )


def _skill(
    clock: ManualClock,
    *,
    with_workspace: bool = True,
    max_candidates: int = 20,
) -> SelectHarvestTargetSkill:
    return SelectHarvestTargetSkill(
        clock=clock,
        reachability_profile_id="offline-test-right-arm-aabb-v1",
        reachability_arm_side="right",
        workspace_min_m=(0.20, -0.60, 0.30) if with_workspace else None,
        workspace_max_m=(0.80, 0.20, 1.20) if with_workspace else None,
        max_candidates=max_candidates,
    )


def _request(
    execution_id: str,
    detection: TomatoDetectionResult,
) -> SelectHarvestTargetRequest:
    return SelectHarvestTargetRequest(
        execution_id=execution_id,
        detection=detection,
        arm_side=ArmSide.RIGHT,
        max_detection_age_s=1.0,
    )


class SelectHarvestTargetSkillTests(unittest.TestCase):
    def test_hard_gates_beat_a_higher_semantic_score(self) -> None:
        clock = ManualClock(start_ns=NOW_NS)
        skill = _skill(clock)
        outside = _target(
            "outside-high-score",
            position_m=(0.95, -0.25, 0.72),
            ripeness=1.0,
            confidence=1.0,
            occlusion=0.0,
        )
        supported = _target(
            "supported",
            position_m=(0.45, -0.25, 0.72),
            ripeness=0.82,
            confidence=0.78,
            occlusion=0.18,
        )
        occluded = _target("occluded", occlusion=0.75)
        try:
            result = skill.run(_request("select-safe", _detection(outside, supported, occluded)))
            self.assertTrue(result.success)
            self.assertEqual(result.output["selected_target_id"], "supported")
            self.assertFalse(result.output["motion_authorized"])
            self.assertTrue(result.output["requires_pregrasp_clearance_check"])
            candidates = {
                item["target_id"]: item for item in result.output["ranked_candidates"]
            }
            self.assertFalse(candidates["outside-high-score"]["eligible"])
            self.assertIn(
                "outside_reachability_profile",
                candidates["outside-high-score"]["rejection_reasons"],
            )
            self.assertFalse(candidates["occluded"]["eligible"])
            self.assertIn(
                "occlusion_above_maximum",
                candidates["occluded"]["rejection_reasons"],
            )
        finally:
            skill.close()

    def test_ties_are_stable_and_revision_ignores_input_order(self) -> None:
        clock = ManualClock(start_ns=NOW_NS)
        first_skill = _skill(clock)
        second_skill = _skill(clock)
        tomato_a = _target("tomato-a")
        tomato_b = _target("tomato-b")
        try:
            first = first_skill.run(
                _request("ordered-b-a", _detection(tomato_b, tomato_a))
            )
            repeated = first_skill.run(
                _request("ordered-b-a", _detection(tomato_a, tomato_b))
            )
            reordered = second_skill.run(
                _request("ordered-a-b", _detection(tomato_a, tomato_b))
            )
            self.assertIs(first, repeated)
            self.assertEqual(first.output["selected_target_id"], "tomato-a")
            self.assertEqual(
                first.output["selection_revision"],
                reordered.output["selection_revision"],
            )
            self.assertEqual(
                [item["target_id"] for item in first.output["ranked_candidates"]],
                ["tomato-a", "tomato-b"],
            )
        finally:
            first_skill.close()
            second_skill.close()

    def test_missing_workspace_profile_fails_reachability_closed(self) -> None:
        clock = ManualClock(start_ns=NOW_NS)
        skill = _skill(clock, with_workspace=False)
        try:
            result = skill.run(_request("unknown-reach", _detection(_target("tomato-1"))))
            self.assertFalse(result.success)
            self.assertIs(
                result.failure_code, AtomicSkillFailureCode.REACHABILITY_UNKNOWN
            )
            self.assertEqual(
                result.output["next_action"],
                HarvestTargetNextAction.REACHABILITY_CONTEXT_REQUIRED.value,
            )
        finally:
            skill.close()

    def test_arm_mismatched_reachability_profile_is_rejected(self) -> None:
        clock = ManualClock(start_ns=NOW_NS)
        skill = _skill(clock)
        request = SelectHarvestTargetRequest(
            execution_id="wrong-arm-profile",
            detection=_detection(_target("tomato-1")),
            arm_side=ArmSide.LEFT,
        )
        try:
            result = skill.run(request)
            self.assertFalse(result.success)
            self.assertIs(result.failure_code, AtomicSkillFailureCode.INVALID_REQUEST)
            self.assertIn("bound to right arm", result.message)
        finally:
            skill.close()

    def test_occlusion_only_failure_requests_active_view(self) -> None:
        clock = ManualClock(start_ns=NOW_NS)
        skill = _skill(clock)
        try:
            result = skill.run(
                _request("active-view", _detection(_target("tomato-1", occlusion=0.8)))
            )
            self.assertFalse(result.success)
            self.assertIs(
                result.failure_code, AtomicSkillFailureCode.VIEWPOINT_INSUFFICIENT
            )
            self.assertEqual(
                result.output["next_action"],
                HarvestTargetNextAction.ACTIVE_VIEW_REQUIRED.value,
            )
        finally:
            skill.close()

    def test_empty_stale_and_oversized_candidate_sets_have_stable_failures(self) -> None:
        cases = (
            (
                "empty",
                _detection(),
                20,
                AtomicSkillFailureCode.NO_HARVESTABLE_TARGET,
            ),
            (
                "stale",
                _detection(
                    _target("stale", sample_monotonic_ns=1),
                    sample_monotonic_ns=1,
                ),
                20,
                AtomicSkillFailureCode.CANDIDATE_SET_STALE,
            ),
            (
                "too-many",
                _detection(_target("one"), _target("two")),
                1,
                AtomicSkillFailureCode.INVALID_REQUEST,
            ),
        )
        for name, detection, limit, expected in cases:
            with self.subTest(name=name):
                skill = _skill(ManualClock(start_ns=NOW_NS), max_candidates=limit)
                try:
                    result = skill.run(_request(f"select-{name}", detection))
                    self.assertFalse(result.success)
                    self.assertIs(result.failure_code, expected)
                finally:
                    skill.close()

    def test_selection_dto_cannot_authorize_motion(self) -> None:
        candidate = HarvestTargetCandidateScore(
            target_id="tomato-1",
            eligible=True,
            score=0.9,
            hard_gates={"workspace_reachable": True},
            score_components={"ripeness": 0.9},
            rejection_reasons=(),
            workspace_margin_m=0.1,
            evidence_sources=("test",),
        )
        with self.assertRaisesRegex(ValueError, "cannot authorize robot motion"):
            HarvestTargetSelection(
                selection_revision="selection-test",
                observation_sequence=12,
                sample_monotonic_ns=NOW_NS,
                selected_target_id="tomato-1",
                next_action=HarvestTargetNextAction.REOBSERVE_SELECTED,
                ranked_candidates=(candidate,),
                motion_authorized=True,
            )


if __name__ == "__main__":
    unittest.main()
