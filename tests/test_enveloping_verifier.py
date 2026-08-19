"""Palm-enclosure evaluation stays independent from runtime control."""

from __future__ import annotations

from dataclasses import replace
import unittest

from module.skills.tomato_pick import (
    EnvelopingTraceFrame,
    EnvelopingVerifierConfig,
    verify_enveloping_trace,
)


def _frame(step: int) -> EnvelopingTraceFrame:
    if step < 3:
        contacts = (False, False, False, False, False)
    elif step < 5:
        contacts = (True, True, True, False, False)
    else:
        contacts = (True, True, True, True, True)
    return EnvelopingTraceFrame(
        palm_supported=True,
        digit_contact=contacts,
        angular_coverage_deg=220.0 if step >= 5 else 0.0,
        thumb_opposition_cos=-0.5 if step >= 5 else 1.0,
        same_hemisphere_score=0.5 if step >= 5 else 1.0,
        palm_load_share=0.3,
        distal_joint_delta_rad=(0.05, 0.05, 0.05, 0.05),
        postbreak=step >= 8,
    )


class EnvelopingVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnvelopingVerifierConfig(
            palm_support_streak_steps=2,
            palm_first_margin_steps=3,
            strict_hold_steps=4,
            require_postbreak_hold=True,
            postbreak_hold_steps=2,
        )

    def test_accepts_palm_first_five_finger_postbreak_hold(self) -> None:
        result = verify_enveloping_trace((_frame(step) for step in range(10)), self.config)

        self.assertTrue(result.strict_success)
        self.assertEqual(result.failure_reasons, ("pass",))
        self.assertEqual(result.first_palm_support_step, 0)
        self.assertEqual(result.first_three_digit_step, 3)
        self.assertEqual(result.envelope_hold_max_streak, 5)
        self.assertEqual(result.postbreak_hold_max_streak, 2)
        self.assertFalse(result.hardware_release_allowed)

    def test_drop_and_idle_distal_joints_are_reported(self) -> None:
        trace = [_frame(step) for step in range(10)]
        trace[8] = replace(
            trace[8],
            dropped=True,
            distal_joint_delta_rad=(0.0, 0.0, 0.0, 0.0),
        )
        trace = [replace(frame, distal_joint_delta_rad=(0.0, 0.0, 0.0, 0.0)) for frame in trace]
        result = verify_enveloping_trace(trace, self.config)

        self.assertFalse(result.strict_success)
        self.assertIn("drop_or_slip", result.failure_reasons)
        self.assertIn("distal_joint_idle", result.failure_reasons)
        self.assertIn("postbreak_hold_short", result.failure_reasons)


if __name__ == "__main__":
    unittest.main()
