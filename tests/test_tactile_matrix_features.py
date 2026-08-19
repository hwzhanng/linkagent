"""Raw G20 matrix features remain independent from the frozen 66D Actor."""

from __future__ import annotations

from dataclasses import replace
import unittest

from module.robotics import MockPickScenario, build_mock_robotics_bundle
from module.skills.tomato_pick import (
    OBSERVATION_DIM,
    PolicyManifest,
    TACTILE_MATRIX_FEATURE_DIM,
    TactileMatrixFeatureConfig,
    TactileMatrixFeatureExtractor,
    fit_tactile_matrix_baseline,
)
from proto import ArmSide


class TactileMatrixFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = build_mock_robotics_bundle(
            side=ArmSide.RIGHT,
            scenario=MockPickScenario(),
        )

    def tearDown(self) -> None:
        self.bundle.close()

    def test_extracts_spatial_features_and_keeps_actor_contract_frozen(self) -> None:
        self.bundle.world.contact = True
        frame = self.bundle.tactile.get_frame(ArmSide.RIGHT)
        features = TactileMatrixFeatureExtractor().extract(
            frame,
            now_ns=self.bundle.clock.monotonic_ns(),
        )

        self.assertEqual(len(features.values), TACTILE_MATRIX_FEATURE_DIM)
        self.assertEqual(features.fingers[0].total_load_raw, 360.0)
        self.assertAlmostEqual(features.fingers[0].contact_area_fraction, 3.0 / 72.0)
        self.assertTrue(all(features.valid_mask))
        self.assertTrue(all(features.contact_mask))

        # The new matrix path is additive; released policy manifests still use
        # the original five-pressure observation layout.
        self.assertEqual(OBSERVATION_DIM, 66)
        self.assertEqual(PolicyManifest.mock_default().tactile_feature_extractor_id, "g20_pressure_5d_v1")
        self.assertEqual(features.feature_extractor_id, "g20_raw_matrix_50d_v1")

    def test_stale_finger_is_zeroed_and_explicitly_flagged(self) -> None:
        self.bundle.world.contact = True
        frame = self.bundle.tactile.get_frame(ArmSide.RIGHT)
        stale_finger = replace(
            frame.fingers[0],
            matrix_sample_monotonic_ns=max(0, self.bundle.clock.monotonic_ns() - 1_000_000_000),
        )
        frame = replace(frame, fingers=(stale_finger, *frame.fingers[1:]))
        result = TactileMatrixFeatureExtractor().extract(
            frame,
            now_ns=self.bundle.clock.monotonic_ns(),
        )

        self.assertFalse(result.valid_mask[0])
        self.assertEqual(result.values[:10], (0.0,) * 9 + (1.0,))
        self.assertTrue(all(result.valid_mask[1:]))

    def test_baseline_removes_static_raw_offset(self) -> None:
        matrix = tuple(tuple(7 for _ in range(6)) for _ in range(12))
        samples = [tuple(matrix for _ in range(5)) for _ in range(5)]
        baseline = fit_tactile_matrix_baseline(samples)
        frame = self.bundle.tactile.get_frame(ArmSide.RIGHT)
        fingers = tuple(replace(finger, matrix_weight_raw=matrix) for finger in frame.fingers)
        frame = replace(frame, fingers=fingers)
        result = TactileMatrixFeatureExtractor(
            TactileMatrixFeatureConfig(contact_taxel_threshold_raw=1.0),
            baseline,
        ).extract(frame, now_ns=self.bundle.clock.monotonic_ns())

        self.assertEqual(result.fingers[0].total_load_raw, 0.0)
        self.assertFalse(any(result.contact_mask))
        self.assertTrue(all(result.valid_mask))


if __name__ == "__main__":
    unittest.main()
