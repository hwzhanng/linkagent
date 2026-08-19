"""Small real-data episode contract and deterministic replay."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from module.robotics import MockPickScenario, build_mock_robotics_bundle
from module.skills.tomato_pick import (
    TACTILE_EPISODE_SCHEMA,
    TactileEpisodeMetadata,
    TactileEpisodeSample,
    TactileMatrixFeatureExtractor,
    read_tactile_episode_jsonl,
    write_tactile_episode_jsonl,
)
from proto import ArmSide, Pose


class TactileEpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = build_mock_robotics_bundle(
            side=ArmSide.RIGHT,
            scenario=MockPickScenario(),
        )
        self.bundle.world.contact = True

    def tearDown(self) -> None:
        self.bundle.close()

    def _sample(self, now_ns: int) -> TactileEpisodeSample:
        hand = self.bundle.hand.get_state(ArmSide.RIGHT)
        tactile = self.bundle.tactile.get_frame(ArmSide.RIGHT)
        features = TactileMatrixFeatureExtractor().extract(tactile, now_ns=now_ns)
        return TactileEpisodeSample.capture(
            hand=hand,
            tactile=tactile,
            now_ns=now_ns,
            phase="grasp",
            object_pose_in_palm=Pose((0.0, 0.0, 0.04), (1.0, 0.0, 0.0, 0.0), "palm"),
            reference_force_n=(1.0, 1.1, 1.2, 1.0, 0.9),
            matrix_features=features,
        )

    def test_jsonl_roundtrip_preserves_raw_matrix_and_units(self) -> None:
        first_time = self.bundle.clock.monotonic_ns()
        first = self._sample(first_time)
        self.bundle.clock.advance_s(0.05)
        second = self._sample(self.bundle.clock.monotonic_ns())
        metadata = TactileEpisodeMetadata(
            episode_id="episode-001",
            side=ArmSide.RIGHT,
            created_utc="2026-08-19T15:00:00+08:00",
            object_description="mock tomato",
            calibration_id="mock_tactile_calibration_v1",
            source_feature_extractor_id="g20_tactile_features_v1",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = write_tactile_episode_jsonl(
                Path(directory) / "episode.jsonl",
                metadata,
                (first, second),
            )
            restored_metadata, restored = read_tactile_episode_jsonl(path)

        self.assertEqual(restored_metadata.schema, TACTILE_EPISODE_SCHEMA)
        self.assertEqual(restored_metadata.side, ArmSide.RIGHT)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].tactile_matrix_raw[0][6][2], 160)
        self.assertEqual(len(restored[0].matrix_features_50d or ()), 50)
        self.assertEqual(restored[0].object_pose_in_palm["frame_id"], "palm")

    def test_capture_rejects_cross_side_state(self) -> None:
        hand = self.bundle.hand.get_state(ArmSide.RIGHT)
        tactile = replace(self.bundle.tactile.get_frame(ArmSide.RIGHT), side=ArmSide.LEFT)
        with self.assertRaisesRegex(ValueError, "same side"):
            TactileEpisodeSample.capture(
                hand=hand,
                tactile=tactile,
                now_ns=self.bundle.clock.monotonic_ns(),
                phase="observe",
            )


if __name__ == "__main__":
    unittest.main()
