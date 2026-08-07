from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from module.skills.tomato_pick.contracts import ACTION_DIM, OBSERVATION_DIM

from script.pretrain_tomato_success_bc import (
    load_success_dataset,
    phase_balanced_indices,
    split_episode_indices,
)


class TomatoSuccessBCTest(unittest.TestCase):
    def _dataset(self, path: Path, *, deterministic: bool = False) -> None:
        phases = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        torch.save(
            {
                "schema_version": 1,
                "observations": torch.zeros((8, OBSERVATION_DIM)),
                "actions": torch.zeros((8, ACTION_DIM)),
                "phases": phases,
                "episode_offsets": torch.tensor([0, 4, 8]),
                "metadata": {"deterministic": deterministic},
            },
            path,
        )

    def test_dataset_validation_and_episode_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "successes.pt"
            self._dataset(path)
            dataset = load_success_dataset(path, minimum_episodes=2, require_stochastic=True)
        train, validation, train_episodes, validation_episodes = split_episode_indices(
            dataset["episode_offsets"], 0.25, 42
        )
        self.assertEqual(len(train_episodes), 1)
        self.assertEqual(len(validation_episodes), 1)
        self.assertTrue(set(train.tolist()).isdisjoint(validation.tolist()))
        self.assertEqual(sorted(train.tolist() + validation.tolist()), list(range(8)))

    def test_deterministic_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "successes.pt"
            self._dataset(path, deterministic=True)
            with self.assertRaisesRegex(ValueError, "stochastic successful actions"):
                load_success_dataset(path, minimum_episodes=2, require_stochastic=True)

    def test_phase_balancing_equalizes_counts(self) -> None:
        phases = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3])
        eligible = torch.arange(len(phases))
        generator = torch.Generator().manual_seed(7)
        balanced = phase_balanced_indices(phases, eligible, generator)
        counts = torch.bincount(phases[balanced], minlength=4)
        self.assertEqual(counts.tolist(), [4, 4, 4, 4])


if __name__ == "__main__":
    unittest.main()
