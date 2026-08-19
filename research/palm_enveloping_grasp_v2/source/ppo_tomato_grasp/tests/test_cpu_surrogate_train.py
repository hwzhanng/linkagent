from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from simulation.ppo_tomato_grasp.cpu_surrogate_train import (
    CpuActorCritic,
    CpuPpoConfig,
    train_cpu_ppo,
)


class CpuSurrogateTrainTests(unittest.TestCase):
    def test_asymmetric_network_shapes(self):
        model = CpuActorCritic()
        observation = torch.zeros((7, 110))
        critic = torch.zeros((7, 148))
        action, raw, log_prob, value = model.act(observation, critic)
        self.assertEqual(tuple(action.shape), (7, 10))
        self.assertEqual(tuple(raw.shape), (7, 10))
        self.assertEqual(tuple(log_prob.shape), (7,))
        self.assertEqual(tuple(value.shape), (7,))
        self.assertTrue(torch.all(action.abs() < 1.0))

    def test_weak_prior_initializes_semantic_mean_without_freezing(self):
        prior = torch.zeros(10)
        prior[0] = 0.70
        prior[4] = 0.60
        model = CpuActorCritic(weak_prior_action=prior)
        action = model.deterministic(torch.zeros((8, 110)))
        self.assertTrue(torch.allclose(action.mean(dim=0), prior, atol=0.01))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_two_update_training_is_finite_and_writes_gated_artifact(self):
        config = CpuPpoConfig(
            num_envs=16,
            horizon=8,
            updates=2,
            epochs=1,
            minibatch_size=64,
            evaluation_interval=2,
            evaluation_envs=16,
            threads=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = train_cpu_ppo(config, output_dir=output)
            self.assertTrue(summary["all_finite"])
            self.assertTrue((output / "latest.pt").is_file())
            self.assertTrue((output / "summary.json").is_file())
            checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
            self.assertTrue(checkpoint["contract"]["simulation_only"])
            self.assertEqual(checkpoint["contract"]["action_dim"], 10)
            self.assertEqual(checkpoint["update"], 2)
            self.assertIn("optimizer", checkpoint)
            self.assertIn("reward_version", checkpoint["environment_contract"])

            legacy_path = output / "legacy.pt"
            checkpoint.pop("environment_contract")
            torch.save(checkpoint, legacy_path)
            continued = replace(config, updates=3)
            with self.assertRaises(ValueError):
                train_cpu_ppo(continued, resume_path=legacy_path)
            migrated = train_cpu_ppo(
                continued,
                resume_path=legacy_path,
                allow_environment_migration=True,
            )
            self.assertTrue(migrated["all_finite"])


if __name__ == "__main__":
    unittest.main()
