from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from simulation.ppo_tomato_grasp.cpu_surrogate_train import CpuActorCritic
from simulation.ppo_tomato_grasp.export_cpu_policy import (
    ENVIRONMENT_CONTRACT,
    TENSOR_CONTRACT,
    export_actor,
)


class CpuPolicyExportTests(unittest.TestCase):
    def test_export_has_exact_parity_and_rejects_bad_shapes(self):
        policy = CpuActorCritic()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            output = root / "actor.ts"
            torch.save(
                {
                    "model": policy.state_dict(),
                    "update": 7,
                    "contract": TENSOR_CONTRACT,
                    "environment_contract": ENVIRONMENT_CONTRACT,
                },
                checkpoint,
            )
            metadata = export_actor(checkpoint, output, parity_samples=128, seed=4)
            self.assertEqual(metadata["maximum_absolute_error"], 0.0)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".ts.json").is_file())
            scripted = torch.jit.load(output)
            self.assertEqual(tuple(scripted(torch.zeros((3, 110))).shape), (3, 10))
            with self.assertRaises(torch.jit.Error):
                scripted(torch.zeros((3, 109)))
            with self.assertRaises(torch.jit.Error):
                scripted(torch.full((3, 110), float("nan")))

    def test_environment_version_mismatch_fails_closed(self):
        policy = CpuActorCritic()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "legacy.pt"
            torch.save(
                {
                    "model": policy.state_dict(),
                    "update": 1,
                    "contract": TENSOR_CONTRACT,
                },
                checkpoint,
            )
            with self.assertRaises(ValueError):
                export_actor(checkpoint, root / "actor.ts", parity_samples=8)


if __name__ == "__main__":
    unittest.main()
