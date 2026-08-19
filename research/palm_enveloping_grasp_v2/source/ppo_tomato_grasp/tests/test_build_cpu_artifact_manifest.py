from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from simulation.ppo_tomato_grasp.build_cpu_artifact_manifest import build_manifest


class CpuArtifactManifestTests(unittest.TestCase):
    def make_bundle(self, root: Path):
        checkpoint = root / "candidate.pt"
        actor = root / "actor.ts"
        metadata = root / "actor.ts.json"
        contract = {"action_dim": 10, "simulation_only": True}
        environment = {"version": "test"}
        torch.save(
            {
                "update": 4,
                "contract": contract,
                "environment_contract": environment,
            },
            checkpoint,
        )
        actor.write_bytes(b"scripted actor fixture")
        metadata.write_text(
            json.dumps(
                {
                    "source_checkpoint": str(checkpoint.resolve()),
                    "source_update": 4,
                    "tensor_contract": contract,
                    "environment_contract": environment,
                    "hardware_command_allowed": False,
                }
            ),
            encoding="utf-8",
        )
        return checkpoint, actor, metadata

    def test_valid_bundle_has_hashes_and_never_releases_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, actor, metadata = self.make_bundle(Path(temporary))
            manifest = build_manifest(checkpoint, actor, [metadata])
            self.assertEqual(len(manifest["artifacts"]), 3)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["artifacts"]))
            self.assertFalse(manifest["hardware_release_allowed"])

    def test_mixed_lineage_and_missing_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, actor, metadata = self.make_bundle(Path(temporary))
            value = json.loads(metadata.read_text(encoding="utf-8"))
            value["source_update"] = 5
            metadata.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_manifest(checkpoint, actor, [metadata])
            with self.assertRaises(ValueError):
                build_manifest(checkpoint, actor, [])

    def test_mixed_environment_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, actor, metadata = self.make_bundle(Path(temporary))
            extra = Path(temporary) / "report.json"
            extra.write_text(
                json.dumps({"environment_contract": {"version": "wrong"}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                build_manifest(checkpoint, actor, [metadata, extra])


if __name__ == "__main__":
    unittest.main()
