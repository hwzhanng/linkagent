from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_promotion_rejects_single_seed_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    import hashlib

    manifest = {
        "schema_version": 1,
        "task": "LinkAgent-TomatoPick-Clean-v5",
        "evaluation": {"deterministic_policy": True, "seeds": [42]},
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "best_summary": {"completed_episodes": 1, "iteration": 1},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "script/promote_tomato_skill.py"),
            "--evaluation-manifest",
            str(path),
            "--output",
            str(tmp_path / "best"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "at least three distinct seeds" in completed.stderr
