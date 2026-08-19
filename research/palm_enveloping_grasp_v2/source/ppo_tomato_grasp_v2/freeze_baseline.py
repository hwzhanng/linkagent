"""Create an auditable, read-only manifest for the v1 baseline.

The command only reads v1 files and writes a manifest under the v2 artifact
namespace.  It never copies, edits, or deletes the protected source/weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "simulation" / "ppo_tomato_grasp"
OUT = V1 / "artifacts" / "enveloping_grasp_v2" / "v1_baseline_manifest.json"

FILES = (
    V1 / "physx_tomato_env.py",
    V1 / "physx_enveloping_env.py",
    V1 / "physx_tomato_train.py",
    V1 / "evaluate_enveloping_checkpoint.py",
    V1 / "artifacts" / "stage4_band1_long" / "best.pt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "read_only_baseline": True,
        })
    payload = {
        "manifest_version": "v2_p0_baseline_freeze_1",
        "root": str(ROOT),
        "protected_paths": [
            str((V1).relative_to(ROOT)),
            str((V1 / "artifacts" / "stage4_band1_long").relative_to(ROOT)),
        ],
        "files": entries,
        "notes": [
            "v1 physics/evaluator/checkpoint are imported only for matched baseline evaluation",
            "v2 changes live in simulation/ppo_tomato_grasp_v2 and artifacts/enveloping_grasp_v2",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
