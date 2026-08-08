"""Promote only the checkpoint selected by a fixed evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    evaluation_path = args.evaluation_manifest.expanduser().resolve()
    if not evaluation_path.is_file():
        raise FileNotFoundError(evaluation_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("schema_version") != 1:
        raise ValueError("evaluation manifest must use schema_version 1")
    if evaluation.get("task") != "LinkAgent-TomatoPick-Clean-v5":
        raise ValueError("evaluation manifest belongs to a different task")
    fixed = evaluation.get("evaluation")
    if not isinstance(fixed, dict):
        raise ValueError("evaluation manifest is missing evaluation settings")
    if fixed.get("deterministic_policy") is not True:
        raise ValueError("best promotion requires deterministic evaluation")
    seeds = fixed.get("seeds")
    if not isinstance(seeds, list) or len(set(seeds)) < 3:
        raise ValueError("best promotion requires at least three distinct seeds")
    best = evaluation.get("best_summary")
    if not isinstance(best, dict) or int(best.get("completed_episodes", 0)) <= 0:
        raise ValueError("evaluation manifest has no completed best summary")
    selection_gate = evaluation.get("selection_gate")
    if not isinstance(selection_gate, dict) or selection_gate.get("passed") is not True:
        raise ValueError("best promotion requires the hardened multi-seed selection gate")
    checkpoint = Path(evaluation["best_checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_sha = evaluation.get("best_checkpoint_sha256")
    if _sha256(checkpoint) != expected_sha:
        raise ValueError("selected checkpoint SHA-256 no longer matches evaluation")

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    iteration = int(best["iteration"])
    skill_id = f"tomato_pick_clean_v5_model{iteration}_best"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "script/export_tomato_skill.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--skill-id",
        skill_id,
        "--status",
        "best_fixed_multiseed_evaluated",
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    copied_evaluation = output / "fixed_evaluation_manifest.json"
    shutil.copy2(evaluation_path, copied_evaluation)
    summary_source = evaluation_path.with_name("summary.md")
    if summary_source.is_file():
        shutil.copy2(summary_source, output / "fixed_evaluation_summary.md")
    failure_source = evaluation_path.with_name("failure_summary.json")
    if failure_source.is_file():
        copied_failure = output / "fixed_evaluation_failures.json"
        shutil.copy2(failure_source, copied_failure)

    skill_manifest_path = output / "skill_manifest.json"
    skill = json.loads(skill_manifest_path.read_text(encoding="utf-8"))
    skill["evaluation"] = {
        "selection_status": "selected_best",
        "fixed_deterministic": True,
        "full_environment_randomization": fixed.get("full_environment_randomization"),
        "seeds": seeds,
        "completed_episodes": int(best["completed_episodes"]),
        "success_rate": float(best["success_rate"]),
        "success_wilson_95_low": float(best["success_wilson_95_low"]),
        "success_wilson_95_high": float(best["success_wilson_95_high"]),
        "drop_rate": float(best["drop_rate"]),
        "detach_rate": float(best["detach_rate"]),
        "retained_rate": float(best["retained_rate"]),
        "source_manifest": copied_evaluation.name,
    }
    deployment = skill["deployment"]
    deployment["profile"] = "simulation_fixed_multiseed_best"
    deployment["blocked_until"] = [
        item
        for item in deployment.get("blocked_until", [])
        if not str(item).startswith("fixed deterministic multi-seed evaluation")
    ]
    skill["artifacts"]["fixed_evaluation_manifest"] = {
        "path": copied_evaluation.name,
        "sha256": _sha256(copied_evaluation),
    }
    if failure_source.is_file():
        skill["artifacts"]["fixed_evaluation_failures"] = {
            "path": copied_failure.name,
            "sha256": _sha256(copied_failure),
        }
    skill_manifest_path.write_text(
        json.dumps(skill, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(skill_manifest_path)


if __name__ == "__main__":
    main()
