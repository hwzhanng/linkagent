"""Replay historical Clean-v5 checkpoints to audit reward guidance.

The sweep is intentionally sequential: Isaac/PhysX results from two concurrent
GPU contexts were previously shown to be invalid.  Each checkpoint is replayed
both in its released Clean-v5 task and as a Physics-v1 compatibility
counterfactual.  The latter is diagnostic only because Physics-v1 changes the
hand manifold, reset distribution, contact physics, stem model, and success
predicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_isaac_root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
if not _isaac_root_value:
    raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout")
ISAAC_ROOT = Path(_isaac_root_value).expanduser().resolve()
ISAAC_PYTHON = ISAAC_ROOT / "env_isaaclab/bin/python"
SOURCE_RUN = (
    PROJECT_ROOT
    / "logs/rsl_rl/tomato_pick_clean_v5/"
    "2026-08-08_02-27-24_tomato_pick_clean_v5_from_scratch"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "logs/qualification/tomato_physics_v1/historical_checkpoint_guidance"
)
DEFAULT_STAGES = (50, 250, 600, 1000, 1350)
SOURCE_TRAINING_LOG = Path("/tmp/tomato_clean_v5_formal_8192.log")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TASKS = {
    "clean_v5": (
        "LinkAgent-TomatoPick-Clean-v5",
        "simulation.tomato_pick.isaac.register_task",
    ),
    "physics_v1": (
        "LinkAgent-TomatoPick-Physics-v1",
        "simulation.tomato_pick.isaac.physics_v1.register_task",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alpha(iteration: int) -> float:
    points = ((0, 0.0), (250, 0.0), (600, 0.35), (1000, 0.75), (1300, 1.0))
    value = max(0, iteration)
    for (left_x, left_y), (right_x, right_y) in zip(points[:-1], points[1:]):
        if value <= right_x:
            fraction = (value - left_x) / (right_x - left_x)
            fraction = min(1.0, max(0.0, fraction))
            fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            return left_y + fraction * (right_y - left_y)
    return 1.0


def _run_evaluation(
    *,
    label: str,
    task: str,
    callback: str,
    checkpoint: Path,
    stage: int,
    output_root: Path,
    num_envs: int,
    steps: int,
    seed: int,
) -> dict:
    output = output_root / label / f"model_{stage}_seed{seed}_{num_envs}.json"
    log = output.with_suffix(".log")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "VIRTUAL_ENV": str(ISAAC_ROOT / "env_isaaclab"),
            "LINKAGENT_ISAACLAB_ROOT": str(ISAAC_ROOT),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": os.pathsep.join(
                (str(PROJECT_ROOT), environment.get("PYTHONPATH", ""))
            ),
            "TOMATO_REWARD_ITERATION_OFFSET": str(stage),
        }
    )
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join(
        (str(ISAAC_ROOT / "env_isaaclab/bin"), environment.get("PATH", ""))
    )
    command = [
        str(ISAAC_PYTHON),
        str(PROJECT_ROOT / "script/evaluate_tomato_checkpoint.py"),
        "--output",
        str(output),
        "--steps",
        str(steps),
        "--num_envs",
        str(num_envs),
        "--task",
        task,
        "--external_callback",
        callback,
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(seed),
        "--headless",
        "--viz",
        "none",
    ]
    print(f"GUIDANCE_EVAL_START label={label} stage={stage}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"evaluation failed label={label} stage={stage}; inspect {log}"
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    if result["checkpoint_sha256"] != _sha256(checkpoint):
        raise RuntimeError(f"checkpoint identity mismatch for {checkpoint}")
    print(
        "GUIDANCE_EVAL_DONE "
        f"label={label} stage={stage} episodes={result['completed_episodes']} "
        f"success={100.0 * result['success_rate']:.3f}% "
        f"detach={100.0 * result['detach_rate']:.3f}% "
        f"return={result['mean_episode_return']:.3f}",
        flush=True,
    )
    return result


def _finite_result(result: dict) -> bool:
    scalar_keys = (
        "success_rate",
        "drop_rate",
        "detach_rate",
        "retained_rate",
        "mean_episode_return",
        "std_episode_return",
        "mean_episode_length",
        "mean_abs_action",
        "action_saturation_rate",
    )
    values = [float(result[key]) for key in scalar_keys]
    values.extend(float(value) for value in result["mean_episode_reward_terms"].values())
    return bool(result["completed_episodes"] > 0 and all(math.isfinite(v) for v in values))


def _source_training_metrics(stages: tuple[int, ...]) -> tuple[dict, int | None]:
    if not SOURCE_TRAINING_LOG.is_file():
        raise FileNotFoundError(SOURCE_TRAINING_LOG)
    text = ANSI.sub(
        "", SOURCE_TRAINING_LOG.read_text(encoding="utf-8", errors="replace")
    )
    records: dict[int, dict[str, float | int]] = {}
    current: dict[str, float | int] | None = None
    metrics = {
        "Metrics/success_rate": "success_rate",
        "Metrics/drop_rate": "drop_rate",
        "Metrics/detach_rate": "detach_rate",
        "Metrics/retained_rate": "retained_rate",
        "Metrics/mean_hold_steps": "mean_hold_steps",
        "Mean action std": "mean_action_std",
        "Mean reward": "mean_reward",
        "Episode_Reward/break_grasped": "break_grasped",
        "Episode_Reward/stable_detach": "stable_detach",
        "Episode_Reward/retention_progress": "retention_progress",
        "Episode_Reward/success": "success_reward",
        "Episode_Reward/drop_failure": "drop_failure",
        "Episode_Reward/excess_force_failure": "excess_force_failure",
    }
    first_success: int | None = None
    for line in text.splitlines():
        match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
        if match:
            if current is not None:
                records[int(current["iteration"])] = current
            current = {"iteration": int(match.group(1))}
            continue
        if current is None:
            continue
        for displayed, key in metrics.items():
            match = re.search(rf"{re.escape(displayed)}:\s*([-+0-9.eE]+)", line)
            if match:
                value = float(match.group(1))
                current[key] = value
                if key == "success_rate" and value > 0.0 and first_success is None:
                    first_success = int(current["iteration"])
                break
    if current is not None:
        records[int(current["iteration"])] = current
    selected = {
        str(stage): records[stage]
        for stage in stages
        if stage in records
    }
    return selected, first_success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stages", type=int, nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    stages = tuple(args.stages)
    if stages != tuple(sorted(set(stages))):
        raise ValueError("stages must be unique and sorted")
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("num-envs and steps must be positive")

    output_root = args.output_root.expanduser().resolve()
    results: dict[str, dict[str, dict]] = {label: {} for label in TASKS}
    for label, (task, callback) in TASKS.items():
        for stage in stages:
            checkpoint = SOURCE_RUN / f"model_{stage}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            output = output_root / label / f"model_{stage}_seed{args.seed}_{args.num_envs}.json"
            if args.skip_existing and output.is_file():
                result = json.loads(output.read_text(encoding="utf-8"))
                if result.get("checkpoint_sha256") != _sha256(checkpoint):
                    raise RuntimeError(f"stale result identity mismatch: {output}")
                print(f"GUIDANCE_EVAL_REUSE label={label} stage={stage}", flush=True)
            else:
                result = _run_evaluation(
                    label=label,
                    task=task,
                    callback=callback,
                    checkpoint=checkpoint,
                    stage=stage,
                    output_root=output_root,
                    num_envs=args.num_envs,
                    steps=args.steps,
                    seed=args.seed,
                )
            results[label][str(stage)] = result

    alphas = [_alpha(stage) for stage in stages]
    source_training, source_first_success = _source_training_metrics(stages)
    clean = results["clean_v5"]
    late_clean = clean[str(stages[-1])]
    early_clean = clean[str(stages[0])]
    checks = {
        "all_results_finite": all(
            _finite_result(result)
            for task_results in results.values()
            for result in task_results.values()
        ),
        "curriculum_alpha_monotone": alphas == sorted(alphas),
        "released_best_reproduces_over_90_percent": late_clean["success_rate"] >= 0.90,
        "released_best_improves_on_early_checkpoint": (
            late_clean["success_rate"] > early_clean["success_rate"]
        ),
        "source_training_first_success_by_update_50": (
            source_first_success is not None and source_first_success <= 50
        ),
        "source_training_stage_records_complete": len(source_training) == len(stages),
        "physics_counterfactual_is_diagnostic_only": True,
    }
    report = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "purpose": "historical reward-guidance and compatibility audit",
        "qualification_rule": (
            "Clean-v5 replay is a pass/fail reference; Physics-v1 replay is a finite-signal "
            "counterfactual only because action semantics and task physics changed."
        ),
        "source_run": str(SOURCE_RUN),
        "stages": list(stages),
        "alignment_alpha": {str(stage): _alpha(stage) for stage in stages},
        "source_training_first_success_iteration": source_first_success,
        "source_training_metrics": source_training,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "results": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary = output_root / "summary.json"
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("GUIDANCE_AUDIT=" + json.dumps({"passed": report["passed"], "checks": checks}))
    print(f"GUIDANCE_AUDIT_PATH={summary}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
