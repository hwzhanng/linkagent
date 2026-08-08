"""Run a fixed-seed checkpoint sweep and write its final manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_isaac_root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
if not _isaac_root_value:
    raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout")
ISAAC_ROOT = Path(_isaac_root_value).expanduser().resolve()
ISAAC_ENV = ISAAC_ROOT / "env_isaaclab"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - radius, center + radius


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 2026))
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of checkpoint evaluations to run concurrently.",
    )
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-url", required=True)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["VIRTUAL_ENV"] = str(ISAAC_ENV)
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join((str(ISAAC_ENV / "bin"), environment.get("PATH", "")))
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT), environment.get("PYTHONPATH", "")))

    results: list[dict] = []
    jobs: list[tuple[int, int, Path, Path, Path]] = []
    for iteration in args.iterations:
        checkpoint = run_dir / f"model_{iteration}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for seed in args.seeds:
            result_path = output_dir / f"model_{iteration}_seed_{seed}.json"
            log_path = output_dir / f"model_{iteration}_seed_{seed}.log"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                results.append(result)
                print(f"[SKIP] model_{iteration} seed {seed} already complete", flush=True)
                continue
            jobs.append((iteration, seed, checkpoint, result_path, log_path))

    def run_evaluation(job: tuple[int, int, Path, Path, Path]) -> dict:
        iteration, seed, checkpoint, result_path, log_path = job
        checkpoint_environment = dict(environment)
        checkpoint_environment["TOMATO_REWARD_ITERATION_OFFSET"] = str(iteration)
        command = [
            str(ISAAC_ENV / "bin" / "python"),
            str(PROJECT_ROOT / "script" / "evaluate_tomato_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(result_path),
            "--num_envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--seed",
            str(seed),
            "--viz",
            "none",
        ]
        print(f"[RUN] model_{iteration} seed {seed}", flush=True)
        with log_path.open("w") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=checkpoint_environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                f"evaluation failed for model_{iteration}, seed {seed}; see {log_path}"
            )
        result = json.loads(result_path.read_text())
        print(
            f"[DONE] model_{iteration} seed {seed}: "
            f"success={result['success_rate']:.4%}, drop={result['drop_rate']:.4%}, "
            f"episodes={result['completed_episodes']}",
            flush=True,
        )
        return result

    if args.workers == 1:
        for job in jobs:
            results.append(run_evaluation(job))
    else:
        print(f"[PARALLEL] workers={args.workers}, pending_jobs={len(jobs)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_evaluation, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())

    summaries: list[dict] = []
    for iteration in args.iterations:
        checkpoint_results = [
            result for result in results if Path(result["checkpoint"]).stem == f"model_{iteration}"
        ]
        successes = sum(int(result["success_count"]) for result in checkpoint_results)
        drops = sum(int(result["drop_count"]) for result in checkpoint_results)
        episodes = sum(int(result["completed_episodes"]) for result in checkpoint_results)
        low, high = wilson_interval(successes, episodes)
        summaries.append(
            {
                "iteration": iteration,
                "checkpoint": str(run_dir / f"model_{iteration}.pt"),
                "seeds": args.seeds,
                "completed_episodes": episodes,
                "success_count": successes,
                "success_rate": successes / episodes,
                "success_wilson_95_low": low,
                "success_wilson_95_high": high,
                "drop_count": drops,
                "drop_rate": drops / episodes,
                "detach_rate": sum(
                    float(result["detach_count"]) for result in checkpoint_results
                )
                / episodes,
                "retained_rate": sum(
                    float(result["retained_count"]) for result in checkpoint_results
                )
                / episodes,
                "mean_episode_return": sum(
                    float(result["mean_episode_return"]) * int(result["completed_episodes"])
                    for result in checkpoint_results
                )
                / episodes,
            }
        )

    ranking = sorted(summaries, key=lambda item: (item["success_rate"], -item["drop_rate"]), reverse=True)
    best = ranking[0]
    best_path = Path(best["checkpoint"])
    manifest = {
        "schema_version": 1,
        "task": "LinkAgent-TomatoPick-Clean-v5",
        "evaluation": {
            "deterministic_policy": True,
            "full_environment_randomization": True,
            "num_envs": args.num_envs,
            "steps_per_seed": args.steps,
            "seeds": args.seeds,
        },
        "source_training_run": str(run_dir),
        "candidate_summaries": summaries,
        "ranking": [item["iteration"] for item in ranking],
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "best_summary": best,
        "wandb_run_id": args.wandb_run_id,
        "wandb_url": args.wandb_url,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Tomato Clean-v5 fixed evaluation",
        "",
        f"Best checkpoint: `{best_path}`",
        f"Success: {best['success_rate']:.2%} "
        f"(95% Wilson CI {best['success_wilson_95_low']:.2%}–{best['success_wilson_95_high']:.2%})",
        f"Drop: {best['drop_rate']:.2%}",
        f"Episodes: {best['completed_episodes']} across seeds {args.seeds}",
        "",
        "| iteration | success | drop | detach | retained | episodes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(summaries, key=lambda value: value["iteration"]):
        lines.append(
            f"| {item['iteration']} | {item['success_rate']:.2%} | {item['drop_rate']:.2%} | "
            f"{item['detach_rate']:.2%} | {item['retained_rate']:.2%} | {item['completed_episodes']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("TOMATO_EVALUATION_MANIFEST=" + json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
