#!/usr/bin/env python3
"""Run independent full-range CPU PPO seeds and summarize reproducibility."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.cpu_surrogate_train import CpuPpoConfig, train_cpu_ppo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=(3101, 3102, 3103))
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="optional directory containing seed_<seed>/latest.pt independent parents",
    )
    parser.add_argument("--allow-environment-migration", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation/ppo_tomato_grasp/artifacts/cpu_reproducibility"),
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("seeds must be nonempty and unique")
    if args.updates < 20 or args.num_envs < 32 or args.threads < 1 or args.epochs < 1:
        raise ValueError("campaign settings are too small for a meaningful run")
    if args.allow_environment_migration and args.resume_dir is None:
        raise ValueError("environment migration requires --resume-dir")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    runs: list[dict[str, object]] = []
    environment_contract: dict[str, object] | None = None
    for seed in args.seeds:
        run_dir = output / f"seed_{seed}"
        resume_path = None
        starting_update = 0
        if args.resume_dir is not None:
            resume_path = (args.resume_dir / f"seed_{seed}" / "latest.pt").resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"missing independent parent checkpoint: {resume_path}")
            parent = torch.load(resume_path, map_location="cpu", weights_only=False)
            starting_update = int(parent["update"])
        config = CpuPpoConfig(
            num_envs=args.num_envs,
            horizon=32,
            updates=starting_update + args.updates,
            epochs=args.epochs,
            minibatch_size=min(4096, args.num_envs * 32),
            curriculum_fraction=1.0,
            seed=seed,
            threads=args.threads,
            evaluation_interval=50,
            evaluation_envs=4096,
            initial_std=0.40,
        )
        summary = train_cpu_ppo(
            config,
            output_dir=run_dir,
            resume_path=resume_path,
            allow_environment_migration=args.allow_environment_migration,
        )
        run_contract = summary.get("environment_contract")
        if not isinstance(run_contract, dict):
            raise RuntimeError(f"seed {seed} did not report an environment contract")
        if environment_contract is None:
            environment_contract = run_contract
        elif run_contract != environment_contract:
            raise RuntimeError(
                f"environment contract changed during campaign: {run_contract!r} != "
                f"{environment_contract!r}"
            )
        evaluation = summary["evaluation"]
        runs.append(
            {
                "seed": seed,
                "elapsed_seconds": summary["elapsed_seconds"],
                "success_rate": evaluation["success_rate"],
                "drop_rate": evaluation["drop_rate"],
                "peak_force_mean_n": evaluation["maximum_force_mean_n"],
                "return_mean": evaluation["return_mean"],
                "checkpoint": str((run_dir / "latest.pt").resolve()),
                "parent_checkpoint": str(resume_path) if resume_path is not None else None,
                "starting_update": starting_update,
            }
        )
        partial = {
            "status": "running" if len(runs) < len(args.seeds) else "complete",
            "runs": runs,
        }
        (output / "campaign_partial.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    success = [float(item["success_rate"]) for item in runs]
    drop = [float(item["drop_rate"]) for item in runs]
    force = [float(item["peak_force_mean_n"]) for item in runs]
    report = {
        "schema_version": 1,
        "status": "cpu_surrogate_reproducibility_only_not_physical_evidence",
        "elapsed_seconds": time.monotonic() - started,
        "configuration": {
            "seeds": args.seeds,
            "additional_updates_per_seed": args.updates,
            "epochs": args.epochs,
            "num_envs": args.num_envs,
            "curriculum_fraction": 1.0,
            "initialization": (
                "independent_parent_checkpoint_migration"
                if args.resume_dir is not None
                else "weak_semantic_prior_trainable"
            ),
            "environment_migration_explicitly_allowed": args.allow_environment_migration,
        },
        "environment_contract": environment_contract,
        "runs": runs,
        "aggregate": {
            "success_rate_mean": statistics.mean(success),
            "success_rate_minimum": min(success),
            "success_rate_population_std": statistics.pstdev(success),
            "drop_rate_mean": statistics.mean(drop),
            "drop_rate_maximum": max(drop),
            "peak_force_mean_n": statistics.mean(force),
            "all_finite": all(math.isfinite(value) for value in (*success, *drop, *force)),
        },
    }
    (output / "campaign_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["aggregate"]["all_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
