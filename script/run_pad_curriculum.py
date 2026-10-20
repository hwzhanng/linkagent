"""Bounded local training continuation, with independent gates between stages.

This orchestrates the existing PPO entrypoint. It does not implement or alter
the policy, optimizer, environment dynamics, or training algorithm.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "simulation/.runtime/run_roto2.sh"
FATAL = ("PhysX ABORT", "Scene state is corrupted", "failed to allocate", "Traceback (most recent call last)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-million", type=float, default=96)
    parser.add_argument("--rotation-million", type=float, default=128)
    parser.add_argument("--hand-posture", choices=("vertical", "palm_up"), default="vertical")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--reset-grasps", type=Path)
    parser.add_argument("--rotation-curriculum", action="store_true",
                        help="Use v6 one-direction regrasp then bidirectional spin; budgets apply to these stages")
    args = parser.parse_args()
    if min(args.contact_million, args.rotation_million) <= 0:
        parser.error("Training budgets must be positive")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error("Missing starting checkpoint")
    reset_args = []
    if args.reset_grasps:
        if not args.reset_grasps.is_file():
            parser.error("Missing reset-grasps file")
        reset_args = ["--reset-grasps", str(args.reset_grasps.resolve())]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    first_stage, second_stage = ("regrasp", "spin") if args.rotation_curriculum else ("contact", "rotation")
    state = {
        "started_at": datetime.now().astimezone().isoformat(),
        "core_architecture_changed": False,
        "task_acceptance_passed": False,
        "initial_checkpoint": str(checkpoint),
        "num_envs": 16384,
        "hand_posture": args.hand_posture,
        "seed": args.seed,
        "reset_grasps_file": str(args.reset_grasps.resolve()) if args.reset_grasps else None,
        "entropy_weight": 0.005,
        "exploration_std": "restored from checkpoint",
        "contact_budget_million": args.contact_million,
        "rotation_budget_million": args.rotation_million,
        "first_stage": first_stage, "second_stage": second_stage,
        "first_stage_gate": ("both independent 64-episode evaluations: >=50% pass after-settling 30deg curriculum, mean strict pad support >=50%, physical failure <=10%; NOT full-task acceptance"
                             if args.rotation_curriculum else "both independent 64-episode nominal evaluations: pad support >=65%, physical failure <=10%"),
        "completed_runs": [],
        "evaluations": [],
    }

    def save(status, **updates):
        state.update(status=status, updated_at=datetime.now().astimezone().isoformat(), **updates)
        tmp = output / "status.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(output / "status.json")
        print(json.dumps({"status": status, **updates}), flush=True)

    def run(name, command):
        save(name, current_log=str(output / (name + ".log")))
        with (output / (name + ".log")).open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        content = (output / (name + ".log")).read_text(errors="replace")
        found = [token for token in FATAL if token in content]
        if found:
            raise RuntimeError(f"Simulator error in {name}: {found}")

    def train(stage, budget, source):
        runs = ROOT / f"logs/roto2_pressure_fruit/jacobian_seed{args.seed}"
        before = set(runs.glob("*/training_result.json"))
        run(stage + "_training", ["bash", str(WRAPPER), "train", "--seed", str(args.seed),
            "--hand-posture", args.hand_posture,
            "--million-steps", str(budget), "--domain-profile", "nominal",
            "--reward-stage", stage, "--entropy-weight", "0.005",
            "--goal-direction", "positive" if stage == "regrasp" else "both",
            "--resume-checkpoint", str(source), "--validate-contracts", *reset_args])
        created = set(runs.glob("*/training_result.json")) - before
        if len(created) != 1:
            raise RuntimeError(f"Expected one completed training run, found {created}")
        result_file = created.pop()
        result = json.loads(result_file.read_text())
        if not result["training_completed"] or not result["finite_parameters"]:
            raise RuntimeError("Training completion/finite-parameter gate failed")
        if not all(v > 0 for v in result["parameter_change_l2"].values()):
            raise RuntimeError("A trainable module did not update")
        state["completed_runs"].append(str(result_file))
        return Path(result["final_checkpoint"])

    def evaluate(stage, source, seed, profile="nominal"):
        name = f"{stage}_{profile}_seed{seed}"
        destination = output / (name + ".json")
        run(name, ["bash", str(WRAPPER), "evaluate", "--seed", str(seed),
            "--hand-posture", args.hand_posture,
            "--domain-profile", profile, "--reward-stage", stage,
            "--goal-direction", "positive" if stage == "regrasp" else "both",
            "--evaluate-checkpoint", str(source), "--evaluation-output", str(destination), *reset_args])
        result = json.loads(destination.read_text())
        if result["episodes"] != 64 or result["pose_metric_max_error_turns"] > 1e-4:
            raise RuntimeError("Independent evaluation evidence gate failed")
        state["evaluations"].append(str(destination))
        return result

    def record(stage, source):
        video_seed = (2061 if stage == "regrasp" else 2063) if args.rotation_curriculum else 2036
        video = output / f"{stage}_seed{video_seed}.mp4"
        run(stage + "_video", ["bash", str(WRAPPER), "record", "--seed", str(video_seed),
            "--hand-posture", args.hand_posture,
            "--domain-profile", "nominal", "--reward-stage", stage,
            "--goal-direction", "positive" if stage == "regrasp" else "both",
            "--evaluate-checkpoint", str(source), "--record-video", str(video), *reset_args])
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
        state["latest_video"] = str(video)

    save("starting")
    try:
        checkpoint = train(first_stage, args.contact_million, checkpoint)
        contact = [evaluate(first_stage, checkpoint, seed) for seed in
                   ((2061, 2062) if args.rotation_curriculum else (2036, 2037))]
        if args.rotation_curriculum:
            gate = all(x["small_angle_curriculum_success_fraction"] >= .5 and
                       x["mean_secure_support_fraction"] >= .5 and x["failure_fraction"] <= .10 for x in contact)
        else:
            gate = all(x["mean_secure_support_fraction"] >= .65 and x["failure_fraction"] <= .10 for x in contact)
        save(first_stage+"_evaluated", latest_checkpoint=str(checkpoint), first_stage_gate_passed=gate)
        record(first_stage, checkpoint)
        if not gate:
            save(first_stage+"_gate_failed", conclusion=f"{first_stage} curriculum is not accepted; {second_stage} stage was not started.")
            return 2
        checkpoint = train(second_stage, args.rotation_million, checkpoint)
        seeds = (2063,2064,2065,2066) if args.rotation_curriculum else (2038,2039,2040,2041)
        rotation = [evaluate(second_stage, checkpoint, seed) for seed in seeds[:2]]
        evaluate(second_stage, checkpoint, seeds[2], "gentle")
        evaluate(second_stage, checkpoint, seeds[3], "full")
        record(second_stage, checkpoint)
        save("completed_pending_behavior_review", latest_checkpoint=str(checkpoint),
             nominal_success_fractions=[x["success_fraction"] for x in rotation],
             conclusion="Review independent pose, contact and video evidence; completion alone is not task acceptance.")
        return 0
    except Exception as exc:
        save("failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
