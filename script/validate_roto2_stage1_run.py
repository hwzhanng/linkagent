"""Evaluate a completed stage1 run and record representative real rollouts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-pid", type=int)
    a = p.parse_args()
    run = a.run.resolve()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / "VALIDATION_PIPELINE.json"
    status = {"training_run": str(run), "status": "waiting_for_training",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "independent_seeds": [6101, 6102], "episodes_per_seed": 64}
    write(status_file, status)
    try:
        if a.wait_pid:
            while Path(f"/proc/{a.wait_pid}").exists():
                time.sleep(30)
        result = json.loads((run / "training_result.json").read_text())
        assert result["training_completed"] and result["finite_parameters"], result
        assert result["training_transitions"] >= 299_000_000 and result["ppo_updates"] >= 2200, result
        cfg = json.loads((run / "run_config.json").read_text())
        assert cfg["pressure_variant"] == "plain" and cfg["reward_stage"] == "baseline"
        assert cfg["resumed_from"] is None
        library = Path(cfg["reset_grasps_file"])
        assert hashlib.sha256(library.read_bytes()).hexdigest() == cfg["reset_grasps_sha256"]
        checkpoint = Path(result["final_checkpoint"])
        status.update(status="evaluating", training_result=result,
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        write(status_file, status)
        common = ["--num-envs", "64", "--pressure-variant", "plain",
                  "--domain-profile", cfg["domain_profile"], "--hand-posture", cfg["hand_posture"],
                  "--reward-stage", "baseline", "--goal-direction", cfg["goal_direction"],
                  "--reset-grasps", str(library), "--evaluate-checkpoint", str(checkpoint)]
        commands = []

        def execute(mode, args, name):
            cmd = ["bash", "simulation/.runtime/run_roto2.sh", mode] + common + args
            commands.append({"command": cmd, "log": str(output / (name + ".log"))})
            write(output / "COMMANDS.json", commands)
            with (output / (name + ".log")).open("w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

        evaluations = []
        for seed in status["independent_seeds"]:
            path = output / f"final_seed{seed}.json"
            execute("evaluate", ["--seed", str(seed), "--evaluation-output", str(path)],
                    f"final_seed{seed}")
            d = json.loads(path.read_text())
            assert d["simulation_physx_rollout_completed"] and d["episodes"] == 64
            assert d["pose_metric_max_error_turns"] <= 1e-4
            evaluations.append(d)
        gate_rows = [{
            "seed": seed, "episodes": d["episodes"],
            "full_turn_successes": sum(r["success"] for r in d["per_episode"]),
            "mean_after_10s_degrees": d["mean_after_10s_directed_turns"] * 360,
            "mean_total_turns": d["mean_directed_rotations"],
            "failure_fraction": d["failure_fraction"],
            "pass": d["mean_after_10s_directed_turns"] > .25 and d["success_fraction"] > 0,
        } for seed, d in zip(status["independent_seeds"], evaluations)]
        write(output / "QUANTITATIVE_GATE.json", {
            "both_independent_seeds_pass": all(r["pass"] for r in gate_rows),
            "rows": gate_rows,
            "definition": "Each seed: mean net progress after10s >90deg and at least one safe true full revolution.",
            "visual_review_pending": True,
            "stage1_accepted": False,
            "note": "Videos require actual visual review; later tactile-constraint and three-variant stages remain open.",
        })
        status["status"] = "recording"
        write(status_file, status)
        rows = sorted(evaluations[0]["per_episode"], key=lambda r: r["after_10s_directed_turns"])
        selected = [("median", rows[len(rows) // 2]),
                    ("best", max(rows, key=lambda r: (r["success"], r["after_10s_directed_turns"])))]
        for label, row in selected:
            video = output / f"final_seed6101_{label}_env{row['env']}.mp4"
            execute("record", ["--seed", "6101", "--video-env-index", str(row["env"]),
                               "--record-video", str(video)], video.stem)
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
                           check=True, stdout=subprocess.DEVNULL)
        status.update(status="completed_pending_visual_review", gate_rows=gate_rows,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write(status_file, status)
    except BaseException as e:
        status.update(status="failed", error=f"{type(e).__name__}: {e}",
                      failed_utc=datetime.now(timezone.utc).isoformat())
        write(status_file, status)
        raise


if __name__ == "__main__":
    main()
