"""Evaluate a completed stage1 run and record representative real rollouts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_video_artifact(video, checkpoint, seed, env_index):
    data = json.loads(video.with_suffix(".json").read_text())
    assert Path(data["checkpoint"]).resolve() == checkpoint.resolve()
    assert data["seed"] == seed and data["env_index"] == env_index
    assert data["episode_completed"] and data["real_time_playback"]
    assert len(data["trace"]) == data["physics_steps_recorded"]
    assert data["trace"][-1]["ended"]
    assert abs(data["directed_turns"] - data["independent_pose_turns"]) < 1e-4
    for index, row in enumerate(data["trace"], 1):
        assert row["step"] == index
        assert all(math.isfinite(x) for x in row["fruit_pose_native"])
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
                   check=True, stdout=subprocess.DEVNULL)
    return {"episode_and_finite_pose_trace_verified": True,
            "independent_pose_metric_verified": True, "video_fully_decoded": True}


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
        assert result.get("cumulative_training_transitions", result["training_transitions"]) >= 299_000_000, result
        assert result.get("cumulative_ppo_updates", result["ppo_updates"]) >= 2200, result
        cfg = json.loads((run / "run_config.json").read_text())
        assert cfg["pressure_variant"] == "plain" and cfg["reward_stage"] == "baseline"
        ancestor, lineage, seen = cfg, [str(run)], {str(run)}
        while ancestor["resumed_from"] is not None:
            parent = Path(ancestor["resumed_from"]).resolve().parent.parent
            assert str(parent) not in seen, "Checkpoint lineage cycle"
            seen.add(str(parent))
            ancestor = json.loads((parent / "run_config.json").read_text())
            assert ancestor["pressure_variant"] == "plain" and ancestor["reward_stage"] == "baseline"
            assert ancestor["reset_grasps_sha256"] == cfg["reset_grasps_sha256"]
            lineage.append(str(parent))
        assert ancestor["exploration_std_override"] == .6, "Expected a fresh0.6-std ancestor"
        status["verified_fresh_training_lineage"] = lineage
        library = Path(cfg["reset_grasps_file"])
        assert hashlib.sha256(library.read_bytes()).hexdigest() == cfg["reset_grasps_sha256"]
        checkpoint = Path(result["final_checkpoint"])
        status.update(status="evaluating", training_result=result,
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        write(status_file, status)
        common = ["--num-envs", "64", "--pressure-variant", "plain",
                  "--domain-profile", cfg["domain_profile"], "--hand-posture", cfg["hand_posture"],
                  "--reward-stage", "baseline", "--goal-direction", cfg["goal_direction"],
                  "--baseline-pose-weight", str(cfg.get("baseline_pose_weight", .01)),
                  "--baseline-approach-weight", str(cfg.get("baseline_approach_weight", 0.)),
                  "--baseline-approach-pad-count", str(cfg.get("baseline_approach_pad_count", 4)),
                  "--baseline-motion-weight", str(cfg.get("baseline_motion_weight", 1.)),
                  "--baseline-centering-weight", str(cfg.get("baseline_centering_weight", .02)),
                  "--baseline-failure-penalty", str(cfg.get("baseline_failure_penalty", 60.)),
                  "--baseline-pad-quality-weight", str(cfg.get("baseline_pad_quality_weight", 0.)),
                  "--baseline-offpad-weight", str(cfg.get("baseline_offpad_weight", 0.)),
                  "--baseline-pressure-weight", str(cfg.get("baseline_pressure_weight", 0.)),
                  "--baseline-contact-motion-floor", str(cfg.get("baseline_contact_motion_floor", 1.)),
                  "--reset-grasps", str(library), "--evaluate-checkpoint", str(checkpoint)]
        commands = []

        def execute(mode, args, name):
            cmd = ["bash", "simulation/.runtime/run_roto2.sh", mode] + common + args
            commands.append({"command": cmd, "log": str(output / (name + ".log"))})
            write(output / "COMMANDS.json", commands)
            started = time.time()
            with (output / (name + ".log")).open("w") as log:
                process = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            commands[-1].update(returncode=process.returncode, started_unix=started)
            write(output / "COMMANDS.json", commands)
            allowed = (0, 143, -15) if mode == "record" else (0,)
            if process.returncode not in allowed:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            if process.returncode != 0:
                # Some camera-enabled runs exit by SIGTERM after writing the
                # full episode and clearing SimulationContext. Retain that
                # nonzero status; accept artifacts only after verification.
                assert "SimulationContext cleared" in (output / (name + ".log")).read_text()
            return started

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
            started = execute("record", ["--seed", "6101", "--video-env-index", str(row["env"]),
                                         "--record-video", str(video)], video.stem)
            assert video.stat().st_mtime >= started
            assert video.with_suffix(".json").stat().st_mtime >= started
            commands[-1]["artifact_verification"] = verify_video_artifact(video, checkpoint, 6101, row["env"])
            write(output / "COMMANDS.json", commands)
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
