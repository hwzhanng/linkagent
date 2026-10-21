"""Matched checkpoint diagnostics; never modifies or stops the training job."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--training-pid", type=int, required=True)
    p.add_argument("--seed", type=int, default=5101)
    a = p.parse_args()
    run, output = a.run.resolve(), a.snapshot.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / "DIAGNOSTIC_PIPELINE.json"
    status = {"status": "waiting_for_snapshot", "diagnostic_only": True,
              "training_run": str(run), "seed": a.seed, "commands": []}

    def save():
        status_file.write_text(json.dumps(status, indent=2) + "\n")

    save()
    try:
        provenance = output / "PROVENANCE.json"
        while not provenance.exists():
            if not Path(f"/proc/{a.training_pid}").exists():
                raise RuntimeError("Training process ended before snapshot was saved")
            time.sleep(15)
        # Snapshot writer publishes this small JSON after its stable copy.
        for attempt in range(3):
            try:
                info = json.loads(provenance.read_text())
                break
            except json.JSONDecodeError:
                if attempt == 2:
                    raise
                time.sleep(1)
        checkpoint = output / "agent.pt"
        assert Path(info["checkpoint"]).resolve().parent.parent == run
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == info["snapshot_sha256"]
        cfg = json.loads((run / "run_config.json").read_text())
        common = ["bash", "simulation/.runtime/run_roto2.sh", "evaluate",
                  "--num-envs", "64", "--seed", str(a.seed),
                  "--pressure-variant", cfg["pressure_variant"],
                  "--domain-profile", cfg["domain_profile"],
                  "--hand-posture", cfg["hand_posture"],
                  "--reward-stage", cfg["reward_stage"],
                  "--baseline-pose-weight", str(cfg.get("baseline_pose_weight", .01)),
                  "--baseline-approach-weight", str(cfg.get("baseline_approach_weight", 0.)),
                  "--baseline-approach-pad-count", str(cfg.get("baseline_approach_pad_count", 4)),
                  "--goal-direction", cfg["goal_direction"],
                  "--reset-grasps", cfg["reset_grasps_file"],
                  "--evaluate-checkpoint", str(checkpoint)]
        status.update(status="evaluating", snapshot=info)
        save()

        def execute(cmd, log):
            row = {"command": cmd, "log": str(log)}
            status["commands"].append(row)
            save()
            with log.open("w") as f:
                result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            row["returncode"] = result.returncode
            save()
            result.check_returncode()

        evaluation = output / f"evaluation_seed{a.seed}.json"
        trace = output / f"action_trace_seed{a.seed}.json"
        execute(common + ["--evaluation-output", str(evaluation)], evaluation.with_suffix(".log"))
        execute(common + ["--trace-policy-output", str(trace)], trace.with_suffix(".log"))
        runtime = ROOT / "simulation/.runtime/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python"
        execute([str(runtime), "-m", "script.plot_roto2_policy_diagnostic",
                 "--evaluation", str(evaluation), "--trace", str(trace),
                 "--output", str(output / "diagnostic.png"),
                 "--label", f"Pose cost{cfg.get('baseline_pose_weight', .01):.2f}, "
                            f"{info['training_transitions'] / 1e6:.2f}M samples, seed{a.seed}"],
                output / "plot.log")
        d = json.loads(evaluation.read_text())
        assert d["simulation_physx_rollout_completed"] and d["episodes"] == 64
        status.update(status="completed", full_turns=sum(r["success"] for r in d["per_episode"]),
                      mean_total_degrees=d["mean_directed_rotations"] * 360,
                      mean_after10s_degrees=d["mean_after_10s_directed_turns"] * 360,
                      physical_failures=sum(r["failed"] for r in d["per_episode"]),
                      plot_verified_matching_pose_traces=True)
        save()
    except BaseException as e:
        status.update(status="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main()
