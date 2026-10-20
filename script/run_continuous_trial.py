"""One bounded v7 PPO trial and fixed independent evaluations; no core changes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "simulation/.runtime/run_roto2.sh"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--million-steps", type=float, default=32)
    parser.add_argument("--seed", type=int, default=48)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.million_steps <= 0:
        parser.error("Existing checkpoint and positive budget required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(started_at=datetime.now().astimezone().isoformat(),
                 core_architecture_changed=False, task_acceptance_passed=False,
                 reward_version="pose_pad_v7_continuous", training_seed=args.seed,
                 num_envs=16384, budget_million=args.million_steps,
                 initial_checkpoint=str(args.checkpoint.resolve()), evaluations=[])

    def save(status, **values):
        state.update(status=status, updated_at=datetime.now().astimezone().isoformat(), **values)
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "status.json")
        print(json.dumps(dict(status=status, **values)), flush=True)

    def run(name, command):
        save(name, current_log=str(output / f"{name}.log"))
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        content = (output / f"{name}.log").read_text(errors="replace")
        for bad in ("PhysX ABORT", "Scene state is corrupted", "failed to allocate", "Traceback (most recent call last)"):
            if bad in content:
                raise RuntimeError(f"{name}: {bad}")

    stopped = threading.Event()

    def monitor():
        with (output / "gpu_samples.jsonl").open("w") as log:
            while not stopped.is_set():
                result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True)
                log.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),
                                          gpu_csv=result.stdout.strip())) + "\n")
                log.flush()
                stopped.wait(10)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    common = ["--hand-posture", "palm_up", "--domain-profile", "nominal",
              "--reward-stage", "continuous", "--goal-direction", "positive",
              "--reset-grasps", str(ROOT / "simulation/roto2_pressure/reset_pad_grasps_palm_up.json")]
    try:
        run_root = ROOT / f"logs/roto2_pressure_fruit/jacobian_seed{args.seed}"
        before = set(run_root.glob("*/training_result.json"))
        run("training", ["bash", str(WRAPPER), "train", "--seed", str(args.seed),
                        "--million-steps", str(args.million_steps), "--entropy-weight", ".005",
                        "--resume-checkpoint", str(args.checkpoint.resolve()), "--validate-contracts", *common])
        added = set(run_root.glob("*/training_result.json")) - before
        if len(added) != 1:
            raise RuntimeError(f"Expected one new training result, found {added}")
        result_path = added.pop()
        result = json.loads(result_path.read_text())
        if not (result["training_completed"] and result["finite_parameters"]
                and all(value > 0 for value in result["parameter_change_l2"].values())):
            raise RuntimeError("Training completion/finite/update check failed")
        checkpoint = result["final_checkpoint"]
        save("training_completed", training_result=str(result_path), latest_checkpoint=checkpoint)
        for seed in (2071, 2072):
            destination = output / f"nominal_seed{seed}.json"
            run(f"evaluate_seed{seed}", ["bash", str(WRAPPER), "evaluate", "--seed", str(seed),
                "--evaluate-checkpoint", checkpoint, "--evaluation-output", str(destination), *common])
            evaluation = json.loads(destination.read_text())
            if evaluation["episodes"] != 64 or evaluation["pose_metric_max_error_turns"] > 1e-4:
                raise RuntimeError("Evaluation sample count or independent pose check failed")
            state["evaluations"].append(str(destination))
        # Recorder writes a same-stem JSON sidecar; keep it distinct from
        # the independent evaluation's 64-episode JSON.
        video = output / "nominal_seed2071_video.mp4"
        run("recording", ["bash", str(WRAPPER), "record", "--seed", "2071",
                         "--evaluate-checkpoint", checkpoint, "--record-video", str(video), *common])
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
        save("completed_pending_behavior_review", latest_video=str(video), training_active=False)
    except Exception as exc:
        save("failed", error=str(exc), training_active=False)
        raise
    finally:
        stopped.set()
        worker.join(timeout=15)


if __name__ == "__main__":
    main()
