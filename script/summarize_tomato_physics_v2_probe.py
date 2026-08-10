"""Validate and summarize the fixed 256-env x 4-update Physics-v2 PPO probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = Path("/tmp/tomato_physics_v2_ppo_256x4.log")
DEFAULT_OUTPUT = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/ppo_256x4_probe.json"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(block: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s+([-+0-9.eE]+)", block)
    if match is None:
        raise RuntimeError(f"missing numerical-probe metric: {label}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    text = ANSI.sub("", args.log.read_text(encoding="utf-8", errors="replace"))
    required = (
        "BOUNDARY=no_hardware_no_formal_training_256x4_only",
        "Number of environments: 256",
        "Training time:",
        "SimulationContext cleared",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"probe did not finish its fixed contract; missing {missing}")
    forbidden = re.search(
        r"Traceback|CUDA out of memory|PhysX[^\n]*overflow|\b(?:nan|inf)\b",
        text,
        flags=re.IGNORECASE,
    )
    if forbidden is not None:
        raise RuntimeError(f"probe log contains error token: {forbidden.group(0)}")

    matches = list(re.finditer(r"Learning iteration (\d+)/4", text))
    if [int(match.group(1)) for match in matches] != [0, 1, 2, 3]:
        raise RuntimeError("probe does not contain exactly learning iterations 0..3")
    iterations = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start():stop]
        values = {
            "iteration": index,
            "steps_per_second": _metric(block, "Steps per second"),
            "value_loss": _metric(block, "Mean value loss"),
            "surrogate_loss": _metric(block, "Mean surrogate loss"),
            "entropy_loss": _metric(block, "Mean entropy loss"),
            "mean_reward": _metric(block, "Mean reward"),
            "mean_action_std": _metric(block, "Mean action std"),
        }
        if not all(math.isfinite(v) for key, v in values.items() if key != "iteration"):
            raise RuntimeError(f"non-finite iteration metrics: {values}")
        iterations.append(values)

    run_match = re.search(r"Logging experiment in directory:\s*(.+)", text)
    exact_match = re.search(r"Exact experiment name requested from command line:\s*(\S+)", text)
    if run_match is None or exact_match is None:
        raise RuntimeError("unable to identify the exact local run directory")
    run = Path(run_match.group(1).strip()) / (
        exact_match.group(1) + "_256x4_seed42_not_formal_training"
    )
    checkpoint = run / "model_3.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000:
        raise RuntimeError(f"final numerical checkpoint is missing or truncated: {checkpoint}")
    urls = re.findall(r"https://wandb\.ai/[^\s\x1b]+", text)
    wandb_url = next((url.rstrip(".,") for url in reversed(urls) if "/runs/" in url), None)

    result = {
        "schema_version": 1,
        "task_id": "LinkAgent-TomatoPick-Physics-v2",
        "qualification": "numerical_probe_only_not_formal_training",
        "num_envs": 256,
        "num_steps_per_env": 32,
        "update_count": 4,
        "total_transitions": 32768,
        "seed": 42,
        "iterations": iterations,
        "run_directory": str(run.resolve()),
        "final_checkpoint": str(checkpoint.resolve()),
        "final_checkpoint_sha256": _sha256(checkpoint),
        "wandb_run_url": wandb_url,
        "log": str(args.log.resolve()),
        "checks": {
            "completed_exactly_four_updates": True,
            "losses_rewards_and_action_std_are_finite": True,
            "no_traceback_nan_inf_overflow_or_oom": True,
            "final_checkpoint_is_nonempty": True,
            "hardware_was_out_of_scope": True,
        },
        "passed": True,
        "formal_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("TOMATO_PHYSICS_V2_PROBE=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
