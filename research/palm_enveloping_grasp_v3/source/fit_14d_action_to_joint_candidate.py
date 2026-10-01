"""Fit the inherited 14-D manifold to a direct-joint P2 geometry candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from simulation.ppo_tomato_grasp.enveloping_action_manifold import (
    ENVELOPING_ACTION_ORDER,
    enveloping_action_to_joint_targets,
)
from simulation.ppo_tomato_grasp.hand_manifold import LEFT_ACTIVE_JOINT_ORDER
from simulation.ppo_tomato_grasp_v3.provenance import atomic_json_write, sha256_file


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--restarts", type=int, default=512)
parser.add_argument("--iterations", type=int, default=1200)
parser.add_argument("--seed", type=int, default=20260824)
args = parser.parse_args()


def main() -> None:
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = json.loads(source.read_text(encoding="utf-8"))
    target = torch.tensor(report["best_candidate"]["active_joint_position_rad"], dtype=torch.float32)
    generator = torch.Generator().manual_seed(args.seed)
    raw = torch.nn.Parameter(torch.randn((args.restarts, 14), generator=generator))
    optimizer = torch.optim.Adam([raw], lr=0.035)
    # Equal normalized joint error prevents the four narrow spread axes from
    # being ignored, while a modest distal multiplier protects the user's
    # explicit requirement that fingertip joints participate.
    lower = torch.tensor([
        0.020000000, 0.027450980, 0.027450980, 0.027450980, 0.027450980,
        -1.000000000, -0.147843137, -0.058117647, 0.029568627, 0.158039216,
        0.000000000, 0.025294118, 0.021176471, 0.021176471, 0.021176471, 0.021176471,
    ])
    upper = torch.tensor([
        0.800000000, 1.317647059, 1.372549020, 1.372549020, 1.317647059,
        0.450000000, -0.117254902, -0.025490196, 0.062196078, 0.174352941,
        1.380000000, 1.250000000, 1.058823529, 1.058823529, 1.037647059, 1.058823529,
    ])
    span = upper - lower
    weight = torch.ones(16)
    weight[[0, 5, 10, 11]] = 1.4
    weight[12:16] = 1.8
    for _ in range(args.iterations):
        action = torch.tanh(raw)
        mapped = enveloping_action_to_joint_targets(
            action,
            residual_span_fraction=0.12,
            distal_residual_span_fraction=0.50,
        )
        normalized_error = (mapped - target) / span
        per_restart = (weight * normalized_error.square()).mean(-1)
        loss = per_restart.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        action = torch.tanh(raw)
        mapped = enveloping_action_to_joint_targets(
            action,
            residual_span_fraction=0.12,
            distal_residual_span_fraction=0.50,
        )
        error = mapped - target
        normalized = error / span
        objective = (weight * normalized.square()).mean(-1)
        index = int(torch.argmin(objective))
        best_error = error[index]
        best_normalized = normalized[index]
        result = {
            "version": "v3_fit_14d_to_direct_joint_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_command_allowed": False,
            "input": str(source),
            "input_sha256": sha256_file(source),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "seed": args.seed,
            "restarts": args.restarts,
            "iterations": args.iterations,
            "action_order": list(ENVELOPING_ACTION_ORDER),
            "joint_order": [name.removeprefix("left_hand_") for name in LEFT_ACTIVE_JOINT_ORDER],
            "target_joint_position_rad": [float(v) for v in target],
            "best_action": [float(v) for v in action[index]],
            "mapped_joint_position_rad": [float(v) for v in mapped[index]],
            "joint_error_rad": [float(v) for v in best_error],
            "joint_normalized_error": [float(v) for v in best_normalized],
            "weighted_normalized_mse": float(objective[index]),
            "rmse_rad": float(torch.sqrt(best_error.square().mean())),
            "max_abs_error_rad": float(best_error.abs().max()),
            "straight_distal_rmse_rad": float(torch.sqrt(best_error[12:16].square().mean())),
            "thumb_joint_rmse_rad": float(torch.sqrt(best_error[[0, 5, 10, 11]].square().mean())),
            "interpretation": "inverse-fit audit only; dynamic geometry must be re-evaluated",
        }
    atomic_json_write(output, result)
    print(result)


if __name__ == "__main__":
    main()
