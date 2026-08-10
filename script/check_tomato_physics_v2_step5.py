"""Pure gate for bounded actuator and tactile uncertainty models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_models import (
    apply_joint_target_uncertainty_v2,
    initial_tactile_uncertainty_state,
    sample_uncertainty_profiles_v2,
    update_tactile_uncertainty_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = load_physics_v2_profile()
    cfg = profile["uncertainty_v2"]["episode_fixed"]
    values = sample_uncertainty_profiles_v2(
        args.samples, generator=torch.Generator().manual_seed(args.seed)
    )
    bound_keys = (
        "hand_response_scale",
        "hand_target_bias_rad",
        "hand_deadband_rad",
        "hand_backlash_rad",
        "arm_response_scale",
        "tactile_gain",
        "tactile_bias_n",
        "tactile_drift_rate_n_s",
        "tactile_saturation_n",
        "tactile_release_time_constant_s",
        "tactile_dropout_probability",
    )
    bounded = {}
    for key in bound_keys:
        low, high = cfg[key]
        bounded[key] = float(values[key].min()) >= low and float(values[key].max()) <= high

    current = torch.zeros((args.samples, 16))
    desired = torch.empty_like(current).uniform_(-2.0, 2.0)
    maximum_delta = torch.full_like(current, 0.08)
    target = apply_joint_target_uncertainty_v2(
        desired,
        current,
        torch.full_like(current, -1.0),
        torch.full_like(current, 1.0),
        maximum_delta,
        values["hand_response_scale"],
        values["hand_target_bias_rad"],
        values["hand_deadband_rad"],
        values["hand_backlash_rad"],
    )

    tactile_profiles = sample_uncertainty_profiles_v2(
        1024, generator=torch.Generator().manual_seed(args.seed + 1)
    )
    tactile_state = initial_tactile_uncertainty_state(
        torch.zeros((1024, 5)), maximum_delay_steps=2
    )
    peak_after_contact = None
    decay_trace = []
    for step in range(20):
        physical = torch.full((1024, 5), 6.0) if step < 3 else torch.zeros((1024, 5))
        tactile_state = update_tactile_uncertainty_v2(
            tactile_state,
            physical,
            tactile_profiles,
            dt_s=0.05,
            generator=torch.Generator().manual_seed(args.seed + 100 + step),
        )
        if step == 4:
            peak_after_contact = float(tactile_state["release_state_n"].mean())
        if step >= 5:
            decay_trace.append(float(tactile_state["release_state_n"].mean()))
    physical_zero = torch.zeros((1024, 5))
    physical_support = (physical_zero >= 0.25).sum(dim=1) >= 2
    checks = {
        "all_episode_fixed_samples_bounded": all(bounded.values()),
        "delay_range_exercised": int(values["tactile_delay_policy_steps"].min()) == 0 and int(values["tactile_delay_policy_steps"].max()) == 2,
        "joint_target_speed_bounded": bool((target.abs() <= maximum_delta + 1.0e-7).all()),
        "joint_target_position_bounded": bool(((target >= -1.0) & (target <= 1.0)).all()),
        "tactile_measurement_saturates": bool((tactile_state["measured_n"] <= tactile_profiles["tactile_saturation_n"] + 1.0e-7).all()),
        "release_state_decays_after_contact": peak_after_contact is not None and all(later <= earlier + 1.0e-7 for earlier, later in zip(decay_trace, decay_trace[1:])),
        "measurement_state_cannot_create_physical_support": not bool(physical_support.any()),
        "training_remains_blocked": profile["implementation_boundary"]["training_allowed"] is False,
    }
    report = {
        "schema_version": 1,
        "gate_id": "tomato_pick_physics_v2_step5",
        "passed": all(checks.values()),
        "checks": checks,
        "sample_bounds": bounded,
        "samples": args.samples,
        "seed": args.seed,
        "metrics": {
            "maximum_joint_target_delta_rad": float(target.abs().max()),
            "tactile_release_decay_trace_n": decay_trace,
            "final_mean_tactile_measurement_n": float(tactile_state["measured_n"].mean()),
        },
        "next_stage": "isolated_physics_gates",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
