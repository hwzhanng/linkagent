#!/usr/bin/env python3
"""Fuzz the future Isaac v1 adapter contract without importing Isaac or CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_ACTIVE_JOINT_ORDER,
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    LEFT_SAFE_VELOCITY_RAD_S,
)
from simulation.ppo_tomato_grasp.isaac_v1_adapter_contract import (
    ISAAC_V1_ADAPTER_ID,
    action_to_active_joint_command,
    canonical_joint_state,
    pack_deployable_actor_si,
    pack_physx_critic_si,
)


def audit(samples: int, seed: int) -> dict[str, object]:
    if samples < 32:
        raise ValueError("samples must be >= 32")
    generator = torch.Generator().manual_seed(seed)
    lower = torch.tensor(LEFT_SAFE_LOWER_RAD)
    upper = torch.tensor(LEFT_SAFE_UPPER_RAD)
    velocity = torch.tensor(LEFT_SAFE_VELOCITY_RAD_S)
    current = lower + torch.rand((samples, 16), generator=generator) * (upper - lower)
    action = 2.0 * torch.rand((samples, 10), generator=generator) - 1.0
    names = ("left_wrist_extra", *reversed(LEFT_ACTIVE_JOINT_ORDER))
    articulation = torch.zeros((samples, len(names)))
    for canonical_index, name in enumerate(LEFT_ACTIVE_JOINT_ORDER):
        articulation[:, names.index(name)] = current[:, canonical_index]
    extracted = canonical_joint_state(names, articulation)
    command = action_to_active_joint_command(action, extracted, names)
    batch = samples
    actor = pack_deployable_actor_si(
        fruit_center_in_palm_m=(torch.rand((batch, 3), generator=generator) - 0.5) * 0.08,
        fruit_semi_axes_m=0.025 + torch.rand((batch, 3), generator=generator) * 0.015,
        fruit_orientation_6d=torch.tensor((1, 0, 0, 0, 1, 0)).float().expand(batch, -1),
        fruit_twist_in_palm_si=(torch.rand((batch, 6), generator=generator) - 0.5) * 0.2,
        canonical_joint_position_rad=current,
        canonical_joint_velocity_rad_s=(torch.rand((batch, 16), generator=generator) - 0.5) * 2.0 * velocity,
        tactile_matrix_features=2.0 * torch.rand((batch, 50), generator=generator) - 1.0,
        previous_action=action,
    )
    critic = pack_physx_critic_si(
        actor,
        physical_fingertip_force_n=torch.rand((batch, 5), generator=generator) * 8.0,
        physical_palm_force_n=torch.rand(batch, generator=generator) * 8.0,
        contact_points_in_palm_m=(torch.rand((batch, 5, 3), generator=generator) - 0.5) * 0.08,
        object_mass_kg=0.05 + torch.rand(batch, generator=generator) * 0.20,
        static_friction=0.25 + torch.rand(batch, generator=generator) * 0.65,
        effective_stiffness_n_m=2500.0 + torch.rand(batch, generator=generator) * 6500.0,
        peak_indentation_m=torch.rand(batch, generator=generator) * 0.006,
        mean_force_exposure_n_s=torch.rand(batch, generator=generator),
        mean_impact_exposure_n=torch.rand(batch, generator=generator),
        damage_score=torch.rand(batch, generator=generator),
        tactile_latent_gain=0.82 + torch.rand((batch, 5), generator=generator) * 0.36,
        tactile_latent_bias_normalized=(torch.rand((batch, 5), generator=generator) - 0.5) * 0.06,
    )
    delta = (command.canonical_targets_rad - current).abs()
    maximum_delta = torch.minimum(velocity * 0.5, torch.ones_like(velocity)) * 0.05
    checks = {
        "name_round_trip_exact": bool(torch.equal(extracted, current)),
        "actor_shape_exact": tuple(actor.shape) == (samples, 110),
        "critic_shape_exact": tuple(critic.shape) == (samples, 148),
        "critic_actor_prefix_exact": bool(torch.equal(critic[:, :110], actor)),
        "all_finite": bool(
            torch.isfinite(actor).all()
            and torch.isfinite(critic).all()
            and torch.isfinite(command.targets_rad).all()
        ),
        "command_inside_safe_ranges": bool(
            (command.canonical_targets_rad >= lower).all()
            and (command.canonical_targets_rad <= upper).all()
        ),
        "command_inside_per_step_velocity": bool((delta <= maximum_delta + 1.0e-7).all()),
        "extra_wrist_not_commanded": names.index("left_wrist_extra")
        not in command.articulation_indices.tolist(),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "status": "isaac_v1_cpu_contract_passed_physx_binding_blocked" if passed else "failed",
        "adapter_id": ISAAC_V1_ADAPTER_ID,
        "samples": samples,
        "seed": seed,
        "checks": checks,
        "passed_cpu_scope": passed,
        "isaac_imported": False,
        "cuda_used": False,
        "hardware_command_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20270825)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed_cpu_scope"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
