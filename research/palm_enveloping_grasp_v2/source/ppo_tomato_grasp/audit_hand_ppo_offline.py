#!/usr/bin/env python3
"""Generate the complete CPU-only validation report for hand PPO development.

This audit deliberately avoids Isaac, CUDA, ROS, SDK, network, and hardware
imports.  Passing it proves tensor/interface consistency and surrogate-task
learnability only; it is not evidence of physical safety or tomato quality.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.cpu_surrogate_env import CpuHandGraspEnv
from simulation.ppo_tomato_grasp.domain_randomization import (
    ROBUSTNESS_BUCKET_ORDER,
    pearson_correlation,
    robustness_bucket_masks,
    sample_hand_grasp_profiles,
)
from simulation.ppo_tomato_grasp.hand_grasp_mdp import (
    FINGER_FORCE_CEILING_N,
    SUCCESS_HOLD_STEPS,
    hand_grasp_transition,
    initial_hand_grasp_state,
)
from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    MAPPING_ID,
    hybrid_action_to_joint_targets,
    synergy_to_joint_targets,
)
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTION_LAYOUT_ID,
    ACTION_ORDER,
    ACTOR_OBSERVATION_DIM,
    ACTOR_OBSERVATION_LAYOUT_ID,
    CRITIC_STATE_DIM,
    CRITIC_STATE_LAYOUT_ID,
    INTERFACE_VERSION,
    TACTILE_FEATURE_LAYOUT_ID,
)


def _manifold_audit(samples: int, generator: torch.Generator) -> dict[str, object]:
    action = 2.0 * torch.rand((samples, ACTION_DIM), generator=generator) - 1.0
    coarse = synergy_to_joint_targets(action[:, :5])
    target = hybrid_action_to_joint_targets(action)
    lower = torch.tensor(LEFT_SAFE_LOWER_RAD)
    upper = torch.tensor(LEFT_SAFE_UPPER_RAD)
    base = synergy_to_joint_targets(torch.zeros((1, 5)))
    columns = []
    for coordinate in range(5):
        probe = torch.zeros((1, 5))
        probe[0, coordinate] = 1.0e-3
        columns.append(((synergy_to_joint_targets(probe) - base) / 1.0e-3)[0])
    jacobian = torch.stack(columns, dim=1)
    return {
        "samples": samples,
        "all_finite": bool(torch.isfinite(target).all()),
        "inside_measured_safe_ranges": bool(
            ((target >= lower - 1.0e-7) & (target <= upper + 1.0e-7)).all()
        ),
        "coarse_local_rank": int(torch.linalg.matrix_rank(jacobian)),
        "maximum_coarse_limit_saturation_fraction": float(
            (((coarse - lower).abs() < 1.0e-6) | ((coarse - upper).abs() < 1.0e-6))
            .float()
            .mean(dim=0)
            .max()
        ),
        "maximum_hybrid_limit_saturation_fraction": float(
            (((target - lower).abs() < 1.0e-6) | ((target - upper).abs() < 1.0e-6))
            .float()
            .mean(dim=0)
            .max()
        ),
    }


def _randomization_audit(samples: int, generator: torch.Generator) -> dict[str, object]:
    profiles = sample_hand_grasp_profiles(
        samples, curriculum_fraction=1.0, generator=generator
    )
    masks = robustness_bucket_masks(profiles)
    mean_radius = profiles["axes_m"][:, :2].mean(dim=1)
    result = {
        "samples": samples,
        "all_finite": all(
            bool(torch.isfinite(value).all())
            for value in profiles.values()
            if value.is_floating_point()
        ),
        "bucket_counts": {name: int(masks[name].sum()) for name in ROBUSTNESS_BUCKET_ORDER},
        "correlations": {
            "radius_to_mass": pearson_correlation(mean_radius, profiles["mass_kg"]),
            "ripeness_to_stiffness": pearson_correlation(
                profiles["ripeness"], profiles["contact_stiffness_n_m"]
            ),
            "wetness_to_static_friction": pearson_correlation(
                profiles["wetness"], profiles["static_friction"]
            ),
        },
    }
    result["all_buckets_nonempty"] = all(result["bucket_counts"].values())
    result["correlation_intent_passed"] = (
        result["correlations"]["radius_to_mass"] > 0.80
        and result["correlations"]["ripeness_to_stiffness"] < -0.50
        and result["correlations"]["wetness_to_static_friction"] < -0.40
    )
    return result


def _run_mdp_trace(force_n: tuple[float, ...], steps: int) -> dict[str, object]:
    reference = torch.zeros(1)
    state = initial_hand_grasp_state(reference)
    total_reward = 0.0
    succeeded = False
    terminated = False
    last = None
    for _ in range(steps):
        last = hand_grasp_transition(
            state,
            surface_gap_m=torch.zeros((1, 5)),
            physical_finger_force_n=torch.tensor((force_n,)),
            physical_palm_force_n=torch.zeros(1),
            center_distance_m=torch.full((1,), 0.010),
            relative_speed_m_s=torch.zeros(1),
            slip_detected=torch.zeros(1, dtype=torch.bool),
            dropped=torch.zeros(1, dtype=torch.bool),
            self_collision=torch.zeros(1, dtype=torch.bool),
            action=torch.zeros((1, ACTION_DIM)),
            action_delta=torch.zeros((1, ACTION_DIM)),
            normalized_joint_limit_proximity=torch.zeros(1),
        )
        state = last.state
        total_reward += float(last.reward[0])
        succeeded |= bool(last.success[0])
        terminated |= bool(last.terminated[0])
    assert last is not None
    return {
        "return": total_reward,
        "success": succeeded,
        "terminated": terminated,
        "damage_score": float(last.state.damage_score[0]),
        "hold_streak": int(last.state.hold_streak[0]),
    }


def _reward_audit() -> dict[str, object]:
    traces = {
        "no_contact": _run_mdp_trace((0.0, 0.0, 0.0, 0.0, 0.0), SUCCESS_HOLD_STEPS),
        "partial_contact": _run_mdp_trace((0.5, 0.5, 0.0, 0.0, 0.0), SUCCESS_HOLD_STEPS),
        "unequal_gentle_enclosure": _run_mdp_trace(
            (0.55, 0.42, 0.68, 0.0, 0.0), SUCCESS_HOLD_STEPS
        ),
        "severe_overload": _run_mdp_trace(
            tuple(1.30 * value for value in FINGER_FORCE_CEILING_N), 1
        ),
    }
    traces["ordering_passed"] = (
        traces["unequal_gentle_enclosure"]["return"]
        > traces["partial_contact"]["return"]
        > traces["no_contact"]["return"]
        and traces["unequal_gentle_enclosure"]["success"]
        and traces["severe_overload"]["terminated"]
        and not traces["severe_overload"]["success"]
    )
    return traces


def _scripted_surrogate_audit(envs: int) -> dict[str, object]:
    stages: dict[str, object] = {}
    # Coarse semantic action: global closure plus thumb opposition.  All shape,
    # spread, and per-digit residual coordinates remain neutral.
    scripted = torch.tensor((0.75, 0.0, 0.0, 0.0, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0))
    for index, fraction in enumerate((0.0, 0.35, 0.70, 1.0)):
        env = CpuHandGraspEnv(
            envs,
            seed=3100 + index,
            curriculum_fraction=fraction,
            auto_reset=False,
        )
        env.reset()
        active = torch.ones(envs, dtype=torch.bool)
        success = torch.zeros(envs, dtype=torch.bool)
        drop = torch.zeros(envs, dtype=torch.bool)
        peak_force = torch.zeros(envs)
        for _ in range(env.episode_steps):
            result = env.step(scripted.expand(envs, -1))
            success |= active & result.success
            drop |= active & result.metrics["drop"]
            peak_force = torch.where(
                active, torch.maximum(peak_force, result.metrics["max_force_n"]), peak_force
            )
            active &= ~(result.terminated | result.truncated)
            if not active.any():
                break
        stages[f"fraction_{fraction:.2f}"] = {
            "success_rate": float(success.float().mean()),
            "drop_rate": float(drop.float().mean()),
            "peak_force_mean_n": float(peak_force.mean()),
        }
    stages["reachability_passed"] = stages["fraction_0.00"]["success_rate"] >= 0.95
    stages["randomization_is_nontrivial"] = (
        stages["fraction_1.00"]["success_rate"] < stages["fraction_0.00"]["success_rate"]
    )
    return stages


def build_report(samples: int, envs: int, seed: int) -> dict[str, object]:
    started = time.monotonic()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    manifold = _manifold_audit(samples, generator)
    randomization = _randomization_audit(samples, generator)
    reward = _reward_audit()
    surrogate = _scripted_surrogate_audit(envs)
    passed = (
        manifold["all_finite"]
        and manifold["inside_measured_safe_ranges"]
        and manifold["coarse_local_rank"] == 5
        and randomization["all_finite"]
        and randomization["all_buckets_nonempty"]
        and randomization["correlation_intent_passed"]
        and reward["ordering_passed"]
        and surrogate["reachability_passed"]
        and surrogate["randomization_is_nontrivial"]
    )
    return {
        "schema_version": 1,
        "status": "cpu_offline_validation_only_not_physical_evidence",
        "generated_unix_time": time.time(),
        "elapsed_seconds": time.monotonic() - started,
        "runtime": {
            "torch_version": torch.__version__,
            "device_used": "cpu",
            "cuda_available_report_only": torch.cuda.is_available(),
            "gpu_or_hardware_commands_issued": False,
        },
        "contract": {
            "interface_version": INTERFACE_VERSION,
            "mapping_id": MAPPING_ID,
            "action_layout": ACTION_LAYOUT_ID,
            "action_order": ACTION_ORDER,
            "action_dim": ACTION_DIM,
            "actor_layout": ACTOR_OBSERVATION_LAYOUT_ID,
            "actor_dim": ACTOR_OBSERVATION_DIM,
            "critic_layout": CRITIC_STATE_LAYOUT_ID,
            "critic_dim": CRITIC_STATE_DIM,
            "tactile_layout": TACTILE_FEATURE_LAYOUT_ID,
        },
        "manifold": manifold,
        "domain_randomization": randomization,
        "reward_state_machine": reward,
        "scripted_surrogate": surrogate,
        "passed_cpu_scope": passed,
        "formal_training_allowed": False,
        "blocking_gates": [
            "Isaac/PhysX collision, contact, and mimic-joint validation",
            "URDF versus real-hand landmark and zero/direction audit",
            "real tactile baseline, scaling, latency, and dropout calibration",
            "tomato force/indentation/damage threshold experiments",
            "shadow-mode and supervised low-speed hardware release review",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.samples < 1000 or args.envs < 32:
        raise ValueError("audit requires at least 1000 samples and 32 environments")
    report = build_report(args.samples, args.envs, args.seed)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["passed_cpu_scope"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
