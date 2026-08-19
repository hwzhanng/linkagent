"""CEM feasibility search for a small-fruit weak geometry prior.

This is not a policy or a final grasp pose. It only answers whether the
calibrated 10D manifold can physically reach a palm-supported 29--30 mm fruit
without overload, and supplies a starting point for tactile/PPO refinement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--steps", type=int, default=140)
parser.add_argument("--seed", type=int, default=47500)
parser.add_argument("--radius-low", type=float, default=0.029)
parser.add_argument("--radius-high", type=float, default=0.030)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from simulation.ppo_tomato_grasp.physx_tomato_env import (
    PhysxTomatoGraspEnv,
    PhysxTomatoGraspEnvCfg,
)


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args.seed)
    cfg = PhysxTomatoGraspEnvCfg()
    cfg.seed = args.seed
    cfg.training_stage = 1
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.events["tomato_geometry"].params.update(
        axis_range_m=(args.radius_low, args.radius_high),
        isotropic_axes=True,
        density_range_kg_m3=(850.0, 850.0),
    )
    cfg.position_jitter = (0.0005, 0.0005, 0.00025)
    cfg.friction_range = (0.65, 0.65)
    cfg.restitution_range = (0.0, 0.0)
    cfg.stiffness_range_n_m = (18000.0, 18000.0)
    cfg.tactile_gain_range = (1.0, 1.0)
    cfg.tactile_bias_range_n = (0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    env = PhysxTomatoGraspEnv(cfg, render_mode=None)

    mean = torch.tensor(
        (0.1623066, -0.2356111, 0.00530165, -0.8699145, 0.9777843,
         -0.0491881, -0.7238308, 0.5984016, -0.8817820, -0.1348379),
        device=env.device,
    )
    std = torch.tensor(
        (0.22, 0.20, 0.16, 0.12, 0.08, 0.18, 0.22, 0.22, 0.18, 0.20),
        device=env.device,
    )
    elite_count = max(8, args.num_envs // 10)
    history: list[dict] = []
    global_best_score = float("-inf")
    global_best = mean.clone()

    for round_index in range(args.rounds):
        env.reset()
        candidates = (mean + torch.randn((args.num_envs, 10), device=env.device) * std).clamp(-1.0, 1.0)
        if round_index == 0:
            candidates[0] = mean
        best_contact_count = torch.zeros(args.num_envs, device=env.device)
        complete_steps = torch.zeros(args.num_envs, device=env.device)
        peak_force = torch.zeros(args.num_envs, device=env.device)
        dropped = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        min_surface_gap = torch.full((args.num_envs,), 1.0, device=env.device)
        for _ in range(args.steps):
            env.step(candidates)
            contacts = (env.contact_forces >= cfg.contact_threshold_n).float().sum(dim=-1)
            palm = env.palm_force >= cfg.palm_support_force_n
            best_contact_count = torch.maximum(best_contact_count, contacts)
            complete_steps += ((contacts == 5) & palm).float()
            peak_force = torch.maximum(peak_force, env.contact_forces.max(dim=-1).values)
            dropped |= env.dropped
            gap = torch.linalg.vector_norm(env._tip_positions() - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1)
            gap = (gap - env.tomato_axes.mean(dim=-1, keepdim=True)).abs().mean(dim=-1)
            min_surface_gap = torch.minimum(min_surface_gap, gap)
        overload = (peak_force >= cfg.severe_overload_scale * env._ceiling.max()).float()
        # A candidate that touches all fingers for one frame is not useful;
        # the target is a quiet 20-step enclosure with no overload/drop.
        # Weight sustained complete contact first, then use contact count and
        # force as tie breakers for CEM.
        score = (
            4.0 * complete_steps.clamp_max(40.0)
            + 2.0 * best_contact_count
            - 1.5 * peak_force.clamp_min(0.0)
            - 120.0 * min_surface_gap
            - 30.0 * overload
            - 40.0 * dropped.float()
        )
        elite_ids = torch.topk(score, elite_count).indices
        elites = candidates[elite_ids]
        mean = 0.25 * mean + 0.75 * elites.mean(dim=0)
        std = (0.25 * std + 0.75 * elites.std(dim=0, unbiased=False)).clamp(0.015, 0.30)
        best_id = int(score.argmax())
        if float(score[best_id]) > global_best_score:
            global_best_score = float(score[best_id])
            global_best = candidates[best_id].clone()
        record = {
            "round": round_index,
            "best_score": float(score.max()),
            "best_contact_count": float(best_contact_count[best_id]),
            "best_complete_steps": float(complete_steps[best_id]),
            "best_peak_force_n": float(peak_force[best_id]),
            "complete_candidate_count": int((complete_steps > 0).sum()),
            "mean": [float(value) for value in mean],
            "std": [float(value) for value in std],
        }
        history.append(record)
        print(record, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "radius_range_m": [args.radius_low, args.radius_high],
                "global_best_score": global_best_score,
                "global_best_action": [float(value) for value in global_best],
                "rounds": history,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
