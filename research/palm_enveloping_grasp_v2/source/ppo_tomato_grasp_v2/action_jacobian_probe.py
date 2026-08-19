"""Finite-difference distal action audit for the v2 safety checkpoint.

This is a read-only simulation diagnostic.  It perturbs one straight-finger
distal action at a time around the actor's action, keeping the palm-first
envelope and the physical fingertip proxy enabled.  The output is intended to
choose the next single-digit curriculum direction; it is not a training
checkpoint and never sends hardware commands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", default="v1_nominal_reliable")
parser.add_argument("--num-envs", type=int, default=9,
                    help="one baseline plus positive/negative case for each of four digits")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=20261070)
parser.add_argument("--distal-delta", type=float, default=0.18)
parser.add_argument("--proximal-delta", type=float, default=0.0)
parser.add_argument("--physical-fingertip-proxy", action="store_true")
parser.add_argument("--fingertip-proxy-radius", type=float, default=0.008)
parser.add_argument("--assist-gain", type=float, default=0.6)
parser.add_argument("--palm-first-coarse-scale", type=float, default=0.25)
parser.add_argument("--palm-first-coarse-max-scale", type=float, default=0.4)
parser.add_argument("--palm-first-ramp-steps", type=int, default=160)
parser.add_argument("--per-digit-near-surface-distance", type=float, default=0.043)
parser.add_argument("--include-synergies", action="store_true",
                    help="also perturb the five global synergy channels +/- delta")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from torch import nn

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.contract import V2_ACTOR_OBSERVATION_DIM, V2_ACTION_DIM
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(V2_ACTOR_OBSERVATION_DIM, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.mean = nn.Linear(128, V2_ACTION_DIM)

    def forward(self, obs):
        return torch.tanh(self.mean(self.body(obs)))


def load_actor(path: Path, device: torch.device) -> Actor:
    checkpoint = torch.load(path.expanduser().resolve(), map_location=device, weights_only=False)
    model = checkpoint["model"]
    actor = Actor().to(device)
    actor.load_state_dict({
        "body.0.weight": model["actor_body.0.weight"],
        "body.0.bias": model["actor_body.0.bias"],
        "body.2.weight": model["actor_body.2.weight"],
        "body.2.bias": model["actor_body.2.bias"],
        "body.4.weight": model["actor_body.4.weight"],
        "body.4.bias": model["actor_body.4.bias"],
        "mean.weight": model["actor_mean.weight"],
        "mean.bias": model["actor_mean.bias"],
    })
    actor.eval()
    return actor


@torch.inference_mode()
def main() -> None:
    if args.num_envs < 9:
        raise ValueError("--num-envs must be at least 9")
    device = torch.device(args.device)
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.episode_length_s = 12.0
    cfg.strict_enveloping_enabled = True
    cfg.strict_require_measured_distal_contact = True
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg.fingertip_collision_proxy_physical = bool(args.physical_fingertip_proxy)
    cfg.fingertip_collision_proxy_radius_m = float(args.fingertip_proxy_radius)
    cfg.virtual_distal_contact_enabled = True
    cfg.palm_first_distal_gate = True
    cfg.palm_first_coarse_scale = float(args.palm_first_coarse_scale)
    cfg.palm_first_coarse_max_scale = float(args.palm_first_coarse_max_scale)
    cfg.palm_first_ramp_steps = int(args.palm_first_ramp_steps)
    cfg.per_digit_distal_gate = True
    cfg.per_digit_distal_link3_threshold_n = 0.04
    cfg.per_digit_distal_link3_stable_steps = 4
    cfg.per_digit_distal_min_episode_step = 20
    cfg.per_digit_distal_near_surface_enabled = True
    cfg.per_digit_distal_near_surface_distance_m = float(args.per_digit_near_surface_distance)
    cfg.per_digit_distal_near_surface_min_episode_step = 20
    cfg.distal_proximal_reach_assist_gain = float(args.assist_gain)
    cfg.distal_proximal_reach_assist_gain_per_digit = None
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    actor = load_actor(args.checkpoint, device)
    obs, _ = env.reset(seed=args.seed)

    # label, target channel, sign, target kind; distal digit targets are
    # represented by their digit index and translated to channels 10..13.
    labels = [("baseline", None, 0.0, "baseline")]
    if args.include_synergies:
        for channel, name in enumerate(("global_flexion", "curl_shape", "balance", "spread", "thumb_opposition")):
            labels.append((f"{name}_plus", channel, 1.0, "synergy"))
            labels.append((f"{name}_minus", channel, -1.0, "synergy"))
    for digit, name in enumerate(("index", "middle", "ring", "little")):
        labels.append((f"{name}_plus", digit, 1.0, "digit"))
        labels.append((f"{name}_minus", digit, -1.0, "digit"))
    if args.num_envs > len(labels):
        labels.extend((f"baseline_extra_{i}", None, 0.0, "baseline") for i in range(args.num_envs - len(labels)))
    labels = labels[:args.num_envs]
    label_by_env = labels

    peak_force = torch.zeros((args.num_envs, 5), device=device)
    measured_peak = torch.zeros((args.num_envs, 5), device=device)
    min_distance = torch.full((args.num_envs, 4), float("inf"), device=device)
    measured_steps = torch.zeros((args.num_envs, 4), device=device)
    palm_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    dropped = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    overloaded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    for _step in range(args.steps):
        action = actor(obs["policy"]).clone()
        for env_id, (_label, target, sign, kind) in enumerate(label_by_env):
            if target is None:
                continue
            if kind == "synergy":
                action[env_id, target] += sign * float(args.distal_delta)
            else:
                action[env_id, 10 + target] += sign * float(args.distal_delta)
                if args.proximal_delta:
                    action[env_id, 6 + target] += sign * float(args.proximal_delta)
        action.clamp_(-1.0, 1.0)
        obs, _reward, terminated, truncated, _info = env.step(action)
        tips = env._tip_positions()[:, 1:]
        distance = torch.linalg.vector_norm(tips - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        peak_force = torch.maximum(peak_force, env.contact_forces)
        measured_peak = torch.maximum(measured_peak, env._v2_measured_link4_force)
        measured_steps += (env._v2_measured_link4_force[:, 1:] >= cfg.contact_threshold_n).float()
        palm_seen |= env.palm_force >= cfg.palm_support_force_n
        dropped |= env.dropped
        overloaded |= env.overloaded
        # DirectRLEnv resets completed environments automatically.  The probe
        # remains a fixed-horizon diagnostic, so do not stop on an early done.
        _ = terminated | truncated

    rows = []
    for env_id, (label, target, sign, kind) in enumerate(label_by_env):
        rows.append({
            "label": label,
            "target_kind": kind,
            "digit": None if kind != "digit" else ("index", "middle", "ring", "little")[target],
            "synergy_channel": None if kind != "synergy" else int(target),
            "sign": sign,
            "distal_delta": float(args.distal_delta),
            "proximal_delta": float(args.proximal_delta),
            "min_tip_distance_m": min_distance[env_id].cpu().tolist(),
            "peak_contact_force_n": peak_force[env_id].cpu().tolist(),
            "peak_measured_link4_force_n": measured_peak[env_id].cpu().tolist(),
            "measured_link4_contact_steps": measured_steps[env_id].cpu().tolist(),
            "palm_support_seen": bool(palm_seen[env_id]),
            "dropped": bool(dropped[env_id]),
            "overloaded": bool(overloaded[env_id]),
        })
    report = {
        "version": "v2_action_jacobian_probe_v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "profile": args.profile,
        "seed": args.seed,
        "steps": args.steps,
        "physical_fingertip_proxy": bool(args.physical_fingertip_proxy),
        "fingertip_proxy_radius_m": args.fingertip_proxy_radius,
        "hardware_command_allowed": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
