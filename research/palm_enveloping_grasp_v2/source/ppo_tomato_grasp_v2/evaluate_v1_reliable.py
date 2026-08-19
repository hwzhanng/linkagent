"""Matched v1 baseline evaluation with reset-safe terminal accounting.

This is the P1/P2 measurement tool for the v2 workstream.  It intentionally
loads the frozen v1 actor and physics environment, but applies a named v2
profile and records every completed episode from a pre-reset terminal
snapshot.  It is therefore suitable for diagnosing whether a future policy
improvement is real rather than a metric/reset artifact.
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", choices=(
    "v1_training_small", "v1_nominal_reliable", "v2_band1_match", "v2_band2_shape",
    "v2_band3_sensor_actuator", "v2_band4_stress", "p2_e0_core", "p2_e1_size",
    "p2_e2_position", "p2_e3_shape", "p2_e4_material", "p2_e5_sensor",
    "p2_e6_actuator", "p2_e7_stress",
), default="v1_nominal_reliable")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=20260820)
parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
parser.add_argument("--coverage-deg", type=float, default=200.0)
parser.add_argument("--hold-steps", type=int, default=10)
parser.add_argument("--safe-force", type=float, default=12.0)
parser.add_argument("--pick-speed", type=float, default=0.02)
parser.add_argument("--pick-displacement", type=float, default=0.035)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
from torch import nn

from simulation.ppo_tomato_grasp.physx_enveloping_env import EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile, profile_definition
from simulation.ppo_tomato_grasp_v2.snapshot_env import TerminalSnapshotEnvelopingEnv


class FrozenV1Actor(nn.Module):
    """Actor architecture used by the v1 110D/14D checkpoint."""

    def __init__(self):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(110, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.actor_mean = nn.Linear(128, 14)
        self.log_std = nn.Parameter(torch.zeros(14))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor_mean(self.actor_body(obs)))


def make_cfg() -> EnvelopingPhysxTomatoGraspEnvCfg:
    cfg = EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args_cli.device
    cfg.training_stage = args_cli.stage
    cfg.strict_enveloping_enabled = True
    cfg.strict_angular_coverage_min_deg = args_cli.coverage_deg
    cfg.strict_hold_steps = args_cli.hold_steps
    cfg.strict_enveloping_require_stem_break = args_cli.stage >= 2
    cfg.pick_speed_m_s = args_cli.pick_speed
    cfg.pick_speed_range_m_s = (args_cli.pick_speed, args_cli.pick_speed)
    cfg.pick_displacement_m = args_cli.pick_displacement
    cfg.safe_force_n = args_cli.safe_force
    cfg.force_ceiling_n = (
        args_cli.safe_force, 0.85 * args_cli.safe_force,
        0.85 * args_cli.safe_force, 0.75 * args_cli.safe_force,
        0.75 * args_cli.safe_force,
    )
    cfg.severe_overload_scale = 1.40
    cfg.episode_length_s = 12.0
    return apply_profile(cfg, args_cli.profile, num_envs=args_cli.num_envs, seed=args_cli.seed)


def _mean(values: torch.Tensor, denominator: torch.Tensor) -> float:
    return float(values.sum().float() / denominator.clamp_min(1.0))


@torch.inference_mode()
def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("num-envs and steps must be positive")
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=args_cli.device, weights_only=False)
    if (checkpoint.get("actor_observation_dim"), checkpoint.get("action_dim")) != (110, 14):
        raise ValueError("checkpoint does not match the frozen v1 110D/14D actor contract")
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    device = torch.device(args_cli.device)
    actor = FrozenV1Actor().to(device)
    actor.load_state_dict(checkpoint["model"], strict=False)
    actor.eval()

    env = TerminalSnapshotEnvelopingEnv(make_cfg())
    obs, _ = env.reset(seed=args_cli.seed)
    completed = torch.zeros((), dtype=torch.long, device=device)
    counters = {name: torch.zeros((), dtype=torch.long, device=device) for name in (
        "success", "strict_success", "drop", "overload", "damage", "timeout",
        "drop_before_pick", "drop_after_pick", "drop_after_break", "pick_started",
        "stem_broken", "post_break_hold", "five_digit", "palm_supported",
        "distal_all", "palm_first", "measured_geometry",
    )}
    max_force_sum = torch.zeros(5, device=device)
    max_palm_sum = torch.zeros((), device=device)
    max_coverage_sum = torch.zeros((), device=device)
    hold_max_sum = torch.zeros((), device=device)
    termination_hist = torch.zeros(6, dtype=torch.long, device=device)
    try:
        for _ in range(args_cli.steps):
            action = actor(obs["policy"])
            obs, _, terminated, truncated, _info = env.step(action)
            done = terminated | truncated
            if bool(done.any()):
                snapshot = env.consume_terminal_snapshots(done)
                n = done.sum().to(dtype=torch.long)
                completed += n
                code = snapshot["termination_code"].clamp(0, 5)
                termination_hist.scatter_add_(0, code, torch.ones_like(code, dtype=torch.long))
                counters["success"] += (code == 1).sum()
                counters["strict_success"] += snapshot["strict_success"].sum()
                counters["drop"] += (code == 2).sum()
                counters["overload"] += (code == 3).sum()
                counters["damage"] += (code == 4).sum()
                counters["timeout"] += (code == 5).sum()
                counters["drop_before_pick"] += snapshot["last_drop_before_pick"].sum()
                counters["drop_after_pick"] += snapshot["last_drop_after_pick"].sum()
                counters["drop_after_break"] += snapshot["last_drop_after_break"].sum()
                counters["pick_started"] += snapshot["pick_started"].sum()
                counters["stem_broken"] += snapshot["stem_broken"].sum()
                counters["post_break_hold"] += (
                    snapshot["stem_broken"]
                    & (snapshot["post_break_hold"] >= env.cfg.success_hold_steps)
                ).sum()
                counters["five_digit"] += (
                    (snapshot["contact_forces"] >= env.cfg.contact_threshold_n).all(-1)
                ).sum()
                counters["palm_supported"] += (
                    snapshot["palm_force"] >= env.cfg.palm_support_force_n
                ).sum()
                counters["distal_all"] += snapshot["strict_distal_participation"].all(-1).sum()
                counters["palm_first"] += (
                    (snapshot["strict_first_palm"] >= 0)
                    & (snapshot["strict_first_three"] >= 0)
                    & (snapshot["strict_first_palm"] + env.cfg.strict_palm_first_margin_steps <= snapshot["strict_first_three"])
                ).sum()
                counters["measured_geometry"] += snapshot["strict_contact_geometry_measured"].any(-1).sum()
                max_force_sum += snapshot["contact_forces"].max(0).values
                max_palm_sum += snapshot["palm_force"].max()
                max_coverage_sum += snapshot["strict_coverage_max"].sum()
                hold_max_sum += snapshot["strict_hold_max"].sum()
            if int(completed) >= args_cli.num_envs:
                break
        den = completed.float().clamp_min(1.0)
        profile = profile_definition(args_cli.profile)
        report = {
            "version": "v2_reliable_terminal_eval_1",
            "checkpoint": str(checkpoint_path),
            "checkpoint_version": checkpoint.get("version"),
            "profile": args_cli.profile,
            "profile_values": profile,
            "profile_resolution": getattr(env.cfg, "v2_profile_resolution", {}),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "stage": args_cli.stage,
            "completed_episodes": int(completed),
            "success_rate": _mean(counters["success"], den),
            "strict_success_rate": _mean(counters["strict_success"], den),
            "drop_rate": _mean(counters["drop"], den),
            "overload_rate": _mean(counters["overload"], den),
            "damage_rate": _mean(counters["damage"], den),
            "timeout_rate": _mean(counters["timeout"], den),
            "drop_before_pick_rate": _mean(counters["drop_before_pick"], den),
            "drop_after_pick_rate": _mean(counters["drop_after_pick"], den),
            "drop_after_break_rate": _mean(counters["drop_after_break"], den),
            "pick_triggered_rate": _mean(counters["pick_started"], den),
            "stem_break_rate": _mean(counters["stem_broken"], den),
            "post_break_hold_rate": _mean(counters["post_break_hold"], den),
            "five_digit_terminal_rate": _mean(counters["five_digit"], den),
            "palm_supported_terminal_rate": _mean(counters["palm_supported"], den),
            "distal_all_terminal_rate": _mean(counters["distal_all"], den),
            "palm_first_terminal_rate": _mean(counters["palm_first"], den),
            "measured_geometry_terminal_rate": _mean(counters["measured_geometry"], den),
            "digit_peak_force_n_mean": [float(v / den) for v in max_force_sum],
            "palm_peak_force_n_mean": float(max_palm_sum / den),
            "strict_coverage_max_mean": float(max_coverage_sum / den),
            "strict_hold_max_mean": float(hold_max_sum / den),
            "termination_histogram_code_0_to_5": [int(v) for v in termination_hist.detach().cpu()],
            "hardware_command_allowed": False,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
