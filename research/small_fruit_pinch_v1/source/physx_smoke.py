#!/usr/bin/env python3
"""小果三指捏取 PhysX 环境的无训练烟雾测试。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--seed", type=int, default=20260831)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.contract import validate_contract
from research.small_fruit_pinch_v1.source.env import SmallFruitPinchEnv, SmallFruitPinchEnvCfg


def main() -> None:
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num-envs 与 steps 必须为正数")
    torch.manual_seed(args.seed)
    cfg = SmallFruitPinchEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    geometry = cfg.events["tomato_geometry"].params
    geometry["axis_range_m"] = tuple(cfg.axis_range_m)
    geometry["density_range_kg_m3"] = tuple(cfg.density_range_kg_m3)
    geometry["accepted_mass_kg"] = tuple(cfg.accepted_mass_kg)
    print("[small-pinch-smoke] construct environment", flush=True)
    env = SmallFruitPinchEnv(cfg, render_mode=None)
    try:
        print("[small-pinch-smoke] reset environment", flush=True)
        obs, _ = env.reset(seed=args.seed)
        validate_contract(
            action_dim=env.cfg.action_space,
            actor_dim=obs["policy"].shape[-1],
            critic_dim=env._get_states().shape[-1],
        )
        action = torch.zeros((args.num_envs, env.cfg.action_space), device=env.device)
        action[:, :5] = torch.tensor((0.15, -0.10, -0.25, -0.10, 0.25), device=env.device)
        action[:, 5:8] = 0.10
        terminal_codes: list[int] = []
        contact_updates = 0
        for step in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(action)
            if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                raise RuntimeError("烟雾测试出现非有限观测或奖励")
            contact_updates += int((env.contact_forces[:, :3] > 0.0).any(dim=-1).sum())
            done = terminated | truncated
            if bool(done.any()):
                terminal_codes.extend(int(value) for value in env.last_termination_code[done].detach().cpu())
            if step == 0:
                print("[small-pinch-smoke] first physics step completed", flush=True)
        report = {
            "schema_version": 1,
            "task": "small_fruit_pinch_v1",
            "mode": "physx_smoke_no_training",
            "num_envs": args.num_envs,
            "steps": args.steps,
            "device": str(env.device),
            "actor_dim": int(obs["policy"].shape[-1]),
            "critic_dim": int(env._get_states().shape[-1]),
            "action_dim": int(env.cfg.action_space),
            "finite_observation_and_reward": True,
            "contact_update_count": contact_updates,
            "terminal_codes": terminal_codes,
            "palm_contact_mean_n": float(env.palm_force.mean()),
            "tim_contact_rate": float(env._verifier.contact[:, :3].all(dim=-1).float().mean()),
            "strict_success_rate": float(env.success.float().mean()),
            "hardware_command_allowed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
