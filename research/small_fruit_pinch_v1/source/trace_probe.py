#!/usr/bin/env python3
"""输出少量环境的每步力/动作轨迹，定位过载发生在接触前还是接触后。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--guided", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv


P1 = (0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.7592, 0.2345, -0.9126, 0.0, 0.0, -0.3269, 0.0349, 0.0, 0.0)


def main() -> None:
    env_cls = NativeSmallFruitPinchEnv
    if args.guided:
        from research.small_fruit_pinch_v1.source.guided_residual_env import GuidedResidualSmallFruitPinchEnv

        env_cls = GuidedResidualSmallFruitPinchEnv
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = env_cls(cfg=cfg)
    try:
        obs, _ = env.reset(seed=20260924)
        action = torch.tensor(P1, device=env.device).expand(args.num_envs, -1)
        best = torch.zeros((args.num_envs, 5), device=env.device)
        for step in range(args.steps):
            obs, _, term, trunc, info = env.step(action)
            force = env._last_physical_force_n.detach().cpu()
            best = torch.maximum(best, env._last_physical_force_n)
            if step < 20 or step % 5 == 0 or bool((force > torch.tensor((2.0, 2.0, 2.0, 2.0, 2.0))).any()):
                print(
                    f"step={step:03d} max_force={force.amax(dim=1).tolist()} "
                    f"mean_force={force.mean(dim=0).tolist()} "
                    f"term={int(term.sum())} max_a={env._executed_actions.detach().cpu().abs().amax(dim=1).tolist()}",
                    flush=True,
                )
        print(f"best={best.detach().cpu().tolist()}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
