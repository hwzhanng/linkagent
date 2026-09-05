#!/usr/bin/env python3
"""在真实 PhysX 中批量搜索左手三指捏取的可达动作区域。

这不是训练，也不保存策略权重。它只用确定的 Sobol 动作样本验证：给定的
果实初始位姿和 14D 小果手部流形，是否实际存在拇指、食指和中指的同时接触。
该关卡不通过时禁止启动长时 PPO，避免把不可达几何误判为算法问题。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--top-k", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import (
    NativeSmallFruitPinchCfg,
    NativeSmallFruitPinchEnv,
)


def _sample_actions(count: int, device: str, seed: int) -> torch.Tensor:
    """在低维协同与三指残差上做可复现覆盖，非随机策略探索。"""

    engine = torch.quasirandom.SobolEngine(dimension=9, scramble=True, seed=seed)
    sample = engine.draw(count).to(device=device) * 2.0 - 1.0
    actions = torch.zeros((count, 14), device=device)
    # 全局闭合始终保持正值；其余维度覆盖闭合曲率、拇指对掌和三根手指残差。
    actions[:, 0] = 0.35 + 0.65 * (sample[:, 0] + 1.0) / 2.0
    actions[:, 1] = sample[:, 1]
    actions[:, 4] = sample[:, 2]
    actions[:, 5:8] = sample[:, 2:5]
    actions[:, 10:14] = sample[:, 5:9]
    return actions


def main() -> None:
    if args.num_envs < 8 or args.steps < 40:
        raise ValueError("可达性扫描至少需要 8 个环境和 40 个控制步")
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        actions = _sample_actions(args.num_envs, env.device, args.seed)
        best_score = torch.full((args.num_envs,), -1.0, device=env.device)
        best_force = torch.zeros((args.num_envs, 3), device=env.device)
        triple_frames = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        for _ in range(args.steps):
            env.step(actions)
            force = env._last_physical_force_n[:, :3]
            # 以最弱一指为主排序：不能用“两个指头很强”掩盖中指没有接触。
            score = torch.amin(force, dim=1)
            update = score > best_score
            best_score = torch.where(update, score, best_score)
            best_force[update] = force[update]
            triple_frames += (force >= 0.08).all(dim=1).long()

        order = torch.argsort(best_score, descending=True)[: args.top_k]
        rows = [
            {
                "env": int(index.item()),
                "min_force_n": round(float(best_score[index].item()), 4),
                "force_n": [round(float(value), 4) for value in best_force[index].tolist()],
                "triple_frames": int(triple_frames[index].item()),
                "action": [round(float(value), 4) for value in actions[index].tolist()],
            }
            for index in order
        ]
        reached = (triple_frames > 0).float().mean().item()
        print(
            "action_search "
            f"triple_contact_candidate_rate={reached:.4f} top={rows}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
