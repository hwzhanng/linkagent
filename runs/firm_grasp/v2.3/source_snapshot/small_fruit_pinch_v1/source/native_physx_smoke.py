#!/usr/bin/env python3
"""原生左手小果捏取环境的最小 PhysX 冒烟测试。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--verification-events", action="store_true")
parser.add_argument("--evidence-json", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from research.small_fruit_pinch_v1.source.native_env import (
    NativeSmallFruitPinchCfg,
    NativeSmallFruitPinchEnv,
)


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        observations, _ = env.reset(seed=42)
        action = observations["policy"].new_zeros((args.num_envs, cfg.action_space))
        for _ in range(5):
            observations, _, _, _, info = env.step(action)
        evidence = {"policy_shape": list(observations["policy"].shape), "critic_shape": list(observations["critic"].shape)}
        if args.verification_events:
            env._verifier.success[0] = True
            env._prepare_verification_support()
            assert bool(env._stem_detached[0]) and not bool(env._audit_natural_detached[0])
            env._verifier.disturbance_due[0] = True
            previous = env._tomato.data.root_lin_vel_w.torch.clone()
            env._apply_verification_impulse()
            immediate = env._tomato.data.root_lin_vel_w.torch.clone()
            expected = env._audit_disturbance_directions[0] * (0.25 * 9.81 * 0.05)
            assert torch.allclose(immediate[0] - previous[0], expected, atol=1.0e-5)
            observations, _, _, _, info = env.step(action)
            evidence.update({
                "probe_injected_stage_flags": True,
                "audit_release_is_not_natural_detachment": not bool(env._audit_terminal["natural_detached"][0]),
                "momentum_delta_m_s": (immediate[0] - previous[0]).tolist(),
                "impulse_ns": env._audit_terminal_impulse_ns[0].tolist(),
                "disturbance_acknowledged_after_physics": bool(env._audit_terminal["disturbance_applied"][0]),
                "zero_contact_support_success": bool(env._audit_terminal["support_retained"][0]),
            })
            assert evidence["disturbance_acknowledged_after_physics"]
            assert not evidence["zero_contact_support_success"]
        if args.evidence_json is not None:
            args.evidence_json.parent.mkdir(parents=True, exist_ok=True)
            with args.evidence_json.open("x", encoding="utf-8") as stream:
                json.dump(evidence, stream, ensure_ascii=False, indent=2)
        print(
            "native_physx_smoke "
            f"policy={tuple(observations['policy'].shape)} "
            f"critic={tuple(observations['critic'].shape)} "
            f"info_keys={sorted(info.keys())}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
