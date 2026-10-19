"""Train a single LinkerHand L20 right hand to continuously rotate fruit.

Examples:
  python -m script.train_fruit_rotation --preflight
  python -m script.train_fruit_rotation --headless --smoke --num-envs 8
  python -m script.train_fruit_rotation --headless --num-envs 256 --updates 6000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
URDF = (ROOT / "botSDK/urdf/01_vendor_original/02_extracted_packages/"
        "L20_8_r_s1/L20_8_r_s1.urdf")


def preflight():
    if not URDF.is_file():
        raise FileNotFoundError(URDF)
    robot = ET.parse(URDF).getroot()
    joints = [joint.attrib["name"] for joint in robot.findall("joint")
              if joint.attrib.get("type") != "fixed"]
    from collections import Counter
    kinds = Counter(j.attrib["type"] for j in robot.findall("joint"))
    from pathlib import Path
    missing_meshes = [
        mesh.attrib["filename"]
        for mesh in robot.findall(".//mesh")
        if not (URDF.parent / mesh.attrib["filename"]).exists()
    ]
    unique_missing = sorted(set(missing_meshes))
    if len(joints) != 21 or unique_missing:
        raise RuntimeError(
            f"Right L20: expected 21 movable joints, got {len(joints)}, "
            f"missing meshes: {unique_missing[:5]}"
        )
    status = {
        "source_urdf": str(URDF),
        "single_hand": True,
        "movable_joint_count": len(joints),
        "joint_types": dict(kinds),
        "actuated_dof": 16,
        "passive_mimic_dof": 5,
        "meshes_resolved": True,
        "isaac_simulation_executed": False,
        "real_sensor_calibrated": False,
    }
    print(json.dumps(status, indent=2))
    return 0


def main():
    if "--preflight" in sys.argv[1:]:
        return preflight()

    # Isaac modules must not import before AppLauncher starts Kit.
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Reset + 20 PhysX steps, no optimization")
    parser.add_argument("--rebuild-asset", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 4096:
        raise ValueError("--num-envs outside 1..4096")
    if not 1 <= args.updates or not 1 <= args.rollout_steps:
        raise ValueError("updates and rollout-steps must be positive")
    if args.rollout_steps * args.num_envs < 8:
        raise ValueError("PPO rollout must contain at least 8 transitions")

    launcher = AppLauncher(args)
    sim_app = launcher.app
    env = None
    try:
        import torch
        from simulation.fruit_rotation.asset import ensure_single_hand_usd
        asset = ensure_single_hand_usd(force=args.rebuild_asset)
        os.environ["LINKAGENT_ROTATION_USD"] = str(asset)
        from simulation.fruit_rotation.env_cfg import FruitRotationEnvCfg
        from simulation.fruit_rotation.env import FruitRotationEnv
        from simulation.fruit_rotation.ppo import PPOSettings, train

        torch.manual_seed(args.seed)
        cfg = FruitRotationEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        env = FruitRotationEnv(cfg)
        if args.smoke:
            obs, _ = env.reset()
            assert obs["policy"].shape == (args.num_envs, 502)
            assert obs["critic"].shape == (args.num_envs, 622)
            for step in range(20):
                obs, reward, failed, timed_out, info = env.step(
                    torch.zeros((args.num_envs, 16), device=env.device)
                )
                if (not bool(torch.isfinite(reward).all())
                        or not bool(torch.isfinite(obs["policy"]).all())):
                    raise RuntimeError(f"Nonfinite fruit rotation simulation at step {step}")
            print(json.dumps({
                "smoke_passed": True,
                "num_envs": args.num_envs,
                "physics_steps": 120,
                "policy_steps": 20,
                "usd": str(asset),
                "failure_terminals": int(failed.sum().item()),
                "policy_shape": list(obs["policy"].shape),
            }, indent=2))
            return 0
        settings = PPOSettings(
            updates=args.updates, rollout_steps=args.rollout_steps
        )
        if (settings.rollout_steps * args.num_envs) % settings.minibatches:
            raise ValueError("num-envs * rollout-steps must divide into 8 minibatches")
        run_dir = args.run_dir or (
            ROOT / "logs/fruit_rotation"
            / time.strftime("%Y%m%d_%H%M%S_single_l20_rotation")
        )
        train(env, settings, run_dir, args.seed, resume=args.resume)
        return 0
    finally:
        if env is not None:
            env.close()
        sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
