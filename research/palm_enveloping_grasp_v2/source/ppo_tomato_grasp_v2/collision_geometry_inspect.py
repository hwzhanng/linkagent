"""Inspect the authored collision shapes used by straight fingertip sensors."""

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
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdGeom, UsdPhysics

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg


def main() -> None:
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg = apply_profile(cfg, "v1_nominal_reliable", num_envs=1, seed=20260905)
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    stage = env.sim.stage
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    names = ("index_link3", "index_link4", "middle_link3", "middle_link4", "ring_link3", "ring_link4", "little_link3", "little_link4")
    rows = []
    for name in names:
        root = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
        for prim in Usd.PrimRange(root):
            schemas = sorted(prim.GetAppliedSchemas())
            has_collision = any("Collision" in s or "collision" in s for s in schemas)
            if prim == root or prim.GetTypeName() in ("Mesh", "Cube", "Capsule", "Sphere") or has_collision:
                item = {"name": name, "path": str(prim.GetPath()), "type": prim.GetTypeName(), "schemas": schemas,
                        "is_instance": prim.IsInstance(), "is_instance_proxy": prim.IsInstanceProxy(),
                        "is_in_prototype": prim.IsInPrototype()}
                extent = prim.GetAttribute("extent")
                if extent and extent.HasValue():
                    item["extent"] = str(extent.Get())
                try:
                    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                    item["bbox_min"] = [float(v) for v in bbox.GetMin()]
                    item["bbox_max"] = [float(v) for v in bbox.GetMax()]
                except Exception as exc:
                    item["bbox_error"] = str(exc)
                try:
                    item["collision_enabled"] = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                except Exception:
                    pass
                rows.append(item)
    report = {"version": "collision_geometry_inspect_v1", "rows": rows, "hardware_command_allowed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
