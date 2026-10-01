"""Read-only PhysX runtime audit of actuated and unactuated hand joints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20260820)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v3.env import V3EnvelopingPhysxTomatoGraspEnv, V3EnvelopingPhysxTomatoGraspEnvCfg


def main() -> None:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cfg = V3EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg = apply_profile(cfg, "v1_nominal_reliable", num_envs=1, seed=args.seed)
    env = V3EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    stiffness = env.robot.root_physx_view.get_dof_stiffnesses()[0].detach().cpu().tolist()
    damping = env.robot.root_physx_view.get_dof_dampings()[0].detach().cpu().tolist()
    positions = env.robot.root_physx_view.get_dof_positions()[0].detach().cpu().tolist()
    velocities = env.robot.root_physx_view.get_dof_velocities()[0].detach().cpu().tolist()
    active = set(int(value) for value in env.active_joint_ids)
    joints = []
    for index, name in enumerate(env.robot.joint_names):
        joints.append({
            "index": index,
            "name": name,
            "active_16d_contract": index in active,
            "runtime_stiffness": float(stiffness[index]),
            "runtime_damping": float(damping[index]),
            "reset_position_rad": float(positions[index]),
            "reset_velocity_rad_s": float(velocities[index]),
        })
    report = {
        "version": "v3_runtime_joint_contract_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_command_allowed": False,
        "total_revolute_dofs": len(env.robot.joint_names),
        "active_contract_dofs": len(env.active_joint_ids),
        "active_joint_names": list(env.active_joint_names),
        "unactuated_contract_joint_names": [row["name"] for row in joints if not row["active_16d_contract"]],
        "joints": joints,
        "interpretation": (
            "A joint outside the 16-DOF action contract may still retain an imported USD drive. "
            "This report distinguishes the policy contract from the actual runtime drive parameters."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "total": report["total_revolute_dofs"],
        "active": report["active_contract_dofs"],
        "outside_contract": report["unactuated_contract_joint_names"],
    }, ensure_ascii=False, indent=2), flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()

