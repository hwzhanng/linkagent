"""Validate and optionally record the Physics-v2 step-1 scaffold gate."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.tomato_pick.physics_v2 import physics_v2_step1_report


def _inherits(path: Path, class_name: str, parent_name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        return False
    node = classes[0]
    bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
    return parent_name in bases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = physics_v2_step1_report()
    wrapper_checks = {
        "environment_preserves_physics_v1_parent": _inherits(
            PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v2/env.py",
            "TomatoPickPhysicsV2Env",
            "TomatoPickPhysicsV1Env",
        ),
        "environment_config_preserves_physics_v1_parent": _inherits(
            PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v2/env_cfg.py",
            "TomatoPickPhysicsV2EnvCfg",
            "TomatoPickPhysicsV1EnvCfg",
        ),
        "ppo_config_has_independent_output_identity": (
            "tomato_pick_physics_v2"
            in (
                PROJECT_ROOT
                / "simulation/tomato_pick/isaac/physics_v2/agents/rsl_rl_ppo_cfg.py"
            ).read_text(encoding="utf-8")
        ),
    }
    report["checks"].update(wrapper_checks)
    report["passed"] = all(report["checks"].values())

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
