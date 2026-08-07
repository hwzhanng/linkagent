"""Review or explicitly execute the isolated C2 Isaac/RSL-RL training plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.training import build_training_plan, execute_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render", action="store_true", help="disable headless mode")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run only after all asset/runtime/physics gates pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print a plan by default; execution always requires ``--execute``."""

    args = _parser().parse_args(argv)
    plan = build_training_plan(
        isaaclab_root=args.isaaclab_root,
        num_environments=args.num_envs,
        seed=args.seed,
        headless=not args.render,
    )
    print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
    if not plan.ready:
        return 2
    if not args.execute:
        return 0
    return execute_training(plan, explicitly_confirmed=True)


if __name__ == "__main__":
    raise SystemExit(main())
