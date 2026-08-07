"""Create a clean PPO fine-tuning checkpoint from a trained tomato policy.

The actor and both observation normalizers are preserved.  The critic can be
kept for same-reward continuation or reset from an architecture-compatible
fresh checkpoint after a reward change.  PPO's Adam state is always emptied so
the requested fine-tuning learning rate is authoritative.  No simulator state
is touched.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Sequence


NORMALIZER_KEYS = (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
    "obs_normalizer.count",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--std", type=float, default=0.20)
    parser.add_argument(
        "--preserve-std",
        action="store_true",
        help="Preserve the source policy's learned per-action standard deviations.",
    )
    parser.add_argument("--keep-critic", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    args = parser.parse_args(argv)
    if not 0.01 <= args.std <= 1.0:
        raise ValueError("--std must lie in [0.01, 1.0]")
    if not 0.0 < args.learning_rate <= 1.0e-2:
        raise ValueError("--learning-rate must lie in (0, 0.01]")

    torch = importlib.import_module("torch")
    source = torch.load(args.source.expanduser().resolve(), map_location="cpu", weights_only=False)
    fresh = torch.load(args.fresh.expanduser().resolve(), map_location="cpu", weights_only=False)
    if source["actor_state_dict"].keys() != fresh["actor_state_dict"].keys():
        raise ValueError("source and fresh actor architectures differ")
    if source["critic_state_dict"].keys() != fresh["critic_state_dict"].keys():
        raise ValueError("source and fresh critic architectures differ")

    actor = {name: value.detach().clone() for name, value in source["actor_state_dict"].items()}
    if not args.preserve_std:
        actor["distribution.log_std_param"] = torch.full_like(
            actor["distribution.log_std_param"], math.log(args.std)
        )
    critic_source = source if args.keep_critic else fresh
    critic = {
        name: value.detach().clone() for name, value in critic_source["critic_state_dict"].items()
    }
    if not args.keep_critic:
        for name in NORMALIZER_KEYS:
            critic[name] = source["critic_state_dict"][name].detach().clone()

    optimizer = fresh["optimizer_state_dict"]
    clean_optimizer = {
        "state": {},
        "param_groups": [dict(group) for group in optimizer["param_groups"]],
    }
    for group in clean_optimizer["param_groups"]:
        group["lr"] = args.learning_rate
    checkpoint = {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "optimizer_state_dict": clean_optimizer,
        "iter": 0,
        "infos": {
            "warmstart_source": str(args.source.expanduser().resolve()),
            "fresh_template": str(args.fresh.expanduser().resolve()),
            "actor_std": (
                "preserved_per_action" if args.preserve_std else args.std
            ),
            "learning_rate": args.learning_rate,
            "fresh_critic": not args.keep_critic,
            "fresh_optimizer": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"TOMATO_ACTOR_WARMSTART={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
