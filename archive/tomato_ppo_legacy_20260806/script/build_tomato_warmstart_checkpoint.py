"""Build an auditable PPO warm-start with fresh Adam state.

The policy and critic parameters are retained, while optimizer moments from a
different reward/action contract are discarded.  No rollout or training is
performed by this script.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.skills.tomato_pick.contracts import OBSERVATION_DIM


def main() -> int:
    torch = importlib.import_module("torch")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--critic-observation-dim",
        type=int,
        default=None,
        help=(
            "expand the critic input to actor observation + existing privileged "
            "state + zero-initialized task extras"
        ),
    )
    parser.add_argument(
        "--actor-observation-dim", type=int, default=OBSERVATION_DIM
    )
    parser.add_argument("--actor-std", type=float, default=None)
    parser.add_argument("--minimum-actor-std", type=float, default=0.05)
    parser.add_argument("--maximum-actor-std", type=float, default=0.80)
    parser.add_argument(
        "--reset-actor-output-row",
        type=int,
        action="append",
        default=[],
        help=(
            "zero one actor output row (weight and bias); may be repeated. "
            "This changes initialization only and leaves the action contract intact"
        ),
    )
    parser.add_argument("--reset-iteration", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("source and output must differ")
    if not 0.0 < args.learning_rate < 1.0:
        raise ValueError("--learning-rate must be in (0, 1)")

    checkpoint = torch.load(source, weights_only=False, map_location="cpu")
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    for state_name in ("actor_state_dict", "critic_state_dict"):
        nonfinite = [
            name
            for name, value in checkpoint[state_name].items()
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
        ]
        if nonfinite:
            raise ValueError(f"{state_name} contains non-finite tensors: {nonfinite}")

    actor_state = checkpoint["actor_state_dict"]
    critic_state = checkpoint["critic_state_dict"]
    output_weight_key = "mlp.6.weight"
    output_bias_key = "mlp.6.bias"
    if args.reset_actor_output_row:
        if output_weight_key not in actor_state or output_bias_key not in actor_state:
            raise ValueError("actor output layer does not match the expected MLP layout")
        output_dim = int(actor_state[output_weight_key].shape[0])
        reset_rows = sorted(set(args.reset_actor_output_row))
        invalid_rows = [row for row in reset_rows if not 0 <= row < output_dim]
        if invalid_rows:
            raise ValueError(
                f"actor output rows must be in [0, {output_dim - 1}]: {invalid_rows}"
            )
        for row in reset_rows:
            actor_state[output_weight_key][row].zero_()
            actor_state[output_bias_key][row].zero_()
    else:
        reset_rows = []
    old_critic_dim = int(critic_state["mlp.0.weight"].shape[1])
    if args.critic_observation_dim is not None:
        target_dim = int(args.critic_observation_dim)
        minimum_target_dim = args.actor_observation_dim + old_critic_dim
        if target_dim < minimum_target_dim:
            raise ValueError(
                "--critic-observation-dim must fit actor observations and the "
                f"existing critic state ({minimum_target_dim} values)"
            )
        if int(actor_state["obs_normalizer._mean"].numel()) != args.actor_observation_dim:
            raise ValueError("actor normalizer does not match --actor-observation-dim")
        old_first_layer = critic_state["mlp.0.weight"]
        expanded_first_layer = torch.zeros(
            (old_first_layer.shape[0], target_dim), dtype=old_first_layer.dtype
        )
        privileged_start = args.actor_observation_dim
        expanded_first_layer[
            :, privileged_start : privileged_start + old_critic_dim
        ] = old_first_layer
        critic_state["mlp.0.weight"] = expanded_first_layer
        for statistic, extra_fill in (
            ("_mean", 0.0),
            ("_var", 1.0),
            ("_std", 1.0),
        ):
            actor_statistic = actor_state[f"obs_normalizer.{statistic}"]
            critic_statistic = critic_state[f"obs_normalizer.{statistic}"]
            if actor_statistic.shape[:-1] != critic_statistic.shape[:-1]:
                raise ValueError("actor and critic normalizer layouts do not match")
            expanded_statistic = torch.full(
                (*critic_statistic.shape[:-1], target_dim),
                extra_fill,
                dtype=critic_statistic.dtype,
            )
            expanded_statistic[..., : args.actor_observation_dim] = actor_statistic
            expanded_statistic[
                ..., privileged_start : privileged_start + old_critic_dim
            ] = critic_statistic
            critic_state[f"obs_normalizer.{statistic}"] = expanded_statistic

    if args.actor_std is not None:
        if not args.minimum_actor_std < args.actor_std < args.maximum_actor_std:
            raise ValueError("--actor-std must be strictly inside its min/max bounds")
        unit = (args.actor_std - args.minimum_actor_std) / (
            args.maximum_actor_std - args.minimum_actor_std
        )
        logit = math.log(unit / (1.0 - unit))
        actor_state["distribution.std_logit_param"].fill_(logit)

    if args.reset_iteration:
        checkpoint["iter"] = 0

    optimizer = checkpoint["optimizer_state_dict"]
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ValueError("optimizer_state_dict has an unexpected layout")
    previous_state_count = len(optimizer.get("state", {}))
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = float(args.learning_rate)
        group["initial_lr"] = float(args.learning_rate)

    infos = checkpoint["infos"] if isinstance(checkpoint["infos"], dict) else {}
    checkpoint["infos"] = {
        **infos,
        "warmstart": {
            "source": str(source),
            "optimizer_state_reset": True,
            "learning_rate": float(args.learning_rate),
            "critic_observation_dim": args.critic_observation_dim,
            "actor_std": args.actor_std,
            "reset_actor_output_rows": reset_rows,
            "iteration_reset": bool(args.reset_iteration),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    report = {
        "source": str(source),
        "output": str(output),
        "iteration": int(checkpoint["iter"]),
        "optimizer_states_removed": previous_state_count,
        "learning_rate": float(args.learning_rate),
        "critic_observation_dim": args.critic_observation_dim,
        "actor_std": args.actor_std,
        "reset_actor_output_rows": reset_rows,
        "iteration_reset": bool(args.reset_iteration),
        "actor_and_critic_finite": True,
    }
    print("TOMATO_WARMSTART=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
