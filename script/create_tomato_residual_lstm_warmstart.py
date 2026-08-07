"""Convert the proven minimal-v3 MLP checkpoint into an exact residual-LSTM warm-start."""

from __future__ import annotations

import argparse
import hashlib
from itertools import chain
from pathlib import Path
from typing import Sequence

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from simulation.tomato_pick.isaac.agents.residual_lstm_model import ResidualLSTMModel


NORMALIZER_KEYS = (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
    "obs_normalizer.count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--fresh",
        type=Path,
        help="Architecture-compatible checkpoint supplying a fresh critic.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    args = parser.parse_args(argv)
    if not 0.0 < args.learning_rate <= 1.0e-3:
        raise ValueError("--learning-rate must lie in (0, 0.001]")

    source_path = args.source.expanduser().resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    fresh_path = args.fresh.expanduser().resolve() if args.fresh else None
    fresh = (
        torch.load(fresh_path, map_location="cpu", weights_only=False)
        if fresh_path is not None
        else source
    )
    obs = TensorDict(
        {"policy": torch.zeros(1, 66), "critic": torch.zeros(1, 83)}, batch_size=[1]
    )
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = ResidualLSTMModel(
        obs,
        obs_groups,
        "actor",
        11,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.60,
            "std_type": "log",
        },
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        freeze_base=True,
        freeze_obs_normalization=True,
    )
    critic = MLPModel(
        obs,
        obs_groups,
        "critic",
        1,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    source_actor = source["actor_state_dict"]
    actor_state = actor.state_dict()
    missing = sorted(set(source_actor) - set(actor_state))
    if missing:
        raise ValueError(f"source actor keys missing from residual actor: {missing}")
    for name, value in source_actor.items():
        if actor_state[name].shape != value.shape:
            raise ValueError(f"actor shape mismatch for {name}: {actor_state[name].shape} != {value.shape}")
        actor_state[name] = value.detach().clone()
    actor.load_state_dict(actor_state, strict=True)

    if critic.state_dict().keys() != source["critic_state_dict"].keys():
        raise ValueError("source critic architecture does not match minimal-v3 critic")
    if critic.state_dict().keys() != fresh["critic_state_dict"].keys():
        raise ValueError("fresh critic architecture does not match minimal-v3 critic")
    critic_state = {
        name: value.detach().clone()
        for name, value in fresh["critic_state_dict"].items()
    }
    if fresh_path is not None:
        for name in NORMALIZER_KEYS:
            critic_state[name] = source["critic_state_dict"][name].detach().clone()
    critic.load_state_dict(critic_state, strict=True)

    # The only trainable actor parameters are the recurrent residual and the
    # source distribution; the base policy and its normalizer remain frozen.
    optimizer = torch.optim.Adam(
        chain(actor.parameters(), critic.parameters()), lr=args.learning_rate
    )
    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": 0,
        "infos": {
            "architecture": "frozen_mlp_plus_zero_initialized_lstm_residual",
            "source_checkpoint": str(source_path),
            "source_sha256": _sha256(source_path),
            "source_iteration": int(source.get("iter", -1)),
            "fresh_optimizer": True,
            "learning_rate": args.learning_rate,
            "base_actor_frozen": True,
            "actor_normalizer_frozen": True,
            "critic_preserved": fresh_path is None,
            "fresh_critic": fresh_path is not None,
            "fresh_template": str(fresh_path) if fresh_path is not None else None,
            "distribution_preserved": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"TOMATO_RESIDUAL_LSTM_WARMSTART={args.output.resolve()}")
    print(f"SOURCE_SHA256={checkpoint['infos']['source_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
