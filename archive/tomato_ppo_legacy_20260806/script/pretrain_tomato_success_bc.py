"""Warm-start the tomato PPO actor from PPO-generated successful episodes.

Only the actor MLP is optimized.  The source actor observation normalizer is
preserved, the action distribution standard deviation is freshly initialized,
and both the critic and PPO Adam optimizer are fresh in the output checkpoint.
The saved checkpoint is directly loadable by the normal RSL-RL runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from itertools import chain
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = importlib.import_module("torch")
TensorDict = importlib.import_module("tensordict").TensorDict
MLPModel = importlib.import_module("rsl_rl.models").MLPModel


ACTOR_OBSERVATION_DIM = 113
CRITIC_OBSERVATION_DIM = 153
ACTION_DIM = 22
NUM_PHASES = 4
FRESH_ACTION_STD = 0.35


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bc-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--ppo-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--minimum-episodes", type=int, default=32)
    parser.add_argument("--minimum-validation-improvement", type=float, default=1.0e-4)
    parser.add_argument(
        "--anchor-contact-to-source",
        action="store_true",
        help="preserve the proven CONTACT policy while cloning successful downstream actions",
    )
    parser.add_argument(
        "--contact-anchor-weight",
        type=float,
        default=4.0,
        help="relative CONTACT source-policy anchor weight when the anchor is enabled",
    )
    parser.add_argument(
        "--allow-deterministic-dataset",
        action="store_true",
        help="testing escape hatch; production self-imitation must use stochastic successes",
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_success_dataset(
    path: Path,
    *,
    minimum_episodes: int,
    require_stochastic: bool,
) -> dict[str, Any]:
    """Load and strictly validate the success-only trajectory dataset."""

    dataset = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(dataset, dict) or dataset.get("schema_version") != 1:
        raise ValueError("success dataset must use schema_version=1")
    required = {"observations", "actions", "phases", "episode_offsets", "metadata"}
    missing = required.difference(dataset)
    if missing:
        raise ValueError(f"success dataset is missing keys: {sorted(missing)}")

    observations = dataset["observations"]
    actions = dataset["actions"]
    phases = dataset["phases"]
    offsets = dataset["episode_offsets"]
    metadata = dataset["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("dataset metadata must be a dictionary")
    if observations.ndim != 2 or observations.shape[1] != ACTOR_OBSERVATION_DIM:
        raise ValueError(f"observations must have shape [N, {ACTOR_OBSERVATION_DIM}]")
    if actions.shape != (observations.shape[0], ACTION_DIM):
        raise ValueError(f"actions must have shape [N, {ACTION_DIM}]")
    if phases.shape != (observations.shape[0],):
        raise ValueError("phases must have shape [N]")
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("episode_offsets must contain at least one episode")
    offsets = offsets.long()
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(observations):
        raise ValueError("episode_offsets must span exactly all transitions")
    if not bool((offsets[1:] > offsets[:-1]).all()):
        raise ValueError("episode_offsets must be strictly increasing")
    episode_count = len(offsets) - 1
    if episode_count < minimum_episodes:
        raise ValueError(
            f"dataset has {episode_count} successful episodes; require at least {minimum_episodes}"
        )
    if not bool(torch.isfinite(observations).all()):
        raise ValueError("observations contain NaN or Inf")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("actions contain NaN or Inf")
    if not bool(((actions >= -1.0) & (actions <= 1.0)).all()):
        raise ValueError("actions exceed [-1, 1]")
    if not bool(((phases >= 0) & (phases < NUM_PHASES)).all()):
        raise ValueError("phases must be integers in [0, 3]")
    phase_counts = torch.bincount(phases.long(), minlength=NUM_PHASES)
    if bool((phase_counts == 0).any()):
        raise ValueError(f"every phase must be present, got counts {phase_counts.tolist()}")
    if require_stochastic and bool(metadata.get("deterministic", False)):
        raise ValueError(
            "production self-imitation requires stochastic successful actions; "
            "deterministic rollouts only reproduce the source mean"
        )

    dataset["observations"] = observations.float().contiguous()
    dataset["actions"] = actions.float().contiguous()
    dataset["phases"] = phases.long().contiguous()
    dataset["episode_offsets"] = offsets.contiguous()
    return dataset


def split_episode_indices(
    offsets: torch.Tensor,
    validation_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Split whole episodes, never individual transitions, into train/validation."""

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    episode_count = len(offsets) - 1
    if episode_count < 2:
        raise ValueError("at least two episodes are required for a split")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(episode_count, generator=generator).tolist()
    validation_count = max(1, min(episode_count - 1, round(episode_count * validation_fraction)))
    validation_episodes = sorted(permutation[:validation_count])
    training_episodes = sorted(permutation[validation_count:])

    def transitions(episodes: list[int]) -> torch.Tensor:
        chunks = [
            torch.arange(int(offsets[index]), int(offsets[index + 1]), dtype=torch.long)
            for index in episodes
        ]
        return torch.cat(chunks)

    return (
        transitions(training_episodes),
        transitions(validation_episodes),
        training_episodes,
        validation_episodes,
    )


def phase_balanced_indices(
    phases: torch.Tensor,
    eligible_indices: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Resample each phase to the largest phase population for one BC epoch."""

    phase_pools = [eligible_indices[phases[eligible_indices] == phase] for phase in range(NUM_PHASES)]
    if any(len(pool) == 0 for pool in phase_pools):
        counts = [len(pool) for pool in phase_pools]
        raise ValueError(f"training split must contain all phases, got {counts}")
    target_count = max(len(pool) for pool in phase_pools)
    balanced = []
    for pool in phase_pools:
        if len(pool) < target_count:
            choice = torch.randint(len(pool), (target_count,), generator=generator)
            balanced.append(pool[choice])
        else:
            balanced.append(pool[torch.randperm(len(pool), generator=generator)])
    joined = torch.cat(balanced)
    return joined[torch.randperm(len(joined), generator=generator)]


def _distribution_cfg() -> dict[str, Any]:
    return {
        "class_name": (
            "simulation.tomato_pick.isaac.agents.squashed_gaussian:"
            "SquashedGaussianDistribution"
        ),
        "init_std": FRESH_ACTION_STD,
        "min_std": 0.05,
        "max_std": 0.80,
        "max_latent_mean": 4.0,
        "epsilon": 1.0e-6,
    }


def _make_models(device: torch.device) -> tuple[MLPModel, MLPModel]:
    dummy = TensorDict(
        {
            "policy": torch.zeros((1, ACTOR_OBSERVATION_DIM), device=device),
            "critic": torch.zeros((1, CRITIC_OBSERVATION_DIM), device=device),
        },
        batch_size=[1],
        device=device,
    )
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(
        dummy,
        groups,
        "actor",
        ACTION_DIM,
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution_cfg(),
    ).to(device)
    critic = MLPModel(
        dummy,
        groups,
        "critic",
        1,
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    ).to(device)
    return actor, critic


@torch.inference_mode()
def _mse_metrics(
    actor: MLPModel,
    observations: torch.Tensor,
    actions: torch.Tensor,
    phases: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    squared_error_sum = 0.0
    element_count = 0
    phase_sum = torch.zeros(NUM_PHASES, dtype=torch.float64)
    phase_count = torch.zeros(NUM_PHASES, dtype=torch.long)
    actor.eval()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_observations = observations[batch_indices].to(device)
        target = actions[batch_indices].to(device)
        tensor_dict = TensorDict({"policy": batch_observations}, batch_size=[len(batch_indices)])
        prediction = actor(tensor_dict)
        error = (prediction - target).square()
        squared_error_sum += float(error.sum())
        element_count += error.numel()
        sample_error = error.mean(dim=1).detach().cpu().double()
        batch_phases = phases[batch_indices]
        for phase in range(NUM_PHASES):
            selected = batch_phases == phase
            if bool(selected.any()):
                phase_sum[phase] += sample_error[selected].sum()
                phase_count[phase] += int(selected.sum())
    return {
        "mse": squared_error_sum / element_count,
        "phase_mse": [
            float(phase_sum[phase] / phase_count[phase]) if phase_count[phase] else None
            for phase in range(NUM_PHASES)
        ],
        "transitions": len(indices),
    }


@torch.inference_mode()
def _predict_actions(
    actor: MLPModel,
    observations: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    predictions = []
    actor.eval()
    for start in range(0, len(observations), batch_size):
        batch = observations[start : start + batch_size].to(device)
        tensor_dict = TensorDict({"policy": batch}, batch_size=[len(batch)])
        predictions.append(actor(tensor_dict).detach().cpu())
    return torch.cat(predictions)


def main() -> int:
    args = _parser().parse_args()
    for path in (args.dataset, args.source_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.resolve() == args.source_checkpoint.resolve():
        raise ValueError("output must not overwrite the source checkpoint")
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch size must be positive")
    if args.minimum_validation_improvement < 0.0:
        raise ValueError("minimum validation improvement must be non-negative")
    if args.contact_anchor_weight <= 0.0:
        raise ValueError("contact-anchor-weight must be positive")
    for value, name in (
        (args.bc_learning_rate, "bc-learning-rate"),
        (args.ppo_learning_rate, "ppo-learning-rate"),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1)")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    dataset = load_success_dataset(
        args.dataset,
        minimum_episodes=args.minimum_episodes,
        require_stochastic=not args.allow_deterministic_dataset,
    )
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in source:
        raise ValueError("source checkpoint does not contain actor_state_dict")
    expected_source_hash = dataset["metadata"].get("source_checkpoint_sha256")
    actual_source_hash = _sha256(args.source_checkpoint)
    if expected_source_hash and expected_source_hash != actual_source_hash:
        raise ValueError("dataset source checkpoint hash does not match --source-checkpoint")

    actor, critic = _make_models(device)
    fresh_std_parameter = actor.state_dict()["distribution.std_logit_param"].clone()
    source_actor_state = source["actor_state_dict"]
    transferred_state = {
        name: value
        for name, value in source_actor_state.items()
        if name != "distribution.std_logit_param"
    }
    missing, unexpected = actor.load_state_dict(transferred_state, strict=False)
    if set(missing) != {"distribution.std_logit_param"} or unexpected:
        raise ValueError(f"unexpected actor transfer mismatch: missing={missing}, unexpected={unexpected}")
    if not torch.equal(actor.state_dict()["distribution.std_logit_param"].cpu(), fresh_std_parameter.cpu()):
        raise RuntimeError("fresh action standard deviation was overwritten")
    normalizer_names = [name for name in source_actor_state if name.startswith("obs_normalizer.")]
    for name in normalizer_names:
        if not torch.equal(actor.state_dict()[name].cpu(), source_actor_state[name].cpu()):
            raise RuntimeError(f"source observation normalization changed while loading: {name}")

    observations = dataset["observations"]
    actions = dataset["actions"]
    phases = dataset["phases"]
    offsets = dataset["episode_offsets"]
    source_mean_actions = _predict_actions(actor, observations, device, args.batch_size)
    training_targets = actions.clone()
    if args.anchor_contact_to_source:
        contact = phases == 0
        training_targets[contact] = source_mean_actions[contact]
    train_indices, validation_indices, train_episodes, validation_episodes = split_episode_indices(
        offsets, args.validation_fraction, args.seed
    )
    for split_name, split_indices in (
        ("training", train_indices),
        ("validation", validation_indices),
    ):
        counts = torch.bincount(phases[split_indices], minlength=NUM_PHASES)
        if bool((counts == 0).any()):
            raise ValueError(f"{split_name} split is missing phases: {counts.tolist()}")

    baseline_train = _mse_metrics(
        actor, observations, training_targets, phases, train_indices, device, args.batch_size
    )
    baseline_validation = _mse_metrics(
        actor, observations, training_targets, phases, validation_indices, device, args.batch_size
    )
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    for parameter in actor.mlp.parameters():
        parameter.requires_grad_(True)
    bc_optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=args.bc_learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 1)
    best_validation_mse = math.inf
    best_mlp_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale_epochs = 0
    epochs_run = 0

    for epoch in range(args.epochs):
        actor.train()
        epoch_indices = phase_balanced_indices(phases, train_indices, generator)
        for start in range(0, len(epoch_indices), args.batch_size):
            batch_indices = epoch_indices[start : start + args.batch_size]
            batch_observations = observations[batch_indices].to(device)
            targets = training_targets[batch_indices].to(device)
            tensor_dict = TensorDict({"policy": batch_observations}, batch_size=[len(batch_indices)])
            predictions = actor(tensor_dict)
            sample_loss = (predictions - targets).square().mean(dim=1)
            if args.anchor_contact_to_source:
                weights = torch.ones_like(sample_loss)
                weights[phases[batch_indices].to(device) == 0] = args.contact_anchor_weight
                loss = (sample_loss * weights).sum() / weights.sum()
            else:
                loss = sample_loss.mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("BC loss became NaN or Inf")
            bc_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.0)
            bc_optimizer.step()

        validation = _mse_metrics(
            actor,
            observations,
            training_targets,
            phases,
            validation_indices,
            device,
            args.batch_size,
        )
        epochs_run = epoch + 1
        if validation["mse"] < best_validation_mse - 1.0e-8:
            best_validation_mse = validation["mse"]
            best_epoch = epoch
            best_mlp_state = {
                name: value.detach().cpu().clone() for name, value in actor.mlp.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_mlp_state is None:
        raise RuntimeError("BC did not produce a finite validation checkpoint")
    actor.mlp.load_state_dict(best_mlp_state)
    final_train = _mse_metrics(
        actor, observations, training_targets, phases, train_indices, device, args.batch_size
    )
    final_validation = _mse_metrics(
        actor,
        observations,
        training_targets,
        phases,
        validation_indices,
        device,
        args.batch_size,
    )
    raw_success_validation = _mse_metrics(
        actor, observations, actions, phases, validation_indices, device, args.batch_size
    )
    validation_improvement = baseline_validation["mse"] - final_validation["mse"]
    if validation_improvement < args.minimum_validation_improvement:
        raise RuntimeError(
            "validation MSE did not improve enough: "
            f"{validation_improvement:.8f} < {args.minimum_validation_improvement:.8f}"
        )

    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    dummy_obs = TensorDict(
        {"policy": observations[:1].to(device)}, batch_size=[1], device=device
    )
    with torch.inference_mode():
        actor(dummy_obs, stochastic_output=True)
        action_std = actor.output_std[0].detach().cpu()
    if not torch.allclose(action_std, torch.full_like(action_std, FRESH_ACTION_STD), atol=1.0e-6):
        raise RuntimeError(f"fresh action std is not {FRESH_ACTION_STD}: {action_std.tolist()}")
    for model_name, model in (("actor", actor), ("critic", critic)):
        nonfinite = [
            name for name, value in model.state_dict().items() if not bool(torch.isfinite(value).all())
        ]
        if nonfinite:
            raise RuntimeError(f"{model_name} contains non-finite state: {nonfinite}")

    ppo_optimizer = torch.optim.Adam(
        chain(actor.parameters(), critic.parameters()), lr=args.ppo_learning_rate
    )
    optimizer_state = ppo_optimizer.state_dict()
    if optimizer_state["state"]:
        raise RuntimeError("fresh PPO optimizer unexpectedly contains moment state")
    report = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": actual_source_hash,
        "success_episodes": len(offsets) - 1,
        "transitions": len(observations),
        "phase_counts": torch.bincount(phases, minlength=NUM_PHASES).tolist(),
        "training_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "baseline_train": baseline_train,
        "baseline_validation": baseline_validation,
        "final_train": final_train,
        "final_validation": final_validation,
        "raw_success_validation": raw_success_validation,
        "validation_improvement": validation_improvement,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "actor_observation_normalizer_preserved": True,
        "action_std": action_std.tolist(),
        "critic_fresh": True,
        "ppo_optimizer_fresh": True,
        "bc_updates": "actor.mlp_only",
        "contact_anchor_to_source": args.anchor_contact_to_source,
        "contact_anchor_weight": args.contact_anchor_weight if args.anchor_contact_to_source else None,
        "seed": args.seed,
    }
    checkpoint = {
        "actor_state_dict": {name: value.detach().cpu() for name, value in actor.state_dict().items()},
        "critic_state_dict": {name: value.detach().cpu() for name, value in critic.state_dict().items()},
        "optimizer_state_dict": optimizer_state,
        "iter": 0,
        "infos": {"success_bc": report},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output)
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = _sha256(args.output)
    print("TOMATO_SUCCESS_BC=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
