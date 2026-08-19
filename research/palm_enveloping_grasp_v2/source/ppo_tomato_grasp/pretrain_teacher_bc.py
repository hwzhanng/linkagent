"""Weighted behavior cloning for the transferred Clean-v5 hand labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


class BCActor(nn.Module):
    """Actor topology shared with the hand-only asymmetric PPO runner."""

    def __init__(self) -> None:
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(110, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.actor_mean = nn.Linear(128, 10)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor_mean(self.actor_body(observation)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initialize", type=Path, default=None)
    parser.add_argument("--coarse-loss-weight", type=float, default=1.0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    datasets = [torch.load(path, map_location="cpu", weights_only=False) for path in args.dataset]
    observation = torch.cat([data["observation"].float() for data in datasets])
    action = torch.cat([data["action"].float() for data in datasets])
    weight = torch.cat([data["weight"].float() for data in datasets])
    if observation.shape[1:] != (110,) or action.shape != (len(observation), 10):
        raise ValueError("dataset contract mismatch")
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(observation), generator=generator)
    split = max(1, int(0.9 * len(order)))
    train_ids, validation_ids = order[:split], order[split:]
    model = BCActor().to(device)
    if args.initialize is not None:
        initial = torch.load(args.initialize, map_location=device, weights_only=False)
        if (initial.get("actor_observation_dim"), initial.get("action_dim"), initial.get("converged")) != (110, 10, True):
            raise ValueError("initializer is not a converged 110D/10D BC checkpoint")
        model.load_state_dict(initial["actor_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    dimension_weight = torch.ones(10, device=device)
    dimension_weight[:5] = args.coarse_loss_weight
    best_loss = math.inf
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = train_ids[torch.randperm(len(train_ids), generator=generator)]
        train_loss_sum = 0.0
        for start in range(0, len(shuffled), args.batch_size):
            ids = shuffled[start : start + args.batch_size]
            obs = observation[ids].to(device)
            target = action[ids].to(device)
            sample_weight = weight[ids].to(device)
            error = (
                (model(obs) - target).square() * dimension_weight
            ).sum(dim=-1) / dimension_weight.sum()
            loss = (error * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss) * len(ids)
        model.eval()
        with torch.inference_mode():
            predicted = model(observation[validation_ids].to(device))
            validation_loss = float(
                (predicted - action[validation_ids].to(device)).square().mean()
            )
        train_loss = train_loss_sum / len(train_ids)
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    initial_loss = history[0]["validation_mse"]
    converged = best_loss < 0.25 * initial_loss and best_loss < 5.0e-3
    payload = {
        "version": "left_hand_teacher_bc_v1",
        "actor_observation_dim": 110,
        "action_dim": 10,
        "actor_state_dict": best_state,
        "dataset": [str(path.resolve()) for path in args.dataset],
        "dataset_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.dataset],
        "epochs": args.epochs,
        "initial_validation_mse": initial_loss,
        "best_validation_mse": best_loss,
        "converged": converged,
        "initialized_from": str(args.initialize.resolve()) if args.initialize is not None else None,
        "coarse_loss_weight": args.coarse_loss_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {key: value for key, value in payload.items() if key != "actor_state_dict"}
    summary["checkpoint"] = str(args.output.resolve())
    summary["checkpoint_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary["last_epoch"] = history[-1]
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print("TEACHER_BC=" + json.dumps(summary), flush=True)
    if not converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
