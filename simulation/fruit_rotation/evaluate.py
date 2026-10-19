"""Deterministic 24-s simulation evaluation for single-hand continuous rotation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .core import PRESSURE_REF_PA
from .ppo import ActorCritic


@torch.no_grad()
def evaluate(env, checkpoint: Path, output: Path, episodes: int | None = None):
    """Evaluate first completed rollout per parallel environment.

    Episode counts > num_envs are not supported by this bounded harness:
    spawn desired evaluation batch with --num-envs first.
    """
    checkpoint = Path(checkpoint)
    data = torch.load(checkpoint, map_location=env.device, weights_only=False)
    model = ActorCritic().to(env.device)
    model.load_state_dict(data["actor_critic"])
    model.eval()
    observations, _ = env.reset()
    n = env.num_envs
    if episodes is not None and episodes > n:
        raise ValueError("For exact independent episodes set --num-envs >= --eval-episodes")
    selected = torch.arange(n, device=env.device) < (episodes or n)
    pending = selected.clone()
    summed_angle = torch.zeros(n, device=env.device)
    signed_progress = torch.zeros(n, device=env.device)
    maximum_proxy = torch.zeros(n, device=env.device)
    supported_steps = torch.zeros(n, device=env.device)
    counted_steps = torch.zeros(n, device=env.device)
    failed = torch.zeros(n, device=env.device, dtype=torch.bool)
    desired = env._goal.clone()
    dt = float(env.cfg.sim.dt * env.cfg.decimation)
    while pending.any():
        action = model.mean(observations["policy"]).tanh()
        observations, _, terminated, truncated, info = env.step(action)
        step_info = info.get("step", {})
        omega = step_info.get("rotation_rate")
        pressure = step_info.get("peak_pressure_pa")
        secure = step_info.get("secure_support")
        if omega is None or pressure is None or secure is None:
            raise RuntimeError("Rotation evaluation telemetry not populated")
        summed_angle[pending] += omega[pending] * dt
        signed_progress[pending] += (
            omega[pending] * desired[pending].sign() * dt
        ).clamp(min=-0.2, max=0.2)
        maximum_proxy[pending] = torch.maximum(
            maximum_proxy[pending], pressure[pending]
        )
        supported_steps[pending] += secure[pending].float()
        counted_steps[pending] += 1
        finished = pending & (terminated | truncated)
        failed[finished] = terminated[finished]
        pending &= ~finished

    useful = selected
    fraction = supported_steps / counted_steps.clamp_min(1)
    passed = (
        (signed_progress >= (2.0 * math.pi))
        & (fraction >= 0.65)
        & (maximum_proxy < env.cfg.maximum_pressure_proxy_pa)
        & ~failed
    )
    payload = {
        "checkpoint": str(checkpoint),
        "episodes": int(useful.sum().item()),
        "target": "signed in-hand rotation >= 360 deg in 24 s, supported >= 65% of steps",
        "success_fraction": passed[useful].float().mean().item(),
        "mean_signed_rotation_rad": signed_progress[useful].mean().item(),
        "mean_absolute_net_rotation_rad": summed_angle[useful].abs().mean().item(),
        "mean_supported_fraction": fraction[useful].mean().item(),
        "maximum_proxy_pressure_pa": maximum_proxy[useful].max().item(),
        "failure_fraction": failed[useful].float().mean().item(),
        "physical_damage_validated": False,
        "fruit_surface_pressure": "96-site Gaussian force/position proxy",
        "notes": "No tissue mechanics, real pressure calibration, or bruise outcome implied",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload
