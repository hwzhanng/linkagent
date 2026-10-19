"""Evaluate EXACTLY one full episode per instance on the 2026 RoTO2 PPO.

Uses the saved pretrained policy/encoder from the upstream Trainer, not a
reimplementation of the Actor. Reports task and held-out full-surface proxy
metrics with correct first-done masks. Never claims actual fruit bruising.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch


@torch.no_grad()
def evaluate_checkpoint(trainer, wrapped, checkpoint: Path, output: Path):
    modules = torch.load(checkpoint, map_location=wrapped.device, weights_only=True)
    for key in ("encoder", "policy"):
        if key not in modules:
            raise ValueError(f"Missing 2026 RoTO2 checkpoint module: {key}")
    agent = trainer.agent
    agent.encoder.load_state_dict(modules["encoder"])
    agent.policy.load_state_dict(modules["policy"])
    if agent.encoder.state_preprocessor is not None and "state_preprocessor" in modules:
        agent.encoder.state_preprocessor.load_state_dict(modules["state_preprocessor"])
    agent.encoder.eval()
    agent.policy.eval()

    states, _ = wrapped.reset(hard=True)
    raw = wrapped._unwrapped
    n = wrapped.num_envs
    pending = torch.ones(n, dtype=torch.bool, device=wrapped.device)
    target_direction = raw._goal.detach().sign()
    signed_progress = torch.zeros(n, device=wrapped.device)
    support = torch.zeros(n, device=wrapped.device)
    pressure_max = torch.zeros(n, device=wrapped.device)
    pressure_sum = torch.zeros(n, device=wrapped.device)
    steps = torch.zeros(n, device=wrapped.device)
    failed = torch.zeros(n, dtype=torch.bool, device=wrapped.device)
    correction_accepted = torch.zeros(n, device=wrapped.device)
    max_steps = int(raw.max_episode_length + 2)
    dt = float(raw.cfg.sim.dt * raw.cfg.decimation)

    for _ in range(max_steps):
        if not bool(pending.any()):
            break
        latents = agent.encoder(states)
        action, _, _ = agent.policy.act(latents, deterministic=True)
        states, _, terminated, truncated, info = wrapped.step(action)
        events = info.get("step", {})
        required = ("rotation_rate", "peak_pressure_pa", "secure_support",
                    "unwrapped_rotation_rad")
        missing = [key for key in required if key not in events]
        if missing:
            raise RuntimeError("Missing current-step evaluation evidence: " + str(missing))
        omega = events["rotation_rate"].reshape(-1)
        proxy_pressure = events["peak_pressure_pa"].reshape(-1)
        support_flag = events["secure_support"].reshape(-1).float()
        signed_progress[pending] += omega[pending] * target_direction[pending] * dt
        support[pending] += support_flag[pending]
        pressure_max[pending] = torch.maximum(
            pressure_max[pending], proxy_pressure[pending]
        )
        pressure_sum[pending] += proxy_pressure[pending]
        correction_accepted[pending] += raw._track.last_accepted[pending].float()
        steps[pending] += 1
        done = (terminated | truncated).reshape(-1)
        failed[pending & done] = terminated.reshape(-1)[pending & done]
        pending &= ~done
    if bool(pending.any()):
        raise RuntimeError("Evaluation episodes did not terminate within episode horizon")
    mean_pressure = pressure_sum / steps.clamp_min(1)
    grip_fraction = support / steps.clamp_min(1)
    success = (
        (signed_progress >= 2.0 * math.pi)
        & (grip_fraction >= 0.65)
        & (pressure_max <= raw.cfg.maximum_pressure_proxy_pa)
        & ~failed
    )
    result = {
        "baseline": "RoTO2_2026_multimodal_rl_unchanged",
        "variant": raw.cfg.pressure_variant,
        "checkpoint": str(checkpoint),
        "episodes": n,
        "success_definition": ">=2pi positive rotation, >=65% secure grip, no fall, below proxy pressure limit",
        "success_fraction": float(success.float().mean()),
        "mean_directed_rotation_radians": float(signed_progress.mean()),
        "mean_directed_rotations": float(signed_progress.mean() / (2 * math.pi)),
        "mean_secure_support_fraction": float(grip_fraction.mean()),
        "mean_peak_pressure_proxy_pa": float(pressure_max.mean()),
        "worst_peak_pressure_proxy_pa": float(pressure_max.max()),
        "mean_per_step_peak_pressure_proxy_pa": float(mean_pressure.mean()),
        "failure_fraction": float(failed.float().mean()),
        "mean_jacobian_acceptance_fraction": float(
            (correction_accepted / steps.clamp_min(1)).mean()
        ),
        "simulation_physx_verified_in_this_chat": False,
        "measured_real_fruit_damage": False,
        "pressure_source": "Gaussian contact-normal-force synthetic surface proxy",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result
