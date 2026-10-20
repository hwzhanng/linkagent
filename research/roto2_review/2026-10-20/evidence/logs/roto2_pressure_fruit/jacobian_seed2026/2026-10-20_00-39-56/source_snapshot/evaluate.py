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
from .reward_geometry import REWARD_VERSION


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
    target_direction = raw._goal.detach().sign().clone()
    initial_pose = raw._fruit.data.root_pose_w.torch.clone()
    pose_trace, active_trace, support_trace = [], [], []
    pad_counts = torch.zeros((n, 5), device=wrapped.device)
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
        pose_trace.append(events["fruit_pose_native"].cpu().numpy().copy())
        active_trace.append(pending.cpu().numpy().copy())
        support_trace.append(support_flag.cpu().numpy().copy())
        pad_counts[pending] += events["pad_contact"][pending].float()
        signed_progress[pending] += events["pose_delta_rad"][pending] * target_direction[pending]
        support[pending] += support_flag[pending]
        pressure_max[pending] = torch.maximum(
            pressure_max[pending], proxy_pressure[pending]
        )
        pressure_sum[pending] += proxy_pressure[pending]
        correction_accepted[pending] += events["jacobian_accepted"][pending].float()
        steps[pending] += 1
        done = (terminated | truncated).reshape(-1)
        failed[pending & done] = terminated.reshape(-1)[pending & done]
        pending &= ~done
    if bool(pending.any()):
        raise RuntimeError("Evaluation episodes did not terminate within episode horizon")
    # Independent offline reconstruction from recorded poses, not reward counters.
    import numpy as np
    from scipy.spatial.transform import Rotation
    poses = np.array(pose_trace)
    all_poses = np.concatenate([initial_pose.cpu().numpy()[None], poses])
    q = all_poses[..., 3:7]
    if not raw._native_xyzw:
        q = q[..., [1,2,3,0]]
    mat = Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(len(q),n,3,3)
    headings = np.arctan2(mat[...,1,0], mat[...,0,0])
    offline_delta = np.arctan2(np.sin(np.diff(headings,axis=0)), np.cos(np.diff(headings,axis=0)))
    active = np.array(active_trace)
    direction = target_direction.cpu().numpy()
    offline_turns = (offline_delta * active).sum(0) * direction / (2*math.pi)
    error = np.max(np.abs(offline_turns - signed_progress.cpu().numpy()/(2*math.pi)))
    if error > 1e-4:
        raise RuntimeError(f'Pose/reward rotation mismatch {error} turns')
    directed_path = np.cumsum(offline_delta*active,axis=0)*direction
    highwater = np.maximum.accumulate(np.maximum(directed_path,0),axis=0)
    increments = np.diff(np.concatenate([np.zeros((1,n)),highwater]),axis=0)
    supported_new_turns = (increments*np.array(support_trace)*active).sum(0)/(2*math.pi)
    upright_fraction = ((mat[1:,...,2,2]>.5)*active).sum(0)/active.sum(0).clip(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), poses=poses, initial_pose=initial_pose.cpu().numpy(),
                        active=active, support=support_trace, target_direction=direction)
    mean_pressure = pressure_sum / steps.clamp_min(1)
    grip_fraction = support / steps.clamp_min(1)
    success = (
        (signed_progress >= 2.0 * math.pi)
        & torch.as_tensor(supported_new_turns >= 1.,device=wrapped.device)
        & torch.as_tensor(upright_fraction >= .95,device=wrapped.device)
        & (grip_fraction >= 0.65)
        & (pressure_max <= raw.cfg.maximum_pressure_proxy_pa)
        & ~failed
    )
    result = {
        "baseline": "RoTO2_2026_multimodal_rl_unchanged",
        "variant": raw.cfg.pressure_variant,
        "checkpoint": str(checkpoint),
        "episodes": n,
        "reward_version": REWARD_VERSION,
        "rotation_source": "independently reconstructed unwrapped material-axis heading from actual quaternion poses",
        "success_definition": ">=1 net directed pose turn AND >=1 new turn with thumb+2 active pads; >=65% pad support; >=95% upright; no physical failure; below pressure limit",
        "pose_metric_max_error_turns": float(error),
        "mean_supported_new_turns": float(supported_new_turns.mean()),
        "mean_per_finger_pad_contact_fraction": (pad_counts / steps[:,None].clamp_min(1)).mean(0).cpu().tolist(),
        "per_episode": [{"env":i,"directed_pose_turns":float(offline_turns[i]),
                         "supported_new_turns":float(supported_new_turns[i]),
                         "pad_support_fraction":float(grip_fraction[i]),
                         "failed":bool(failed[i]),"success":bool(success[i]),
                         "steps":int(steps[i])} for i in range(n)],
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
        "mean_episode_steps": float(steps.mean()),
        "simulation_physx_rollout_completed": True,
        "measured_real_fruit_damage": False,
        "pressure_source": "Gaussian contact-normal-force synthetic surface proxy",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result
