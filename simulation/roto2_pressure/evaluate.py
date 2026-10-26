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
from .reward_geometry import reward_version


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
    target_speed = raw._goal.detach().abs().clone()
    initial_pose = raw._fruit.data.root_pose_w.torch.clone()
    pose_trace, active_trace, support_trace, pad_trace = [], [], [], []
    legacy_support_trace = []
    legacy_support = torch.zeros(n, device=wrapped.device)
    pad_counts = torch.zeros((n, 5), device=wrapped.device)
    pad_quality_sum = pad_counts.clone()
    approach_quality_sum = pad_counts.clone()
    signed_progress = torch.zeros(n, device=wrapped.device)
    support = torch.zeros(n, device=wrapped.device)
    pressure_max = torch.zeros(n, device=wrapped.device)
    pressure_sum = torch.zeros(n, device=wrapped.device)
    steps = torch.zeros(n, device=wrapped.device)
    failed = torch.zeros(n, dtype=torch.bool, device=wrapped.device)
    correction_accepted = torch.zeros(n, device=wrapped.device)
    reward_term_totals = {}
    max_steps = int(raw.max_episode_length + 2)
    dt = float(raw.cfg.sim.dt * raw.cfg.decimation)

    for _ in range(max_steps):
        if not bool(pending.any()):
            break
        latents = agent.encoder(states)
        action, _, _ = agent.policy.act(latents, deterministic=True)
        states, _, terminated, truncated, info = wrapped.step(action)
        events = info.get("step", {})
        for key, value in events.get("reward_terms", {}).items():
            if key not in reward_term_totals:
                reward_term_totals[key] = torch.zeros(n, device=wrapped.device)
            reward_term_totals[key][pending] += value[pending]
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
        support_trace.append((support_flag.bool() & events["pose_step_valid"]).float().cpu().numpy().copy())
        legacy_flag = events.get("legacy_thumb_support", events["secure_support"]).bool()
        legacy_support_trace.append((legacy_flag & events["pose_step_valid"]).cpu().numpy().copy())
        legacy_support[pending] += legacy_flag[pending].float()
        pad_trace.append(events["pad_contact"].cpu().numpy().copy())
        pad_counts[pending] += events["pad_contact"][pending].float()
        pad_quality_sum[pending] += events["pad_quality"][pending]
        approach_quality_sum[pending] += events["pad_approach_quality"][pending]
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
    legacy_supported_turns = (increments*np.array(legacy_support_trace)*active).sum(0)/(2*math.pi)
    upright_fraction = ((mat[1:,...,2,2]>.5)*active).sum(0)/active.sum(0).clip(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), poses=poses, initial_pose=initial_pose.cpu().numpy(),
                        active=active, support=support_trace, legacy_support=legacy_support_trace,
                        pads=pad_trace, target_direction=direction, target_speed=target_speed.cpu().numpy())
    mean_pressure = pressure_sum / steps.clamp_min(1)
    grip_fraction = support / steps.clamp_min(1)
    success = (
        (signed_progress >= 2.0 * math.pi)
        & torch.as_tensor(supported_new_turns >= 1.,device=wrapped.device)
        & torch.as_tensor(upright_fraction >= .95,device=wrapped.device)
        & torch.as_tensor((np.abs(offline_delta)*active).max(0) < .5,device=wrapped.device)
        & (grip_fraction >= 0.65)
        & (pressure_max <= raw.cfg.maximum_pressure_proxy_pa)
        & ~failed
    )
    legacy_fraction = legacy_support / steps.clamp_min(1)
    legacy_success = (
        (signed_progress >= 2.0 * math.pi)
        & torch.as_tensor(legacy_supported_turns >= 1., device=wrapped.device)
        & torch.as_tensor(upright_fraction >= .95, device=wrapped.device)
        & torch.as_tensor((np.abs(offline_delta)*active).max(0) < .5, device=wrapped.device)
        & (legacy_fraction >= .65) & (pressure_max <= raw.cfg.maximum_pressure_proxy_pa) & ~failed
    )
    support_definition = (
        "any >=3 effective pads OR stable object (center displacement<25mm, linear speed<0.12m/s, height drop<15mm); tilt cosine>0.5"
        if raw.cfg.reward_stage == "baseline" else "thumb+>=2 other effective pads; tilt cosine>0.5")
    settling_steps = round(1. / dt)
    after_settle_turns = (offline_delta[settling_steps:]*active[settling_steps:]).sum(0)*direction/(2*math.pi)
    late_steps = round(10. / dt)
    late_turns = (offline_delta[late_steps:]*active[late_steps:]).sum(0)*direction/(2*math.pi)
    small_angle_success = ((after_settle_turns >= 1/12) & (grip_fraction.cpu().numpy() >= .5)
                          & (upright_fraction >= .95) & ~failed.cpu().numpy()
                          & ((np.abs(offline_delta)*active).max(0) < .5)
                          & (pressure_max.cpu().numpy() <= raw.cfg.maximum_pressure_proxy_pa))
    result = {
        "baseline": "RoTO2_2026_multimodal_rl_unchanged",
        "variant": raw.cfg.pressure_variant,
        "checkpoint": str(checkpoint),
        "episodes": n,
        "reward_version": reward_version(raw.cfg.reward_stage), "evaluation_reward_stage": raw.cfg.reward_stage,
        "baseline_pose_weight": raw.cfg.baseline_pose_weight,
        "baseline_approach_weight": raw.cfg.baseline_approach_weight,
        "baseline_approach_pad_count": raw.cfg.baseline_approach_pad_count,
        "baseline_motion_weight": raw.cfg.baseline_motion_weight,
        "baseline_centering_weight": raw.cfg.baseline_centering_weight,
        "baseline_failure_penalty": raw.cfg.baseline_failure_penalty,
        "baseline_pad_quality_weight": raw.cfg.baseline_pad_quality_weight,
        "baseline_offpad_weight": raw.cfg.baseline_offpad_weight,
        "baseline_pressure_weight": raw.cfg.baseline_pressure_weight,
        "baseline_contact_motion_floor": raw.cfg.baseline_contact_motion_floor,
        "domain_profile": raw.cfg.domain_profile,
        "hand_posture": raw.cfg.hand_posture,
        "goal_direction": raw.cfg.goal_direction,
        "target_angular_speed_rad_s": raw.cfg.target_angular_speed_rad_s,
        "mean_reward_term_totals": {k:float(v.mean()) for k,v in reward_term_totals.items()},
        "small_angle_curriculum_success_fraction": float(small_angle_success.mean()),
        "small_angle_curriculum_definition": "after first 1s: >=30deg net target-direction pose rotation, >=50% task-defined support, >=95% upright, no failure, pressure below limit, no pose jump; NOT full-task success",
        "mean_after_settling_directed_turns": float(after_settle_turns.mean()),
        "mean_after_10s_directed_turns": float(late_turns.mean()),
        "direction_results": {str(int(s)): {"episodes":int((direction==s).sum()),
            "mean_directed_turns":float(offline_turns[direction==s].mean()),
            "mean_after_settling_directed_turns":float(after_settle_turns[direction==s].mean())}
            for s in (-1,1) if (direction==s).any()},
        "rotation_source": "independently reconstructed unwrapped material-axis heading from actual quaternion poses",
        "support_definition": support_definition,
        "metric_schema_note": "pad_support_fraction is a historical compatibility alias for task_support_fraction; use support_definition and legacy_thumb_support_fraction to interpret it",
        "success_definition": ">=1 net directed pose turn AND >=1 new turn with task-defined support; >=65% task-defined support; >=95% upright; no physical failure; below pressure limit; no pose jump",
        "legacy_success_definition": ">=1 net directed pose turn AND >=1 new turn with thumb+2 active pads; >=65% legacy support; >=95% upright; no physical failure; below pressure limit; no pose jump",
        "legacy_success_fraction": float(legacy_success.float().mean()),
        "mean_legacy_thumb_support_fraction": float(legacy_fraction.mean()),
        "pose_metric_max_error_turns": float(error),
        "maximum_pose_heading_step_rad": float((np.abs(offline_delta)*active).max()),
        "mean_supported_new_turns": float(supported_new_turns.mean()),
        "mean_per_finger_pad_quality": (pad_quality_sum / steps[:,None].clamp_min(1)).mean(0).cpu().tolist(),
        "mean_per_finger_approach_quality": (approach_quality_sum / steps[:,None].clamp_min(1)).mean(0).cpu().tolist(),
        "mean_per_finger_pad_contact_fraction": (pad_counts / steps[:,None].clamp_min(1)).mean(0).cpu().tolist(),
        "per_episode": [{"env":i,"directed_pose_turns":float(offline_turns[i]),
                         "goal_direction":float(direction[i]),
                         "target_speed_rad_s":float(target_speed[i]),
                         "after_settling_directed_turns":float(after_settle_turns[i]),
                         "after_10s_directed_turns":float(late_turns[i]),
                         "upright_fraction":float(upright_fraction[i]),
                         "small_angle_curriculum_success":bool(small_angle_success[i]),
                         "supported_new_turns":float(supported_new_turns[i]),
                         "pad_support_fraction":float(grip_fraction[i]),
                         "task_support_fraction":float(grip_fraction[i]),
                         "legacy_thumb_support_fraction":float(legacy_fraction[i]),
                         "legacy_supported_new_turns":float(legacy_supported_turns[i]),
                         "legacy_success":bool(legacy_success[i]),
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
