"""Run the v2 staged tactile teacher without PPO (diagnostic only)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument("--action-prior", required=True, help="14 comma-separated weak prior")
p.add_argument("--profile", default="v1_nominal_reliable")
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--steps", type=int, default=280)
p.add_argument("--seed", type=int, default=20261024)
p.add_argument("--tactile-gain", type=float, default=0.08)
p.add_argument("--max-feedback-step", type=float, default=0.03)
p.add_argument("--root-phase-steps", type=int, default=12)
p.add_argument("--proximal-phase-steps", type=int, default=40)
p.add_argument("--distal-phase-steps", type=int, default=90)
p.add_argument("--thumb-first-steps", type=int, default=0)
p.add_argument("--thumb-first-global-scale", type=float, default=0.35)
p.add_argument("--coverage-delta", type=str, default="0.0,0.0,0.0,0.0")
p.add_argument("--motion-smoothing-alpha", type=float, default=1.0)
p.add_argument("--physical-fingertip-proxy", action="store_true")
p.add_argument("--disable-virtual-distal-contact", action="store_true")
p.add_argument("--fingertip-proxy-radius", type=float, default=None)
p.add_argument("--strict-require-measured-distal-contact", action="store_true")
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher, EnvelopingTeacherCfg
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg


@torch.inference_mode()
def main() -> None:
    prior = [float(v.strip()) for v in args.action_prior.split(",") if v.strip()]
    if len(prior) != 14:
        raise ValueError("--action-prior requires 14 values")
    coverage_delta = tuple(float(v.strip()) for v in args.coverage_delta.split(",") if v.strip())
    if len(coverage_delta) != 4 or any(abs(v) > 0.30 for v in coverage_delta):
        raise ValueError("--coverage-delta requires four values in [-0.30,0.30]")
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.strict_enveloping_enabled = True
    cfg.pregrasp_drop_grace_steps = 180
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    cfg.fingertip_collision_proxy_physical = bool(args.physical_fingertip_proxy)
    if args.fingertip_proxy_radius is not None:
        cfg.fingertip_collision_proxy_radius_m = float(args.fingertip_proxy_radius)
    if args.disable_virtual_distal_contact:
        cfg.virtual_distal_contact_enabled = False
    cfg.strict_require_measured_distal_contact = bool(args.strict_require_measured_distal_contact)
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    teacher = EnvelopingTeacher(
        args.num_envs,
        env.device,
        EnvelopingTeacherCfg(
            distal_prior_action=tuple(prior[10:]),
            tactile_gain=args.tactile_gain,
            max_feedback_step=args.max_feedback_step,
            root_phase_steps=args.root_phase_steps,
            proximal_phase_steps=args.proximal_phase_steps,
            distal_phase_steps=args.distal_phase_steps,
            thumb_first_phase_steps=max(0, int(args.thumb_first_steps)),
            thumb_first_global_scale=float(args.thumb_first_global_scale),
            thumb_first_coverage_delta=coverage_delta,
            motion_smoothing_alpha=float(args.motion_smoothing_alpha),
        ),
    )
    teacher.root_prior[:] = torch.tensor(prior, device=env.device).unsqueeze(0)
    teacher.residual_prior[:] = torch.tensor(prior[5:10], device=env.device).unsqueeze(0)
    done_count = success = drop = overload = timeout = distal_all = distal_contact_all = 0
    hold = coverage = palm = 0.0
    phase_counts = torch.zeros(4, dtype=torch.long, device=env.device)
    for _ in range(args.steps):
        action = teacher.command(env.contact_forces, env.palm_force)
        phase_counts += torch.bincount(teacher.phase.clamp(0, 3), minlength=4)
        _, _, term, trunc, _ = env.step(action)
        done = term | trunc
        if bool(done.any()):
            snap = env.consume_terminal_snapshots(done)
            code = snap["termination_code"]
            done_count += int(done.sum())
            success += int((code == 1).sum())
            drop += int((code == 2).sum())
            overload += int((code == 3).sum())
            timeout += int((code == 5).sum())
            distal_all += int(snap["strict_distal_participation"].all(-1).sum())
            distal_contact_all += int(snap["distal_contact_observed"].all(-1).sum())
            hold += float(snap["strict_hold_max"].float().sum())
            coverage += float(snap["strict_coverage_max"].float().sum())
            palm += float((snap["palm_force"] >= env.cfg.palm_support_force_n).float().sum())
            teacher.reset(torch.nonzero(done, as_tuple=False).flatten())
    den = max(done_count, 1)
    out = {
        "version": "v2_teacher_probe_v1",
        "profile": args.profile,
        "seed": args.seed,
        "completed_episodes": done_count,
        "success_rate": success / den,
        "drop_rate": drop / den,
        "overload_rate": overload / den,
        "timeout_rate": timeout / den,
        "distal_all_rate": distal_all / den,
        "distal_contact_all_rate": distal_contact_all / den,
        "strict_hold_max_mean": hold / den,
        "strict_coverage_max_mean_deg": coverage / den,
        "palm_support_rate": palm / den,
        "teacher_phase_counts": phase_counts.cpu().tolist(),
        "teacher_thumb_first_steps": int(args.thumb_first_steps),
        "teacher_coverage_delta": list(coverage_delta),
        "teacher_motion_smoothing_alpha": float(args.motion_smoothing_alpha),
        "hardware_command_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
