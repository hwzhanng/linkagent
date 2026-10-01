"""Nominal residual-PPO trainer for the v3 direct flexion contract.

The policy learns a bounded correction on top of the scripted event teacher;
success is judged by the env's own strict telemetry (sequential gate aware).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
DEFAULT_SCENARIO = Path(__file__).resolve().parent / "scenarios/v3_large100_nominal.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
parser.add_argument("--override", type=Path)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--updates", type=int, default=300)
parser.add_argument("--horizon", type=int, default=96)
parser.add_argument("--seed", type=int, default=20260822)
parser.add_argument("--residual-scale", type=float, default=0.10)
parser.add_argument("--lr", type=float, default=3.0e-4)
parser.add_argument("--entropy-coef", type=float, default=0.003)
parser.add_argument("--eval-interval", type=int, default=20)
parser.add_argument("--save-interval", type=int, default=10)
parser.add_argument("--log-std-init", type=float, default=-1.0)
parser.add_argument("--resume", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
from torch import nn
from torch.distributions import Normal

from simulation.ppo_tomato_grasp_v2.contract import V2_ACTION_DIM, V2_ACTOR_OBSERVATION_DIM
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v3.direct_joint_teacher import (
    DirectJointEventTeacher,
    DirectJointTeacherCfg,
)
from simulation.ppo_tomato_grasp_v3.env import (
    V3EnvelopingPhysxTomatoGraspEnv,
    V3EnvelopingPhysxTomatoGraspEnvCfg,
)
from simulation.ppo_tomato_grasp_v3.provenance import sha256_file
from simulation.ppo_tomato_grasp_v3.scenario import (
    apply_contact_collision_mode,
    canonical_hash,
    load_scenario,
    merge_scenario_override,
)

OBS_DIM = V2_ACTOR_OBSERVATION_DIM
ACT_DIM = V2_ACTION_DIM


def make_cfg(scenario: dict) -> V3EnvelopingPhysxTomatoGraspEnvCfg:
    cfg = V3EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args_cli.device
    cfg.training_stage = 1
    cfg.strict_enveloping_enabled = True
    cfg.strict_enveloping_require_stem_break = False
    cfg = apply_profile(cfg, scenario["evaluation"]["profile"], num_envs=args_cli.num_envs, seed=args_cli.seed)
    task, obj = scenario["task"], scenario["object"]
    geometry, strict = scenario["contact_geometry"], scenario["strict"]
    safety, controller = scenario["safety"], scenario["controller"]
    physics = scenario["physics"]
    cfg.position_jitter = tuple(float(v) for v in task["initial_position_jitter_m"])
    cfg.axis_range_m = (float(obj["semi_axes_m"][0]), float(obj["semi_axes_m"][0]))
    cfg.isotropic_axes = True
    cfg.accepted_mass_kg = tuple(float(v) for v in obj["accepted_mass_kg"])
    cfg.density_range_kg_m3 = (float(obj["density_kg_m3"]), float(obj["density_kg_m3"]))
    cfg.events["tomato_geometry"].params.update(
        axis_range_m=cfg.axis_range_m,
        isotropic_axes=True,
        accepted_mass_kg=cfg.accepted_mass_kg,
        density_range_kg_m3=cfg.density_range_kg_m3,
    )
    cfg.tomato_contact_offset_m = float(obj["contact_offset_m"])
    cfg.tomato_rest_offset_m = float(obj["rest_offset_m"])
    cfg.tomato_reset_clearance_m = float(obj["reset_clearance_m"])
    cfg.friction_range = (float(obj["friction"]), float(obj["friction"]))
    cfg.restitution_range = (float(obj["restitution"]), float(obj["restitution"]))
    cfg.tomato_cfg.spawn.physics_material.compliant_contact_stiffness = float(obj["contact_stiffness_n_m"])
    apply_contact_collision_mode(cfg, geometry)
    cfg.fingertip_collision_proxy_radius_m = float(geometry["fingertip_proxy_radius_m"])
    cfg.fingertip_proxy_contact_offset_m = float(geometry["proxy_contact_offset_m"])
    cfg.fingertip_proxy_rest_offset_m = float(geometry["proxy_rest_offset_m"])
    cfg.virtual_distal_contact_enabled = bool(geometry["virtual_distal_contact_for_shaping"])
    cfg.palm_cup_inner_half_extent_m = tuple(float(v) for v in geometry["palm_cup_inner_half_extent_m"])
    cfg.palm_cup_rim_height_m = float(geometry["palm_cup_rim_height_m"])
    cfg.strict_angular_coverage_min_deg = float(strict["angular_coverage_min_deg"])
    cfg.strict_thumb_opposition_cos_max = float(strict["thumb_opposition_cos_max"])
    cfg.strict_same_hemisphere_score_max = float(strict["same_hemisphere_score_max"])
    cfg.strict_palm_load_share_min = float(strict["palm_load_share"][0])
    cfg.strict_palm_load_share_max = float(strict["palm_load_share"][1])
    cfg.strict_palm_first_margin_steps = int(strict["palm_first_margin_steps"])
    cfg.strict_hold_steps = int(strict["diagnostic_hold_steps"])
    cfg.v3_sequential_contact_gate = bool(strict.get("sequential_contact_allowed", False))
    cfg.safe_force_n = float(safety["safe_force_n"])
    cfg.force_ceiling_n = tuple(float(v) for v in safety["force_ceiling_n"])
    cfg.command_rate_limit_rad_s = float(safety["command_rate_limit_rad_s"])
    cfg.distal_action_filter_alpha = float(controller["distal_filter_alpha"])
    cfg.v3_urdf_terminal_mimic_enabled = bool(physics["urdf_terminal_mimic_enabled"])
    cfg.v3_urdf_mimic_servo_rate_rad_s = float(physics["urdf_mimic_servo_rate_rad_s"])
    cfg.v3_action_contract = str(controller["action_contract"])
    cfg.v3_direct_servo_zero_velocity = bool(physics["direct_servo_zero_velocity"])
    cfg.position_servo_enabled = bool(physics["direct_position_servo_enabled"])
    cfg.v3_filter_fingertip_proxy_support_collision = bool(
        physics["filter_fingertip_proxy_support_collision"]
    )
    cfg.v3_disable_palm_cup_rim_collision = bool(physics["disable_palm_cup_rim_collision"])
    cfg.v3_joint_drive_effort_limit_nm = float(physics["joint_drive_effort_limit_nm"])
    cfg.v3_joint_drive_stiffness_nm_rad = float(physics["joint_drive_stiffness_nm_rad"])
    cfg.v3_joint_drive_damping_nm_s_rad = float(physics["joint_drive_damping_nm_s_rad"])
    cfg.v3_mimic_drive_effort_limit_nm = float(physics["mimic_drive_effort_limit_nm"])
    cfg.v3_mimic_drive_stiffness_nm_rad = float(physics["mimic_drive_stiffness_nm_rad"])
    cfg.v3_mimic_drive_damping_nm_s_rad = float(physics["mimic_drive_damping_nm_s_rad"])
    cfg.v3_mimic_drive_armature_kg_m2 = float(physics["mimic_drive_armature_kg_m2"])
    cfg.v3_hand_disable_gravity = bool(physics["hand_disable_gravity"])
    cfg.v3_use_tomato_filtered_contact_force = bool(physics["use_tomato_filtered_contact_force"])
    cfg.v3_contact_history_substeps = int(physics["contact_history_substeps"])
    required_steps = (
        int(controller["approach_steps"]) + 4 * int(controller["per_digit_steps"])
        + int(controller["hold_steps"]) + 20
    )
    cfg.episode_length_s = max(24.0, required_steps * float(cfg.policy_dt_s))
    return cfg


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.mean = nn.Linear(128, ACT_DIM)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean(self.body(observation)))


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.value = nn.Linear(128, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(self.body(observation))


def checkpoint_payload(actor: Actor, extra: dict | None = None) -> dict:
    model = {
        "actor_body.0.weight": actor.body[0].weight.data,
        "actor_body.0.bias": actor.body[0].bias.data,
        "actor_body.2.weight": actor.body[2].weight.data,
        "actor_body.2.bias": actor.body[2].bias.data,
        "actor_body.4.weight": actor.body[4].weight.data,
        "actor_body.4.bias": actor.body[4].bias.data,
        "actor_mean.weight": actor.mean.weight.data,
        "actor_mean.bias": actor.mean.bias.data,
    }
    payload = {
        "actor_observation_dim": OBS_DIM,
        "action_dim": ACT_DIM,
        "model": {k: v.detach().cpu() for k, v in model.items()},
    }
    if extra:
        payload.update(extra)
    return payload


@torch.inference_mode()
def greedy_eval(env, teacher, actor, obs, steps: int, scale: float) -> dict:
    actor.eval()
    sims, covs, seqs = [], [], []
    for _ in range(steps):
        base = teacher.command(
            env.episode_length_buf,
            env._v2_measured_link4_force,
            env.palm_force,
            torch.linalg.vector_norm(env.tomato.data.root_lin_vel_w, dim=-1),
            torch.linalg.vector_norm(
                env._tip_positions() - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1
            ),
            env.robot.data.joint_pos[:, env.active_joint_ids],
        )
        residual = actor(obs)
        action = (base + scale * residual).clamp(-1.0, 1.0)
        observation, _, _, _, _ = env.step(action)
        obs = (
            observation["policy"] if isinstance(observation, dict) else observation
        ).detach()
        sims.append(env._v3_max_simultaneous_physical.float().mean().item())
        covs.append(env._v3_physical_coverage_current.mean().item())
        seqs.append(env._v3_strict_success_latched.float().mean().item())
    actor.train()
    return {
        "sim_digits_mean": sum(sims) / len(sims),
        "coverage_current_mean": sum(covs) / len(covs),
        "seq_success_rate": sum(seqs) / len(seqs),
    }


def main() -> None:
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = args_cli.scenario.expanduser().resolve()
    scenario, scenario_hash = load_scenario(scenario_path)
    if args_cli.override is not None:
        override_path = args_cli.override.expanduser().resolve()
        override = json.loads(override_path.read_text(encoding="utf-8"))
        scenario, scenario_hash = merge_scenario_override(scenario, override)

    torch.manual_seed(args_cli.seed)
    device = torch.device(args_cli.device)
    cfg = make_cfg(scenario)
    env = V3EnvelopingPhysxTomatoGraspEnv(cfg)
    observation, _ = env.reset(seed=args_cli.seed)

    controller_cfg_block = scenario["controller"]["direct_teacher"]
    teacher = DirectJointEventTeacher(
        args_cli.num_envs,
        device,
        tuple(float(v) for v in scenario["controller"]["target_action"]),
        DirectJointTeacherCfg(
            coarse_steps=int(controller_cfg_block["coarse_steps"]),
            distal_steps=int(controller_cfg_block["distal_steps"]),
            force_lower_n=tuple(float(v) for v in controller_cfg_block["force_lower_n"]),
            force_upper_n=tuple(float(v) for v in controller_cfg_block["force_upper_n"]),
            retreat_action_step=float(controller_cfg_block["retreat_action_step"]),
            hold_closure_margin=float(controller_cfg_block["hold_closure_margin"]),
            palm_support_threshold_n=float(controller_cfg_block["palm_support_threshold_n"]),
            precontact_fraction=float(controller_cfg_block["precontact_fraction"]),
            contact_latch_steps=int(controller_cfg_block["contact_latch_steps"]),
            lost_contact_grace_steps=int(controller_cfg_block["lost_contact_grace_steps"]),
            reacquire_action_step=float(controller_cfg_block["reacquire_action_step"]),
            max_closure_extension=float(controller_cfg_block["max_closure_extension"]),
            contact_detect_n=tuple(float(v) for v in controller_cfg_block["contact_detect_n"]),
            contact_latch_start_step=int(controller_cfg_block["contact_latch_start_step"]),
            search_probe_steps=int(controller_cfg_block["search_probe_steps"]),
            high_force_dwell_steps=int(controller_cfg_block["high_force_dwell_steps"]),
            digit_start_steps=(
                None
                if controller_cfg_block.get("digit_start_steps") is None
                else tuple(int(v) for v in controller_cfg_block["digit_start_steps"])
            ),
        ),
    )

    actor = Actor().to(device)
    critic = Critic().to(device)
    nn.init.uniform_(actor.mean.weight, -1.0e-3, 1.0e-3)
    nn.init.zeros_(actor.mean.bias)
    log_std = nn.Parameter(
        torch.full((ACT_DIM,), float(args_cli.log_std_init), device=device)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": actor.parameters()},
            {"params": critic.parameters()},
            {"params": [log_std]},
        ],
        lr=args_cli.lr,
    )

    lower = torch.tensor(
        [float(v) for v in controller_cfg_block["force_lower_n"]], device=device
    )
    upper = torch.tensor(
        [float(v) for v in controller_cfg_block["force_upper_n"]], device=device
    )
    cov_target = float(scenario["strict"]["angular_coverage_min_deg"])
    scale = float(args_cli.residual_scale)

    def shape_reward() -> torch.Tensor:
        palm_ok = (env.palm_force >= 0.3).float()
        fingers = env._v2_measured_link4_force
        contacts = (fingers >= 0.04).sum(-1).float()
        band = ((fingers >= lower) & (fingers <= upper)).float().mean(-1)
        all_five_recent = env._v3_gate_current["all_five_physical"].float()
        speed01 = (
            torch.linalg.vector_norm(env.tomato.data.root_lin_vel_w, dim=-1) / 0.05
        ).clamp(0.0, 1.0)
        cov01 = (env._v3_physical_coverage_max / cov_target).clamp(0.0, 1.0)
        bad = (env.overloaded.float() + env.dropped.float()).clamp(max=1.0)
        fresh_success = env._v3_strict_success_latched.float()
        return (
            0.40 * palm_ok
            + 0.12 * contacts
            + 0.30 * all_five_recent
            + 0.12 * band
            + 0.10 * cov01
            - 0.60 * speed01 * palm_ok
            - 10.0 * bad
            + 8.0 * fresh_success
        )

    metrics_path = output_dir / "metrics.jsonl"
    best_score = -1.0e9
    skipped_updates = 0
    start_update = 0
    resume_path = output_dir / "latest.pt"
    if args_cli.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state"])
        critic.load_state_dict(ckpt["critic_state"])
        with torch.no_grad():
            log_std.copy_(ckpt["log_std"].to(device))
            log_std.clamp_(-2.0, -0.7)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = int(ckpt.get("update", -1)) + 1
        best_score = float(ckpt.get("best_score", -1.0e9))
        print(json.dumps({"resume": True, "start_update": start_update}), flush=True)

    def full_payload(score: float, update: int) -> dict:
        return checkpoint_payload(
            actor,
            {
                "score": score,
                "update": update,
                "best_score": best_score,
                "actor_state": actor.state_dict(),
                "critic_state": critic.state_dict(),
                "log_std": log_std.detach().cpu(),
                "optimizer": optimizer.state_dict(),
            },
        )

    backup = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    critic_backup = {k: v.detach().clone() for k, v in critic.state_dict().items()}

    config_record = {
        "version": "v3_residual_ppo_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_command_allowed": False,
        "scenario_sha256": scenario_hash,
        "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args_cli).items()
        },
        "residual_scale": scale,
    }
    (output_dir / "config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")

    obs_dict = observation
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    for update in range(start_update, start_update + args_cli.updates):
        storage = {
            key: torch.zeros((args_cli.horizon, args_cli.num_envs), device=device)
            for key in ("rewards", "values", "logprobs", "dones", "terms")
        }
        act_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, ACT_DIM), device=device)
        obs_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, OBS_DIM), device=device)
        episode_returns = []
        with torch.inference_mode():
            for step in range(args_cli.horizon):
                obs_buf[step] = obs
                mean = actor(obs)
                std = log_std.exp().expand_as(mean)
                dist = Normal(mean, std)
                raw = dist.sample()
                logprob = dist.log_prob(raw).sum(-1)
                value = critic(obs).squeeze(-1)
                base = teacher.command(
                    env.episode_length_buf,
                    env._v2_measured_link4_force,
                    env.palm_force,
                    torch.linalg.vector_norm(env.tomato.data.root_lin_vel_w, dim=-1),
                    torch.linalg.vector_norm(
                        env._tip_positions() - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1
                    ),
                    env.robot.data.joint_pos[:, env.active_joint_ids],
                )
                action = (base + scale * raw.clamp(-2.0, 2.0)).clamp(-1.0, 1.0)
                act_buf[step] = action
                storage["values"][step] = value
                storage["logprobs"][step] = logprob
                observation, _, terminated, truncated, _ = env.step(action)
                obs = (
                    observation["policy"] if isinstance(observation, dict) else observation
                ).detach()
                storage["rewards"][step] = shape_reward()
                storage["dones"][step] = (terminated | truncated).float()
                storage["terms"][step] = terminated.float()
            next_value = critic(obs).squeeze(-1) * (1.0 - storage["terms"][-1])

        advantages = torch.zeros_like(storage["rewards"])
        gae = torch.zeros_like(next_value)
        for step in reversed(range(args_cli.horizon)):
            mask = 1.0 - storage["dones"][step]
            gae = (
                storage["rewards"][step]
                + 0.99 * next_value * mask
                - storage["values"][step]
                + 0.95 * 0.99 * gae * mask
            )
            advantages[step] = gae
            next_value = storage["values"][step]
        returns = advantages + storage["values"]
        flat_adv = advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / flat_adv.std().clamp_min(1.0e-6)
        size = args_cli.horizon * args_cli.num_envs
        flat_obs = obs_buf.reshape(size, OBS_DIM)
        flat_act = act_buf.reshape(size, ACT_DIM)
        flat_log = storage["logprobs"].reshape(-1)
        flat_val = storage["values"].reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_adv = flat_adv.detach()

        actor.train()
        for _ in range(4):
            perm = torch.randperm(size, device=device)
            for start in range(0, size, size // 4):
                idx = perm[start : start + size // 4]
                mean = actor(flat_obs[idx])
                std = log_std.exp().expand_as(mean)
                dist = Normal(mean, std)
                new_log = dist.log_prob(flat_act[idx]).sum(-1)
                ratio = (new_log - flat_log[idx]).exp()
                surr = torch.minimum(
                    ratio * flat_adv[idx],
                    ratio.clamp(0.8, 1.2) * flat_adv[idx],
                )
                value_loss = 0.5 * (critic(flat_obs[idx]).squeeze(-1) - flat_ret[idx]) ** 2
                loss = (
                    -surr.mean()
                    + 0.5 * value_loss.mean()
                    - args_cli.entropy_coef * dist.entropy().mean()
                )
                if not torch.isfinite(loss):
                    skipped_updates += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()) + [log_std], 1.0
                )
                optimizer.step()

        with torch.no_grad():
            log_std.clamp_(-2.0, -0.7)
        params_bad = not all(
            torch.isfinite(p).all() for p in list(actor.parameters()) + list(critic.parameters())
        )
        if params_bad:
            actor.load_state_dict(backup)
            critic.load_state_dict(critic_backup)
            skipped_updates += 1
        backup = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        critic_backup = {k: v.detach().clone() for k, v in critic.state_dict().items()}
        env.reset()

        record = {
            "update": update,
            "reward_mean": storage["rewards"].mean().item(),
            "skipped_updates": skipped_updates,
            "log_std_mean": log_std.exp().mean().item(),
        }
        if (update + 1) % args_cli.eval_interval == 0 or update == args_cli.updates - 1:
            greedy = greedy_eval(env, teacher, actor, obs, steps=150, scale=scale)
            score = greedy["sim_digits_mean"] + 2.0 * greedy["seq_success_rate"]
            record.update({"eval": greedy, "score": score})
            torch.save(full_payload(score, update), output_dir / "latest.pt")
            if score > best_score:
                best_score = score
                torch.save(full_payload(score, update), output_dir / "best.pt")
            print(json.dumps({"update": update, "eval": greedy}), flush=True)
        if (update + 1) % args_cli.save_interval == 0:
            torch.save(full_payload(best_score, update), output_dir / "latest.pt")
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    env.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
