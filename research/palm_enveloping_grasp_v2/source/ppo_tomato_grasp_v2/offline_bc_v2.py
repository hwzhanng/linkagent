"""Offline observation-conditioned behavior cloning for the v2 hand task.

This is a simulation-only warm-start gate.  It deliberately stores the
teacher-labelled observations before fitting the actor, so the result can be
audited independently from the online PPO/teacher blend.  It is not a final
controller and never enables a hardware command path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--profile", choices=("v1_nominal_reliable", "v2_band1_match"), default="v2_band1_match")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--collect-steps", type=int, default=240)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--learning-rate", type=float, default=2.0e-3)
parser.add_argument("--seed", type=int, default=20261141)
parser.add_argument("--teacher-prior-ramp-steps", type=int, default=120)
parser.add_argument("--thumb-first", action="store_true",
                    help="use an observable low-impulse thumb-opposition then four-finger closure schedule")
parser.add_argument("--thumb-first-steps", type=int, default=80)
parser.add_argument("--finger-ramp-steps", type=int, default=180)
parser.add_argument("--teacher-thumb-first-global-scale", type=float, default=0.35)
parser.add_argument("--teacher-coverage-delta", type=str, default="0.0,0.0,0.0,0.0",
                    help="signed bounded balance, spread, opposition, thumb-residual delta")
parser.add_argument("--teacher-motion-smoothing-alpha", type=float, default=1.0)
parser.add_argument(
    "--sequential-fingers", action="store_true",
    help="after thumb opposition, ramp index/middle/ring/little proximal+distal channels one finger at a time",
)
parser.add_argument(
    "--teacher-prior-action", type=str,
    default="1.0,-1.0,0.5314251780509949,-0.5596628189086914,1.0,-0.01167266070842743,0.58189457654953,0.9941583275794983,0.11048947274684906,0.05933154374361038,-0.6171775460243225,0.6542128324508667,-0.24184483289718628,0.5135747194290161",
)
parser.add_argument("--teacher-thumb-weight", type=float, default=4.0)
parser.add_argument("--teacher-distal-weight", type=float, default=4.0)
parser.add_argument("--source-checkpoint", type=Path, default=None)
parser.add_argument(
    "--behavior-checkpoint", type=Path, default=None,
    help="optional actor used to collect off-policy states for a DAgger-style pass; labels remain teacher actions",
)
parser.add_argument(
    "--behavior-blend", type=float, default=0.0,
    help="fraction of behavior actor in the executed collection action (0 keeps pure teacher rollout)",
)
parser.add_argument("--physical-fingertip-proxy", action="store_true")
parser.add_argument("--proxy-only-collision", action="store_true")
parser.add_argument("--disable-virtual-distal-contact", action="store_true")
parser.add_argument("--strict-require-measured-distal-contact", action="store_true")
parser.add_argument("--fingertip-proxy-radius", type=float, default=0.006)
parser.add_argument("--proxy-contact-offset", type=float, default=0.0003)
parser.add_argument("--proxy-rest-offset", type=float, default=0.0001)
parser.add_argument("--tomato-contact-offset", type=float, default=0.0003)
parser.add_argument("--tomato-rest-offset", type=float, default=0.0001)
parser.add_argument("--cup-half-extent", type=str, default="0.040,0.035")
parser.add_argument("--assist-per-digit", type=str, default="0.45,0.45,0.30,0.25")
parser.add_argument("--distal-filter-alpha", type=float, default=0.999)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
from torch import nn

from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher, EnvelopingTeacherCfg
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile, profile_definition
from simulation.ppo_tomato_grasp_v2.contract import V2_ACTOR_OBSERVATION_DIM, V2_ACTION_DIM, V2_CRITIC_STATE_DIM
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg


def parse_values(value: str, count: int, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != count:
        raise ValueError(f"{name} requires {count} comma-separated values")
    return values


def parse_signed_coverage_delta(value: str, name: str = "--teacher-coverage-delta") -> tuple[float, float, float, float]:
    values = parse_values(value, 4, name)
    if any(abs(item) > 0.30 for item in values):
        raise ValueError(f"{name} values must lie in [-0.30, 0.30]")
    return values  # type: ignore[return-value]


class BCActorCritic(nn.Module):
    """Checkpoint-compatible actor/critic shell; only actor rows are fitted."""

    def __init__(self) -> None:
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(V2_ACTOR_OBSERVATION_DIM, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.critic_body = nn.Sequential(
            nn.Linear(V2_CRITIC_STATE_DIM, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(128, V2_ACTION_DIM)
        self.value = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((V2_ACTION_DIM,), math.log(0.20)))

    def actor(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor_mean(self.actor_body(observation)))


def make_cfg() -> V2EnvelopingPhysxTomatoGraspEnvCfg:
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args_cli.device
    apply_profile(cfg, args_cli.profile, num_envs=args_cli.num_envs, seed=args_cli.seed)
    cfg.training_stage = 1
    # Keep the collector alive for the complete labelled schedule.  The base
    # profile defaults to 10 s (200 policy steps at 20 Hz); collecting 240
    # steps would otherwise time out at step 199 and append fresh open-hand
    # labels to the tail of the dataset.  This is collection-only and does
    # not alter the evaluation/training defaults.
    cfg.episode_length_s = max(10.0, (int(args_cli.collect_steps) + 10) * cfg.policy_dt_s)
    cfg.strict_enveloping_enabled = True
    cfg.strict_enveloping_require_stem_break = False
    cfg.strict_angular_coverage_min_deg = 120.0
    cfg.strict_hold_steps = 8
    # The dataset collector must expose the complete observable schedule.  A
    # normal safety ceiling would reset an episode before the late thumb/finger
    # labels are recorded, creating the all-open label bias seen in t0w.  This
    # relaxed ceiling is collection-only; actor-only evaluation always uses
    # the normal v2 safety limits and is the authoritative gate.
    cfg.safe_force_n = 1000.0
    cfg.force_ceiling_n = (1000.0, 1000.0, 1000.0, 1000.0, 1000.0)
    cfg.severe_overload_scale = 2.0
    cfg.pregrasp_drop_grace_steps = max(120, int(args_cli.collect_steps) + 1)
    # Keep the collection semantics identical to evaluation: proxy-only
    # means the authored fingertip pads are actual PhysX colliders, otherwise
    # the teacher dataset would contain a false all-zero distal signal.
    cfg.fingertip_collision_proxy_physical = bool(
        args_cli.physical_fingertip_proxy or args_cli.proxy_only_collision
    )
    cfg.proxy_only_collision = bool(args_cli.proxy_only_collision)
    cfg.fingertip_collision_proxy_radius_m = float(args_cli.fingertip_proxy_radius)
    cfg.fingertip_proxy_contact_offset_m = float(args_cli.proxy_contact_offset)
    cfg.fingertip_proxy_rest_offset_m = float(args_cli.proxy_rest_offset)
    cfg.tomato_contact_offset_m = float(args_cli.tomato_contact_offset)
    cfg.tomato_rest_offset_m = float(args_cli.tomato_rest_offset)
    cfg.virtual_distal_contact_enabled = not bool(args_cli.disable_virtual_distal_contact)
    cfg.strict_require_measured_distal_contact = bool(args_cli.strict_require_measured_distal_contact)
    cfg.distal_action_filter_alpha = float(args_cli.distal_filter_alpha)
    cfg.distal_proximal_reach_assist_gain = 0.0
    cfg.distal_proximal_reach_assist_gain_per_digit = parse_values(args_cli.assist_per_digit, 4, "--assist-per-digit")
    cup = parse_values(args_cli.cup_half_extent, 2, "--cup-half-extent")
    cfg.palm_cup_inner_half_extent_m = cup
    cfg.palm_cup_rim_height_m = 0.02
    # Keep authored event parameters synchronized after profile resolution.
    cfg.events["v2_fingertip_collision_proxies"].params.update({
        "radius_m": cfg.fingertip_collision_proxy_radius_m,
        "physical_collision": cfg.fingertip_collision_proxy_physical,
        "contact_offset_m": cfg.fingertip_proxy_contact_offset_m,
        "rest_offset_m": cfg.fingertip_proxy_rest_offset_m,
    })
    cfg.events["tomato_geometry"].params.update({
        "contact_offset_m": cfg.tomato_contact_offset_m,
        "rest_offset_m": cfg.tomato_rest_offset_m,
    })
    return cfg


def load_source(model: BCActorCritic, path: Path | None, device: torch.device) -> None:
    if path is None:
        return
    checkpoint = torch.load(path.expanduser().resolve(), map_location=device, weights_only=False)
    source = checkpoint.get("model", checkpoint)
    own = model.state_dict()
    compatible = {key: value for key, value in source.items() if key in own and own[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)


@torch.inference_mode()
def collect(
    env: V2EnvelopingPhysxTomatoGraspEnv,
    teacher: EnvelopingTeacher,
    steps: int,
    behavior_model: BCActorCritic | None = None,
):
    obs, _ = env.reset(seed=args_cli.seed)
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    executed_actions: list[torch.Tensor] = []
    episode_steps: list[torch.Tensor] = []
    teacher_phases: list[torch.Tensor] = []
    measured_link4_forces: list[torch.Tensor] = []
    palm_forces: list[torch.Tensor] = []
    for _ in range(steps):
        actor_obs = obs["policy"].clone()
        episode_steps.append(env.episode_length_buf.detach().clone())
        measured_link4_forces.append(env._v2_measured_link4_force.detach().clone())
        palm_forces.append(env.palm_force.detach().clone())
        teacher_action = teacher.command(
            env.contact_forces,
            env.palm_force,
            episode_step=env.episode_length_buf,
            measured_digit_force_n=env._v2_measured_link4_force,
        ).clone()
        # The phase is an audit label, never an actor input.  Persisting it
        # makes it possible to reject datasets dominated by open-hand or
        # pre-contact labels without mistaking a high BC loss for a physics
        # failure.
        teacher_phases.append(teacher.phase.detach().clone())
        if args_cli.thumb_first:
            # The transform is clocked only by the public episode step.  It
            # intentionally does not read teacher.phase/support_latched, so
            # the saved labels remain a valid observation-conditioned target.
            open_action = teacher.open_action
            episode_step = env.episode_length_buf.float()
            thumb_fraction = ((episode_step - 8.0) / max(float(args_cli.thumb_first_steps), 1.0)).clamp(0.0, 1.0)
            finger_fraction = ((episode_step - float(args_cli.thumb_first_steps)) / max(float(args_cli.finger_ramp_steps), 1.0)).clamp(0.0, 1.0)
            # Use the audited prior as the schedule target rather than the
            # teacher's private support-latched output.  Force feedback still
            # runs in the teacher above, but a missing/private latch must not
            # silently turn every thumb label back into the open action.
            prior_action = torch.cat(
                (
                    teacher.root_prior[:, :10],
                    teacher.residual_prior.new_tensor(tuple(teacher.cfg.distal_prior_action or (0.0, 0.0, 0.0, 0.0)))
                    .unsqueeze(0).expand_as(teacher.distal_trim),
                ), dim=-1,
            )
            scheduled = open_action.clone()
            # Opposition, thumb residual, and global thumb curl start first;
            # straight-finger roots/residuals enter gradually afterwards.
            scheduled[:, 4] = open_action[:, 4] + thumb_fraction * (prior_action[:, 4] - open_action[:, 4])
            scheduled[:, 5] = open_action[:, 5] + thumb_fraction * (prior_action[:, 5] - open_action[:, 5])
            scheduled[:, 0] = open_action[:, 0] + thumb_fraction * (prior_action[:, 0] - open_action[:, 0])
            if args_cli.sequential_fingers:
                # The direct PhysX probe reaches the thumb only after the
                # straight fingers are inserted in a low-impulse sequence.
                # Keep each digit's proximal and distal residual causal and
                # expose the order through the public episode clock.
                for digit in range(4):
                    start = float(args_cli.thumb_first_steps + digit * max(int(args_cli.finger_ramp_steps), 1))
                    fraction = ((episode_step - start) / max(float(args_cli.finger_ramp_steps), 1.0)).clamp(0.0, 1.0)
                    scheduled[:, 1 + digit] = open_action[:, 1 + digit] + fraction * (prior_action[:, 1 + digit] - open_action[:, 1 + digit])
                    scheduled[:, 6 + digit] = open_action[:, 6 + digit] + fraction * (prior_action[:, 6 + digit] - open_action[:, 6 + digit])
                    scheduled[:, 10 + digit] = open_action[:, 10 + digit] + fraction * (prior_action[:, 10 + digit] - open_action[:, 10 + digit])
            else:
                scheduled[:, 1:5] = open_action[:, 1:5] + finger_fraction.unsqueeze(-1) * (prior_action[:, 1:5] - open_action[:, 1:5])
                scheduled[:, 6:10] = open_action[:, 6:10] + finger_fraction.unsqueeze(-1) * (prior_action[:, 6:10] - open_action[:, 6:10])
                scheduled[:, 10:14] = open_action[:, 10:14] + finger_fraction.unsqueeze(-1) * (prior_action[:, 10:14] - open_action[:, 10:14])
            teacher_action = scheduled.clamp(-1.0, 1.0)
        observations.append(actor_obs.detach())
        actions.append(teacher_action.detach())
        executed_action = teacher_action
        if behavior_model is not None and args_cli.behavior_blend > 0.0:
            behavior_action = behavior_model.actor(actor_obs)
            blend = float(args_cli.behavior_blend)
            executed_action = ((1.0 - blend) * teacher_action + blend * behavior_action).clamp(-1.0, 1.0)
        executed_actions.append(executed_action.detach().clone())
        obs, _reward, terminated, truncated, _info = env.step(executed_action)
        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
        if done_ids.numel() > 0:
            teacher.reset(done_ids)
    return {
        "observation": torch.cat(observations, dim=0),
        "action": torch.cat(actions, dim=0),
        "executed_action": torch.cat(executed_actions, dim=0),
        "episode_step": torch.cat(episode_steps, dim=0),
        "teacher_phase": torch.cat(teacher_phases, dim=0),
        "measured_link4_force_n": torch.cat(measured_link4_forces, dim=0),
        "palm_force_n": torch.cat(palm_forces, dim=0),
    }


def main() -> None:
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    device = torch.device(args_cli.device)
    env = V2EnvelopingPhysxTomatoGraspEnv(make_cfg())
    prior = parse_values(args_cli.teacher_prior_action, 14, "--teacher-prior-action")
    teacher_cfg = EnvelopingTeacherCfg(
        distal_prior_action=tuple(prior[10:]),
        prior_ramp_steps=max(0, int(args_cli.teacher_prior_ramp_steps)),
        tactile_gain=0.08,
        max_feedback_step=0.03,
        thumb_first_phase_steps=max(0, int(args_cli.thumb_first_steps) if args_cli.thumb_first else 0),
        thumb_first_global_scale=float(args_cli.teacher_thumb_first_global_scale),
        thumb_first_coverage_delta=parse_signed_coverage_delta(args_cli.teacher_coverage_delta),
        motion_smoothing_alpha=float(args_cli.teacher_motion_smoothing_alpha),
    )
    teacher = EnvelopingTeacher(args_cli.num_envs, device, teacher_cfg)
    prior_tensor = torch.tensor(prior, device=device, dtype=torch.float32)
    teacher.root_prior[:, :10] = prior_tensor[:10]
    teacher.residual_prior[:] = prior_tensor[5:10]
    behavior_model = None
    if args_cli.behavior_checkpoint is not None:
        behavior_model = BCActorCritic().to(device)
        load_source(behavior_model, args_cli.behavior_checkpoint, device)
        behavior_model.eval()
    trace = collect(env, teacher, args_cli.collect_steps, behavior_model)
    observations = trace["observation"]
    actions = trace["action"]
    # Keep the saved dataset on CPU: it is the audit artifact, not a hidden
    # replay buffer.  Fitting below uses shuffled GPU minibatches.
    dataset_path = args_cli.output_dir / "teacher_dataset.pt"
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": "v2_offline_teacher_dataset_1",
        "observation": observations.cpu(),
        "action": actions.cpu(),
        "executed_action": trace["executed_action"].cpu(),
        "episode_step": trace["episode_step"].cpu(),
        "teacher_phase": trace["teacher_phase"].cpu(),
        "measured_link4_force_n": trace["measured_link4_force_n"].cpu(),
        "palm_force_n": trace["palm_force_n"].cpu(),
        "profile": args_cli.profile,
        "profile_values": profile_definition(args_cli.profile),
        "teacher_prior_action": list(prior),
        "thumb_first": bool(args_cli.thumb_first),
        "sequential_fingers": bool(args_cli.sequential_fingers),
        "thumb_first_steps": int(args_cli.thumb_first_steps),
        "finger_ramp_steps": int(args_cli.finger_ramp_steps),
        "teacher_thumb_first_global_scale": float(args_cli.teacher_thumb_first_global_scale),
        "teacher_coverage_delta": list(parse_signed_coverage_delta(args_cli.teacher_coverage_delta)),
        "teacher_motion_smoothing_alpha": float(args_cli.teacher_motion_smoothing_alpha),
        "behavior_checkpoint": str(args_cli.behavior_checkpoint.resolve()) if args_cli.behavior_checkpoint else None,
        "behavior_blend": float(args_cli.behavior_blend),
        "episode_length_s": float(env.cfg.episode_length_s),
        "max_episode_length": int(env.max_episode_length),
        "resolved_trace_config": {
            "fingertip_collision_proxy_physical": cfg_value(env.cfg, "fingertip_collision_proxy_physical"),
            "proxy_only_collision": cfg_value(env.cfg, "proxy_only_collision"),
            "fingertip_proxy_radius_m": cfg_value(env.cfg, "fingertip_collision_proxy_radius_m"),
            "fingertip_proxy_contact_offset_m": cfg_value(env.cfg, "fingertip_proxy_contact_offset_m"),
            "fingertip_proxy_rest_offset_m": cfg_value(env.cfg, "fingertip_proxy_rest_offset_m"),
            "tomato_contact_offset_m": cfg_value(env.cfg, "tomato_contact_offset_m"),
            "tomato_rest_offset_m": cfg_value(env.cfg, "tomato_rest_offset_m"),
            "palm_cup_inner_half_extent_m": list(cfg_value(env.cfg, "palm_cup_inner_half_extent_m")),
            "distal_action_filter_alpha": cfg_value(env.cfg, "distal_action_filter_alpha"),
            "distal_proximal_reach_assist_gain_per_digit": list(cfg_value(env.cfg, "distal_proximal_reach_assist_gain_per_digit")),
            "virtual_distal_contact_enabled": cfg_value(env.cfg, "virtual_distal_contact_enabled"),
            "strict_require_measured_distal_contact": cfg_value(env.cfg, "strict_require_measured_distal_contact"),
        },
        "hardware_command_allowed": False,
    }, dataset_path)

    model = BCActorCritic().to(device)
    load_source(model, args_cli.source_checkpoint, device)
    model.train()
    optimizer = torch.optim.Adam(model.actor_body.parameters(), lr=args_cli.learning_rate)
    optimizer.add_param_group({"params": model.actor_mean.parameters()})
    weights = torch.ones(V2_ACTION_DIM, device=device)
    weights[4] = float(args_cli.teacher_thumb_weight)
    weights[10:] = float(args_cli.teacher_distal_weight)
    x = observations.to(device)
    y = actions.to(device)
    generator = torch.Generator(device=device).manual_seed(args_cli.seed + 17)
    history: list[float] = []
    for epoch in range(1, args_cli.epochs + 1):
        order = torch.randperm(x.shape[0], generator=generator, device=device)
        epoch_loss = torch.zeros((), device=device)
        batches = 0
        for start in range(0, x.shape[0], args_cli.batch_size):
            idx = order[start:start + args_cli.batch_size]
            prediction = model.actor(x[idx])
            loss = ((prediction - y[idx]).square() * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.actor_body.parameters(), 1.0)
            nn.utils.clip_grad_norm_(model.actor_mean.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.detach()
            batches += 1
        value = float((epoch_loss / max(batches, 1)).cpu())
        history.append(value)
        if epoch == 1 or epoch % 5 == 0 or epoch == args_cli.epochs:
            print(json.dumps({"epoch": epoch, "bc_loss": value}, ensure_ascii=False), flush=True)
    model.eval()
    with torch.inference_mode():
        final_prediction = model.actor(x)
        final_error = (final_prediction - y).square().mean(0).cpu().tolist()
    checkpoint = {
        "version": "palm_enveloping_grasp_v2",
        "task": "palm_supported_five_finger_enveloping",
        "stage": 1,
        "profile": args_cli.profile,
        "actor_observation_dim": V2_ACTOR_OBSERVATION_DIM,
        "critic_state_dim": V2_CRITIC_STATE_DIM,
        "action_dim": V2_ACTION_DIM,
        "contract_version": "palm_enveloping_grasp_v2.0",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": 0,
        "total_steps": int(x.shape[0]),
        "metrics": {"offline_bc_loss_history": history, "offline_bc_action_mse": final_error},
        "resolved_config": {
            "profile_values": profile_definition(args_cli.profile),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "collect_steps": args_cli.collect_steps,
            "episode_length_s": float(env.cfg.episode_length_s),
            "max_episode_length": int(env.max_episode_length),
            "epochs": args_cli.epochs,
            "teacher_prior_action": list(prior),
            "teacher_prior_ramp_steps": args_cli.teacher_prior_ramp_steps,
            "thumb_first": bool(args_cli.thumb_first),
            "sequential_fingers": bool(args_cli.sequential_fingers),
            "thumb_first_steps": int(args_cli.thumb_first_steps),
            "finger_ramp_steps": int(args_cli.finger_ramp_steps),
            "behavior_checkpoint": str(args_cli.behavior_checkpoint.resolve()) if args_cli.behavior_checkpoint else None,
            "behavior_blend": float(args_cli.behavior_blend),
            "teacher_thumb_weight": args_cli.teacher_thumb_weight,
            "teacher_distal_weight": args_cli.teacher_distal_weight,
            "fingertip_collision_proxy_physical": cfg_value(env.cfg, "fingertip_collision_proxy_physical"),
            "proxy_only_collision": cfg_value(env.cfg, "proxy_only_collision"),
            "fingertip_proxy_radius_m": cfg_value(env.cfg, "fingertip_collision_proxy_radius_m"),
            "fingertip_proxy_contact_offset_m": cfg_value(env.cfg, "fingertip_proxy_contact_offset_m"),
            "fingertip_proxy_rest_offset_m": cfg_value(env.cfg, "fingertip_proxy_rest_offset_m"),
            "distal_action_filter_alpha": cfg_value(env.cfg, "distal_action_filter_alpha"),
            "distal_proximal_reach_assist_gain_per_digit": list(cfg_value(env.cfg, "distal_proximal_reach_assist_gain_per_digit")),
            "palm_cup_inner_half_extent_m": list(cfg_value(env.cfg, "palm_cup_inner_half_extent_m")),
            "virtual_distal_contact_enabled": cfg_value(env.cfg, "virtual_distal_contact_enabled"),
            "strict_require_measured_distal_contact": cfg_value(env.cfg, "strict_require_measured_distal_contact"),
        },
        "source_checkpoint": str(args_cli.source_checkpoint.resolve()) if args_cli.source_checkpoint else None,
        "dataset": str(dataset_path.resolve()),
        "hardware_command_allowed": False,
    }
    checkpoint_path = args_cli.output_dir / "bc.pt"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "version": "v2_offline_bc_1",
        "dataset": str(dataset_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "samples": int(x.shape[0]),
        "observation_dim": int(x.shape[-1]),
        "action_dim": int(y.shape[-1]),
        "initial_action_mean": actions.mean(0).cpu().tolist(),
        "final_action_mse": final_error,
        "final_bc_loss": history[-1],
        "profile": args_cli.profile,
        "thumb_first": bool(args_cli.thumb_first),
        "sequential_fingers": bool(args_cli.sequential_fingers),
        "hardware_command_allowed": False,
    }
    (args_cli.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    os._exit(0)


def cfg_value(cfg, name: str):
    value = getattr(cfg, name)
    if isinstance(value, tuple):
        return tuple(float(v) for v in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


if __name__ == "__main__":
    main()
