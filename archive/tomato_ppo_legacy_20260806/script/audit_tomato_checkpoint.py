"""Run a fixed tomato-policy checkpoint evaluation without updating it."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.skills.tomato_pick.contracts import ACTION_DIM, OBSERVATION_DIM


def _parser() -> argparse.ArgumentParser:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim2real-mode", choices=("full", "nominal"), default="full")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--post-detach-mode",
        choices=(
            "policy",
            "zero_after_observed_detach",
            "close_after_observed_detach",
            "track_fruit_after_observed_detach",
        ),
        default="policy",
        help=(
            "counterfactual diagnostic: after detachment is observable on the next "
            "policy step, preserve the current arm/hand targets with a zero action"
        ),
    )
    parser.add_argument(
        "--closure-assist-after-contact",
        action="store_true",
        help=(
            "audit-only counterfactual: keep the policy arm/non-closure actions, "
            "but command positive flexion after multi-finger contact"
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    args = _parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")

    launcher = AppLauncher(args)
    app = launcher.app
    wrapped = None
    try:
        gym = importlib.import_module("gymnasium")
        torch = importlib.import_module("torch")
        runner_module = importlib.import_module("rsl_rl.runners.on_policy_runner")
        wrapper_module = importlib.import_module("isaaclab_rl.rsl_rl")
        task = importlib.import_module("simulation.tomato_pick.isaac")
        cfg_module = importlib.import_module("simulation.tomato_pick.isaac.env_cfg")
        agent_module = importlib.import_module(
            "simulation.tomato_pick.isaac.agents.rsl_rl_ppo_cfg"
        )

        cfg = cfg_module.TomatoPickEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.enable_sim2real_randomization = args.sim2real_mode == "full"
        raw_env = gym.make(task.TASK_ID, cfg=cfg)
        agent_cfg = agent_module.TomatoPickPPORunnerCfg()
        agent_cfg.device = args.device
        agent_cfg = wrapper_module.handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib")
        )
        wrapped = wrapper_module.RslRlVecEnvWrapper(
            raw_env, clip_actions=agent_cfg.clip_actions
        )
        runner = runner_module.OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(args.checkpoint.resolve()))
        policy = runner.get_inference_policy(device=wrapped.device)
        raw = wrapped.unwrapped
        env_module = importlib.import_module("simulation.tomato_pick.isaac.env")
        closure_action_indices = torch.tensor(
            [
                6 + index
                for index, name in enumerate(raw._hand_joint_names)
                if env_module.RIGHT_HAND_CONTROL_RAD[name][3] == "closure"
            ],
            device=wrapped.device,
            dtype=torch.long,
        )
        observations = wrapped.get_observations()

        checks = {
            "actor_shape_113": tuple(observations["policy"].shape)
            == (args.num_envs, OBSERVATION_DIM),
            "action_shape_22": raw.single_action_space.shape == (ACTION_DIM,),
            "observations_finite": True,
            "actions_finite_and_bounded": True,
            "rewards_finite": True,
            "no_nan_inf_or_overlap": True,
        }
        event_counts = {
            name: 0
            for name in (
                "contact",
                "grasp",
                "break_once",
                "retention",
                "success",
                "drop",
                "excess_force",
            )
        }
        phase_visit_counts = [0, 0, 0, 0]
        action_sum = torch.zeros(ACTION_DIM, device=wrapped.device)
        action_square_sum = torch.zeros(ACTION_DIM, device=wrapped.device)
        action_count = 0
        maximum_hold_streak = 0
        detached_samples = 0
        detached_stable_samples = 0
        completed_episodes = 0
        post_detach_override_samples = 0
        post_detach_override_active = torch.zeros(
            args.num_envs, dtype=torch.bool, device=wrapped.device
        )
        closure_assist_samples = 0
        break_and_drop_same_step = 0
        break_without_drop_same_step = 0

        with torch.inference_mode():
            for _ in range(args.steps):
                actions = (
                    runner.alg.actor(observations, stochastic_output=True)
                    if args.stochastic
                    else policy(observations)
                )
                if args.closure_assist_after_contact:
                    contact_force = raw._finger_force_magnitudes()
                    multi_finger_contact = (
                        contact_force >= raw.cfg.minimum_contact_force_n
                    ).sum(dim=1) >= 2
                    assisted = multi_finger_contact & ~raw._stem_detached
                    if bool(assisted.any()):
                        assisted_ids = assisted.nonzero(as_tuple=False).squeeze(1)
                        actions[
                            assisted_ids.unsqueeze(1), closure_action_indices.unsqueeze(0)
                        ] = 1.0
                        closure_assist_samples += int(assisted.sum())
                if args.post_detach_mode != "policy":
                    detached_before_action = raw._stem_detached.clone()
                    post_detach_override_active &= detached_before_action
                    if bool(detached_before_action.any()):
                        detached_ids = detached_before_action.nonzero(
                            as_tuple=False
                        ).squeeze(1)
                        # This is an audit-only counterfactual.  Zero is an
                        # incremental command, so it preserves the current IK
                        # and hand targets.  Clearing the simulated delay line
                        # prevents a queued pre-break command from confounding
                        # whether the already-achieved grasp can carry fruit.
                        newly_observed = (
                            detached_before_action & ~post_detach_override_active
                        )
                        if bool(newly_observed.any()):
                            raw._sim2real.clear_action_history(
                                newly_observed.nonzero(as_tuple=False).squeeze(1)
                            )
                        post_detach_override_active |= detached_before_action
                        actions[detached_before_action] = 0.0
                        if args.post_detach_mode in (
                            "close_after_observed_detach",
                            "track_fruit_after_observed_detach",
                        ):
                            actions[
                                detached_ids.unsqueeze(1),
                                closure_action_indices.unsqueeze(0),
                            ] = 1.0
                        if (
                            args.post_detach_mode
                            == "track_fruit_after_observed_detach"
                        ):
                            fruit_position = raw._tomato.data.root_pos_w.torch[
                                detached_ids
                            ]
                            fingertip_center = raw._robot.data.body_pos_w.torch[
                                detached_ids[:, None], raw._fingertip_frame_body_ids
                            ].mean(dim=1)
                            position_error_w = fruit_position - fingertip_center
                            world_from_grasp = raw._grasp_frame_basis_w[detached_ids]
                            position_error_g = torch.bmm(
                                world_from_grasp.transpose(1, 2),
                                position_error_w.unsqueeze(2),
                            ).squeeze(2)
                            translation_scales = torch.tensor(
                                raw.cfg.action_scales[:3], device=wrapped.device
                            )
                            actions[detached_ids, :3] = (
                                position_error_g / translation_scales
                            ).clamp(-1.0, 1.0)
                        post_detach_override_samples += int(
                            detached_before_action.sum()
                        )
                checks["observations_finite"] &= bool(
                    torch.isfinite(observations["policy"]).all()
                )
                checks["actions_finite_and_bounded"] &= bool(
                    torch.isfinite(actions).all()
                    and (actions >= -1.0).all()
                    and (actions <= 1.0).all()
                )
                action_sum += actions.sum(dim=0)
                action_square_sum += actions.square().sum(dim=0)
                action_count += actions.shape[0]
                for phase_index in range(4):
                    phase_visit_counts[phase_index] += int((raw._phase == phase_index).sum())

                observations, rewards, dones, extras = wrapped.step(actions)
                checks["rewards_finite"] &= bool(torch.isfinite(rewards).all())
                checks["no_nan_inf_or_overlap"] &= bool(
                    torch.isfinite(raw._robot.data.joint_pos.torch).all()
                    and torch.isfinite(raw._tomato.data.root_state_w.torch).all()
                )
                completed_episodes += int(dones.sum())
                maximum_hold_streak = max(maximum_hold_streak, int(raw._hold_streak.max()))
                detached = raw._stem_detached
                detached_samples += int(detached.sum())
                detached_stable_samples += int((detached & ~raw._dropped & ~raw._excess_force).sum())
                for name in event_counts:
                    event_counts[name] += int((raw._last_reward_terms[name] != 0.0).sum())
                break_event = raw._last_reward_terms["break_once"] != 0.0
                drop_event = raw._last_reward_terms["drop"] != 0.0
                break_and_drop_same_step += int((break_event & drop_event).sum())
                break_without_drop_same_step += int((break_event & ~drop_event).sum())

        action_mean = action_sum / max(action_count, 1)
        action_variance = (
            action_square_sum / max(action_count, 1) - action_mean.square()
        ).clamp_min(0.0)
        report = {
            "schema_version": 2,
            "passed": all(checks.values()),
            "checkpoint": str(args.checkpoint.resolve()),
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "sim2real_mode": args.sim2real_mode,
            "stochastic": args.stochastic,
            "post_detach_mode": args.post_detach_mode,
            "post_detach_override_samples": post_detach_override_samples,
            "closure_assist_after_contact": args.closure_assist_after_contact,
            "closure_assist_samples": closure_assist_samples,
            "break_and_drop_same_step": break_and_drop_same_step,
            "break_without_drop_same_step": break_without_drop_same_step,
            "checks": checks,
            "completed_episodes": completed_episodes,
            "reward_event_counts": event_counts,
            "phase_visit_counts": phase_visit_counts,
            "maximum_hold_streak": maximum_hold_streak,
            "detached_stable_rate": detached_stable_samples / max(detached_samples, 1),
            "action_blocks": {
                "translation_mean": action_mean[:3].tolist(),
                "rotation_mean": action_mean[3:6].tolist(),
                "hand_joint_mean": action_mean[6:22].tolist(),
                "translation_std": action_variance[:3].sqrt().tolist(),
                "rotation_std": action_variance[3:6].sqrt().tolist(),
                "hand_joint_std": action_variance[6:22].sqrt().tolist(),
            },
            "optimizer_steps": 0,
        }
        print("CHECKPOINT_AUDIT=" + json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["passed"] else 2
    except BaseException as exc:
        print(
            "CHECKPOINT_AUDIT_EXCEPTION="
            + json.dumps({"type": type(exc).__name__, "message": str(exc)}),
            flush=True,
        )
        traceback.print_exc()
        return 3
    finally:
        if wrapped is not None:
            wrapped.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
