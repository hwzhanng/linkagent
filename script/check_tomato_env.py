"""Run the final, non-training Isaac gate for the minimal tomato MDP."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or args.steps < 4:
        raise ValueError("--num-envs must be positive and --steps must be at least 4")

    launcher = AppLauncher(args)
    app = launcher.app
    raw_env = None
    try:
        gym = importlib.import_module("gymnasium")
        torch = importlib.import_module("torch")
        task = importlib.import_module("simulation.tomato_pick.isaac")
        cfg_module = importlib.import_module("simulation.tomato_pick.isaac.env_cfg")

        cfg = cfg_module.TomatoPickEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        raw_env = gym.make(task.TASK_ID, cfg=cfg)
        env = raw_env.unwrapped
        obs, _ = raw_env.reset(seed=args.seed)

        initial_targets = env._joint_targets[:, env._hand_joint_ids].clone()
        fruit = env._tomato.data.root_pos_w.torch
        env_ids = torch.arange(args.num_envs, device=env.device)
        clearance = env._hand_collision_clearance_for(
            fruit, env._fruit_axes_m, env_ids
        )
        pull_axis = env._grasp_basis_w[:, :, 2]
        fruit_attachment = fruit - pull_axis * env._fruit_axes_m[:, 2:3]
        plant_anchor = env._stem_plant_anchor_pos_w
        visible_center = env._stem_visual_positions_w
        stem_vector = plant_anchor - fruit_attachment
        stem_distance = torch.linalg.vector_norm(stem_vector, dim=1)
        stem_axis_cosine = (stem_vector * pull_axis).sum(dim=1) / stem_distance.clamp_min(1.0e-8)
        visible_center_error = torch.linalg.vector_norm(
            visible_center - 0.5 * (fruit_attachment + plant_anchor), dim=1
        )
        fruit_endpoint_error = torch.linalg.vector_norm(
            env._stem_rest_attachment_pos_w - fruit_attachment, dim=1
        )
        ee_position = env._robot.data.body_pos_w.torch[:, env._ee_body_id]
        calibrated_center_g = torch.tensor(
            cfg.closed_grasp_fruit_center_m, device=env.device
        ).expand(args.num_envs, -1)
        calibrated_center_w = ee_position + torch.bmm(
            env._grasp_basis_w, calibrated_center_g.unsqueeze(2)
        ).squeeze(2)
        required_translation_g = torch.bmm(
            env._grasp_basis_w.transpose(1, 2),
            (fruit - calibrated_center_w).unsqueeze(2),
        ).squeeze(2)
        checks = {
            "actor_shape": tuple(obs["policy"].shape) == (args.num_envs, 66),
            "critic_shape": tuple(obs["critic"].shape) == (args.num_envs, 83),
            "action_shape": env.single_action_space.shape == (11,),
            "finite_reset_observation": bool(
                torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
            ),
            "reset_clearance": bool(
                (clearance >= cfg.minimum_initial_clearance_m - 1.0e-4).all()
            ),
            "zero_initial_contact": True,
            "zero_action_keeps_pregrasp": True,
            "cartesian_action_changes_command": False,
            "synergy_action_changes_targets": False,
            "visible_stem_is_on_plant_side": bool((stem_axis_cosine < -0.999).all()),
            "visible_stem_touches_fruit_top": bool(
                (visible_center_error < 1.0e-6).all()
                and (fruit_endpoint_error < 1.0e-6).all()
                and torch.allclose(
                    stem_distance,
                    torch.full_like(stem_distance, cfg.stem_visual_height_m),
                    atol=1.0e-6,
                )
            ),
            "reset_is_inside_cartesian_workspace": bool(
                (
                    required_translation_g.abs()
                    < torch.tensor(cfg.max_residual_translation_m, device=env.device)
                ).all()
            ),
            "attachment_breaks_under_axial_pull": False,
            "attachment_survives_axial_compression": False,
            "finite_rollout": True,
            "termination_masks_disjoint": True,
        }
        maximum_force = 0.0
        maximum_stem_force = 0.0
        first_detached_step = None
        maximum_hand_delta = 0.0
        maximum_ee_command_delta = 0.0
        contact_envs: set[int] = set()
        detached_envs: set[int] = set()

        zero = torch.zeros((args.num_envs, 11), device=env.device)
        obs, reward, terminated, truncated, _ = raw_env.step(zero)
        initial_force = float(env._last_physical_force_n.max())
        initial_stem_force = float(
            torch.linalg.vector_norm(env._last_stem_force_w, dim=1).max()
        )
        initial_detached_count = int(env._stem_detached.sum())
        checks["zero_initial_contact"] = initial_force <= 0.05
        checks["zero_action_keeps_pregrasp"] = bool(
            torch.allclose(
                env._joint_targets[:, env._hand_joint_ids], initial_targets, atol=1.0e-6
            )
        )

        for step in range(1, args.steps):
            action = torch.zeros((args.num_envs, 11), device=env.device)
            # This is wiring excitation, not a training expert.  It asks for a
            # modest horizontal palm approach and closure, then a modest
            # downward axial pull/twist.
            if step < args.steps // 2:
                action[:, 0] = 0.35
                action[:, 6] = 0.45
                action[:, 7] = 0.20
                action[:, 8] = 0.35
                action[:, 10] = 0.35
            else:
                action[:, 2] = 0.35
                action[:, 5] = 0.25
                action[:, 6] = 0.55
                action[:, 7] = 0.30
                action[:, 8] = 0.45
                action[:, 10] = 0.45
            obs, reward, terminated, truncated, _ = raw_env.step(action)
            maximum_force = max(maximum_force, float(env._last_physical_force_n.max()))
            maximum_stem_force = max(
                maximum_stem_force,
                float(torch.linalg.vector_norm(env._last_stem_force_w, dim=1).max()),
            )
            if first_detached_step is None and bool(env._stem_detached.any()):
                first_detached_step = step
            maximum_hand_delta = max(
                maximum_hand_delta,
                float(
                    (env._joint_targets[:, env._hand_joint_ids] - initial_targets)
                    .abs()
                    .max()
                ),
            )
            maximum_ee_command_delta = max(
                maximum_ee_command_delta,
                float(
                    (env._ee_command_pose_w[:, :3] - env._ee_reference_pose_w[:, :3])
                    .abs()
                    .max()
                ),
            )
            contact_envs.update(
                env._last_physical_force_n.amax(dim=1).gt(0.05).nonzero().flatten().tolist()
            )
            detached_envs.update(env._stem_detached.nonzero().flatten().tolist())
            checks["finite_rollout"] &= bool(
                torch.isfinite(obs["policy"]).all()
                and torch.isfinite(obs["critic"]).all()
                and torch.isfinite(reward).all()
                and torch.isfinite(env._robot.data.joint_pos.torch).all()
                and torch.isfinite(env._robot.data.joint_vel.torch).all()
            )
            checks["termination_masks_disjoint"] &= not bool((terminated & truncated).any())

        checks["cartesian_action_changes_command"] = maximum_ee_command_delta > 1.0e-4
        checks["synergy_action_changes_targets"] = maximum_hand_delta > 1.0e-4

        # Isolate the stem integration: displace one still-attached fruit away
        # from the plant.  This checks the sign convention and one-shot event,
        # not a scripted grasp or a policy demonstration.
        candidate = (~env._stem_detached).nonzero().flatten()
        if len(candidate) >= 2:
            pull_env_id = candidate[:1]
            compression_env_id = candidate[1:2]
            selected = torch.cat((pull_env_id, compression_env_id))
            pose = env._tomato.data.root_pose_w.torch[selected].clone()
            pose[0, :3] += 0.08 * env._grasp_basis_w[pull_env_id, :, 2][0]
            pose[1, :3] -= 0.08 * env._grasp_basis_w[compression_env_id, :, 2][0]
            env._tomato.write_root_pose_to_sim_index(root_pose=pose, env_ids=selected)
            env._tomato.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((2, 6), device=env.device), env_ids=selected
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(dt=0.0)
            env._stem_break_event.zero_()
            for _ in range(cfg.stem_break_persistence_substeps):
                # Hold the two counterfactual poses quasi-statically.  The
                # attachment integrator normally changes fruit velocity after
                # each substep; resetting it here isolates persistence and
                # force-direction semantics from free-body acceleration.
                env._tomato.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((2, 6), device=env.device),
                    env_ids=selected,
                )
                env._apply_attachment_wrench()
            checks["attachment_breaks_under_axial_pull"] = bool(
                env._stem_detached[pull_env_id].all()
            )
            checks["attachment_survives_axial_compression"] = bool(
                (~env._stem_detached[compression_env_id]).all()
            )
        report = {
            "schema_version": 1,
            "passed": all(checks.values()),
            "task_id": task.TASK_ID,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "checks": checks,
            "minimum_initial_clearance_m": float(clearance.min()),
            "first_fruit_position_w_m": fruit[0].tolist(),
            "first_ee_position_w_m": ee_position[0].tolist(),
            "first_stem_anchor_position_w_m": plant_anchor[0].tolist(),
            "first_environment_origin_w_m": env.scene.env_origins[0].tolist(),
            "maximum_required_reset_translation_m": required_translation_g.abs()
            .amax(dim=0)
            .tolist(),
            "maximum_visible_stem_axis_cosine": float(stem_axis_cosine.max()),
            "maximum_visible_stem_center_error_m": float(visible_center_error.max()),
            "maximum_fruit_stem_endpoint_error_m": float(fruit_endpoint_error.max()),
            "validated_hand_collision_body_count": len(env._hand_collision_body_ids),
            "initial_contact_force_n": initial_force,
            "initial_stem_force_n": initial_stem_force,
            "initial_detached_count": initial_detached_count,
            "first_detached_step": first_detached_step,
            "maximum_contact_force_n": maximum_force,
            "maximum_stem_force_n": maximum_stem_force,
            "contact_environment_count": len(contact_envs),
            "detached_environment_count": len(detached_envs),
            "maximum_hand_target_delta_rad": maximum_hand_delta,
            "maximum_ee_command_delta_m": maximum_ee_command_delta,
            "training_started": False,
        }
        print("TOMATO_ENV_GATE=" + json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["passed"] else 2
    except BaseException as exc:
        print(
            "TOMATO_ENV_GATE_EXCEPTION="
            + json.dumps({"type": type(exc).__name__, "message": str(exc)}),
            flush=True,
        )
        traceback.print_exc()
        return 3
    finally:
        if raw_env is not None:
            raw_env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
