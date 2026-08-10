"""Run the non-training integrated gate for the Physics-v2 Isaac task."""

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

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "logs/qualification/tomato_physics_v2/integrated_environment_gate.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--closed-center-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the grasp-frame closed fruit center for calibration only.",
    )
    parser.add_argument(
        "--nominal-reset",
        action="store_true",
        help="Use a near-nominal deterministic reset for grasp-center calibration.",
    )
    parser.add_argument(
        "--legacy-center-pregrasp",
        action="store_true",
        help="Keep the inherited center hand pose while sampling fruit.",
    )
    parser.add_argument(
        "--action-mode",
        choices=(
            "random",
            "zero",
            "grasp-sweep",
            "reach-close",
            "reach-stress",
            "collision-stress",
        ),
        default="random",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("--num-envs and --steps must be positive")

    launcher = AppLauncher(args)
    app = launcher.app
    raw_env = None
    try:
        gym = importlib.import_module("gymnasium")
        torch = importlib.import_module("torch")
        task = importlib.import_module("simulation.tomato_pick.isaac.physics_v2")
        cfg_module = importlib.import_module(
            "simulation.tomato_pick.isaac.physics_v2.env_cfg"
        )

        cfg = cfg_module.TomatoPickPhysicsV2EnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        if args.nominal_reset:
            cfg.reset_arm_joint_ellipsoid_radius_rad = (1.0e-6,) * 7
            cfg.reset_arm_correlated_modes_rad = ()
            cfg.reset_arm_correlated_mode_coefficient_ranges = ()
            cfg.reset_fruit_stem_tilt_rad_max = 0.0
            cfg.reset_fruit_yaw_rad = (-1.0e-6, 1.0e-6)
            cfg.reset_fruit_approach_yaw_rad = (-1.0e-6, 1.0e-6)
            cfg.reset_fruit_approach_pitch_rad = (-1.0e-6, 1.0e-6)
            cfg.maximum_closure_translation_norm_m = 0.20
            cfg.minimum_closure_translation_headroom_fraction = 0.05
        if args.legacy_center_pregrasp:
            cfg.open_hand_before_workspace_reset = False
        if args.closed_center_m is not None:
            cfg.closed_grasp_fruit_center_m = tuple(args.closed_center_m)
        raw_env = gym.make(task.TASK_ID, cfg=cfg)
        env = raw_env.unwrapped
        # Gym's seed does not reset every direct torch sampler used by the
        # inherited workspace reset.  Make calibration comparisons repeatable
        # across separate processes and center overrides.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        obs, _ = raw_env.reset(seed=args.seed)

        checks = {
            "task_is_registered": task.TASK_ID in gym.registry,
            "training_remains_blocked": (
                task.IMPLEMENTATION_STATUS.endswith("training_blocked")
            ),
            "actor_shape": tuple(obs["policy"].shape) == (args.num_envs, 66),
            "critic_shape": tuple(obs["critic"].shape) == (args.num_envs, 83),
            "action_shape": env.single_action_space.shape == (11,),
            "finite_reset": bool(
                torch.isfinite(obs["policy"]).all()
                and torch.isfinite(obs["critic"]).all()
            ),
            "physics_v2_robot_asset_selected": (
                str(cfg.robot.spawn.usd_path).endswith("_physics_v2.usda")
            ),
            "fruit_profiles_cover_all_envs": (
                env._fruit_effective_stiffness_n_m.shape == (args.num_envs,)
                and env._stem_torsional_limit_nm.shape == (args.num_envs,)
            ),
            "uncertainty_profiles_cover_all_envs": all(
                values.shape[0] == args.num_envs
                for values in env._uncertainty_profiles.values()
            ),
            "damage_state_cover_all_envs": all(
                values.shape[0] == args.num_envs
                for values in env._fruit_damage_state.values()
            ),
            "finite_rollout": True,
            "finite_v2_diagnostics": True,
            "termination_masks_disjoint": True,
        }

        generator = torch.Generator(device=env.device).manual_seed(args.seed + 1)
        maximum_reward_abs = 0.0
        maximum_measured_force_n = 0.0
        maximum_physical_force_n = 0.0
        maximum_damage_score = 0.0
        minimum_closed_center_error_m = float("inf")
        initial_ee_position_w = env._robot.data.body_pos_w.torch[
            :, env._ee_body_id
        ].clone()
        initial_ee_quaternion_w = env._robot.data.body_quat_w.torch[
            :, env._ee_body_id
        ].clone()
        initial_hand_joint_position = env._robot.data.joint_pos.torch[
            :, env._hand_joint_ids
        ].clone()
        maximum_ee_displacement_m = 0.0
        maximum_hand_joint_displacement_rad = 0.0
        maximum_enclosure_score = 0.0
        terminated_count = 0
        truncated_count = 0
        contact_envs: set[int] = set()
        done_reason_counts = {
            "success": 0,
            "drop": 0,
            "excess_force": 0,
            "out_of_bounds": 0,
            "timeout": 0,
        }
        grasp_trajectory = None
        if args.action_mode in ("grasp-sweep", "reach-close", "reach-stress"):
            grasp_trajectory = importlib.import_module(
                "script.render_tomato_hand_5d_manifold_v2"
            ).trajectory()
        for step_index in range(args.steps):
            if args.action_mode in ("reach-close", "reach-stress"):
                actions = torch.zeros((args.num_envs, 11), device=env.device)
                fruit_position_w = env._tomato.data.root_pos_w.torch
                ee_position_w = env._robot.data.body_pos_w.torch[:, env._ee_body_id]
                closed_center_g = torch.tensor(
                    cfg.closed_grasp_fruit_center_m, device=env.device
                ).expand(args.num_envs, -1)
                desired_ee_position_w = fruit_position_w - torch.bmm(
                    env._grasp_basis_w, closed_center_g.unsqueeze(2)
                ).squeeze(2)
                error_g = torch.bmm(
                    env._grasp_basis_w.transpose(1, 2),
                    (desired_ee_position_w - ee_position_w).unsqueeze(2),
                ).squeeze(2)
                minimum_closed_center_error_m = min(
                    minimum_closed_center_error_m,
                    float(torch.linalg.vector_norm(error_g, dim=1).min()),
                )
                translation_scales = torch.tensor(
                    cfg.action_scales[:3], device=env.device
                )
                actions[:, :3] = (error_g / translation_scales).clamp(-0.5, 0.5)
                closure_start = max(1, args.steps // 3)
                closure_duration = max(1, args.steps // 3)
                closure_index = min(
                    len(grasp_trajectory) - 1,
                    max(0, step_index - closure_start)
                    * len(grasp_trajectory)
                    // closure_duration,
                )
                actions[:, 6:11] = torch.tensor(
                    grasp_trajectory[closure_index], device=env.device
                )
                if (
                    args.action_mode == "reach-stress"
                    and step_index >= 2 * args.steps // 3
                ):
                    actions[:, 6:11] = torch.tensor(
                        (1.0, 0.0, 0.0, 0.0, 1.0), device=env.device
                    )
            elif grasp_trajectory is not None:
                frame_index = min(
                    len(grasp_trajectory) - 1,
                    step_index * len(grasp_trajectory) // args.steps,
                )
                actions = torch.zeros((args.num_envs, 11), device=env.device)
                actions[:, 6:11] = torch.tensor(
                    grasp_trajectory[frame_index], device=env.device
                )
            elif args.action_mode == "collision-stress":
                actions = torch.zeros((args.num_envs, 11), device=env.device)
                actions[:, 6:11] = torch.tensor(
                    (1.0, 0.0, 0.0, 0.0, 1.0), device=env.device
                )
            elif args.action_mode == "zero":
                actions = torch.zeros((args.num_envs, 11), device=env.device)
            else:
                actions = 0.05 * (
                    2.0
                    * torch.rand(
                        (args.num_envs, 11),
                        generator=generator,
                        device=env.device,
                    )
                    - 1.0
                )
            obs, reward, terminated, truncated, info = raw_env.step(actions)
            checks["finite_rollout"] &= bool(
                torch.isfinite(obs["policy"]).all()
                and torch.isfinite(obs["critic"]).all()
                and torch.isfinite(reward).all()
                and torch.isfinite(env._robot.data.joint_pos.torch).all()
                and torch.isfinite(env._robot.data.joint_vel.torch).all()
            )
            checks["finite_v2_diagnostics"] &= bool(
                all(
                    torch.isfinite(values).all()
                    for values in env._fruit_damage_state.values()
                )
                and all(
                    torch.isfinite(values).all()
                    for values in env._stem_load_components_v2.values()
                )
                and torch.isfinite(
                    env._tactile_uncertainty_state["measured_n"]
                ).all()
            )
            checks["termination_masks_disjoint"] &= not bool(
                (terminated & truncated).any()
            )
            maximum_reward_abs = max(maximum_reward_abs, float(reward.abs().max()))
            maximum_measured_force_n = max(
                maximum_measured_force_n,
                float(env._tactile_uncertainty_state["measured_n"].max()),
            )
            maximum_physical_force_n = max(
                maximum_physical_force_n,
                float(env._last_physical_force_n.max()),
            )
            contact_envs.update(
                env._last_physical_force_n.amax(dim=1)
                .gt(0.05)
                .nonzero()
                .flatten()
                .tolist()
            )
            maximum_damage_score = max(
                maximum_damage_score,
                float(env._fruit_damage_state["damage_score"].max()),
            )
            maximum_ee_displacement_m = max(
                maximum_ee_displacement_m,
                float(
                    torch.linalg.vector_norm(
                        env._robot.data.body_pos_w.torch[:, env._ee_body_id]
                        - initial_ee_position_w,
                        dim=1,
                    ).max()
                ),
            )
            maximum_enclosure_score = max(
                maximum_enclosure_score, float(env._enclosure_score().max())
            )
            maximum_hand_joint_displacement_rad = max(
                maximum_hand_joint_displacement_rad,
                float(
                    (
                        env._robot.data.joint_pos.torch[:, env._hand_joint_ids]
                        - initial_hand_joint_position
                    ).abs().max()
                ),
            )
            terminated_count += int(terminated.sum())
            truncated_count += int(truncated.sum())
            step_info = info.get("step", {})
            for reason in done_reason_counts:
                done_reason_counts[reason] += int(
                    step_info.get(f"{reason}_count", 0)
                )

        if args.action_mode in ("reach-close", "reach-stress"):
            checks["scripted_contact_observed"] = len(contact_envs) > 0

        report = {
            "schema_version": 1,
            "passed": all(checks.values()),
            "task_id": task.TASK_ID,
            "implementation_status": task.IMPLEMENTATION_STATUS,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "action_mode": args.action_mode,
            "device": str(env.device),
            "checks": checks,
            "maximum_reward_abs": maximum_reward_abs,
            "maximum_measured_force_n": maximum_measured_force_n,
            "maximum_physical_force_n": maximum_physical_force_n,
            "maximum_damage_score": maximum_damage_score,
            "minimum_closed_center_error_m": (
                minimum_closed_center_error_m
                if minimum_closed_center_error_m != float("inf")
                else None
            ),
            "maximum_ee_displacement_m": maximum_ee_displacement_m,
            "maximum_enclosure_score": maximum_enclosure_score,
            "maximum_hand_joint_displacement_rad": (
                maximum_hand_joint_displacement_rad
            ),
            "final_fruit_center_in_grasp_frame_m": torch.bmm(
                env._grasp_basis_w.transpose(1, 2),
                (
                    env._tomato.data.root_pos_w.torch
                    - env._robot.data.body_pos_w.torch[:, env._ee_body_id]
                ).unsqueeze(2),
            ).squeeze(2)[0].tolist(),
            "final_fingertip_center_distance_m": float(
                torch.linalg.vector_norm(
                    env._robot.data.body_pos_w.torch[
                        :, env._fingertip_contact_body_ids
                    ].mean(dim=1)
                    - env._tomato.data.root_pos_w.torch,
                    dim=1,
                ).min()
            ),
            "final_fingertip_center_in_grasp_frame_m": torch.bmm(
                env._grasp_basis_w.transpose(1, 2),
                (
                    env._robot.data.body_pos_w.torch[
                        :, env._fingertip_contact_body_ids
                    ].mean(dim=1)
                    - env._robot.data.body_pos_w.torch[:, env._ee_body_id]
                ).unsqueeze(2),
            ).squeeze(2)[0].tolist(),
            "final_fingertip_positions_in_grasp_frame_m": torch.bmm(
                env._grasp_basis_w.transpose(1, 2).unsqueeze(1).expand(-1, 5, -1, -1).reshape(-1, 3, 3),
                (
                    env._robot.data.body_pos_w.torch[
                        :, env._fingertip_contact_body_ids
                    ]
                    - env._robot.data.body_pos_w.torch[:, env._ee_body_id].unsqueeze(1)
                ).reshape(-1, 3, 1),
            ).reshape(args.num_envs, 5, 3)[0].tolist(),
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "contact_env_count": len(contact_envs),
            "contact_env_fraction": len(contact_envs) / args.num_envs,
            "done_reason_counts": done_reason_counts,
            "initial_reset_fruit_local_m": env._reset_workspace_fruit_local_m[
                0
            ].tolist(),
            "initial_reset_closure_translation_norm_m": float(
                torch.linalg.vector_norm(
                    env._reset_workspace_fruit_local_m[0]
                    - torch.tensor(
                        cfg.closed_grasp_fruit_center_m, device=env.device
                    )
                )
            ),
            "grasp_basis_w": env._grasp_basis_w[0].tolist(),
            "initial_ee_quaternion_wxyz": initial_ee_quaternion_w[0].tolist(),
            "final_ee_quaternion_wxyz": env._robot.data.body_quat_w.torch[
                0, env._ee_body_id
            ].tolist(),
            "training_started": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "TOMATO_PHYSICS_V2_INTEGRATED_GATE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )
        return 0 if report["passed"] else 2
    except BaseException as exc:
        report = {
            "schema_version": 1,
            "passed": False,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "training_started": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "TOMATO_PHYSICS_V2_INTEGRATED_GATE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )
        return 1
    finally:
        if raw_env is not None:
            raw_env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
