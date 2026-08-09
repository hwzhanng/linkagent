"""Render the actual Physics-v1 A7/palm/fruit reset workspace as a point cloud."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _range_summary(array, numpy_module) -> dict[str, list[float]]:
    return {
        "minimum": numpy_module.min(array, axis=0).tolist(),
        "p05": numpy_module.quantile(array, 0.05, axis=0).tolist(),
        "median": numpy_module.median(array, axis=0).tolist(),
        "p95": numpy_module.quantile(array, 0.95, axis=0).tolist(),
        "maximum": numpy_module.max(array, axis=0).tolist(),
    }


def _set_equal_3d(ax, points, numpy_module) -> None:
    lower = numpy_module.min(points, axis=0)
    upper = numpy_module.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.55 * float(numpy_module.max(upper - lower))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def main(argv: Sequence[str] | None = None) -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--joint-radii-deg",
        type=float,
        nargs=7,
        metavar=("A1", "A2", "A3", "A4", "A5", "A6", "A7"),
        help="override the seven reset ellipsoid radii for a candidate cloud",
    )
    parser.add_argument(
        "--artifact-tag",
        default="",
        help="optional filesystem-safe label appended to output artifacts",
    )
    parser.add_argument(
        "--disable-correlated-modes",
        action="store_true",
        help="render the base joint ellipsoid alone for a same-seed A/B cloud",
    )
    parser.add_argument(
        "--correlated-mode-scale",
        type=float,
        default=1.0,
        help="diagnostic-only scale applied to every configured correlated joint mode",
    )
    parser.add_argument(
        "--closure-headroom-fraction",
        type=float,
        help="override the runtime Cartesian closure-headroom gate for diagnostics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "logs/diagnostics/tomato_physics_v1_reset_workspace",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 64:
        raise ValueError("at least 64 environments are required for an honest point cloud")

    launcher = AppLauncher(args)
    app = launcher.app
    raw_env = None
    try:
        gym = importlib.import_module("gymnasium")
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        plt = importlib.import_module("matplotlib.pyplot")
        task = importlib.import_module("simulation.tomato_pick.isaac.physics_v1")
        cfg_module = importlib.import_module(
            "simulation.tomato_pick.isaac.physics_v1.env_cfg"
        )

        cfg = cfg_module.TomatoPickPhysicsV1EnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        cfg.sim.device = args.device
        if args.joint_radii_deg is not None:
            if any(value <= 0.0 or not math.isfinite(value) for value in args.joint_radii_deg):
                raise ValueError("all joint radii must be finite and positive")
            cfg.reset_arm_joint_ellipsoid_radius_rad = tuple(
                math.radians(value) for value in args.joint_radii_deg
            )
        if args.disable_correlated_modes:
            cfg.reset_arm_correlated_modes_rad = ()
        elif args.correlated_mode_scale != 1.0:
            if not 0.0 < args.correlated_mode_scale <= 2.0:
                raise ValueError("correlated mode scale must lie in (0, 2]")
            cfg.reset_arm_correlated_modes_rad = tuple(
                tuple(args.correlated_mode_scale * value for value in mode)
                for mode in cfg.reset_arm_correlated_modes_rad
            )
        if args.closure_headroom_fraction is not None:
            if not 0.0 <= args.closure_headroom_fraction < 1.0:
                raise ValueError("closure headroom fraction must lie in [0, 1)")
            cfg.minimum_closure_translation_headroom_fraction = (
                args.closure_headroom_fraction
            )
        raw_env = gym.make(task.TASK_ID, cfg=cfg)
        env = raw_env.unwrapped
        try:
            observation, _ = raw_env.reset(seed=args.seed)
        except BaseException as error:
            print(
                "RESET_WORKSPACE_INITIALIZATION_FAILED="
                + json.dumps(
                    {
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise
        if observation["policy"].shape != (args.num_envs, 66):
            raise RuntimeError("Physics-v1 actor observation shape drifted")

        origin = env.scene.env_origins
        palm = env._robot.data.body_pos_w.torch[:, env._ee_body_id] - origin
        fruit = env._tomato.data.root_pos_w.torch - origin
        arm_offset = env._reset_workspace_arm_joint_offset_rad
        base_arm_offset = env._reset_workspace_base_arm_joint_offset_rad
        mode_coefficient = env._reset_workspace_mode_coefficients
        local_fruit = env._reset_workspace_fruit_local_m
        tilt = env._reset_workspace_fruit_tilt_rad
        clearance = env._reset_workspace_clearance_m
        translation_headroom = env._reset_workspace_translation_headroom_m
        pull_axis = env._grasp_basis_w[:, :, 2]
        force = env._physical_finger_force()
        tensors = (
            palm,
            fruit,
            arm_offset,
            base_arm_offset,
            mode_coefficient,
            local_fruit,
            tilt,
            clearance,
            translation_headroom,
            pull_axis,
            force,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("reset workspace contains NaN or Inf")
        if bool((clearance < cfg.minimum_initial_clearance_m).any()):
            raise RuntimeError("reset workspace contains an initially intersecting fruit")

        # Verify the complete pre-grasp-to-closed-grasp motion, not merely the
        # legality of the reset pose.  The goal preserves palm orientation and
        # moves the fruit from its sampled local offset into the configured
        # closed-grasp centre.  IK is constrained to retain an explicit joint
        # margin rather than merely clamping at the physical limit.
        all_env_ids = torch.arange(args.num_envs, device=env.device)
        initial_ee_pose = env._robot.data.body_pose_w.torch[
            :, env._ee_body_id
        ].clone()
        closed_center = torch.tensor(
            cfg.closed_grasp_fruit_center_m, device=env.device
        ).expand(args.num_envs, -1)
        closure_translation_local = local_fruit - closed_center
        closure_target_position = initial_ee_pose[:, :3] + torch.bmm(
            env._grasp_basis_w, closure_translation_local.unsqueeze(2)
        ).squeeze(2)
        closure_target_quaternion = initial_ee_pose[:, 3:7].clone()
        physical_limits = env._robot.data.joint_pos_limits.torch[
            all_env_ids[:, None], env._arm_joint_ids
        ]
        physical_span = physical_limits[..., 1] - physical_limits[..., 0]
        closure_margin = cfg.minimum_closure_joint_limit_margin_fraction
        closure_lower = physical_limits[..., 0] + closure_margin * physical_span
        closure_upper = physical_limits[..., 1] - closure_margin * physical_span
        identity6 = torch.eye(6, device=env.device).expand(args.num_envs, -1, -1)
        jacobian_ids = [
            index + env._robot.num_base_dofs for index in env._arm_joint_ids
        ]
        for _ in range(40):
            current_pose = env._robot.data.body_pose_w.torch[:, env._ee_body_id]
            from isaaclab.utils.math import compute_pose_error

            position_error, rotation_error = compute_pose_error(
                current_pose[:, :3],
                current_pose[:, 3:7],
                closure_target_position,
                closure_target_quaternion,
            )
            error = torch.cat((position_error, rotation_error), dim=1)
            jacobian = torch.nan_to_num(
                env._robot.data.body_link_jacobian_w.torch[
                    :, env._ee_jacobian_id, :, jacobian_ids
                ]
            )
            dls = torch.linalg.solve(
                torch.bmm(jacobian, jacobian.transpose(1, 2))
                + cfg.ik_damping**2 * identity6,
                error.unsqueeze(2),
            )
            dq = torch.bmm(jacobian.transpose(1, 2), dls).squeeze(2)
            dq = torch.nan_to_num(dq)
            largest = dq.abs().amax(dim=1, keepdim=True)
            dq *= 0.08 / largest.clamp_min(0.08)
            joint_position = env._robot.data.joint_pos.torch[
                :, env._arm_joint_ids
            ] + dq
            joint_position = torch.maximum(
                closure_lower, torch.minimum(closure_upper, joint_position)
            )
            env._robot.write_joint_position_to_sim_index(
                position=joint_position,
                joint_ids=env._arm_joint_ids,
                env_ids=all_env_ids,
            )
            env._robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(joint_position),
                joint_ids=env._arm_joint_ids,
                env_ids=all_env_ids,
            )
            env.sim.forward()
            env.scene.update(dt=0.0)

        closure_pose = env._robot.data.body_pose_w.torch[:, env._ee_body_id]
        closure_position_error = torch.linalg.vector_norm(
            closure_pose[:, :3] - closure_target_position, dim=1
        )
        closure_joint_position = env._robot.data.joint_pos.torch[
            :, env._arm_joint_ids
        ]
        closure_joint_margin_fraction = torch.minimum(
            (closure_joint_position - physical_limits[..., 0]) / physical_span,
            (physical_limits[..., 1] - closure_joint_position) / physical_span,
        ).amin(dim=1)
        closure_ik_reachable = (
            closure_position_error <= cfg.maximum_closure_ik_position_error_m
        ) & (
            closure_joint_margin_fraction
            >= cfg.minimum_closure_joint_limit_margin_fraction - 1.0e-5
        )
        if not bool(closure_ik_reachable.all()):
            failed_indices = torch.nonzero(~closure_ik_reachable).squeeze(1)
            failed_indices = failed_indices[:16]
            print(
                "CLOSURE_IK_GATE_FAILED="
                + json.dumps(
                    {
                        "reachable": int(closure_ik_reachable.sum()),
                        "total": args.num_envs,
                        "position_error_m_maximum": float(closure_position_error.max()),
                        "position_error_m_p95": float(torch.quantile(closure_position_error, 0.95)),
                        "joint_margin_fraction_minimum": float(closure_joint_margin_fraction.min()),
                        "joint_margin_fraction_p05": float(torch.quantile(closure_joint_margin_fraction, 0.05)),
                        "failed_samples": [
                            {
                                "sample": int(index),
                                "palm_xyz_m": [
                                    float(value) for value in palm[index]
                                ],
                                "closure_translation_local_m": [
                                    float(value)
                                    for value in closure_translation_local[index]
                                ],
                                "mode_coefficient": [
                                    float(value)
                                    for value in mode_coefficient[index]
                                ],
                                "position_error_m": float(
                                    closure_position_error[index]
                                ),
                                "joint_margin_fraction": float(
                                    closure_joint_margin_fraction[index]
                                ),
                            }
                            for index in failed_indices
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise RuntimeError(
                "reset workspace contains a closure pose without the required IK margin"
            )

        palm_np = palm.detach().cpu().numpy()
        fruit_np = fruit.detach().cpu().numpy()
        offset_np = arm_offset.detach().cpu().numpy()
        base_offset_np = base_arm_offset.detach().cpu().numpy()
        mode_coefficient_np = mode_coefficient.detach().cpu().numpy()
        local_np = local_fruit.detach().cpu().numpy()
        tilt_np = torch.rad2deg(tilt).detach().cpu().numpy()
        clearance_np = clearance.detach().cpu().numpy()
        translation_headroom_np = translation_headroom.detach().cpu().numpy()
        closure_position_error_np = closure_position_error.detach().cpu().numpy()
        closure_joint_margin_np = closure_joint_margin_fraction.detach().cpu().numpy()
        pull_np = pull_axis.detach().cpu().numpy()
        force_np = force.detach().cpu().numpy()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_tag = args.artifact_tag.strip().replace(" ", "_")
        if artifact_tag and not all(
            character.isalnum() or character in "_-" for character in artifact_tag
        ):
            raise ValueError("--artifact-tag may contain only letters, digits, '-' and '_'")
        suffix = f"_{artifact_tag}" if artifact_tag else ""
        stem = f"physics_v1_reset_workspace_seed{args.seed}_{args.num_envs}{suffix}"
        csv_path = (args.output_dir / f"{stem}.csv").resolve()
        json_path = (args.output_dir / f"{stem}.json").resolve()
        png_path = (args.output_dir / f"{stem}.png").resolve()
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "sample_id",
                "palm_x_m", "palm_y_m", "palm_z_m",
                "fruit_x_m", "fruit_y_m", "fruit_z_m",
                "fruit_local_approach_m", "fruit_local_lateral_m", "fruit_local_pull_m",
                "fruit_tilt_deg", "clearance_m",
                "pull_x", "pull_y", "pull_z",
                "maximum_reset_contact_force_n",
                "closure_translation_headroom_approach_m",
                "closure_translation_headroom_lateral_m",
                "closure_translation_headroom_pull_m",
                "closure_ik_position_error_m",
                "closure_joint_limit_margin_fraction",
            ] + [f"a7_joint_{index + 1}_offset_rad" for index in range(7)]
            fieldnames += [
                f"a7_joint_{index + 1}_base_ellipsoid_offset_rad"
                for index in range(7)
            ]
            fieldnames += [
                f"correlated_mode_{index + 1}_coefficient"
                for index in range(mode_coefficient_np.shape[1])
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(args.num_envs):
                row = {
                    "sample_id": index,
                    "palm_x_m": palm_np[index, 0],
                    "palm_y_m": palm_np[index, 1],
                    "palm_z_m": palm_np[index, 2],
                    "fruit_x_m": fruit_np[index, 0],
                    "fruit_y_m": fruit_np[index, 1],
                    "fruit_z_m": fruit_np[index, 2],
                    "fruit_local_approach_m": local_np[index, 0],
                    "fruit_local_lateral_m": local_np[index, 1],
                    "fruit_local_pull_m": local_np[index, 2],
                    "fruit_tilt_deg": tilt_np[index],
                    "clearance_m": clearance_np[index],
                    "pull_x": pull_np[index, 0],
                    "pull_y": pull_np[index, 1],
                    "pull_z": pull_np[index, 2],
                    "maximum_reset_contact_force_n": force_np[index].max(),
                    "closure_translation_headroom_approach_m": translation_headroom_np[index, 0],
                    "closure_translation_headroom_lateral_m": translation_headroom_np[index, 1],
                    "closure_translation_headroom_pull_m": translation_headroom_np[index, 2],
                    "closure_ik_position_error_m": closure_position_error_np[index],
                    "closure_joint_limit_margin_fraction": closure_joint_margin_np[index],
                }
                row.update(
                    {
                        f"a7_joint_{joint + 1}_offset_rad": offset_np[index, joint]
                        for joint in range(7)
                    }
                )
                row.update(
                    {
                        f"a7_joint_{joint + 1}_base_ellipsoid_offset_rad": base_offset_np[index, joint]
                        for joint in range(7)
                    }
                )
                row.update(
                    {
                        f"correlated_mode_{mode + 1}_coefficient": mode_coefficient_np[index, mode]
                        for mode in range(mode_coefficient_np.shape[1])
                    }
                )
                writer.writerow(row)

        combined = np.concatenate((palm_np, fruit_np), axis=0)
        summary = {
            "schema_version": 1,
            "task_id": task.TASK_ID,
            "reset_workspace_version": cfg_module.RESET_WORKSPACE.get("version", 1),
            "seed": args.seed,
            "sample_count": args.num_envs,
            "artifact_tag": artifact_tag,
            "arm_joint_ellipsoid_radius_deg_in_a7_order": [
                math.degrees(value)
                for value in cfg.reset_arm_joint_ellipsoid_radius_rad
            ],
            "coordinate_frame": "robot environment frame in metres",
            "palm_position_m": _range_summary(palm_np, np),
            "fruit_position_m": _range_summary(fruit_np, np),
            "fruit_minus_palm_m": _range_summary(fruit_np - palm_np, np),
            "a7_joint_offset_rad": _range_summary(offset_np, np),
            "a7_base_ellipsoid_offset_rad": _range_summary(base_offset_np, np),
            "correlated_mode_coefficient": _range_summary(mode_coefficient_np, np),
            "fruit_tilt_deg": {
                "minimum": float(tilt_np.min()),
                "median": float(np.median(tilt_np)),
                "p95": float(np.quantile(tilt_np, 0.95)),
                "maximum": float(tilt_np.max()),
            },
            "initial_clearance_m": {
                "minimum": float(clearance_np.min()),
                "median": float(np.median(clearance_np)),
                "maximum": float(clearance_np.max()),
            },
            "closure_translation_headroom_m": _range_summary(
                translation_headroom_np, np
            ),
            "closure_translation_norm_m": _range_summary(
                np.linalg.norm(
                    local_np - np.asarray(cfg.closed_grasp_fruit_center_m), axis=1
                ),
                np,
            ),
            "closure_ik": {
                "reachable_fraction": float(closure_ik_reachable.float().mean()),
                "position_error_m_maximum": float(closure_position_error_np.max()),
                "position_error_m_p95": float(np.quantile(closure_position_error_np, 0.95)),
                "joint_limit_margin_fraction_minimum": float(closure_joint_margin_np.min()),
                "joint_limit_margin_fraction_p05": float(np.quantile(closure_joint_margin_np, 0.05)),
            },
            "maximum_reset_contact_force_n": float(force_np.max()),
            "finite": True,
            "training_started": False,
            "csv_path": str(csv_path),
            "png_path": str(png_path),
        }
        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

        blue = "#1769AA"
        orange = "#D97706"
        figure = plt.figure(figsize=(13.6, 10.2), facecolor="white")
        grid = figure.add_gridspec(2, 2, hspace=0.26, wspace=0.20)
        ax3d = figure.add_subplot(grid[0, 0], projection="3d")
        ax_xy = figure.add_subplot(grid[0, 1])
        ax_xz = figure.add_subplot(grid[1, 0])
        ax_yz = figure.add_subplot(grid[1, 1])
        ax3d.scatter(
            palm_np[:, 0], palm_np[:, 1], palm_np[:, 2],
            s=9, c=blue, alpha=0.34, marker="o", linewidths=0,
            label="Palm reset",
        )
        ax3d.scatter(
            fruit_np[:, 0], fruit_np[:, 1], fruit_np[:, 2],
            s=12, c=orange, alpha=0.42, marker="x", linewidths=0.7,
            label="Fruit reset",
        )
        ax3d.set_xlabel("X forward (m)")
        ax3d.set_ylabel("Y lateral (m)")
        ax3d.set_zlabel("Z height (m)")
        ax3d.set_title("3D sampled reset cloud", loc="left", weight="bold")
        _set_equal_3d(ax3d, combined, np)
        ax3d.legend(loc="upper left", frameon=False)

        projections = (
            (ax_xy, 0, 1, "Top view: X–Y", "X forward (m)", "Y lateral (m)"),
            (ax_xz, 0, 2, "Side view: X–Z", "X forward (m)", "Z height (m)"),
            (ax_yz, 1, 2, "Front view: Y–Z", "Y lateral (m)", "Z height (m)"),
        )
        for axis, horizontal, vertical, title, xlabel, ylabel in projections:
            axis.scatter(
                palm_np[:, horizontal], palm_np[:, vertical],
                s=8, c=blue, alpha=0.28, marker="o", linewidths=0,
            )
            axis.scatter(
                fruit_np[:, horizontal], fruit_np[:, vertical],
                s=10, c=orange, alpha=0.38, marker="x", linewidths=0.6,
            )
            axis.set_title(title, loc="left", weight="bold")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(True, color="#E5E7EB", linewidth=0.7)
            projected = combined[:, [horizontal, vertical]]
            lower = np.min(projected, axis=0)
            upper = np.max(projected, axis=0)
            padding = np.maximum(0.04, 0.08 * (upper - lower))
            axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            axis.set_aspect("equal", adjustable="box")

        figure.suptitle(
            f"Physics-v1 reachable reset workspace v{cfg_module.RESET_WORKSPACE.get('version', 1)}",
            x=0.06, y=0.985, ha="left", fontsize=17, weight="bold", color="#111827",
        )
        figure.text(
            0.06,
            0.952,
            f"{args.num_envs:,} actual USD forward-kinematics samples · seed {args.seed} · "
            "blue circles are palms, orange crosses are corresponding fruits",
            ha="left", va="top", fontsize=10.5, color="#4B5563",
        )
        figure.savefig(png_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        print("TOMATO_PHYSICS_V1_RESET_CLOUD=" + json.dumps(summary, sort_keys=True))
        print(f"PNG={png_path}")
        print(f"CSV={csv_path}")
        print(f"JSON={json_path}")
        return 0
    except BaseException as error:
        print(
            "RESET_WORKSPACE_DIAGNOSTIC_FAILED="
            + json.dumps(
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    finally:
        if raw_env is not None:
            raw_env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
