"""Run the non-training Isaac gate for the independent Physics-v1 task."""

from __future__ import annotations

import argparse
import importlib
import json
import math
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
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-scale", type=float, default=0.45)
    parser.add_argument("--disable-mimics", action="store_true")
    parser.add_argument("--disable-self-collision", action="store_true")
    parser.add_argument("--disable-tomato-collision", action="store_true")
    parser.add_argument("--grasp-sweep", action="store_true")
    parser.add_argument("--collision-stress", action="store_true")
    parser.add_argument("--disable-contact-detail", action="store_true")
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 4:
        raise ValueError("--num-envs must be positive and --steps must be at least 4")
    if args.grasp_sweep and args.collision_stress:
        raise ValueError("select at most one scripted action mode")

    launcher = AppLauncher(args)
    app = launcher.app
    raw_env = None
    env = None
    try:
        gym = importlib.import_module("gymnasium")
        torch = importlib.import_module("torch")
        task = importlib.import_module("simulation.tomato_pick.isaac.physics_v1")
        cfg_module = importlib.import_module(
            "simulation.tomato_pick.isaac.physics_v1.env_cfg"
        )
        tomato_constants = importlib.import_module(
            "simulation.tomato_pick.isaac.env"
        )
        physx = importlib.import_module("omni.physx")
        PhysicsSchemaTools = importlib.import_module("pxr.PhysicsSchemaTools")
        Sdf = importlib.import_module("pxr.Sdf")
        Usd = importlib.import_module("pxr.Usd")
        UsdPhysics = importlib.import_module("pxr.UsdPhysics")

        hand_self_contact_pairs: dict[tuple[str, str], dict[str, float | int]] = {}

        def accumulate_hand_self_contacts(contact_headers, contact_data) -> None:
            for header in contact_headers:
                collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
                collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
                if "/Robot/" not in collider0 or "/Robot/" not in collider1:
                    continue
                relative0 = collider0.split("/Robot/", 1)[-1]
                relative1 = collider1.split("/Robot/", 1)[-1]
                pair = tuple(sorted((relative0, relative1)))
                record = hand_self_contact_pairs.setdefault(
                    pair,
                    {
                        "report_count": 0,
                        "contact_point_count": 0,
                        "minimum_separation_m": math.inf,
                        "maximum_impulse_n_s": 0.0,
                    },
                )
                record["report_count"] = int(record["report_count"]) + 1
                start = header.contact_data_offset
                stop = start + header.num_contact_data
                record["contact_point_count"] = int(
                    record["contact_point_count"]
                ) + int(header.num_contact_data)
                for contact_index in range(start, stop):
                    contact = contact_data[contact_index]
                    impulse = contact.impulse
                    impulse_norm = math.sqrt(sum(float(value) ** 2 for value in impulse))
                    record["minimum_separation_m"] = min(
                        float(record["minimum_separation_m"]),
                        float(contact.separation),
                    )
                    record["maximum_impulse_n_s"] = max(
                        float(record["maximum_impulse_n_s"]), impulse_norm
                    )

        physx_simulation = physx.get_physx_simulation_interface()
        contact_subscription = physx_simulation.subscribe_contact_report_events(
            accumulate_hand_self_contacts
        )

        cfg = cfg_module.TomatoPickPhysicsV1EnvCfg()
        cfg.detailed_step_diagnostics = True
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        if args.disable_mimics:
            cfg.events.hand_mimics = None
        if args.disable_self_collision:
            cfg.robot.spawn.articulation_props.enabled_self_collisions = False
            cfg.events.hand_self_collision_filters = None
        if args.disable_tomato_collision:
            cfg.tomato.spawn.collision_props.collision_enabled = False
        if args.disable_contact_detail:
            for sensor in cfg.fingertip_contacts.values():
                sensor.track_contact_points = False
                sensor.track_friction_forces = False
        raw_env = gym.make(task.TASK_ID, cfg=cfg)
        env = raw_env.unwrapped
        obs, _ = raw_env.reset(seed=args.seed)
        accumulate_hand_self_contacts(*physx_simulation.get_contact_report())
        robot_path = "/World/envs/env_0/Robot"
        adjacent_body_pairs = set()
        runtime_robot_overlap_pairs = set()
        query_robot_overlaps = None
        if args.num_envs == 1:
            stage = env.sim.stage
            root = stage.GetPrimAtPath(robot_path)
            for prim in Usd.PrimRange(root):
                if not prim.IsA(UsdPhysics.Joint):
                    continue
                joint = UsdPhysics.Joint(prim)
                body0 = joint.GetBody0Rel().GetTargets()
                body1 = joint.GetBody1Rel().GetTargets()
                if len(body0) == len(body1) == 1:
                    adjacent_body_pairs.add(
                        frozenset((str(body0[0]), str(body1[0])))
                    )

            collider_body: dict[str, str] = {}
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                ancestor = prim
                while ancestor and ancestor.IsValid() and str(ancestor.GetPath()) != "/":
                    if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                        collider_body[str(prim.GetPath())] = str(ancestor.GetPath())
                        break
                    ancestor = ancestor.GetParent()

            query_interface = physx.get_physx_scene_query_interface()

            def query_robot_overlaps():
                overlap_pairs = set()
                for collider_path, query_body_path in collider_body.items():
                    hits = []

                    def report_overlap(hit) -> bool:
                        hits.append(str(hit.rigid_body))
                        return True

                    encoded_path = PhysicsSchemaTools.encodeSdfPath(
                        Sdf.Path(collider_path)
                    )
                    query_interface.overlap_shape(
                        encoded_path[0], encoded_path[1], report_overlap, False
                    )
                    for hit_body_path in hits:
                        if (
                            hit_body_path == query_body_path
                            or not hit_body_path.startswith(robot_path)
                        ):
                            continue
                        overlap_pairs.add(
                            frozenset((query_body_path, hit_body_path))
                        )
                return overlap_pairs

            runtime_robot_overlap_pairs.update(query_robot_overlaps())
        all_env_ids = torch.arange(args.num_envs, device=env.device)
        initial_hand_fruit_clearance_m = env._hand_collision_clearance_for(
            env._tomato.data.root_pos_w.torch,
            env._fruit_axes_m,
            all_env_ids,
        )
        reset_joint_radius = torch.tensor(
            cfg.reset_arm_joint_ellipsoid_radius_rad, device=env.device
        )
        reset_normalized_joint_radius = torch.linalg.vector_norm(
            env._reset_workspace_base_arm_joint_offset_rad / reset_joint_radius,
            dim=1,
        )
        reset_mode_coefficient = env._reset_workspace_mode_coefficients
        reset_mode_ranges = torch.tensor(
            cfg.reset_arm_correlated_mode_coefficient_ranges,
            device=env.device,
        )
        reset_arm_position = env._nominal_joint_targets[:, env._arm_joint_ids] + env._reset_workspace_arm_joint_offset_rad
        reset_limits = env._robot.data.joint_pos_limits.torch[
            all_env_ids[:, None], env._arm_joint_ids
        ]
        reset_margin = cfg.reset_joint_limit_margin_fraction * (
            reset_limits[..., 1] - reset_limits[..., 0]
        )
        reset_lower = reset_limits[..., 0] + reset_margin
        reset_upper = reset_limits[..., 1] - reset_margin
        initial_reset_contact_force_n = env._physical_finger_force()
        closed_grasp_center = torch.tensor(
            cfg.closed_grasp_fruit_center_m, device=env.device
        )
        reset_closure_translation = (
            env._reset_workspace_fruit_local_m - closed_grasp_center
        )
        reset_closure_distance = torch.linalg.vector_norm(
            reset_closure_translation, dim=1
        )
        required_translation_headroom = (
            cfg.minimum_closure_translation_headroom_fraction
            * torch.tensor(cfg.max_residual_translation_m, device=env.device)
        )

        checks = {
            "formal_training_is_ready_but_not_started": (
                task.IMPLEMENTATION_STATUS.endswith("ready_not_started")
                and cfg_module.PROFILE["training_gate"]["ready"] is True
                and cfg_module.PROFILE["training_gate"][
                    "formal_training_started"
                ] is False
            ),
            "actor_shape": tuple(obs["policy"].shape) == (args.num_envs, 66),
            "critic_shape": tuple(obs["critic"].shape) == (args.num_envs, 83),
            "action_shape": env.single_action_space.shape == (11,),
            "finite_reset": bool(
                torch.isfinite(obs["policy"]).all()
                and torch.isfinite(obs["critic"]).all()
            ),
            "reset_workspace_is_continuous_and_bounded": bool(
                torch.isfinite(env._reset_workspace_arm_joint_offset_rad).all()
                and (reset_normalized_joint_radius <= 1.0 + 1.0e-5).all()
                and (
                    reset_mode_coefficient
                    >= reset_mode_ranges[:, 0] - 1.0e-6
                ).all()
                and (
                    reset_mode_coefficient
                    <= reset_mode_ranges[:, 1] + 1.0e-6
                ).all()
                and (reset_arm_position >= reset_lower).all()
                and (reset_arm_position <= reset_upper).all()
                and (env._reset_workspace_arm_joint_offset_rad.abs().sum(dim=1) > 0.0).all()
            ),
            "reset_workspace_clearance_is_valid": bool(
                (
                    env._reset_workspace_clearance_m
                    >= cfg.minimum_initial_clearance_m
                ).all()
            ),
            "reset_workspace_fruit_tilt_is_bounded": bool(
                (env._reset_workspace_fruit_tilt_rad >= 0.0).all()
                and (
                    env._reset_workspace_fruit_tilt_rad
                    <= cfg.reset_fruit_stem_tilt_rad_max + 1.0e-6
                ).all()
            ),
            "reset_workspace_has_no_initial_contact": bool(
                (initial_reset_contact_force_n <= 1.0e-4).all()
            ),
            "reset_workspace_preserves_closure_margin": bool(
                (
                    reset_closure_distance
                    <= cfg.maximum_closure_translation_norm_m + 1.0e-6
                ).all()
                and (
                    env._reset_workspace_translation_headroom_m
                    >= required_translation_headroom - 1.0e-6
                ).all()
            ),
            "independent_actuator_groups": set(cfg.robot.actuators)
            == {
                "arms",
                "g20_active_right",
                "g20_passive_followers_right",
                "g20_park_left",
            },
            "passive_followers_have_no_position_drive": (
                cfg.robot.actuators["g20_passive_followers_right"].stiffness == 0.0
                and cfg.robot.actuators["g20_passive_followers_right"].damping == 0.0
            ),
            "native_compliant_mimics_present": (
                getattr(env, "_physics_v1_authored_mimic_count", 0)
                == (0 if args.disable_mimics else 5 * args.num_envs)
                and (
                    args.disable_mimics
                    or env._physics_v1_authored_mimic_side == "right"
                )
                and (
                    args.disable_mimics
                    or (
                        env._physics_v1_mimics_baked is True
                        and env._physics_v1_runtime_authored_mimic_count == 0
                    )
                )
                and env._robot.num_fixed_tendons == 0
            ),
            "articulation_self_collision_is_enabled": (
                cfg.robot.spawn.articulation_props.enabled_self_collisions
                is (not args.disable_self_collision)
            ),
            "adjacent_hand_collision_filter_is_exact": (
                not hasattr(env, "_physics_v1_hand_self_collision_filter")
                if args.disable_self_collision
                else (
                    env._physics_v1_hand_self_collision_filter["enabled"] is True
                    and env._physics_v1_hand_self_collision_filter[
                        "policy"
                    ] == "right_distal_envelope_and_palm_only"
                    and env._physics_v1_hand_self_collision_filter[
                        "pairs_per_robot"
                    ] == 2426
                    and env._physics_v1_hand_self_collision_filter[
                        "asset_baked"
                    ] is True
                    and env._physics_v1_hand_self_collision_filter[
                        "runtime_authored_pair_count"
                    ] == 0
                    and env._physics_v1_hand_self_collision_filter[
                        "baked_pair_count"
                    ] == 2426
                    and env._physics_v1_hand_self_collision_filter[
                        "verified_robot_instance_count"
                    ] == args.num_envs
                    and env._physics_v1_hand_self_collision_filter[
                        "adjacent_joint_pair_count"
                    ] == 54
                    and env._physics_v1_hand_self_collision_filter[
                        "allowed_nonadjacent_hand_pair_count"
                    ] == 59
                    and env._physics_v1_hand_self_collision_filter[
                        "contact_reporter_body_count"
                    ] >= 28
                    and env._physics_v1_hand_self_collision_filter[
                        "unique_relative_pair_count"
                    ] == 2426
                    and env._physics_v1_hand_self_collision_filter[
                        "external_contact_body_count"
                    ] == 12
                    and env._physics_v1_hand_self_collision_filter[
                        "self_collision_envelope_body_count"
                    ] == 12
                    and env._physics_v1_hand_self_collision_filter[
                        "nonadjacent_hand_pairs_enabled"
                    ] is True
                )
            ),
            "finger_material_reaches_collision_shapes": (
                env._physics_v1_hand_material_binding["bound_body_count"] > 0
                and env._physics_v1_hand_material_binding[
                    "inherited_collider_count"
                ]
                > 0
                and env._physics_v1_hand_material_binding["asset_baked"] is True
                and env._physics_v1_hand_material_binding[
                    "runtime_authored_binding_count"
                ]
                == 0
            ),
            "correlated_fruit_material_bank_is_bounded": (
                1
                <= env._physics_v1_fruit_material_binding["material_count"]
                <= 16
                and env._physics_v1_fruit_material_binding["bound_fruit_count"]
                == args.num_envs
                and env._physics_v1_fruit_material_binding[
                    "inherited_collider_count"
                ]
                >= args.num_envs
            ),
            "contact_telemetry_capacity_is_explicit": all(
                sensor.max_contact_data_count_per_prim == 64
                and sensor.track_contact_points
                and not sensor.track_friction_forces
                and not sensor.track_pose
                for sensor in cfg.fingertip_contacts.values()
            ),
            "finite_rollout": True,
            "hand_target_rate_limited": True,
            "effective_hand_velocity_respects_asset_and_linkage": bool(
                (
                    env._effective_hand_velocity_rad_s
                    <= torch.tensor(
                        cfg.hand_joint_max_velocity_rad_s, device=env.device
                    )
                    + 1.0e-6
                ).all()
                and (
                    env._effective_hand_velocity_rad_s
                    <= env._robot.data.joint_vel_limits.torch[
                        0, env._hand_joint_ids
                    ]
                    + 1.0e-6
                ).all()
                and all(
                    float(
                        env._effective_hand_velocity_rad_s[
                            env._hand_joint_ids.index(
                                env._robot.joint_names.index(driver_name)
                            )
                        ]
                    )
                    <= referred_limit + 1.0e-6
                    for driver_name, referred_limit in (
                        env._linkage_driver_velocity_caps_rad_s.items()
                    )
                )
            ),
            "active_hand_velocity_within_envelope": True,
            "passive_linkage_stays_bounded": True,
            "termination_masks_disjoint": True,
            "one_substep_cannot_break_stem": False,
            "sustained_multiaxial_load_can_break_stem": False,
        }
        max_contact_force_n = 0.0
        max_joint_velocity_rad_s = 0.0
        max_hand_target_step_rad = 0.0
        max_linkage_residual_rad = 0.0
        linkage_residual_by_joint: dict[str, float] = {}
        linkage_peak_detail_by_joint: dict[str, dict[str, float | int]] = {}
        maximum_active_hand_velocity_rad_s = 0.0
        maximum_pre_reset_fingertip_force_n = 0.0
        minimum_contact_radius_m = float("inf")
        maximum_contact_radius_m = 0.0
        maximum_invalid_contact_force_n = 0.0
        maximum_hand_target_error_rad = 0.0
        per_joint_active_velocity_rad_s = torch.zeros(
            len(env._hand_joint_ids), device=env.device
        )
        terminated_count = 0
        truncated_count = 0
        done_reason_counts = {
            "success": 0,
            "drop": 0,
            "excess_force": 0,
            "out_of_bounds": 0,
            "timeout": 0,
        }

        previous_targets = env._joint_targets[:, env._hand_joint_ids].clone()
        generator = torch.Generator(device=env.device).manual_seed(args.seed + 1)
        grasp_trajectory = None
        if args.grasp_sweep:
            grasp_trajectory = importlib.import_module(
                "script.render_tomato_hand_5d_manifold_v2"
            ).trajectory()
        for step_index in range(args.steps):
            if grasp_trajectory is None:
                if args.collision_stress:
                    action = torch.zeros((args.num_envs, 11), device=env.device)
                    action[:, 6:11] = torch.tensor(
                        (1.0, 0.0, 0.0, 0.0, 1.0), device=env.device
                    )
                else:
                    action = args.action_scale * (
                        2.0
                        * torch.rand(
                            (args.num_envs, 11),
                            generator=generator,
                            device=env.device,
                        )
                        - 1.0
                    )
            else:
                frame_index = min(
                    len(grasp_trajectory) - 1,
                    step_index * len(grasp_trajectory) // args.steps,
                )
                action = torch.zeros((args.num_envs, 11), device=env.device)
                action[:, 6:11] = torch.tensor(
                    grasp_trajectory[frame_index], device=env.device
                )
            obs, reward, terminated, truncated, info = raw_env.step(action)
            accumulate_hand_self_contacts(*physx_simulation.get_contact_report())
            if query_robot_overlaps is not None:
                runtime_robot_overlap_pairs.update(query_robot_overlaps())
            continuing = ~(terminated | truncated)
            terminated_count += int(terminated.sum())
            truncated_count += int(truncated.sum())
            step_info = info.get("step", {})
            for reason in done_reason_counts:
                done_reason_counts[reason] += int(
                    step_info.get(f"{reason}_count", 0)
                )
            maximum_pre_reset_fingertip_force_n = max(
                maximum_pre_reset_fingertip_force_n,
                float(step_info.get("maximum_fingertip_force_n", 0.0)),
            )
            maximum_invalid_contact_force_n = max(
                maximum_invalid_contact_force_n,
                float(step_info.get("maximum_invalid_contact_force_n", 0.0)),
            )
            if "minimum_contact_radius_m" in step_info:
                minimum_contact_radius_m = min(
                    minimum_contact_radius_m,
                    float(step_info["minimum_contact_radius_m"]),
                )
                maximum_contact_radius_m = max(
                    maximum_contact_radius_m,
                    float(step_info["maximum_contact_radius_m"]),
                )
            current_targets = env._joint_targets[:, env._hand_joint_ids]
            hand_target_error = (
                env._robot.data.joint_pos.torch[:, env._hand_joint_ids]
                - current_targets
            ).abs()
            if bool(continuing.any()):
                maximum_hand_target_error_rad = max(
                    maximum_hand_target_error_rad,
                    float(hand_target_error[continuing].max()),
                )
            target_delta = (current_targets - previous_targets).abs()
            limit = (
                env._effective_hand_velocity_rad_s
                * float(cfg.sim.dt)
                * float(cfg.decimation)
            )
            if bool(continuing.any()):
                continuing_delta = target_delta[continuing]
                checks["hand_target_rate_limited"] &= bool(
                    (continuing_delta <= limit.unsqueeze(0) + 1.0e-6).all()
                )
                max_hand_target_step_rad = max(
                    max_hand_target_step_rad, float(continuing_delta.max())
                )
            previous_targets = current_targets.clone()
            max_contact_force_n = max(
                max_contact_force_n, float(env._last_physical_force_n.max())
            )
            max_joint_velocity_rad_s = max(
                max_joint_velocity_rad_s,
                float(env._robot.data.joint_vel.torch.abs().max()),
            )
            if bool(continuing.any()):
                active_velocity = env._robot.data.joint_vel.torch[
                    continuing
                ][:, env._hand_joint_ids].abs()
                maximum_active_hand_velocity_rad_s = max(
                    maximum_active_hand_velocity_rad_s,
                    float(active_velocity.max()),
                )
                per_joint_active_velocity_rad_s = torch.maximum(
                    per_joint_active_velocity_rad_s,
                    active_velocity.amax(dim=0),
                )
                checks["active_hand_velocity_within_envelope"] &= bool(
                    (
                        per_joint_active_velocity_rad_s
                        <= 1.25 * env._effective_hand_velocity_rad_s + 0.05
                    ).all()
                )
            for follower_name in tomato_constants.RIGHT_HAND_MIMIC_RAD:
                side = (
                    "right_hand_"
                    if follower_name.startswith("right_hand_")
                    else "left_hand_"
                )
                right_name = follower_name.replace("left_hand_", "right_hand_", 1)
                driver_right, multiplier, _legacy_offset = (
                    tomato_constants.RIGHT_HAND_MIMIC_RAD[right_name]
                )
                driver_name = driver_right.replace("right_hand_", side, 1)
                follower_id = env._robot.joint_names.index(follower_name)
                driver_id = env._robot.joint_names.index(driver_name)
                residual = (
                    env._robot.data.joint_pos.torch[:, follower_id]
                    - multiplier * env._robot.data.joint_pos.torch[:, driver_id]
                    - cfg.passive_mimic_offset_rad
                )
                if bool(continuing.any()):
                    continuing_residual = residual[continuing]
                    continuing_env_ids = torch.nonzero(
                        continuing, as_tuple=False
                    ).squeeze(1)
                    peak_local_index = int(
                        continuing_residual.abs().argmax().item()
                    )
                    peak_env_id = int(
                        continuing_env_ids[peak_local_index].item()
                    )
                    peak_abs_residual = float(
                        continuing_residual[peak_local_index].abs().item()
                    )
                    if peak_abs_residual > linkage_residual_by_joint.get(
                        follower_name, 0.0
                    ):
                        linkage_peak_detail_by_joint[follower_name] = {
                            "step": step_index,
                            "env_id": peak_env_id,
                            "residual_rad": float(residual[peak_env_id].item()),
                            "driver_position_rad": float(
                                env._robot.data.joint_pos.torch[
                                    peak_env_id, driver_id
                                ].item()
                            ),
                            "follower_position_rad": float(
                                env._robot.data.joint_pos.torch[
                                    peak_env_id, follower_id
                                ].item()
                            ),
                            "driver_velocity_rad_s": float(
                                env._robot.data.joint_vel.torch[
                                    peak_env_id, driver_id
                                ].item()
                            ),
                            "follower_velocity_rad_s": float(
                                env._robot.data.joint_vel.torch[
                                    peak_env_id, follower_id
                                ].item()
                            ),
                        }
                    max_linkage_residual_rad = max(
                        max_linkage_residual_rad,
                        peak_abs_residual,
                    )
                    linkage_residual_by_joint[follower_name] = max(
                        linkage_residual_by_joint.get(follower_name, 0.0),
                        peak_abs_residual,
                    )
                    checks["passive_linkage_stays_bounded"] &= bool(
                        (
                            continuing_residual.abs()
                            <= cfg.passive_mimic_max_tracking_error_rad
                        ).all()
                    )
            checks["finite_rollout"] &= bool(
                torch.isfinite(obs["policy"]).all()
                and torch.isfinite(obs["critic"]).all()
                and torch.isfinite(reward).all()
                and torch.isfinite(env._robot.data.joint_pos.torch).all()
                and torch.isfinite(env._robot.data.joint_vel.torch).all()
                and torch.isfinite(env._stem_damage).all()
            )
            checks["termination_masks_disjoint"] &= not bool(
                (terminated & truncated).any()
            )

        # Exercise the runtime load-duration path without training or a
        # scripted policy.  Solve a static displacement that produces a unit
        # tensile load ratio after gravity compensation.  The old 1-metre
        # displacement saturated the 120 N safety clamp and could legitimately
        # integrate more than one unit of damage in one substep; it did not
        # test the intended duration prior.  At unit load the first 120 Hz
        # substep must survive and the held load must eventually detach.
        env_id = torch.tensor([0], device=env.device)
        pose = env._tomato.data.root_pose_w.torch[env_id].clone()
        pull_axis = env._grasp_basis_w[env_id, :, 2]
        omega_squared = float(cfg.stem_linear_natural_frequency_rad_s) ** 2
        unit_tension_displacement = (
            env._stem_tensile_limit_n[env_id].unsqueeze(1)
            / (env._fruit_mass_kg[env_id].unsqueeze(1) * omega_squared)
        ) * pull_axis - env._gravity_w.unsqueeze(0) / omega_squared
        pose[:, :3] += unit_tension_displacement
        env._tomato.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_id)
        env._tomato.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=env.device), env_ids=env_id
        )
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
        env._stem_break_event[env_id] = False
        env._stem_detached[env_id] = False
        env._stem_damage[env_id] = 0.0
        env._apply_attachment_wrench()
        first_load_ratio = float(env._stem_combined_load_ratio[env_id].item())
        checks["one_substep_cannot_break_stem"] = not bool(
            env._stem_detached[env_id].item()
        )
        first_damage = float(env._stem_damage[env_id].item())
        for _ in range(24):
            env._tomato.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=env.device), env_ids=env_id
            )
            env._apply_attachment_wrench()
            if bool(env._stem_detached[env_id].item()):
                break
        checks["sustained_multiaxial_load_can_break_stem"] = bool(
            env._stem_detached[env_id].item()
        )

        report = {
            "schema_version": 1,
            "passed": all(checks.values()),
            "task_id": task.TASK_ID,
            "implementation_status": task.IMPLEMENTATION_STATUS,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "action_scale": args.action_scale,
            "action_mode": (
                "grasp_sweep"
                if args.grasp_sweep
                else "collision_stress"
                if args.collision_stress
                else "random"
            ),
            "mimics_disabled_for_ablation": args.disable_mimics,
            "self_collision_disabled_for_ablation": args.disable_self_collision,
            "tomato_collision_disabled_for_ablation": args.disable_tomato_collision,
            "contact_detail_disabled_for_ablation": args.disable_contact_detail,
            "checks": checks,
            "fruit_mass_kg": {
                "minimum": float(env._fruit_mass_kg.min()),
                "maximum": float(env._fruit_mass_kg.max()),
            },
            "initial_hand_fruit_clearance_m": {
                "minimum": float(initial_hand_fruit_clearance_m.min()),
                "maximum": float(initial_hand_fruit_clearance_m.max()),
            },
            "reset_workspace": {
                "maximum_normalized_joint_radius": float(
                    reset_normalized_joint_radius.max()
                ),
                "minimum_clearance_m": float(
                    env._reset_workspace_clearance_m.min()
                ),
                "maximum_clearance_m": float(
                    env._reset_workspace_clearance_m.max()
                ),
                "maximum_fruit_tilt_deg": float(
                    torch.rad2deg(env._reset_workspace_fruit_tilt_rad).max()
                ),
                "maximum_initial_contact_force_n": float(
                    initial_reset_contact_force_n.max()
                ),
                "maximum_closure_translation_norm_m": float(
                    reset_closure_distance.max()
                ),
                "minimum_closure_translation_headroom_m": (
                    env._reset_workspace_translation_headroom_m.amin(dim=0).tolist()
                ),
            },
            "stem_tensile_limit_n": {
                "minimum": float(env._stem_tensile_limit_n.min()),
                "maximum": float(env._stem_tensile_limit_n.max()),
            },
            "first_counterfactual_damage": first_damage,
            "first_counterfactual_load_ratio": first_load_ratio,
            "maximum_contact_force_n": max_contact_force_n,
            "maximum_pre_reset_fingertip_force_n": (
                maximum_pre_reset_fingertip_force_n
            ),
            "contact_radius_m": {
                "minimum": (
                    minimum_contact_radius_m
                    if minimum_contact_radius_m != float("inf")
                    else None
                ),
                "maximum": maximum_contact_radius_m or None,
            },
            "maximum_invalid_contact_force_n": maximum_invalid_contact_force_n,
            "per_joint_active_velocity_rad_s": {
                name: float(value)
                for name, value in zip(
                    env._hand_joint_names, per_joint_active_velocity_rad_s
                )
            },
            "per_joint_effective_velocity_limit_rad_s": {
                name: float(value)
                for name, value in zip(
                    env._hand_joint_names, env._effective_hand_velocity_rad_s
                )
            },
            "linkage_driver_velocity_caps_rad_s": (
                env._linkage_driver_velocity_caps_rad_s
            ),
            "maximum_joint_velocity_rad_s": max_joint_velocity_rad_s,
            "maximum_active_hand_velocity_rad_s": maximum_active_hand_velocity_rad_s,
            "maximum_hand_target_step_rad": max_hand_target_step_rad,
            "maximum_hand_target_error_rad": maximum_hand_target_error_rad,
            "final_hand_target_error_rad": {
                name: float(value)
                for name, value in zip(
                    env._hand_joint_names,
                    (
                        env._robot.data.joint_pos.torch[0, env._hand_joint_ids]
                        - env._joint_targets[0, env._hand_joint_ids]
                    ).abs(),
                )
            },
            "maximum_linkage_residual_rad": max_linkage_residual_rad,
            "linkage_residual_by_joint_rad": linkage_residual_by_joint,
            "linkage_peak_detail_by_joint": linkage_peak_detail_by_joint,
            "final_linkage_residual_by_joint_rad": {
                follower_name: float(
                    (
                        env._robot.data.joint_pos.torch[
                            0, env._robot.joint_names.index(follower_name)
                        ]
                        - multiplier
                        * env._robot.data.joint_pos.torch[
                            0, env._robot.joint_names.index(driver_name)
                        ]
                        - cfg.passive_mimic_offset_rad
                    ).abs()
                )
                for follower_name, (
                    driver_name,
                    multiplier,
                    _legacy_offset,
                ) in tomato_constants.RIGHT_HAND_MIMIC_RAD.items()
            },
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "done_reason_counts": done_reason_counts,
            "hand_material_binding": env._physics_v1_hand_material_binding,
            "hand_self_collision_filter": (
                getattr(env, "_physics_v1_hand_self_collision_filter", None)
            ),
            "hand_self_contact_pairs": [
                {
                    "body0": pair[0],
                    "body1": pair[1],
                    **{
                        name: (None if value == math.inf else value)
                        for name, value in values.items()
                    },
                }
                for pair, values in sorted(
                    hand_self_contact_pairs.items(),
                    key=lambda item: float(item[1]["maximum_impulse_n_s"]),
                    reverse=True,
                )[:64]
            ],
            "runtime_robot_overlap_pairs": [
                {
                    "body0": body0.removeprefix(f"{robot_path}/"),
                    "body1": body1.removeprefix(f"{robot_path}/"),
                    "direct_joint_neighbors": pair in adjacent_body_pairs,
                }
                for pair in sorted(
                    runtime_robot_overlap_pairs,
                    key=lambda value: sorted(value),
                )
                for body0, body1 in [sorted(pair)]
            ],
            "fruit_material_binding": env._physics_v1_fruit_material_binding,
            "contact_material_bins": {
                "minimum": int(
                    env._prestartup_contact_material_profiles["material_bin"].min()
                ),
                "maximum": int(
                    env._prestartup_contact_material_profiles["material_bin"].max()
                ),
                "unique": int(
                    torch.unique(
                        env._prestartup_contact_material_profiles["material_bin"]
                    ).numel()
                ),
            },
            "training_started": False,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print("TOMATO_PHYSICS_V1_GATE=" + json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["passed"] else 2
    except BaseException as exc:
        diagnostics = {}
        if env is not None:
            try:
                torch = importlib.import_module("torch")
                tensors = {
                    "joint_pos": env._robot.data.joint_pos.torch,
                    "joint_vel": env._robot.data.joint_vel.torch,
                    "fruit_pose": env._tomato.data.root_pose_w.torch,
                    "fruit_velocity": torch.cat(
                        (
                            env._tomato.data.root_lin_vel_w.torch,
                            env._tomato.data.root_ang_vel_w.torch,
                        ),
                        dim=1,
                    ),
                    "finger_force": env._last_physical_force_n,
                    "stem_force": env._last_stem_force_w,
                    "stem_torque": env._last_stem_torque_w,
                }
                for name, value in tensors.items():
                    finite = torch.isfinite(value)
                    diagnostics[name] = {
                        "all_finite": bool(finite.all()),
                        "finite_max_abs": (
                            float(value[finite].abs().max()) if bool(finite.any()) else None
                        ),
                        "nonfinite_count": int((~finite).sum()),
                    }
            except BaseException as diagnostic_exc:
                diagnostics = {"diagnostic_error": str(diagnostic_exc)}
        print(
            "TOMATO_PHYSICS_V1_GATE_EXCEPTION="
            + json.dumps(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "diagnostics": diagnostics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        traceback.print_exc()
        return 3
    finally:
        if "contact_subscription" in locals():
            contact_subscription = None
        if raw_env is not None:
            raw_env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
