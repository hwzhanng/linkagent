"""Run a no-training Isaac smoke gate for Physics-v2 public-prior mechanics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "logs/qualification/tomato_physics_v2/isaac_smoke_gate.json"
)

from script.physics_v2_audit_inputs import (  # noqa: E402
    AUDITED_RUNTIME_INPUTS,
    input_sha256,
)


def _all_finite(value: Any, torch) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item, torch) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item, torch) for item in value)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 16:
        raise ValueError("--num-envs must be positive and --steps at least 16")

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
        physics_module = importlib.import_module(
            "simulation.tomato_pick.physics_v2"
        )
        math_utils = importlib.import_module("isaaclab.utils.math")
        profile = physics_module.load_physics_v2_profile()
        engineering = profile["engineering_prior_data"]
        cfg = cfg_module.TomatoPickPhysicsV2EnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.detailed_step_diagnostics = True
        raw_env = gym.make(task.TASK_ID, cfg=cfg)
        env = raw_env.unwrapped
        observation, _ = raw_env.reset(seed=args.seed)
        finite_observation = _all_finite(observation, torch)
        finite_reward = True
        termination_disjoint = True
        maximum_stem_force_n = 0.0
        maximum_stem_torque_nm = 0.0
        maximum_joint_velocity_rad_s = 0.0
        minimum_integrity = 1.0
        maximum_cumulative_twist_rad = 0.0
        maximum_effective_twist_rad = 0.0
        action_dim = int(raw_env.action_space.shape[-1])
        for step in range(args.steps):
            phase = 2.0 * torch.pi * float(step) / float(args.steps)
            actions = torch.zeros(
                (args.num_envs, action_dim), device=env.device
            )
            # Small bounded rotational and hand motion exercises the new
            # torsional state without turning this smoke gate into training.
            actions[:, 5] = 0.20 * torch.sin(torch.tensor(phase, device=env.device))
            actions[:, 6:] = 0.25 * torch.sin(
                torch.tensor(phase + 0.5, device=env.device)
            )
            observation, reward, terminated, truncated, _info = raw_env.step(
                actions
            )
            finite_observation &= _all_finite(observation, torch)
            finite_reward &= bool(torch.isfinite(reward).all())
            termination_disjoint &= not bool((terminated & truncated).any())
            maximum_stem_force_n = max(
                maximum_stem_force_n,
                float(torch.linalg.vector_norm(env._last_stem_force_w, dim=1).max()),
            )
            maximum_stem_torque_nm = max(
                maximum_stem_torque_nm,
                float(torch.linalg.vector_norm(env._last_stem_torque_w, dim=1).max()),
            )
            maximum_joint_velocity_rad_s = max(
                maximum_joint_velocity_rad_s,
                float(env._robot.data.joint_vel.torch.abs().max()),
            )
            minimum_integrity = min(
                minimum_integrity, float(env._stem_integrity.min())
            )
            maximum_cumulative_twist_rad = max(
                maximum_cumulative_twist_rad,
                float(env._stem_cumulative_twist_rad.max()),
            )
            maximum_effective_twist_rad = max(
                maximum_effective_twist_rad,
                float(env._stem_effective_twist_rad.max()),
            )

        profiles = env._prestartup_physics_v2_profiles
        contact_materials = env._prestartup_contact_material_profiles
        category_count = torch.bincount(
            env._task_orientation_category, minlength=4
        )
        sampled_delays = {
            int(value) for value in torch.unique(env._action_delay).tolist()
        }
        fruit_pose = env._tomato.data.root_pose_w.torch
        ee_pose = env._robot.data.body_pose_w.torch[:, env._ee_body_id]
        fingertip_center = env._robot.data.body_pos_w.torch[
            :, env._fingertip_contact_body_ids
        ].mean(dim=1)
        expected_tip_center_relative = (
            math_utils.quat_apply_inverse(
                ee_pose[:, 3:7],
                fruit_pose[:, :3]
                + env._perceived_fruit_position_error_w
                - fingertip_center,
            )
            / 0.15
        ).clamp(-1.0, 1.0)
        expected_position_error_w, expected_rotation_error_w = (
            math_utils.compute_pose_error(
                ee_pose[:, :3],
                ee_pose[:, 3:7],
                env._ee_command_pose_w[:, :3],
                env._ee_command_pose_w[:, 3:7],
            )
        )
        expected_controller_tracking_error = (
            torch.cat(
                (
                    torch.bmm(
                        env._grasp_basis_w.transpose(1, 2),
                        expected_position_error_w.unsqueeze(2),
                    ).squeeze(2),
                    torch.bmm(
                        env._grasp_basis_w.transpose(1, 2),
                        expected_rotation_error_w.unsqueeze(2),
                    ).squeeze(2),
                ),
                dim=1,
            )
            / torch.tensor(
                cfg.action_scales,
                device=env.device,
                dtype=expected_position_error_w.dtype,
            ).unsqueeze(0)
        ).clamp(-1.0, 1.0)
        material_bins = torch.unique(contact_materials["material_bin"])
        expected_opposition_target_n = torch.maximum(
            torch.full_like(
                env._fruit_mass_kg, cfg.grasp_full_contact_force_n
            ),
            cfg.load_bearing_opposition_safety_factor
            * env._fruit_mass_kg
            * 9.81
            / (
                2.0
                * contact_materials["effective_pair_static_friction"].to(
                    env.device
                )
            ),
        )
        checks = {
            "task_identity_is_physics_v2": task.TASK_ID
            == "LinkAgent-TomatoPick-Physics-v2-PublicPrior-v1",
            "finite_reset_and_rollout_observations": finite_observation,
            "finite_rollout_rewards": finite_reward,
            "termination_masks_disjoint": termination_disjoint,
            "morphology_density_gate_enforced": bool(
                (profiles["density_kg_m3"] >= 800.0).all()
                and (profiles["density_kg_m3"] <= 1200.0).all()
            ),
            "public_axial_strength_replaces_4_to_8N": float(
                env._stem_tensile_limit_n.mean()
            )
            > 30.0,
            "torsion_limit_is_finite_positive": bool(
                torch.isfinite(env._stem_torsion_limit_nm).all()
                and (env._stem_torsion_limit_nm > 0.0).all()
            ),
            "integrity_remains_finite_bounded": bool(
                torch.isfinite(env._stem_integrity).all()
                and (env._stem_integrity >= 0.0).all()
                and (env._stem_integrity <= 1.0).all()
            ),
            "all_runtime_physics_tensors_are_finite": all(
                bool(torch.isfinite(value).all())
                for value in (
                    env._robot.data.joint_pos.torch,
                    env._robot.data.joint_vel.torch,
                    env._tomato.data.root_pose_w.torch,
                    env._tomato.data.root_lin_vel_w.torch,
                    env._tomato.data.root_ang_vel_w.torch,
                    env._last_stem_force_w,
                    env._last_stem_torque_w,
                    env._stem_damage,
                )
            ),
            "runtime_stays_inside_explicit_force_torque_clamps": (
                maximum_stem_force_n <= cfg.maximum_stem_force_n + 1.0e-4
                and maximum_stem_torque_nm
                <= cfg.maximum_stem_torque_nm + 1.0e-4
            ),
            "world_orientation_category_assigned_per_env": int(
                category_count.sum()
            )
            == args.num_envs,
            "continuous_tilt_inside_15deg_engineering_support": bool(
                (profiles["orientation_polar_tilt_rad"] >= 0.0).all()
                and (
                    profiles["orientation_polar_tilt_rad"]
                    <= math.radians(15.0)
                ).all()
            ),
            "episode_fixed_fatigue_parameters_are_bounded": bool(
                (profiles["stem_subcritical_onset_ratio"] >= 0.45).all()
                and (profiles["stem_subcritical_onset_ratio"] <= 0.75).all()
                and (profiles["stem_subcritical_damage_exponent"] >= 1.5).all()
                and (profiles["stem_subcritical_damage_exponent"] <= 3.0).all()
                and (profiles["stem_critical_duration_s"] >= 0.20).all()
                and (profiles["stem_critical_duration_s"] <= 1.20).all()
            ),
            "engineering_pair_friction_material_bank_is_bounded": bool(
                (
                    contact_materials["effective_pair_static_friction"]
                    >= 0.30
                ).all()
                and (
                    contact_materials["effective_pair_static_friction"]
                    <= 0.70
                ).all()
                and (contact_materials["material_static_friction"] >= 0.0).all()
                and (contact_materials["material_dynamic_friction"] >= 0.0).all()
            ),
            "effective_contact_surrogate_is_bounded": bool(
                (contact_materials["material_stiffness_n_m"] >= 3000.0).all()
                and (
                    contact_materials["material_stiffness_n_m"] <= 12000.0
                ).all()
                and (contact_materials["material_damping_n_s_m"] >= 10.0).all()
                and (contact_materials["material_damping_n_s_m"] <= 60.0).all()
            ),
            "a7_l20_action_delay_surrogate_is_zero_to_two_steps": tuple(
                cfg.action_delay_steps
            )
            == (0, 2),
            "action_history_covers_maximum_sampled_delay": (
                env._action_history.shape[0] >= max(cfg.action_delay_steps) + 1
                and sampled_delays.issubset({0, 1, 2})
                and (args.num_envs < 64 or sampled_delays == {0, 1, 2})
            ),
            "vector_gate_exercises_multiple_material_profiles": (
                args.num_envs < 64 or int(material_bins.numel()) >= 4
            ),
            "load_bearing_opposition_target_matches_runtime_physics": bool(
                torch.allclose(
                    env._load_bearing_opposition_target_n,
                    expected_opposition_target_n,
                    rtol=1.0e-6,
                    atol=1.0e-6,
                )
                and torch.isfinite(
                    env._load_bearing_opposition_target_n
                ).all()
                and (env._load_bearing_opposition_target_n > 0.0).all()
                and (
                    env._load_bearing_opposition_target_n
                    <= cfg.maximum_fingertip_force_n
                ).all()
            ),
            "v2_actor_has_deployable_motion_tracking_and_command_history": (
                tuple(observation["policy"].shape)
                == (args.num_envs, cfg.observation_space)
                and cfg.observation_space == 103
                and cfg.state_space == 120
                and torch.allclose(
                    observation["policy"][:, 69:72],
                    expected_tip_center_relative,
                    rtol=1.0e-6,
                    atol=1.0e-6,
                )
                and torch.allclose(
                    observation["policy"][:, 75:81],
                    expected_controller_tracking_error,
                    rtol=1.0e-6,
                    atol=1.0e-6,
                )
                and torch.allclose(
                    observation["policy"][:, -22:],
                    env._action_history[:2]
                    .permute(1, 0, 2)
                    .reshape(args.num_envs, 22),
                )
                and profile["training_observation"][
                    "controller_tracking_error_exposed_to_actor"
                ]
                is True
                and profile["training_observation"][
                    "controller_tracking_error_history_steps"
                ]
                == 1
                and profile["training_observation"][
                    "privileged_velocity_exposed_to_actor"
                ]
                is False
                and profile["training_observation"][
                    "privileged_action_delay_exposed_to_actor"
                ]
                is False
                and profile["training_observation"][
                    "privileged_stem_wrench_exposed_to_actor"
                ]
                is False
                and profile["training_observation"][
                    "privileged_detachment_exposed_to_actor"
                ]
                is False
            ),
            "mode_aware_capacity_curriculum_is_training_only": (
                not cfg.discovery_capacity_curriculum_enabled
                and cfg.discovery_tensile_shear_initial_scale == 1.0
                and cfg.discovery_bending_torsion_initial_scale == 0.10
                and cfg.discovery_bending_torsion_requires_secure_support
            ),
            "secure_retain_predicate_is_not_weakened": (
                cfg.thumb_opposition_required
                and cfg.minimum_supported_fingertips >= 2
                and cfg.maximum_retained_distance_m <= 0.12
                and cfg.maximum_retained_relative_speed_m_s <= 0.25
                and cfg.success_hold_steps >= 20
            ),
            "paper_strategy_audit_is_diagnostic_and_nontrivial": (
                profile["paper_strategy_audit"][
                    "diagnostic_only_not_rewarded"
                ]
                and math.isclose(
                    cfg.paper_strategy_effective_twist_minimum_rad,
                    2.0 * math.pi,
                )
                and cfg.paper_strategy_bending_utilization_minimum == 0.50
                and cfg.paper_strategy_final_axial_pull_n == (1.0, 15.29)
                and bool(
                    (
                        env._stem_effective_twist_rad
                        <= env._stem_cumulative_twist_rad + 1.0e-6
                    ).all()
                )
            ),
            "engineering_prior_is_hash_pinned": cfg.engineering_prior_profile_id
            == "tomato_pick_physics_v2_engineering_prior_v1",
            "v1_profile_not_mutated_at_runtime": cfg.physics_profile_id
            == "tomato_pick_physics_v2_public_prior_v1",
        }
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "qualification": "physics_v2_public_prior_isaac_smoke",
            "task_id": task.TASK_ID,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "passed": all(checks.values()),
            "candidate_id": profile["training"]["first_success_gate"][
                "candidate_id"
            ],
            "input_sha256": input_sha256(
                PROJECT_ROOT,
                (*AUDITED_RUNTIME_INPUTS, "script/check_tomato_physics_v2.py"),
            ),
            "checks": checks,
            "profile_summary": {
                "fruit_mass_kg": {
                    "mean": float(profiles["mass_kg"].mean()),
                    "minimum": float(profiles["mass_kg"].min()),
                    "maximum": float(profiles["mass_kg"].max()),
                },
                "density_kg_m3": {
                    "mean": float(profiles["density_kg_m3"].mean()),
                    "minimum": float(profiles["density_kg_m3"].min()),
                    "maximum": float(profiles["density_kg_m3"].max()),
                },
                "axial_limit_n": {
                    "mean": float(env._stem_tensile_limit_n.mean()),
                    "minimum": float(env._stem_tensile_limit_n.min()),
                    "maximum": float(env._stem_tensile_limit_n.max()),
                },
                "bending_limit_nm": {
                    "mean": float(env._stem_bending_limit_nm.mean()),
                    "minimum": float(env._stem_bending_limit_nm.min()),
                    "maximum": float(env._stem_bending_limit_nm.max()),
                },
                "torsion_limit_nm": {
                    "mean": float(env._stem_torsion_limit_nm.mean()),
                    "minimum": float(env._stem_torsion_limit_nm.min()),
                    "maximum": float(env._stem_torsion_limit_nm.max()),
                },
                "orientation_counts_front_back_left_right": [
                    int(value) for value in category_count
                ],
                "orientation_polar_tilt_deg": {
                    "mean": float(
                        torch.rad2deg(
                            profiles["orientation_polar_tilt_rad"]
                        ).mean()
                    ),
                    "minimum": float(
                        torch.rad2deg(
                            profiles["orientation_polar_tilt_rad"]
                        ).min()
                    ),
                    "maximum": float(
                        torch.rad2deg(
                            profiles["orientation_polar_tilt_rad"]
                        ).max()
                    ),
                },
                "fatigue_time_constant_s": {
                    "mean": float(profiles["stem_critical_duration_s"].mean()),
                    "minimum": float(profiles["stem_critical_duration_s"].min()),
                    "maximum": float(profiles["stem_critical_duration_s"].max()),
                },
                "effective_pair_static_friction": {
                    "mean": float(
                        contact_materials[
                            "effective_pair_static_friction"
                        ].mean()
                    ),
                    "minimum": float(
                        contact_materials[
                            "effective_pair_static_friction"
                        ].min()
                    ),
                    "maximum": float(
                        contact_materials[
                            "effective_pair_static_friction"
                        ].max()
                    ),
                },
                "load_bearing_opposition_target_n": {
                    "mean": float(
                        env._load_bearing_opposition_target_n.mean()
                    ),
                    "minimum": float(
                        env._load_bearing_opposition_target_n.min()
                    ),
                    "maximum": float(
                        env._load_bearing_opposition_target_n.max()
                    ),
                },
                "action_delay_steps_uniform_inclusive": engineering[
                    "a7_l20_effective_actuator"
                ]["action_delay_steps_uniform_inclusive"],
                "sampled_action_delay_steps": sorted(sampled_delays),
                "unique_contact_material_bins": int(material_bins.numel()),
                "runtime_envelope": {
                    "maximum_stem_force_n": maximum_stem_force_n,
                    "maximum_stem_torque_nm": maximum_stem_torque_nm,
                    "maximum_joint_velocity_rad_s": maximum_joint_velocity_rad_s,
                    "minimum_integrity": minimum_integrity,
                    "maximum_cumulative_twist_rad": (
                        maximum_cumulative_twist_rad
                    ),
                    "maximum_effective_twist_rad": maximum_effective_twist_rad,
                },
            },
            "training_started": False,
            "hardware_connected": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    except BaseException as exc:
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "qualification": "physics_v2_public_prior_isaac_smoke",
            "task_id": "LinkAgent-TomatoPick-Physics-v2-PublicPrior-v1",
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_started": False,
            "hardware_connected": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    finally:
        if raw_env is not None:
            raw_env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
