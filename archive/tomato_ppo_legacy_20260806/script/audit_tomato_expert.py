"""Audit physical reachability with an observation-only 22-D expert.

This executable is deliberately outside the task and training packages.  It
never changes the environment, never updates a policy, and never becomes part
of deployment.  Its only purpose is to answer a binary engineering question:
can the frozen 22-D action interface complete the current tomato-pick physics?

The controller reads the same 113-D observation that the deployable actor
receives.  Simulator truth is used only after stepping to score outcomes and
grasp quality; it never selects an action.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.skills.tomato_pick.contracts import ACTION_DIM, OBSERVATION_DIM


FLEXION_ACTIONS = (
    (6, 17),   # thumb joint2 / joint3
    (7, 18),   # index joint1 / joint2
    (8, 19),   # middle joint1 / joint2
    (9, 20),   # ring joint1 / joint2
    (10, 21),  # little joint1 / joint2
)
OPPOSITION_ACTIONS = (11, 16)  # thumb joint0 / joint1
SPREAD_ACTIONS = (12, 13, 14, 15)
HAND_POSITION_OBSERVATION = slice(26, 42)
PRESSURE_OBSERVATION = slice(58, 63)
PRESSURE_DELTA_OBSERVATION = slice(63, 68)
GRAVITY_OBSERVATION = slice(68, 71)


def _parser() -> argparse.ArgumentParser:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim2real-mode", choices=("nominal", "full"), default="nominal")
    parser.add_argument(
        "--approach-caps",
        type=float,
        nargs="+",
        default=(0.30, 0.50),
        help="maximum absolute normalized Cartesian approach action",
    )
    parser.add_argument(
        "--spread-targets",
        type=float,
        nargs="+",
        default=(0.00, 0.14, 0.25),
        help="normalized G20 spread targets used by the audit profiles",
    )
    parser.add_argument(
        "--opposition-targets",
        type=float,
        nargs="+",
        default=(0.0,),
        help="normalized thumb opposition targets used by the audit profiles",
    )
    parser.add_argument(
        "--lift-steps-m",
        type=float,
        nargs="+",
        default=(0.0, 0.0005),
        help="per-policy-step anti-gravity translation while forming a grasp",
    )
    parser.add_argument(
        "--vertical-offsets-m",
        type=float,
        nargs="+",
        default=(0.0,),
        help="world-up offset of the contact target; negative values place the hand lower",
    )
    parser.add_argument(
        "--wrist-y-offsets-rad",
        type=float,
        nargs="+",
        default=(0.0,),
        help="grasp-frame local-y wrist offsets used by the geometry scan",
    )
    parser.add_argument(
        "--opposition-offsets-m",
        type=float,
        nargs="+",
        default=(0.0,),
        help="grasp-frame x offsets for centring fruit between thumb and fingers",
    )
    parser.add_argument(
        "--lateral-offsets-m",
        type=float,
        nargs="+",
        default=(0.0,),
        help="grasp-frame y offsets for the geometry scan",
    )
    parser.add_argument("--thumb-force-target-n", type=float, default=6.5)
    parser.add_argument("--support-force-target-n", type=float, default=1.2)
    parser.add_argument(
        "--hand-control-mode",
        choices=("pressure", "absolute_shape"),
        default="pressure",
    )
    parser.add_argument(
        "--closure-targets",
        type=float,
        nargs="+",
        default=(0.20,),
        help="normalized flexion target used only by absolute_shape mode",
    )
    parser.add_argument("--pull-action", type=float, default=0.25)
    parser.add_argument("--twist-action", type=float, default=0.20)
    parser.add_argument("--detach-command-steps", type=int, default=20)
    parser.add_argument(
        "--fracture-mode",
        choices=("normal", "disabled_for_diagnosis"),
        default="normal",
        help="counterfactual only: disable fracture to inspect pre-fracture grasp reachability",
    )
    parser.add_argument(
        "--audit-all-hand-contacts",
        action="store_true",
        help=(
            "diagnose which of the 26 right-hand rigid bodies is touching the tomato "
            "at the exact fracture substep"
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")
    if args.detach_command_steps < 1:
        raise ValueError("--detach-command-steps must be positive")
    for name, values in (
        ("approach caps", args.approach_caps),
        ("spread targets", args.spread_targets),
        ("opposition targets", args.opposition_targets),
        ("lift steps", args.lift_steps_m),
        ("vertical offsets", args.vertical_offsets_m),
        ("wrist y offsets", args.wrist_y_offsets_rad),
        ("closure targets", args.closure_targets),
        ("opposition offsets", args.opposition_offsets_m),
        ("lateral offsets", args.lateral_offsets_m),
    ):
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} must contain finite values")
    if not all(0.0 < value <= 1.0 for value in args.approach_caps):
        raise ValueError("approach caps must lie in (0, 1]")
    if not all(-1.0 <= value <= 1.0 for value in args.spread_targets):
        raise ValueError("spread targets must lie in [-1, 1]")
    if not all(-1.0 <= value <= 1.0 for value in args.opposition_targets):
        raise ValueError("opposition targets must lie in [-1, 1]")
    if not all(0.0 <= value <= 0.002 for value in args.lift_steps_m):
        raise ValueError("lift steps must lie in [0, 0.002] metres")
    if not all(-0.05 <= value <= 0.05 for value in args.vertical_offsets_m):
        raise ValueError("vertical offsets must lie in [-0.05, 0.05] metres")
    if not all(-0.30 <= value <= 0.30 for value in args.wrist_y_offsets_rad):
        raise ValueError("wrist y offsets must lie in [-0.30, 0.30] radians")
    if not all(0.0 <= value <= 1.0 for value in args.closure_targets):
        raise ValueError("closure targets must lie in [0, 1]")
    if not all(-0.05 <= value <= 0.05 for value in args.opposition_offsets_m):
        raise ValueError("opposition offsets must lie in [-0.05, 0.05] metres")
    if not all(-0.03 <= value <= 0.03 for value in args.lateral_offsets_m):
        raise ValueError("lateral offsets must lie in [-0.03, 0.03] metres")
    if not 0.0 < args.thumb_force_target_n < 15.0:
        raise ValueError("thumb force target must lie in (0, 15) N")
    if not 0.0 < args.support_force_target_n < 15.0:
        raise ValueError("support force target must lie in (0, 15) N")
    if not 0.0 < args.pull_action <= 1.0 or not 0.0 <= args.twist_action <= 1.0:
        raise ValueError("pull/twist actions must lie in their normalized bounds")


def _profile_table(args: argparse.Namespace) -> list[dict[str, float]]:
    return [
        {
            "approach_cap": float(approach),
            "spread_target": float(spread),
            "opposition_target": float(opposition),
            "lift_step_m": float(lift),
            "vertical_offset_m": float(vertical),
            "wrist_y_offset_rad": float(wrist_y),
            "closure_target": float(closure),
            "opposition_offset_m": float(opposition_offset),
            "lateral_offset_m": float(lateral_offset),
        }
        for (
            approach,
            spread,
            opposition,
            lift,
            vertical,
            wrist_y,
            closure,
            opposition_offset,
            lateral_offset,
        ) in itertools.product(
            args.approach_caps,
            args.spread_targets,
            args.opposition_targets,
            args.lift_steps_m,
            args.vertical_offsets_m,
            args.wrist_y_offsets_rad,
            args.closure_targets,
            args.opposition_offsets_m,
            args.lateral_offsets_m,
        )
    ]


def _desired_hand_positions(
    torch, raw, spread_targets, opposition_targets, closure_targets
):
    """Return neutral opposition and a selected spread in radians.

    Flexion is intentionally excluded because it is pressure controlled.
    """

    env_module = importlib.import_module("simulation.tomato_pick.isaac.env")
    desired = raw._robot.data.joint_pos.torch[:, raw._hand_joint_ids].clone()
    for column, name in enumerate(raw._hand_joint_names):
        lower, neutral, upper, source = env_module.RIGHT_HAND_CONTROL_RAD[name]
        if source == "closure":
            desired[:, column] = lower + closure_targets * (upper - lower)
        elif source == "opposition":
            value = opposition_targets
            desired[:, column] = torch.where(
                value >= 0.0,
                neutral + value * (upper - neutral),
                neutral + (-value) * (lower - neutral),
            )
        elif source == "spread":
            value = spread_targets
            desired[:, column] = torch.where(
                value >= 0.0,
                neutral + value * (upper - neutral),
                neutral + (-value) * (lower - neutral),
            )
    return desired


def _right_hand_body_paths(body_root: str) -> dict[str, str]:
    """Return the exact nested USD rigid-body paths for the right G20."""

    paths = {"right_hand_base_link": body_root}
    parent = body_root
    for index in range(6):
        name = f"right_hand_thumb_link{index}"
        parent = f"{parent}/{name}"
        paths[name] = parent
    for finger in ("index", "middle", "ring", "little"):
        parent = body_root
        for index in range(5):
            name = f"right_hand_{finger}_link{index}"
            parent = f"{parent}/{name}"
            paths[name] = parent
    return paths


def main() -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    args = _parser().parse_args()
    _validate_args(args)
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
        cfg.enable_sim2real_randomization = args.sim2real_mode == "full"
        env_module = importlib.import_module("simulation.tomato_pick.isaac.env")
        original_setup_scene = None
        if args.audit_all_hand_contacts:
            sensor_module = importlib.import_module("isaaclab.sensors")
            env_class = env_module.TomatoPickEnv
            original_setup_scene = env_class._setup_scene

            def setup_scene_with_contact_audit(self) -> None:
                original_setup_scene(self)
                audit_sensors = {}
                for body_name, prim_path in _right_hand_body_paths(
                    cfg_module.RIGHT_HAND_BODY_ROOT
                ).items():
                    sensor_cfg = sensor_module.ContactSensorCfg(
                        prim_path=prim_path,
                        update_period=1.0 / 120.0,
                        history_length=0,
                        filter_prim_paths_expr=[f"{self.scene.env_regex_ns}/Tomato"],
                        track_pose=False,
                        track_contact_points=False,
                        track_friction_forces=False,
                    )
                    sensor = sensor_module.ContactSensor(sensor_cfg)
                    audit_sensors[body_name] = sensor
                    self.scene.sensors[f"audit_{body_name}_tomato_contact"] = sensor
                self._audit_all_hand_contact_sensors = audit_sensors

            env_class._setup_scene = setup_scene_with_contact_audit
        try:
            raw_env = gym.make(task.TASK_ID, cfg=cfg)
        finally:
            if original_setup_scene is not None:
                env_class._setup_scene = original_setup_scene
        raw = raw_env.unwrapped
        observations, _ = raw_env.reset(seed=args.seed)

        all_hand_contact_sensors = {}
        break_contact_audit = {
            "events": 0,
            "tip_force_magnitude_sum_n": 0.0,
            "non_tip_force_magnitude_sum_n": 0.0,
            "events_with_only_unobserved_contact": 0,
            "body_contact_event_counts": {},
            "body_force_magnitude_sum_n": {},
            "body_maximum_force_n": {},
            "dominant_body_event_counts": {},
            "tip_view_validation_samples": 0,
            "existing_tip_force_magnitude_sum_n": 0.0,
            "diagnostic_tip_force_magnitude_sum_n": 0.0,
            "maximum_tip_view_difference_n": 0.0,
        }
        if args.audit_all_hand_contacts:
            all_hand_contact_sensors = raw._audit_all_hand_contact_sensors
            for body_name in all_hand_contact_sensors:
                break_contact_audit["body_contact_event_counts"][body_name] = 0
                break_contact_audit["body_force_magnitude_sum_n"][body_name] = 0.0
                break_contact_audit["body_maximum_force_n"][body_name] = 0.0
                break_contact_audit["dominant_body_event_counts"][body_name] = 0

            observed_tip_bodies = set(
                cfg_module.RIGHT_FINGERTIP_CONTACT_BODY_NAMES.values()
            )
            body_names = tuple(all_hand_contact_sensors)
            tip_columns = torch.tensor(
                [name in observed_tip_bodies for name in body_names],
                dtype=torch.bool,
                device=raw.device,
            )
            observed_tip_columns = torch.tensor(
                [body_names.index(name) for name in cfg_module.RIGHT_FINGERTIP_CONTACT_BODY_NAMES.values()],
                dtype=torch.long,
                device=raw.device,
            )
            rolling_force_max = torch.zeros(
                (args.num_envs, len(body_names)), device=raw.device
            )
            original_pre_physics_step = raw._pre_physics_step

            def audited_pre_physics_step(actions) -> None:
                original_pre_physics_step(actions)
                rolling_force_max.zero_()

            raw._pre_physics_step = audited_pre_physics_step
            original_apply_stem_wrench = raw._apply_stem_wrench

            def audited_apply_stem_wrench() -> None:
                force_by_body = {}
                for body_name, sensor in all_hand_contact_sensors.items():
                    sensor._update_buffers_impl()
                    force_by_body[body_name] = torch.linalg.vector_norm(
                        sensor.data.force_matrix_w.torch[:, 0, 0], dim=1
                    )
                force_matrix = torch.stack(
                    [force_by_body[name] for name in body_names], dim=1
                )
                rolling_force_max.copy_(torch.maximum(rolling_force_max, force_matrix))

                existing_tip_force = raw._instantaneous_finger_force_magnitudes()
                diagnostic_tip_force = force_matrix[:, observed_tip_columns]
                validation_mask = (
                    existing_tip_force.amax(dim=1) >= 0.10
                ) | (diagnostic_tip_force.amax(dim=1) >= 0.10)
                if bool(validation_mask.any()):
                    break_contact_audit["tip_view_validation_samples"] += int(
                        validation_mask.sum()
                    )
                    break_contact_audit["existing_tip_force_magnitude_sum_n"] += float(
                        existing_tip_force[validation_mask].sum()
                    )
                    break_contact_audit["diagnostic_tip_force_magnitude_sum_n"] += float(
                        diagnostic_tip_force[validation_mask].sum()
                    )
                    break_contact_audit["maximum_tip_view_difference_n"] = max(
                        break_contact_audit["maximum_tip_view_difference_n"],
                        float(
                            torch.abs(
                                existing_tip_force[validation_mask]
                                - diagnostic_tip_force[validation_mask]
                            ).max()
                        ),
                    )

                threshold_before = raw._stem_threshold_reached_this_step.clone()
                original_apply_stem_wrench()
                newly_thresholded = (
                    ~threshold_before & raw._stem_threshold_reached_this_step
                )
                if not bool(newly_thresholded.any()):
                    return

                selected = rolling_force_max[newly_thresholded]
                break_contact_audit["events"] += int(newly_thresholded.sum())
                break_contact_audit["tip_force_magnitude_sum_n"] += float(
                    selected[:, tip_columns].sum()
                )
                break_contact_audit["non_tip_force_magnitude_sum_n"] += float(
                    selected[:, ~tip_columns].sum()
                )
                break_contact_audit["events_with_only_unobserved_contact"] += int(
                    (
                        (selected[:, ~tip_columns].amax(dim=1) >= 0.10)
                        & (selected[:, tip_columns].amax(dim=1) < 0.10)
                    ).sum()
                )
                dominant_columns = selected.argmax(dim=1)
                for column, body_name in enumerate(body_names):
                    values = selected[:, column]
                    break_contact_audit["body_contact_event_counts"][body_name] += int(
                        (values >= 0.10).sum()
                    )
                    break_contact_audit["body_force_magnitude_sum_n"][body_name] += float(
                        values.sum()
                    )
                    break_contact_audit["body_maximum_force_n"][body_name] = max(
                        break_contact_audit["body_maximum_force_n"][body_name],
                        float(values.max()),
                    )
                    break_contact_audit["dominant_body_event_counts"][body_name] += int(
                        (dominant_columns == column).sum()
                    )

            raw._apply_stem_wrench = audited_apply_stem_wrench

        profiles = _profile_table(args)
        profile_id = torch.arange(args.num_envs, device=raw.device) % len(profiles)
        approach_cap = torch.tensor(
            [profiles[int(index)]["approach_cap"] for index in profile_id.cpu()],
            device=raw.device,
        )
        spread_target = torch.tensor(
            [profiles[int(index)]["spread_target"] for index in profile_id.cpu()],
            device=raw.device,
        )
        opposition_target = torch.tensor(
            [profiles[int(index)]["opposition_target"] for index in profile_id.cpu()],
            device=raw.device,
        )
        lift_step_m = torch.tensor(
            [profiles[int(index)]["lift_step_m"] for index in profile_id.cpu()],
            device=raw.device,
        )
        vertical_offset_m = torch.tensor(
            [profiles[int(index)]["vertical_offset_m"] for index in profile_id.cpu()],
            device=raw.device,
        )
        wrist_y_offset_rad = torch.tensor(
            [profiles[int(index)]["wrist_y_offset_rad"] for index in profile_id.cpu()],
            device=raw.device,
        )
        closure_target = torch.tensor(
            [profiles[int(index)]["closure_target"] for index in profile_id.cpu()],
            device=raw.device,
        )
        opposition_offset_m = torch.tensor(
            [profiles[int(index)]["opposition_offset_m"] for index in profile_id.cpu()],
            device=raw.device,
        )
        lateral_offset_m = torch.tensor(
            [profiles[int(index)]["lateral_offset_m"] for index in profile_id.cpu()],
            device=raw.device,
        )
        pressure_target = torch.tensor(
            (
                args.thumb_force_target_n,
                args.support_force_target_n,
                args.support_force_target_n,
                args.support_force_target_n,
                args.support_force_target_n,
            ),
            device=raw.device,
        )
        hand_lower = torch.tensor(
            [
                importlib.import_module("simulation.tomato_pick.isaac.env")
                .RIGHT_HAND_CONTROL_RAD[name][0]
                for name in raw._hand_joint_names
            ],
            device=raw.device,
        )
        hand_upper = torch.tensor(
            [
                importlib.import_module("simulation.tomato_pick.isaac.env")
                .RIGHT_HAND_CONTROL_RAD[name][2]
                for name in raw._hand_joint_names
            ],
            device=raw.device,
        )

        # Controller-local state is derived only from the deployable
        # observation and its own short streaks.  0=approach, 1=envelop,
        # 2=pull/twist, 3=hold.
        controller_stage = torch.zeros(args.num_envs, dtype=torch.long, device=raw.device)
        load_streak = torch.zeros_like(controller_stage)
        stage_steps = torch.zeros_like(controller_stage)
        checks = {
            "observation_shape_113": tuple(observations["policy"].shape)
            == (args.num_envs, OBSERVATION_DIM),
            "action_shape_22": raw.single_action_space.shape == (ACTION_DIM,),
            "observations_finite": True,
            "actions_finite_and_bounded": True,
            "rewards_finite": True,
            "physics_finite": True,
        }
        per_profile = [
            {
                "episodes": 0,
                "contact": 0,
                "grasp": 0,
                "detach": 0,
                "success": 0,
                "drop": 0,
                "excess_force": 0,
                "timeout": 0,
            }
            for _ in profiles
        ]
        maximum_hold_streak = 0
        maximum_contact_streak = 0
        q_at_load_sum = 0.0
        q_at_load_count = 0
        force_closure_samples = 0
        force_closure_opposition_sum = 0.0
        force_closure_resultant_sum = 0.0
        contact_truth_samples = 0
        positive_support_samples = 0
        contact_q_sum = 0.0
        best_contact_q = -math.inf
        best_contact_geometry: dict[str, object] = {}
        maximum_applied_stem_force_n = 0.0
        maximum_quasistatic_stem_force_n = 0.0
        maximum_applied_stem_torque_nm = 0.0
        maximum_elastic_stem_torque_nm = 0.0
        nominal_break_force_n = raw._break_force_n.clone()
        nominal_break_torque_nm = raw._break_torque_nm.clone()
        quasistatic_failure_samples = 0
        quasistatic_failure_with_grasp_samples = 0
        quasistatic_failure_with_positive_support_samples = 0

        with torch.inference_mode():
            for _ in range(args.steps):
                if args.fracture_mode == "disabled_for_diagnosis":
                    # This is an explicit counterfactual, never a training
                    # setting.  It isolates whether the hand can form a grasp
                    # before the current failure model fires.
                    raw._break_force_n.fill_(1.0e6)
                    raw._break_torque_nm.fill_(1.0e6)
                policy_observation = observations["policy"]
                actions = torch.zeros(
                    (args.num_envs, ACTION_DIM), dtype=torch.float32, device=raw.device
                )
                active = ~raw._reset_pending
                pressure = policy_observation[:, PRESSURE_OBSERVATION] * 20.0
                pressure_delta = policy_observation[:, PRESSURE_DELTA_OBSERVATION] * 20.0
                first_contact = pressure.max(dim=1).values >= 0.20
                newly_enveloping = active & (controller_stage == 0) & first_contact
                controller_stage[newly_enveloping] = 1
                stage_steps[newly_enveloping] = 0

                approaching = active & (controller_stage == 0)
                enveloping = active & (controller_stage == 1)
                positioning = approaching | enveloping
                if bool(positioning.any()):
                    pose_error = policy_observation[:, :6]
                    scales = torch.tensor(raw.cfg.action_scales[:6], device=raw.device)
                    world_up_g = -policy_observation[:, GRAVITY_OBSERVATION]
                    target_offset_g = world_up_g * vertical_offset_m.unsqueeze(1)
                    target_offset_g[:, 0] += opposition_offset_m
                    target_offset_g[:, 1] += lateral_offset_m
                    target_rotation_g = torch.zeros(
                        (args.num_envs, 3), device=raw.device
                    )
                    target_rotation_g[:, 1] = wrist_y_offset_rad
                    error_to_target = pose_error.clone()
                    error_to_target[:, :3] -= target_offset_g
                    error_to_target[:, 3:6] -= target_rotation_g
                    pose_action = -0.12 * error_to_target / scales
                    pose_action[:, :3] = torch.maximum(
                        -approach_cap.unsqueeze(1),
                        torch.minimum(approach_cap.unsqueeze(1), pose_action[:, :3]),
                    )
                    pose_action[:, 3:6] = pose_action[:, 3:6].clamp(-0.20, 0.20)
                    actions[positioning, :6] = pose_action[positioning]

                # Opposition and spread converge to physical targets through
                # the same incremental 16-joint interface exposed to PPO.
                hand_norm = policy_observation[:, HAND_POSITION_OBSERVATION]
                hand_position = hand_lower + 0.5 * (hand_norm + 1.0) * (
                    hand_upper - hand_lower
                )
                desired_hand = _desired_hand_positions(
                    torch,
                    raw,
                    spread_target,
                    opposition_target,
                    closure_target,
                )
                shape_action = ((desired_hand - hand_position) / 0.025).clamp(-0.20, 0.20)
                actions[:, OPPOSITION_ACTIONS[0]] = shape_action[:, 5]
                actions[:, OPPOSITION_ACTIONS[1]] = shape_action[:, 10]
                actions[:, SPREAD_ACTIONS[0]] = shape_action[:, 6]
                actions[:, SPREAD_ACTIONS[1]] = shape_action[:, 7]
                actions[:, SPREAD_ACTIONS[2]] = shape_action[:, 8]
                actions[:, SPREAD_ACTIONS[3]] = shape_action[:, 9]

                hand_active = active & (controller_stage >= 1)
                pressure_action = (
                    0.10 * (pressure_target.unsqueeze(0) - pressure)
                    - 0.05 * pressure_delta
                ).clamp(-0.12, 0.25)
                if args.hand_control_mode == "pressure":
                    for finger, pair in enumerate(FLEXION_ACTIONS):
                        actions[hand_active, pair[0]] = pressure_action[hand_active, finger]
                        actions[hand_active, pair[1]] = pressure_action[hand_active, finger]
                else:
                    for pair in FLEXION_ACTIONS:
                        actions[hand_active, pair[0]] = shape_action[hand_active, pair[0] - 6]
                        actions[hand_active, pair[1]] = shape_action[hand_active, pair[1] - 6]

                if bool(enveloping.any()):
                    # A small anti-gravity translation biases the contact
                    # normals toward supporting, rather than pushing down on,
                    # the fruit.  It is expressed using the observed gravity
                    # direction and remains inside the 6-D action interface.
                    world_up_g = -policy_observation[:, GRAVITY_OBSERVATION]
                    translation_scale = torch.tensor(
                        raw.cfg.action_scales[:3], device=raw.device
                    )
                    lift_action = world_up_g * (
                        lift_step_m / translation_scale.mean()
                    ).unsqueeze(1)
                    actions[enveloping, :3] += lift_action[enveloping].clamp(-0.15, 0.15)
                    actions[enveloping] = actions[enveloping].clamp(-1.0, 1.0)

                contact_count = (pressure >= raw.cfg.minimum_contact_force_n).sum(dim=1)
                acceptable_load = (
                    (contact_count >= 2)
                    & (pressure.sum(dim=1) >= 8.0)
                    & (pressure.sum(dim=1) <= 14.0)
                    & (pressure.max(dim=1).values < 12.0)
                )
                load_streak = torch.where(
                    active & (controller_stage == 1) & acceptable_load,
                    load_streak + 1,
                    torch.zeros_like(load_streak),
                )
                begin_pull = active & (controller_stage == 1) & (load_streak >= 4)
                if bool(begin_pull.any()):
                    q = raw._gravity_support_ratio()
                    q_at_load_sum += float(q[begin_pull].sum())
                    q_at_load_count += int(begin_pull.sum())
                    opposition, resultant, _ = raw._grasp_force_closure_quality()
                    force_closure_samples += int(begin_pull.sum())
                    force_closure_opposition_sum += float(opposition[begin_pull].sum())
                    force_closure_resultant_sum += float(resultant[begin_pull].sum())
                    controller_stage[begin_pull] = 2
                    stage_steps[begin_pull] = 0
                    load_streak[begin_pull] = 0

                pulling = active & (controller_stage == 2)
                actions[pulling, 2] = args.pull_action
                actions[pulling, 5] = args.twist_action
                if bool(pulling.any()):
                    world_up_g = -policy_observation[:, GRAVITY_OBSERVATION]
                    translation_scale = torch.tensor(
                        raw.cfg.action_scales[:3], device=raw.device
                    )
                    lift_action = world_up_g * (0.0004 / translation_scale.mean())
                    actions[pulling, :3] += lift_action[pulling].clamp(-0.10, 0.10)
                    actions[pulling] = actions[pulling].clamp(-1.0, 1.0)
                finish_pull = pulling & (stage_steps >= args.detach_command_steps)
                controller_stage[finish_pull] = 3
                stage_steps[finish_pull] = 0

                holding = active & (controller_stage == 3)
                # Keep Cartesian velocity continuous for one modest upward
                # step while pressure control retains the achieved hand shape.
                if bool(holding.any()):
                    world_up_g = -policy_observation[:, GRAVITY_OBSERVATION]
                    translation_scale = torch.tensor(
                        raw.cfg.action_scales[:3], device=raw.device
                    )
                    hold_lift = world_up_g * (0.0003 / translation_scale.mean())
                    actions[holding, :3] = hold_lift[holding].clamp(-0.08, 0.08)

                actions[~active] = 0.0
                actions.clamp_(-1.0, 1.0)
                checks["observations_finite"] &= bool(torch.isfinite(policy_observation).all())
                checks["actions_finite_and_bounded"] &= bool(
                    torch.isfinite(actions).all()
                    and (actions >= -1.0).all()
                    and (actions <= 1.0).all()
                )

                observations, reward, terminated, truncated, _ = raw_env.step(actions)
                done = terminated | truncated
                checks["rewards_finite"] &= bool(torch.isfinite(reward).all())
                checks["physics_finite"] &= bool(
                    torch.isfinite(raw._robot.data.joint_pos.torch).all()
                    and torch.isfinite(raw._tomato.data.root_state_w.torch).all()
                )
                maximum_hold_streak = max(maximum_hold_streak, int(raw._hold_streak.max()))
                maximum_contact_streak = max(
                    maximum_contact_streak, int(raw._contact_streak.max())
                )

                # Read simulator truth only after the action has been fixed.
                # This snapshot diagnoses whether the collision geometry can
                # put any contacted pad below the fruit centre and create an
                # upward supporting resultant.
                force_on_fruit_w = -torch.stack(
                    [
                        sensor.data.force_matrix_w.torch[:, 0, 0]
                        for sensor in raw._contact_sensors.values()
                    ],
                    dim=1,
                )
                exact_force_n = torch.linalg.vector_norm(force_on_fruit_w, dim=2)
                any_true_contact = exact_force_n.max(dim=1).values >= 0.10
                support_q = raw._gravity_support_ratio()
                if bool(any_true_contact.any()):
                    contact_truth_samples += int(any_true_contact.sum())
                    positive_support_samples += int(
                        (any_true_contact & (support_q > 0.0)).sum()
                    )
                    contact_q_sum += float(support_q[any_true_contact].sum())
                    gravity_w = torch.tensor(raw.cfg.sim.gravity, device=raw.device)
                    position_error = (
                        raw._stem_anchor_pos_w - raw._tomato.data.root_pos_w.torch
                    )
                    quasistatic_force = raw._fruit_mass_kg.unsqueeze(1) * (
                        raw.cfg.stem_linear_natural_frequency_rad_s**2
                        * position_error
                        - gravity_w.unsqueeze(0)
                    )
                    quat_box_minus = importlib.import_module(
                        "isaaclab.utils.math"
                    ).quat_box_minus
                    rotation_error = quat_box_minus(
                        raw._stem_anchor_quat_w,
                        raw._tomato.data.root_pose_w.torch[:, 3:7],
                    )
                    inertia_proxy = (
                        0.4
                        * raw._fruit_mass_kg
                        * torch.square(raw._fruit_axes_m).mean(dim=1)
                    )
                    elastic_torque = inertia_proxy.unsqueeze(1) * (
                        raw.cfg.stem_angular_natural_frequency_rad_s**2
                        * rotation_error
                    )
                    maximum_applied_stem_force_n = max(
                        maximum_applied_stem_force_n,
                        float(
                            torch.linalg.vector_norm(
                                raw._last_stem_force_w[any_true_contact], dim=1
                            ).max()
                        ),
                    )
                    maximum_quasistatic_stem_force_n = max(
                        maximum_quasistatic_stem_force_n,
                        float(
                            torch.linalg.vector_norm(
                                quasistatic_force[any_true_contact], dim=1
                            ).max()
                        ),
                    )
                    maximum_applied_stem_torque_nm = max(
                        maximum_applied_stem_torque_nm,
                        float(
                            torch.linalg.vector_norm(
                                raw._last_stem_torque_w[any_true_contact], dim=1
                            ).max()
                        ),
                    )
                    maximum_elastic_stem_torque_nm = max(
                        maximum_elastic_stem_torque_nm,
                        float(
                            torch.linalg.vector_norm(
                                elastic_torque[any_true_contact], dim=1
                            ).max()
                        ),
                    )
                    quasistatic_failure = any_true_contact & (
                        (
                            torch.linalg.vector_norm(quasistatic_force, dim=1)
                            >= nominal_break_force_n
                        )
                        | (
                            torch.linalg.vector_norm(elastic_torque, dim=1)
                            >= nominal_break_torque_nm
                        )
                    )
                    quasistatic_failure_samples += int(quasistatic_failure.sum())
                    quasistatic_failure_with_grasp_samples += int(
                        (quasistatic_failure & raw._grasp_stable).sum()
                    )
                    quasistatic_failure_with_positive_support_samples += int(
                        (quasistatic_failure & (support_q > 0.5)).sum()
                    )
                    candidate = torch.where(
                        any_true_contact,
                        support_q,
                        torch.full_like(support_q, -torch.inf),
                    )
                    candidate_q, candidate_env = candidate.max(dim=0)
                    if float(candidate_q) > best_contact_q:
                        env_index = int(candidate_env)
                        best_contact_q = float(candidate_q)
                        fruit_position = raw._tomato.data.root_pos_w.torch[env_index]
                        pad_position = raw._robot.data.body_pos_w.torch[
                            env_index, raw._fingertip_contact_body_ids
                        ]
                        terminal_position = raw._robot.data.body_pos_w.torch[
                            env_index, raw._fingertip_frame_body_ids
                        ]
                        grasp_from_world = raw._grasp_frame_basis_w[
                            env_index
                        ].transpose(0, 1)
                        gravity_w = torch.tensor(raw.cfg.sim.gravity, device=raw.device)
                        world_up = -gravity_w / torch.linalg.vector_norm(gravity_w)
                        best_contact_geometry = {
                            "environment": env_index,
                            "profile": profiles[int(profile_id[env_index])],
                            "q_up": best_contact_q,
                            "fruit_position_w": fruit_position.cpu().tolist(),
                            "pad_minus_fruit_grasp_m": torch.matmul(
                                grasp_from_world,
                                (pad_position - fruit_position).transpose(0, 1),
                            ).transpose(0, 1).cpu().tolist(),
                            "pad_height_above_fruit_m": torch.matmul(
                                pad_position - fruit_position, world_up
                            ).cpu().tolist(),
                            "terminal_minus_fruit_grasp_m": torch.matmul(
                                grasp_from_world,
                                (terminal_position - fruit_position).transpose(0, 1),
                            ).transpose(0, 1).cpu().tolist(),
                            "terminal_height_above_fruit_m": torch.matmul(
                                terminal_position - fruit_position, world_up
                            ).cpu().tolist(),
                            "gravity_direction_grasp": torch.matmul(
                                grasp_from_world,
                                gravity_w / torch.linalg.vector_norm(gravity_w),
                            ).cpu().tolist(),
                            "force_on_fruit_w_n": force_on_fruit_w[
                                env_index
                            ].cpu().tolist(),
                            "force_magnitude_n": exact_force_n[
                                env_index
                            ].cpu().tolist(),
                        }

                terms = raw._last_reward_terms
                contact_event = terms["contact"] > 0.0
                grasp_event = terms["grasp"] > 0.0
                success_event = terms["success"] > 0.0
                drop_event = terms["drop"] < 0.0
                excess_event = terms["excess_force"] < 0.0
                timeout_event = terms["timeout"] < 0.0
                detach_event = raw._stem_detached | success_event | drop_event
                for index in range(len(profiles)):
                    selected = profile_id == index
                    result = per_profile[index]
                    result["contact"] += int((selected & contact_event).sum())
                    result["grasp"] += int((selected & grasp_event).sum())
                    result["detach"] += int((selected & detach_event).sum())
                    result["success"] += int((selected & success_event).sum())
                    result["drop"] += int((selected & drop_event).sum())
                    result["excess_force"] += int((selected & excess_event).sum())
                    result["timeout"] += int((selected & timeout_event).sum())
                    result["episodes"] += int((selected & done).sum())

                controller_stage[done] = 0
                load_streak[done] = 0
                stage_steps[done] = 0
                stage_steps[active & ~done] += 1

        aggregate = {
            key: sum(int(result[key]) for result in per_profile)
            for key in per_profile[0]
        }
        ranked = sorted(
            (
                {"profile": profiles[index], **result}
                for index, result in enumerate(per_profile)
            ),
            key=lambda item: (
                item["success"],
                -item["drop"],
                item["grasp"],
            ),
            reverse=True,
        )
        report = {
            "schema_version": 1,
            "passed": all(checks.values()),
            "controller_inputs": "deployable_observation_113d_only",
            "simulator_truth_use": "outcome_and_quality_audit_only",
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "sim2real_mode": args.sim2real_mode,
            "fracture_mode": args.fracture_mode,
            "hand_control_mode": args.hand_control_mode,
            "checks": checks,
            "aggregate": aggregate,
            "maximum_contact_streak": maximum_contact_streak,
            "maximum_hold_streak": maximum_hold_streak,
            "load_entry": {
                "samples": q_at_load_count,
                "mean_gravity_support_ratio": q_at_load_sum / max(q_at_load_count, 1),
                "mean_opposition_cosine": force_closure_opposition_sum
                / max(force_closure_samples, 1),
                "mean_resultant_ratio": force_closure_resultant_sum
                / max(force_closure_samples, 1),
            },
            "contact_support_truth": {
                "samples": contact_truth_samples,
                "positive_fraction": positive_support_samples
                / max(contact_truth_samples, 1),
                "mean_q_up": contact_q_sum / max(contact_truth_samples, 1),
                "maximum_q_up": best_contact_q,
                "best_geometry": best_contact_geometry,
            },
            "stem_load_diagnosis": {
                "maximum_applied_force_n": maximum_applied_stem_force_n,
                "maximum_quasistatic_force_n": maximum_quasistatic_stem_force_n,
                "maximum_applied_torque_nm": maximum_applied_stem_torque_nm,
                "maximum_elastic_torque_nm": maximum_elastic_stem_torque_nm,
                "quasistatic_failure_samples": quasistatic_failure_samples,
                "quasistatic_failure_with_grasp_samples": (
                    quasistatic_failure_with_grasp_samples
                ),
                "quasistatic_failure_with_q_above_0_5_samples": (
                    quasistatic_failure_with_positive_support_samples
                ),
            },
            "all_hand_contact_at_fracture": break_contact_audit,
            "top_profiles": ranked[: min(12, len(ranked))],
        }
        print("TOMATO_EXPERT_AUDIT=" + json.dumps(report, sort_keys=True))
        return 0 if report["passed"] else 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if raw_env is not None:
            raw_env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
