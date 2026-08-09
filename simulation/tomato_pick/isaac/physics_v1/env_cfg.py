"""Independent Isaac configuration for the calibrated Physics-v1 task."""

from __future__ import annotations

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.configclass import configclass

from simulation.tomato_pick.isaac.env_cfg import (
    ARM_PATTERN,
    ROBOT_USD,
    TomatoPickEnvCfg,
)
from simulation.tomato_pick.physics_v1 import load_physics_v1_profile

from .assets import (
    author_correlated_fruit_profiles,
    verify_baked_hand_physics_material,
    verify_baked_passive_hand_mimics,
    verify_baked_hand_self_collision_filters,
)


PROFILE = load_physics_v1_profile()
HAND = PROFILE["hand"]
ARM = PROFILE["arm"]
CONTACT = PROFILE["contact"]
SECURE_SUCCESS = PROFILE["secure_success"]
RESET_WORKSPACE = PROFILE["reset_workspace"]
_CLEAN_V5_CFG = TomatoPickEnvCfg()
PHYSICS_V1_ROBOT_USD = ROBOT_USD.with_name(
    "linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda"
)
RIGHT_HAND_INDEPENDENT_PATTERN = (
    r"^right_hand_(thumb_joint[0-3]|"
    r"(index|middle|ring|little)_joint[0-2])$"
)
RIGHT_HAND_FOLLOWER_PATTERN = (
    r"^right_hand_(thumb_joint4|(index|middle|ring|little)_joint3)$"
)
LEFT_HAND_ALL_PATTERN = r"^left_hand_.*_joint[0-4]$"


@configclass
class TomatoPickPhysicsV1EventCfg:
    hand_mimics = EventTerm(
        func=verify_baked_passive_hand_mimics,
        mode="prestartup",
        params={"asset_name": "robot", "hand_side": "right"},
    )
    hand_material = EventTerm(
        func=verify_baked_hand_physics_material,
        mode="prestartup",
        params={"asset_name": "robot"},
    )
    hand_self_collision_filters = EventTerm(
        func=verify_baked_hand_self_collision_filters,
        mode="prestartup",
        params={"asset_name": "robot"},
    )
    fruit_profile = EventTerm(
        func=author_correlated_fruit_profiles,
        mode="prestartup",
        params={"asset_name": "tomato"},
    )


@configclass
class TomatoPickPhysicsV1EnvCfg(TomatoPickEnvCfg):
    """Clean-v5 interface with independent, calibration-gated physics."""

    events = TomatoPickPhysicsV1EventCfg()
    robot = copy.deepcopy(_CLEAN_V5_CFG.robot)
    robot.spawn.usd_path = str(PHYSICS_V1_ROBOT_USD)
    # The USD contains instance-proxy collision meshes.  The generic spawn
    # binder cannot author on those proxies, so the prestartup event binds this
    # material on the non-instanced hand rigid-body parents instead.
    robot.spawn.physics_material = None
    # Clean-v5 intentionally disabled self collision. Physics-v1 enables it
    # only in this independent overlay; the exact matrix is baked once into an
    # independent derived USD and verified at prestartup. Cross-digit and distal-palm
    # pairs remain active so the thumb cannot pass through the fingers, while
    # adjacent/same-digit and out-of-scope coarse robot geometry are filtered.
    robot.spawn.articulation_props.enabled_self_collisions = True
    robot.actuators = {
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[ARM_PATTERN],
            stiffness=ARM["provisional_stiffness_nm_rad"],
            damping=ARM["provisional_damping_nm_s_rad"],
        ),
        "g20_active_right": ImplicitActuatorCfg(
            joint_names_expr=[RIGHT_HAND_INDEPENDENT_PATTERN],
            effort_limit_sim=HAND["active_drive_effort_limit_nm"],
            stiffness=HAND["active_drive_stiffness_nm_rad"],
            damping=HAND["active_drive_damping_nm_s_rad"],
            armature=HAND["active_drive_armature_kg_m2"],
            friction=0.02,
        ),
        # These joints are mechanically passive in L20.  Position-drive gains
        # are zero; native compliant PhysX mimic joints authored at pre-startup
        # provide the driver/follower linkage.
        "g20_passive_followers_right": ImplicitActuatorCfg(
            joint_names_expr=[RIGHT_HAND_FOLLOWER_PATTERN],
            effort_limit_sim=0.01,
            stiffness=0.0,
            damping=0.0,
            friction=0.002,
        ),
        # Physics-v1 is a single-right-hand task.  The unused left hand is
        # parked by ordinary position drives and has all self collision
        # filtered.  It is not falsely advertised as a second passive L20.
        "g20_park_left": ImplicitActuatorCfg(
            joint_names_expr=[LEFT_HAND_ALL_PATTERN],
            effort_limit_sim=HAND["active_drive_effort_limit_nm"],
            stiffness=HAND["active_drive_stiffness_nm_rad"],
            damping=HAND["active_drive_damping_nm_s_rad"],
            armature=HAND["active_drive_armature_kg_m2"],
            friction=0.02,
        ),
    }

    tomato = copy.deepcopy(_CLEAN_V5_CFG.tomato)
    tomato.spawn.rigid_props.max_depenetration_velocity = CONTACT[
        "maximum_depenetration_velocity_m_s"
    ]
    tomato.spawn.rigid_props.linear_damping = 0.08
    tomato.spawn.rigid_props.angular_damping = 0.08
    tomato.spawn.collision_props.contact_offset = CONTACT["contact_offset_m"]
    tomato.spawn.collision_props.rest_offset = CONTACT["rest_offset_m"]
    tomato.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=CONTACT["tomato_static_friction"],
        dynamic_friction=CONTACT["tomato_dynamic_friction"],
        restitution=CONTACT["restitution"],
        friction_combine_mode=CONTACT["friction_combine_mode"],
        restitution_combine_mode=CONTACT["restitution_combine_mode"],
        compliant_contact_stiffness=CONTACT["compliant_contact_stiffness_n_m"],
        compliant_contact_damping=CONTACT["compliant_contact_damping_n_s_m"],
    )

    fingertip_contacts = copy.deepcopy(_CLEAN_V5_CFG.fingertip_contacts)
    for sensor in fingertip_contacts.values():
        sensor.track_contact_points = True
        sensor.track_friction_forces = True
        sensor.max_contact_data_count_per_prim = 64

    maximum_arm_joint_delta_rad = ARM[
        "maximum_joint_target_delta_rad_per_policy_step"
    ]
    maximum_arm_target_lead_rad = ARM["maximum_joint_target_lead_rad"]
    hand_joint_max_velocity_rad_s = tuple(
        math.radians(value) for value in HAND["joint_max_velocity_deg_s_in_policy_order"]
    )
    passive_mimic_offset_rad = HAND["passive_mimic_offset_rad"]
    passive_mimic_driver_velocity_safety_factor = HAND[
        "passive_mimic_driver_velocity_safety_factor"
    ]
    passive_mimic_max_tracking_error_rad = HAND[
        "passive_mimic_max_tracking_error_rad"
    ]
    maximum_retained_distance_m = SECURE_SUCCESS["maximum_center_distance_m"]
    maximum_retained_relative_speed_m_s = SECURE_SUCCESS[
        "maximum_relative_speed_m_s"
    ]
    minimum_support_force_n = SECURE_SUCCESS["minimum_support_force_n"]
    minimum_supported_fingertips = SECURE_SUCCESS[
        "minimum_supported_fingertips"
    ]
    thumb_opposition_required = SECURE_SUCCESS["thumb_opposition_required"]
    maximum_fingertip_force_n = 8.0
    contact_point_maximum_surface_gap_m = CONTACT[
        "contact_point_maximum_surface_gap_m"
    ]
    reset_arm_joint_ellipsoid_radius_rad = tuple(
        math.radians(value)
        for value in RESET_WORKSPACE[
            "arm_joint_ellipsoid_radius_deg_in_a7_order"
        ]
    )
    reset_arm_correlated_modes_rad = tuple(
        tuple(math.radians(value) for value in mode)
        for mode in RESET_WORKSPACE.get(
            "arm_joint_correlated_modes_deg_in_a7_order", []
        )
    )
    reset_arm_correlated_mode_coefficient_ranges = tuple(
        tuple(float(value) for value in bounds)
        for bounds in RESET_WORKSPACE.get(
            "arm_joint_correlated_mode_coefficient_ranges", []
        )
    )
    reset_joint_limit_margin_fraction = RESET_WORKSPACE[
        "joint_limit_margin_fraction"
    ]
    reset_fruit_approach_distance_m = tuple(
        RESET_WORKSPACE["fruit_approach_distance_m"]
    )
    reset_fruit_approach_yaw_rad = tuple(
        math.radians(value) for value in RESET_WORKSPACE["fruit_approach_yaw_deg"]
    )
    reset_fruit_approach_pitch_rad = tuple(
        math.radians(value) for value in RESET_WORKSPACE["fruit_approach_pitch_deg"]
    )
    reset_fruit_stem_tilt_rad_max = math.radians(
        RESET_WORKSPACE["fruit_stem_tilt_deg_max"]
    )
    reset_fruit_yaw_rad = tuple(
        math.radians(value) for value in RESET_WORKSPACE["fruit_yaw_deg"]
    )
    minimum_initial_clearance_m = RESET_WORKSPACE[
        "minimum_initial_clearance_m"
    ]
    reset_fruit_sampling_max_attempts = RESET_WORKSPACE[
        "fruit_sampling_max_attempts"
    ]
    maximum_closure_translation_norm_m = RESET_WORKSPACE[
        "maximum_closure_translation_norm_m"
    ]
    minimum_closure_translation_headroom_fraction = RESET_WORKSPACE[
        "minimum_closure_translation_headroom_fraction"
    ]
    minimum_closure_joint_limit_margin_fraction = RESET_WORKSPACE[
        "minimum_closure_joint_limit_margin_fraction"
    ]
    maximum_closure_ik_position_error_m = RESET_WORKSPACE[
        "maximum_closure_ik_position_error_m"
    ]

    stem_linear_natural_frequency_rad_s = 30.0
    stem_angular_natural_frequency_rad_s = 12.0
    maximum_stem_force_n = 120.0
    maximum_stem_torque_nm = 0.8
    stem_damage_onset_ratio = PROFILE["stem_prior"]["damage_onset_ratio"]
    stem_damage_recovery_time_s = PROFILE["stem_prior"]["damage_recovery_time_s"]


__all__ = ["TomatoPickPhysicsV1EnvCfg"]
