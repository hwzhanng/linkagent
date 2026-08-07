"""Isaac Lab configuration for one-arm local tomato picking."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from module.skills.tomato_pick.phase_contract import (
    CONTACT_FORCE_THRESHOLD_N,
    GRASP_MINIMUM_TOTAL_NORMAL_FORCE_N,
)
from module.skills.tomato_pick.contracts import ACTION_DIM, OBSERVATION_DIM
from simulation.tomato_pick.initialization import (
    LEFT_ARM_PARK_JOINTS_RAD,
    PREGRASP_RESET_SPEC,
    RIGHT_ARM_PREGRASP_JOINTS_RAD,
)

from .assets import randomize_correlated_ellipsoid_geometry


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ROBOT_USD = (
    REPOSITORY_ROOT
    / "botSDK/urdf/02_project_derived/isaac_assets/"
    "linkbot_bimanual_sdk_training_mass1kg/linkbot_bimanual_sdk_training_mass1kg.usda"
)
ARM_PATTERN = r"^[LR][1-7]_Joint$"
HAND_INDEPENDENT_PATTERN = (
    r"^(left|right)_hand_(thumb_joint[0-3]|"
    r"(index|middle|ring|little)_joint[0-2])$"
)
HAND_FOLLOWER_PATTERN = (
    r"^(left|right)_hand_(thumb_joint4|"
    r"(index|middle|ring|little)_joint3)$"
)
RIGHT_HAND_BODY_ROOT = (
    "/World/envs/env_.*/Robot/Geometry/base_link/R1_Link/R2_Link/R3_Link/"
    "R4_Link/R5_Link/R6_Link/R7_Link/R8_Link/right_hand_base_link"
)
RIGHT_FINGERTIP_PATHS = {
    "thumb": RIGHT_HAND_BODY_ROOT
    + "/right_hand_thumb_link0/right_hand_thumb_link1/right_hand_thumb_link2/"
    "right_hand_thumb_link3/right_hand_thumb_link4",
    "index": RIGHT_HAND_BODY_ROOT
    + "/right_hand_index_link0/right_hand_index_link1/right_hand_index_link2/"
    "right_hand_index_link3",
    "middle": RIGHT_HAND_BODY_ROOT
    + "/right_hand_middle_link0/right_hand_middle_link1/right_hand_middle_link2/"
    "right_hand_middle_link3",
    "ring": RIGHT_HAND_BODY_ROOT
    + "/right_hand_ring_link0/right_hand_ring_link1/right_hand_ring_link2/"
    "right_hand_ring_link3",
    "little": RIGHT_HAND_BODY_ROOT
    + "/right_hand_little_link0/right_hand_little_link1/right_hand_little_link2/"
    "right_hand_little_link3",
}
RIGHT_FINGERTIP_FRAME_BODY_NAMES = {
    "thumb": "right_hand_thumb_link5",
    "index": "right_hand_index_link4",
    "middle": "right_hand_middle_link4",
    "ring": "right_hand_ring_link4",
    "little": "right_hand_little_link4",
}
RIGHT_FINGERTIP_CONTACT_BODY_NAMES = {
    "thumb": "right_hand_thumb_link4",
    "index": "right_hand_index_link3",
    "middle": "right_hand_middle_link3",
    "ring": "right_hand_ring_link3",
    "little": "right_hand_little_link3",
}
RIGHT_HAND_BODY_NAMES = (
    "right_hand_base_link",
    *(f"right_hand_thumb_link{index}" for index in range(6)),
    *(
        f"right_hand_{finger}_link{index}"
        for finger in ("index", "middle", "ring", "little")
        for index in range(5)
    ),
)


@configclass
class TomatoPickEventCfg:
    fruit_geometry = EventTerm(
        func=randomize_correlated_ellipsoid_geometry,
        mode="prestartup",
        params={
            "asset_name": "tomato",
            "axis_ranges_m": {
                "x": (0.025, 0.035),
                "y": (0.025, 0.035),
                "z": (0.025, 0.035),
            },
            "density_range_kg_m3": (650.0, 1050.0),
            "accepted_mass_kg": (0.03, 0.50),
            "maximum_axis_ratio": 1.6,
        },
    )


@configclass
class TomatoPickEnvCfg(DirectRLEnvCfg):
    """One active environment configuration; PPO remains launch-gated."""

    # One global task horizon.  Progress labels never terminate a rollout.
    episode_length_s = 8.0
    decimation = 6
    action_space = ACTION_DIM
    observation_space = OBSERVATION_DIM
    # The asymmetric critic must be at least as observable as the deployed
    # actor.  The old 33-D state omitted phase progress, end-effector error,
    # action history, and the force-closure signals used by the reward and
    # transition gates.  That made successful, dropped, and near-timeout
    # trajectories indistinguishable to the value function.  Keep the actor's
    # frozen 113-D contract and append 33 exact task-state values plus seven
    # training-only force-closure/state-machine values.
    state_space = 153
    debug_vis = False

    sim = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        # The PhysX default (163,840 patches) overflows around 2,048 cloned
        # dexterous-hand environments.  One million patches matches the scale
        # used by Isaac Lab contact-rich manipulation tasks and still fits the
        # 32 GB RTX 5090 budget.
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="min",
            # Tomato material buckets already contain the intended effective
            # sim-to-real friction range.  A unit baseline on unassigned hand
            # colliders makes multiply preserve that range instead of silently
            # multiplying it by 0.55/0.45 a second time.
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.02,
        ),
    )
    scene = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=1.8,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    events = TomatoPickEventCfg()
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
    )
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            activate_contact_sensors=True,
        ),
        actuators={
            "arm_imported_drives": ImplicitActuatorCfg(
                joint_names_expr=[ARM_PATTERN],
                # The imported 34.9 N m/rad drive leaves roughly 0.1 rad of
                # gravity sag with the fixed G20 mass profile.  The hardware
                # position controller is substantially stiffer; use a stable
                # position-servo proxy while retaining the USD effort limits.
                stiffness=1200.0,
                damping=80.0,
            ),
            "g20_independent_drives": ImplicitActuatorCfg(
                joint_names_expr=[HAND_INDEPENDENT_PATTERN],
                effort_limit_sim=1.0,
                stiffness=12.0,
                damping=0.6,
                friction=0.01,
            ),
            "g20_follower_drives": ImplicitActuatorCfg(
                joint_names_expr=[HAND_FOLLOWER_PATTERN],
                effort_limit_sim=1.0,
                stiffness=12.0,
                damping=0.6,
                friction=0.01,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            joint_pos={
                **{
                    f"R{index}_Joint": value
                    for index, value in enumerate(RIGHT_ARM_PREGRASP_JOINTS_RAD, start=1)
                },
                **{
                    f"L{index}_Joint": value
                    for index, value in enumerate(LEFT_ARM_PARK_JOINTS_RAD, start=1)
                },
            },
        ),
    )
    tomato = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tomato",
        spawn=sim_utils.MeshSphereCfg(
            radius=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05,
                angular_damping=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0002,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.55,
                dynamic_friction=0.45,
                restitution=0.02,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.06, 0.03),
                roughness=0.72,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.24, 1.17)),
    )
    stem = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Stem",
        spawn=sim_utils.MeshCapsuleCfg(
            radius=0.003,
            height=0.05,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.42, 0.06),
                roughness=0.8,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.24, 1.23)),
    )
    fingertip_contacts = {
        name: ContactSensorCfg(
            prim_path=prim_path,
            update_period=1.0 / 120.0,
            history_length=0,
            # Canonical Isaac Lab namespace form; env.py expands it before
            # manually constructing the environment-aligned contact view.
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
            track_pose=True,
            track_contact_points=False,
            track_friction_forces=False,
        )
        for name, prim_path in RIGHT_FINGERTIP_PATHS.items()
    }
    enable_fingertip_contacts = True
    # The physical G20 publishes tactile frames at up to 200 FPS and the
    # deployable feature extractor consumes the latest valid frame directly.
    # Sensor delay, gain and bias are already modelled by Sim2RealRuntime, so
    # an additional policy-rate EMA would double-count latency.  Zero means a
    # current 20 Hz sample; non-zero values are reserved for measured hardware
    # filters only.
    tactile_policy_sample_retention = 0.0
    minimum_hand_inertia_kg_m2 = 2.5e-7
    # Local grasp-frame translation, local rotation-vector, then the sixteen
    # independently commanded G20 joints.  These are controller increments,
    # not phase scripts; every dimension remains available throughout a run.
    action_scales = (
        0.004,
        0.004,
        0.006,
        0.040,
        0.040,
        0.050,
        *((0.025,) * 16),
    )
    max_residual_translation_m = (0.06, 0.06, 0.12)
    max_residual_rotation_rad = (0.60, 0.60, 1.20)
    ik_position_damping = 0.015
    ik_rotation_damping = 0.080
    # Resolve the redundant seventh arm DOF away from straight-elbow
    # singularities without changing PPO's requested Cartesian translation.
    ik_nullspace_posture_gain = 1.00
    # Rotation tracking and redundant-posture bias are secondary tasks.  Give
    # them their own budget so they can never consume the full joint-rate
    # budget and suppress PPO's Cartesian translation.
    maximum_arm_secondary_joint_delta_rad = 0.020
    # One 120 Hz substep may request up to 0.04 rad (4.8 rad/s), still below
    # every arm joint velocity limit while giving the 20 Hz Cartesian policy
    # enough authority to realize its 2--3 mm command against gravity.
    maximum_arm_joint_delta_rad = 0.070
    maximum_arm_target_lead_rad = 0.120
    right_arm_pregrasp_joint_pos_rad = RIGHT_ARM_PREGRASP_JOINTS_RAD
    left_arm_park_joint_pos_rad = LEFT_ARM_PARK_JOINTS_RAD
    initial_fruit_clearance_m = PREGRASP_RESET_SPEC.fruit_clearance_m
    # Start outside the open G20 aperture with zero contact, while keeping the
    # fruit inside the now-wider Cartesian workspace.  PPO must actively
    # approach the fruit; reset geometry must neither make contact nor decide
    # fracture.  The contact reference is shifted by the same amount.
    fruit_precontact_distance_range_m = (0.06, 0.09)
    fruit_precontact_lateral_offset_m = 0.010
    collision_retry_clearance_increment_m = (
        PREGRASP_RESET_SPEC.collision_retry_clearance_increment_m
    )
    maximum_collision_retries = PREGRASP_RESET_SPEC.maximum_collision_retries
    pregrasp_small_fruit_lateral_radius_m = PREGRASP_RESET_SPEC.small_fruit_lateral_radius_m
    pregrasp_large_fruit_lateral_radius_m = PREGRASP_RESET_SPEC.large_fruit_lateral_radius_m
    pregrasp_small_fruit_closure = PREGRASP_RESET_SPEC.small_fruit_closure
    pregrasp_large_fruit_closure = PREGRASP_RESET_SPEC.large_fruit_closure
    pregrasp_thumb_opposition = PREGRASP_RESET_SPEC.thumb_opposition
    pregrasp_finger_spread = PREGRASP_RESET_SPEC.finger_spread
    reset_minimum_robot_settle_steps = PREGRASP_RESET_SPEC.minimum_robot_settle_steps
    reset_required_stable_steps = PREGRASP_RESET_SPEC.required_stable_steps
    reset_maximum_steps = PREGRASP_RESET_SPEC.maximum_reset_steps
    reset_arm_joint_position_tolerance_rad = (
        PREGRASP_RESET_SPEC.arm_joint_position_tolerance_rad
    )
    reset_hand_joint_position_tolerance_rad = (
        PREGRASP_RESET_SPEC.hand_joint_position_tolerance_rad
    )
    reset_arm_joint_velocity_tolerance_rad_s = (
        PREGRASP_RESET_SPEC.arm_joint_velocity_tolerance_rad_s
    )
    reset_hand_joint_step_delta_tolerance_rad = (
        PREGRASP_RESET_SPEC.hand_joint_step_delta_tolerance_rad
    )
    reset_fruit_linear_velocity_tolerance_m_s = (
        PREGRASP_RESET_SPEC.fruit_linear_velocity_tolerance_m_s
    )
    reset_fruit_angular_velocity_tolerance_rad_s = (
        PREGRASP_RESET_SPEC.fruit_angular_velocity_tolerance_rad_s
    )
    reset_maximum_initial_contact_force_n = PREGRASP_RESET_SPEC.maximum_initial_contact_force_n
    reset_maximum_anchor_displacement_m = PREGRASP_RESET_SPEC.maximum_anchor_displacement_m
    minimum_arm_joint_limit_margin_fraction = (
        PREGRASP_RESET_SPEC.minimum_arm_joint_limit_margin_fraction
    )
    reset_staging_offset_m = (0.45, 0.0, 0.50)
    stem_visual_height_m = 0.05
    # Kept only for the frozen observation/manifest layout.  The first three
    # entries are logging labels; only the final retention window is a success
    # invariant, never an action mask or phase timeout.
    phase_step_limits = (30, 30, 40, 20)
    success_hold_steps = 20
    fruit_axis_lower_m = (0.025, 0.025, 0.025)
    fruit_axis_upper_m = (0.035, 0.035, 0.035)
    # Full target deployment distribution is active from rollout one.  Set
    # false only for the separate deterministic audit, never as a curriculum.
    enable_sim2real_randomization = True
    # Stiff enough for millimetre-scale pre-fracture compliance under finger
    # loads while remaining stable at the 120 Hz physics step.
    stem_linear_natural_frequency_rad_s = 50.0
    stem_linear_damping_ratio = 1.0
    stem_angular_natural_frequency_rad_s = 15.0
    stem_angular_damping_ratio = 1.0
    maximum_stem_force_n = 100.0
    maximum_stem_torque_nm = 0.30
    enable_stem_wrench = True
    # A tomato is detached by sustained tension along the peduncle or twist
    # about it.  Three 120 Hz substeps reject lateral impact transients while
    # remaining much shorter than one 20 Hz policy interval.
    stem_break_persistence_substeps = 3
    maximum_fingertip_force_n = 15.0
    minimum_contact_force_n = CONTACT_FORCE_THRESHOLD_N
    minimum_grasp_total_normal_force_n = GRASP_MINIMUM_TOTAL_NORMAL_FORCE_N
    # Force balance is a soft, hand-shape-agnostic grasp signal.  It does not
    # require a particular thumb/finger combination and never gates fracture.
    grasp_force_resultant_score_ceiling = 0.85
    soft_contact_force_n = 8.0
    # A retained fruit is centered among the G20 fingertips, not at the wrist
    # frame.  The nominal fingertip-center distance is about 3--4 cm; 12 cm
    # leaves more than two large-fruit radii of clearance before declaring a
    # true separation from the hand.
    maximum_retained_fruit_to_fingertip_center_distance_m = 0.12
    strict_runtime_checks = True

    # Outcome-driven shaping.  Contact and grasp are truthful one-shot events
    # computed from their own physical predicates, so they guide exploration
    # without prescribing a sequence.  Fracture alone is never rewarded:
    # detaching a fruit that is then dropped must not earn progress credit.
    approach_potential_distance_scale_m = 0.05
    reward_approach_progress_scale = 0.50
    reward_contact_scale = 1.0
    reward_grasp_progress_scale = 1.0
    reward_grasp_scale = 2.0
    reward_stem_load_progress_scale = 1.0
    reward_break_once_scale = 0.0
    # Must match PPO gamma so approach, grasp, stem-load and HOLD shaping are
    # policy-invariant discounted potentials.  Completion/failure returns the
    # corresponding potential to zero instead of preserving dense credit.
    reward_potential_discount = 0.99
    reward_retention_scale = 0.50
    reward_success_scale = 20.0
    # Every terminal failure has the same outcome value.  Dense potentials are
    # returned to zero on failure, so neither waiting nor a failed fracture can
    # harvest positive progress credit.
    reward_drop_scale = -20.0
    reward_excess_force_scale = -20.0
    reward_timeout_scale = -20.0
    # Do not penalize fruit/hand relative velocity before fracture: an attached
    # fruit must move relative to the approaching and pulling hand.  True
    # post-detachment loss is already measured by the one-second HOLD window
    # and fingertip-envelope drop test used by the frozen success contract.
    reward_soft_contact_force_scale = -0.10
    reward_joint_limit_scale = -0.05
    reward_action_magnitude_scale = -0.0005
    reward_action_rate_scale = -0.002
__all__ = [
    "RIGHT_FINGERTIP_CONTACT_BODY_NAMES",
    "RIGHT_FINGERTIP_PATHS",
    "TomatoPickEnvCfg",
]
