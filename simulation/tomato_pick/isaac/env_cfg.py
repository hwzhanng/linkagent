"""Isaac Lab configuration for the minimal tomato-picking MDP."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from module.skills.tomato_pick import ACTION_DIM, CRITIC_STATE_DIM, OBSERVATION_DIM
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
    r"^(left|right)_hand_(thumb_joint4|(index|middle|ring|little)_joint3)$"
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
    """One active environment; stages are metrics, never controls."""

    episode_length_s = 8.0
    # The 8 s limit is an explicit unsuccessful task outcome.  It must not be
    # value-bootstrapped, otherwise discounting makes safe waiting preferable
    # to an earlier physical attempt.
    is_finite_horizon = True
    decimation = 6
    action_space = ACTION_DIM
    observation_space = OBSERVATION_DIM
    state_space = CRITIC_STATE_DIM
    debug_vis = False

    # Keep recordings centered on the hand--fruit interaction.  The default
    # Isaac Lab camera is a world-scale overview and made failures impossible
    # to inspect frame by frame.  This changes visualization only.
    viewer = ViewerCfg(
        eye=(1.35, -0.58, 2.10),
        lookat=(0.97, -0.385, 1.93),
        origin_type="world",
        env_index=0,
        resolution=(1280, 720),
    )

    sim = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**21),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="min",
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
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[ARM_PATTERN],
                stiffness=1200.0,
                damping=80.0,
            ),
            "g20_independent": ImplicitActuatorCfg(
                joint_names_expr=[HAND_INDEPENDENT_PATTERN],
                effort_limit_sim=1.0,
                stiffness=12.0,
                damping=0.6,
                friction=0.01,
            ),
            "g20_followers": ImplicitActuatorCfg(
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
                static_friction=0.60,
                dynamic_friction=0.50,
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
        # A cylinder has an unambiguous end plane.  This lets the lower end be
        # placed exactly on the fruit attachment point; the previous capsule
        # mixed its cylinder height and cap radius and appeared to float.
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.003,
            height=0.05,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.42, 0.06), roughness=0.8
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.24, 1.23)),
    )
    fingertip_contacts = {
        name: ContactSensorCfg(
            prim_path=prim_path,
            update_period=1.0 / 120.0,
            history_length=0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
            track_pose=True,
            track_contact_points=False,
            track_friction_forces=False,
        )
        for name, prim_path in RIGHT_FINGERTIP_PATHS.items()
    }

    action_scales = (0.004, 0.004, 0.006, 0.040, 0.040, 0.060)
    # The verified reset needs about 9 cm of forward travel to reach the
    # closure envelope.  Keep generous local authority without exposing the
    # global arm workspace to PPO.
    max_residual_translation_m = (0.20, 0.08, 0.15)
    max_residual_rotation_rad = (0.80, 0.80, 1.40)
    ik_damping = 0.035
    ik_nullspace_gain = 0.05
    maximum_arm_joint_delta_rad = 0.055
    maximum_arm_target_lead_rad = 0.10
    minimum_joint_limit_margin_fraction = PREGRASP_RESET_SPEC.joint_limit_margin_fraction

    closed_grasp_fruit_center_m = PREGRASP_RESET_SPEC.closed_grasp_fruit_center_m
    pregrasp_distance_range_m = PREGRASP_RESET_SPEC.fruit_distance_outside_grasp_m
    pregrasp_lateral_offset_range_m = PREGRASP_RESET_SPEC.lateral_offset_range_m
    pregrasp_axial_noise_m = PREGRASP_RESET_SPEC.axial_noise_m
    pregrasp_arm_joint_noise_deg = PREGRASP_RESET_SPEC.arm_joint_noise_deg
    minimum_initial_clearance_m = PREGRASP_RESET_SPEC.minimum_clearance_m
    stem_visual_height_m = 0.05
    success_hold_steps = 20
    # Detach credit is delayed until several complete policy steps of stable,
    # opposed retention.  The final success definition remains 20 steps.
    stable_detach_credit_steps = 3
    maximum_retained_distance_m = 0.12
    maximum_retained_relative_speed_m_s = 0.25
    maximum_fingertip_force_n = 15.0
    grasp_full_contact_force_n = 0.60
    reach_distance_scale_m = 0.30

    stem_linear_natural_frequency_rad_s = 45.0
    stem_linear_damping_ratio = 1.0
    stem_angular_natural_frequency_rad_s = 15.0
    stem_angular_damping_ratio = 1.0
    maximum_stem_force_n = 100.0
    maximum_stem_torque_nm = 0.30
    break_force_range_n = (3.0, 7.0)
    # Physics runs at 120 Hz and the policy at 20 Hz (decimation=6).  Requiring
    # seven consecutive overloaded substeps (58.3 ms) guarantees the actor one
    # observation/action opportunity before breakage becomes irreversible.
    stem_break_persistence_substeps = 7

    action_delay_steps = (0, 1)
    tactile_force_gain = (0.85, 1.15)
    tactile_force_bias_n = (-0.15, 0.15)
    target_position_noise_m = 0.006
    fruit_axis_relative_noise = 0.08
    strict_runtime_checks = True


__all__ = [
    "RIGHT_FINGERTIP_CONTACT_BODY_NAMES",
    "RIGHT_FINGERTIP_FRAME_BODY_NAMES",
    "RIGHT_FINGERTIP_PATHS",
    "RIGHT_HAND_BODY_NAMES",
    "TomatoPickEnvCfg",
]
