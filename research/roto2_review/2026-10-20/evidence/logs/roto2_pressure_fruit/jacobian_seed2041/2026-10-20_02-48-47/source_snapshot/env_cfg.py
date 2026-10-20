"""Isaac Lab 3.x direct RL configuration: fixed wrist, one right L20 hand."""
from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from simulation.tomato_pick.isaac.assets import randomize_correlated_ellipsoid_geometry
from .core import ACTION_DIM, POLICY_DIM

ASSET_USD = os.environ.get("LINKAGENT_ROTATION_USD")
if not ASSET_USD or not Path(ASSET_USD).is_file():
    raise RuntimeError("Launch via script/train_fruit_rotation.py to build and select single L20 USD")


def randomize_rotation_fruit(env, env_ids, *, asset_name, axis_ranges_m,
                             density_range_kg_m3, accepted_mass_kg, maximum_axis_ratio):
    """Keep the sampled ellipsoid's geometry, density, mass and inertia consistent."""
    import math
    from pxr import Gf, UsdPhysics
    randomize_correlated_ellipsoid_geometry(
        env, env_ids, asset_name=asset_name, axis_ranges_m=axis_ranges_m,
        density_range_kg_m3=density_range_kg_m3, accepted_mass_kg=accepted_mass_kg,
        maximum_axis_ratio=maximum_axis_ratio)
    axes = env._prestartup_fruit_axes_m
    masses = (4 * math.pi / 3) * axes.prod(-1) * env._prestartup_fruit_density_kg_m3
    paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    for path, dimensions, mass in zip(paths, axes, masses):
        x, y, z = [float(v) for v in dimensions]
        m = float(mass)
        api = UsdPhysics.MassAPI.Apply(env.sim.stage.GetPrimAtPath(path))
        api.CreateMassAttr().Set(m)
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(
            m * (y*y + z*z) / 5, m * (x*x + z*z) / 5, m * (x*x + y*y) / 5))
    env._prestartup_fruit_mass_kg = masses


@configclass
class RotationEvents:
    geometry = EventTerm(
        func=randomize_rotation_fruit,
        mode="prestartup",
        params={
            "asset_name": "fruit",
            "axis_ranges_m": {
                "x": (0.029, 0.035),
                "y": (0.029, 0.035),
                "z": (0.029, 0.035),
            },
            "density_range_kg_m3": (850.0, 1050.0),
            "accepted_mass_kg": (0.08, 0.19),
            "maximum_axis_ratio": 1.20,
        },
    )


@configclass
class FruitRotationEnvCfg(DirectRLEnvCfg):
    episode_length_s = 24.0
    is_finite_horizon = True
    decimation = 6
    # Direct task follows RoTO 2.0's proprioception+tactile dictionary contract.
    obs_list = ("prop", "tactile")
    obs_stack = 4
    tactile_cfg = {"binary_tactile": False, "binary_threshold": 0.01}
    pixel_cfg = None
    num_eval_envs = 16
    num_actions = 16
    debug = False
    domain_profile = "full"  # launch-time randomization curriculum
    reward_stage = "rotation"  # "contact" curriculum or full "rotation" objective
    pressure_variant = "jacobian"  # "plain" | "scalar" | "jacobian"
    reset_grasps_file = str(Path(__file__).with_name("reset_pad_grasps.json"))
    reset_joint_noise_rad = 0.018
    reset_position_noise_m = 0.002
    pressure_relief_weight = 0.25
    action_space = ACTION_DIM
    observation_space = POLICY_DIM
    state_space = 0
    debug_vis = False
    viewer = ViewerCfg(
        eye=(0.60, -0.58, 1.18), lookat=(0.0, 0.0, 0.89),
        origin_type="world", env_index=0,
    )
    sim = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.6,
            friction_combine_mode="average", restitution=0.01,
        ),
    )
    scene = InteractiveSceneCfg(
        num_envs=256, env_spacing=1.0,
        replicate_physics=False, clone_in_fabric=False,
    )
    events = RotationEvents()
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane",
        collision_group=-1, debug_vis=False,
    )
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ASSET_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=0.20,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            # The task explicitly activates its six nested contact bodies.
            activate_contact_sensors=False,
        ),
        actuators={
            "g20_active": ImplicitActuatorCfg(
                joint_names_expr=[
                    r"^(thumb_joint[0-3]|(index|middle|ring|little)_joint[0-2])$"
                ],
                effort_limit_sim=0.45,
                stiffness=10.0,
                damping=0.8,
                armature=0.002,
                friction=0.02,
            ),
            "g20_followers": ImplicitActuatorCfg(
                joint_names_expr=[
                    r"^(thumb_joint4|(index|middle|ring|little)_joint3)$"
                ],
                effort_limit_sim=0.01,
                stiffness=0.0,
                damping=0.0,
                friction=0.002,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.75)),
    )
    fruit = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Fruit",
        spawn=sim_utils.MeshSphereCfg(
            radius=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.20,
                linear_damping=0.08, angular_damping=0.10,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0015, rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.75, dynamic_friction=0.60,
                friction_combine_mode="average", restitution=0.01,
                compliant_contact_stiffness=5000.0,
                compliant_contact_damping=25.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.87, 0.14, 0.07), roughness=0.65,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.94)),
    )
    maximum_joint_target_delta_rad = 0.035
    maximum_joint_target_lead_rad = 0.11
    target_angular_speed_rad_s = (0.45, 0.80)
    pressure_noise_std_fraction = 0.025
    tactile_noise_std_fraction = 0.035
    sparse_panel_dropout_probability = 0.25
    minimum_support_force_n = 0.15
    maximum_contact_force_n = 8.0
    maximum_fruit_center_displacement_m = 0.105
    maximum_pressure_proxy_pa = 18000.0
    overpressure_termination_steps = 3
