"""Palm-up, palm-supported tomato grasp environment with a stem pick task.

The first PhysX tomato environment for HAND_ONLY_TOMATO_PPO_PLAN.md: the left
wrist is fixed with the palm facing up, an ellipsoid rigid tomato rests in the
palm support region, a virtual spring-damper stem connects the tomato top to a
plant anchor above the hand, and a fixed slow pick trajectory moves the anchor
after a stable enclosure forms so the stem breaks under physical load.  The
policy controls only the ten hand DOFs (five synergy coordinates plus five
digit residuals) through the frozen 10D/110D/163D contract.

Key differences from ``physx_sphere_env.py``:

- the tomato is a per-env scaled ellipsoid resting on the palm (no capture
  assist; the palm carries the weight, exactly the plan's §5 rationale);
- the stem is a virtual spring-damper migrated from the team Clean-v5
  environment with no gravity compensation (the palm supports the rest
  weight, the stem only loads during the pick);
- the palm (base_link) has its own PhysX contact sensor;
- success is decided by the independent eight-condition Verifier (plan §6).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass

from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_ACTIVE_JOINT_ORDER,
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    LEFT_SAFE_VELOCITY_RAD_S,
    hybrid_action_to_joint_targets,
)
from simulation.ppo_tomato_grasp.enveloping_action_manifold import enveloping_action_to_joint_targets
from simulation.ppo_tomato_grasp.retention_reflex import adaptive_retention_correction
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    POLICY_RATE_HZ,
    TOMATO_CRITIC_STATE_DIM,
    pack_actor_observation,
    pack_tomato_critic_state,
)
from simulation.ppo_tomato_grasp.tomato_stem_physics import (
    evaluate_success_conditions,
    evaluate_stage1_hold_conditions,
    quat_mul_wxyz,
    quat_rotate_vector_wxyz,
    quat_to_rot_matrix_6d,
    stem_attachment_wrench,
    stem_load_components,
    update_stem_damage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAND_USD = PROJECT_ROOT / (
    "simulation/ppo_tomato_grasp/artifacts/physx_assets/left_hand_urdf/"
    "left_hand_l20_physx.usd"
)
# Top surface of the base_link palm block (mesh bbox z extent).  The tomato
# rests on this plane; the value is fixed by the vendor asset geometry.
PALM_TOP_Z_M = 0.169


def author_tomato_ellipsoid_geometry(
    env,
    env_ids,
    *,
    asset_name: str,
    axis_range_m: tuple[float, float] = (0.025, 0.035),
    density_range_kg_m3: tuple[float, float] = (700.0, 1050.0),
    accepted_mass_kg: tuple[float, float] = (0.050, 0.180),
    maximum_axis_ratio: float = 1.6,
    isotropic_axes: bool = False,
    contact_offset_m: float = 0.002,
    rest_offset_m: float = 0.0,
) -> None:
    """Author one ellipsoid tomato per environment before physics starts.

    Migrated from the team Clean-v5 prestartup geometry event.  The tomato is
    spawned as a unit-radius mesh sphere and non-uniform USD scale turns it
    into an ellipsoid with semi-axes 25-35 mm and a 50-180 g mass prior
    (plan §5).  Impossible size/mass combinations are rejected.  The sampled
    axes and density are stored on the environment for per-instance mass
    assignment, actor measurement, and the privileged critic.
    """

    if env.sim.is_playing():
        raise RuntimeError("tomato geometry can only be authored before physics starts")
    if env_ids is not None:
        raise ValueError("prestartup tomato geometry must cover every environment")
    count = env.scene.num_envs
    low, high = axis_range_m
    axes = torch.empty((count, 3), dtype=torch.float32)
    density = torch.empty(count, dtype=torch.float32)
    accepted = torch.zeros(count, dtype=torch.bool)
    for _ in range(64):
        pending = (~accepted).nonzero(as_tuple=False).squeeze(-1)
        if pending.numel() == 0:
            break
        if isotropic_axes:
            radius = low + torch.rand((pending.numel(), 1)) * (high - low)
            samples = radius.expand(-1, 3).clone()
        else:
            samples = low + torch.rand((pending.numel(), 3)) * (high - low)
        sampled_density = density_range_kg_m3[0] + torch.rand(pending.numel()) * (
            density_range_kg_m3[1] - density_range_kg_m3[0]
        )
        mass = 4.0 / 3.0 * math.pi * samples.prod(dim=-1) * sampled_density
        ratio = samples.max(dim=-1).values / samples.min(dim=-1).values
        valid = (
            (mass >= accepted_mass_kg[0])
            & (mass <= accepted_mass_kg[1])
            & (ratio <= maximum_axis_ratio)
        )
        if valid.any():
            target_ids = pending[valid]
            axes[target_ids] = samples[valid]
            density[target_ids] = sampled_density[valid]
            accepted[target_ids] = True
    if not bool(accepted.all()):
        raise RuntimeError("could not sample correlated tomato geometry in 64 attempts")

    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    prim_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(prim_paths) != count:
        raise RuntimeError(f"expected {count} tomato prims, found {len(prim_paths)}")
    for index, prim_path in enumerate(prim_paths):
        prim = env.sim.stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        scale_op = next(
            (op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale),
            None,
        )
        if scale_op is None:
            scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(*(float(value) for value in axes[index])))
        mesh = env.sim.stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
        if mesh.IsValid():
            UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Set("convexHull")
            collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
        else:
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr().Set(float(contact_offset_m))
        collision.CreateRestOffsetAttr().Set(float(rest_offset_m))

    env._tomato_axes_m = axes
    env._tomato_density_kg_m3 = density


@configclass
class PhysxTomatoSceneCfg(InteractiveSceneCfg):
    """Scene assets: one filtered PhysX contact channel per digit plus palm."""

    thumb_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_link5",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=True,
        max_contact_data_count_per_prim=32,
        update_period=0.0,
    )
    index_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/index_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=True,
        max_contact_data_count_per_prim=32,
        update_period=0.0,
    )
    middle_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/middle_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=True,
        max_contact_data_count_per_prim=32,
        update_period=0.0,
    )
    ring_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ring_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=True,
        max_contact_data_count_per_prim=32,
        update_period=0.0,
    )
    little_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/little_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=True,
        max_contact_data_count_per_prim=32,
        update_period=0.0,
    )
    thumb_contact_link4: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=False,
        max_contact_data_count_per_prim=4,
        update_period=0.0,
    )
    index_contact_link3: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/index_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=False,
        max_contact_data_count_per_prim=4,
        update_period=0.0,
    )
    middle_contact_link3: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/middle_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=False,
        max_contact_data_count_per_prim=4,
        update_period=0.0,
    )
    ring_contact_link3: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ring_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=False,
        max_contact_data_count_per_prim=4,
        update_period=0.0,
    )
    little_contact_link3: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/little_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=False,
        max_contact_data_count_per_prim=4,
        update_period=0.0,
    )
    palm_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
        track_contact_points=True,
        max_contact_data_count_per_prim=32,
        update_period=0.0,
    )


@configclass
class PhysxTomatoGraspEnvCfg(DirectRLEnvCfg):
    """Palm-up palm-supported tomato enclosure and stem-break task."""

    seed = 42
    # 120 Hz PhysX / 20 Hz policy contract (plan section 9).
    decimation = 6
    episode_length_s = 10.0
    action_space = ACTION_DIM
    observation_space = ACTOR_OBSERVATION_DIM
    state_space = TOMATO_CRITIC_STATE_DIM
    # Keep the resolved geometry prior first-class on the config as well as
    # in the pre-startup USD authoring event.  Evaluation/training reports
    # and explicit CLI overrides must not rely on ad-hoc attributes that a
    # configclass may discard.
    axis_range_m: tuple[float, float] = (0.025, 0.035)
    isotropic_axes: bool = False
    accepted_mass_kg: tuple[float, float] = (0.050, 0.180)
    density_range_kg_m3: tuple[float, float] = (700.0, 1050.0)

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        # GPU rigid-patch allocator; the USD-cloned scene needs it for the
        # same reason as the team Clean-v5 configuration.
        physx=PhysxCfg(gpu_max_rigid_patch_count=2**21),
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(HAND_USD),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "left_hand": IdealPDActuatorCfg(
                joint_names_expr=[
                    r"thumb_joint[0-3]",
                    r"(index|middle|ring|little)_joint[0-2]",
                ],
                effort_limit_sim=0.02,
                velocity_limit_sim=0.85,
                stiffness=0.06,
                damping=0.006,
            )
        },
    )

    # Unit-radius mesh sphere; the prestartup event scales each environment
    # copy into a 25-35 mm semi-axis ellipsoid with a 50-180 g mass prior.
    tomato_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tomato",
        spawn=sim_utils.MeshSphereCfg(
            radius=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.15,
                # A soft fruit dissipates rotation substantially faster than
                # a hard hollow sphere.  The 1.0 s^-1 first prior is retained
                # as a randomizable sim value pending real spin-down data.
                angular_damping=1.0,
                # A falling tomato can exceed 1 m/s before the first palm
                # contact; the default depenetration cap would let it tunnel
                # through the palm collision mesh.
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.075),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.70,
                dynamic_friction=0.60,
                restitution=0.0,
                compliant_contact_stiffness=18000.0,
                compliant_contact_damping=180.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.06, 0.03), roughness=0.72
            ),
            activate_contact_sensors=True,
        ),
        # The tomato rests on the palm top surface; per-env z is assigned at
        # reset from the sampled polar semi-axis.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.03, 0.005, 0.20)),
    )

    scene: PhysxTomatoSceneCfg = PhysxTomatoSceneCfg(
        num_envs=16,
        env_spacing=1.5,
        # USD-level per-instance ellipsoid scaling (the prestartup geometry
        # event) requires classic USD replication; fabric replication would
        # share one scaled source across every environment.
        replicate_physics=False,
        clone_in_fabric=False,
    )

    events = {
        "tomato_geometry": EventTerm(
            func=author_tomato_ellipsoid_geometry,
            mode="prestartup",
            params={
                "asset_name": "tomato",
                "axis_range_m": (0.025, 0.035),
                "density_range_kg_m3": (700.0, 1050.0),
                "accepted_mass_kg": (0.050, 0.180),
                "maximum_axis_ratio": 1.6,
                "isotropic_axes": False,
                "contact_offset_m": 0.002,
                "rest_offset_m": 0.0,
            },
        )
    }

    # Resting geometry.  The palm support point is the fixed palm-top surface
    # under the tomato; the actor's fruit-center observation is normalized
    # around it instead of the palm origin so the block does not saturate.
    # The imported base_link origin is not the centre of its usable palm
    # surface.  A reset collision sweep found that x=0 intersects the four
    # straight fingers, while x=+35..45 mm is supported without penetration
    # (about 0.027 mm first-step drift at x=+35 mm).  Use the middle of that
    # measured band as the palm-centred task frame.  Repeated five-channel
    # probes rejected the earlier y=-50 mm assumption: it could make thumb
    # contact, but not a simultaneous five-digit enclosure.  The measured
    # common workspace is centred at y=0 with the configured small jitter.
    palm_top_z_m = PALM_TOP_Z_M
    # The imported vendor palm mesh has small convex-decomposition gaps.  A
    # thin collision-only insert under the tomato makes the support region
    # continuous while remaining part of the base_link rigid body (so the
    # palm contact sensor still reports the real support reaction).
    # Cover the complete randomized palm-center region.  The previous 64 mm
    # x 100 mm insert was narrower than the 25--35 mm tomato plus the 4 mm
    # position jitter, so a valid reset could roll off the support before the
    # first close action.  This is still a thin collision representation of
    # the palm, not a kinematic capture volume.
    palm_support_surface_size_m = (0.160, 0.200, 0.006)
    # The vendor base-link collision contains holes large enough for the
    # 25--35 mm fruit to fall through.  Approximate the shallow anatomical
    # palm cup with low edge pads (thenar/hypothenar/palm edge), not a closed
    # capture volume.  These dimensions remain a sim prior pending scan or
    # physical palm-surface measurement.
    palm_cup_inner_half_extent_m = (0.055, 0.045)
    palm_cup_rim_thickness_m = 0.006
    # First uncalibrated palm-geometry prior.  A 20 mm open edge pad is the
    # smallest height that passes the required 20 Hz / one-second retention
    # feasibility gate; it is not a closed capture volume and remains a
    # sim2real calibration item pending a palm scan.
    palm_cup_rim_height_m = 0.020
    # Explicit support collision margins.  These are kept separate from the
    # tomato and fingertip margins so a diagnostic can remove speculative
    # lateral impulses without changing the default training prior.
    palm_support_contact_offset_m = 0.002
    palm_support_rest_offset_m = 0.0
    tomato_rest_xy = (0.040, 0.000)
    tomato_reset_clearance_m = 0.001
    # Optional simulation-only pregrasp settling window.  When enabled, the
    # reset fruit is placed on the support plane with zero velocity for a
    # bounded number of physics steps before normal dynamics are released.
    # This removes an artificial 1 mm drop impulse from the training signal;
    # it is not a grasp/success shortcut and remains disabled by default.
    pregrasp_settle_steps = 0
    position_jitter = (0.004, 0.004, 0.002)
    grasp_reference_scale_m = 0.08

    # Contact gates (Newtons; reporter forces from the converted fingertip
    # meshes are lower than calibrated taxel values).
    contact_threshold_n = 0.04
    support_force_n = 0.10
    palm_support_force_n = 0.30
    safe_force_n = 6.0
    force_ceiling_n = (6.0, 7.0, 7.0, 6.0, 5.0)
    severe_overload_scale = 1.25
    target_force_n = 0.65
    success_hold_steps = 20
    # Three consecutive policy frames (100 ms) reject one-frame contact
    # transients while allowing the fixed pick to begin before the enclosure
    # drifts out of the narrow five-fingertip overlap region.
    prepick_hold_steps = 3
    prepick_recovery_after_steps = 160
    prepick_recovery_min_steps = 2
    success_bonus = 10.0

    # Stem spring-damper (migrated from the team Clean-v5 overlay).  No
    # gravity compensation: the palm carries the resting weight.
    stem_linear_natural_frequency_rad_s = 30.0
    stem_linear_damping_ratio = 1.0
    stem_angular_natural_frequency_rad_s = 12.0
    stem_angular_damping_ratio = 1.0
    maximum_stem_force_n = 120.0
    maximum_stem_torque_nm = 0.8
    pregrasp_stem_wrench_scale = 0.03
    stem_damage_onset_ratio = 0.55
    stem_damage_recovery_time_s = 1.5
    stem_critical_duration_s = (0.15, 0.30)
    # Stage-2 "easy" stem band (plan §8): low and narrow break thresholds.
    stem_tensile_limit_n = (0.50, 0.90)
    stem_shear_to_tension_ratio = 0.8
    stem_bending_lever_m = 0.02
    stem_torsion_to_bending_ratio = 0.5

    # Fixed slow pick trajectory (plan §5): in the hand-fixed frame, upward
    # hand motion relative to a stationary plant is represented by the plant
    # anchor moving downward.  The previous positive-z anchor motion pulled
    # the fruit out of the upward-facing palm—the opposite relative motion.
    # The policy keeps commanding the fingers throughout; nothing is frozen.
    pick_speed_m_s = 0.02
    pick_displacement_m = 0.035
    pick_twist_rate_rad_s = 0.0
    pick_twist_angle_rad = 0.0
    # Physics-baseline switches.  The active palm-supported branch uses a
    # downward plant-anchor motion and lets the palm carry the resting fruit
    # weight.  The team Clean-v5 reference instead compensated gravity in the
    # stem and moved the anchor upward.  Keep both modes explicit so an
    # ablation can be reproduced without touching the archived implementation.
    stem_gravity_compensation = False
    pick_direction_z_sign = -1.0
    # Optional post-pick tactile reflex.  PPO still chooses the full 14D
    # action; this small bounded residual only nudges each distal joint toward
    # its own low-force target when a measured contact is lost, and backs off
    # when a digit exceeds its soft ceiling.  It is disabled for historical
    # checkpoints and enabled only in the dedicated reflex ablation/training.
    tactile_reflex_enabled = False
    tactile_reflex_gain_rad_per_n = 0.045
    tactile_reflex_max_correction_rad = 0.16
    tactile_reflex_target_force_n = (0.55, 0.42, 0.42, 0.36, 0.36)
    tactile_reflex_high_force_n = (2.80, 2.20, 2.20, 2.00, 2.00)
    # Safer post-break reflex candidate.  Unlike the historical tactile
    # ablation, this path is low-pass filtered, rate-limited by the normal
    # target servo, and uses both force error and measured slip speed.  It is
    # disabled by default so old checkpoints remain bitwise-compatible.
    adaptive_retention_reflex_enabled = False
    adaptive_retention_force_gain_rad_per_n = 0.012
    adaptive_retention_slip_gain_rad_per_m_s = 0.018
    adaptive_retention_slip_threshold_m_s = 0.04
    adaptive_retention_max_correction_rad = 0.018
    adaptive_retention_filter_alpha = 0.75
    adaptive_retention_target_force_n = (0.55, 0.42, 0.42, 0.36, 0.36)
    adaptive_retention_high_force_n = (2.80, 2.20, 2.20, 2.00, 2.00)
    post_break_target_latch = False
    post_break_command_mix = 0.15
    retention_only = False
    # Optional runtime bank of measured valid enclosure joint states.  It is
    # injected by the retention-course driver and is intentionally None for
    # ordinary PPO/evaluation runs; the fallback action below preserves the
    # historical contract.
    retention_seed_joint_positions = None
    retention_start_action = (
        0.30, -0.15, -0.63, -0.73, 0.82,
        -0.13, -0.86, 0.08, -0.90, 0.02,
        0.20, 0.20, 0.12, 0.33,
    )

    # Retention / verifier thresholds (plan §6).
    max_center_distance_m = 0.080
    max_relative_speed_m_s = 0.35
    drop_height_m = 0.10
    # Optional curriculum grace before palm-supported enclosure.  The
    # historical v1 default remains zero; v2 may use a bounded window so a
    # policy can establish palm support before a numerically transient fruit
    # fall terminates the episode.  This never weakens the success gate.
    pregrasp_drop_grace_steps = 0
    damage_limit = 0.40

    # Curriculum stage: 0 = contact/physics smoke gate, 1 = palm-supported
    # five-finger enclosure and hold, 2 = enable the fixed stem pick.  The
    # default is stage 1 so a PPO run cannot accidentally optimize a stem
    # break before it has learned the hand-only enclosure skill.
    training_stage = 1
    minimum_enclosing_digits = 5

    # Reward weights (initial values; tuned per curriculum stage).
    proximity_reward_weight = 2.0
    fingertip_proximity_reward_weight = 0.8
    contact_reward_weight = 2.0
    new_digit_reward_weight = 0.4
    thumb_opposition_reward_weight = 0.5
    palm_support_reward_weight = 0.8
    secure_reward_weight = 2.5
    stable_hold_reward_weight = 2.0
    lost_enclosure_penalty_weight = 1.0
    drop_penalty_weight = 10.0
    balance_penalty_weight = 0.3
    speed_penalty_weight = 0.50
    wait_penalty_weight = 0.03
    jitter_penalty_weight = 0.02
    concentration_penalty_weight = 0.30
    overload_penalty_weight = 10.0
    impact_penalty_weight = 0.20
    damage_penalty_weight = 0.50
    joint_penalty_weight = 0.05
    saturation_penalty_weight = 0.05
    unstable_pick_penalty_weight = 0.20
    break_bonus = 3.0
    # After the stem releases, keep a dense per-digit retention signal.  The
    # old reward only paid for the strict latch, so a policy could reach a
    # good pre-pick enclosure and then lose one or two fingers during lift
    # without receiving a useful corrective gradient before the drop.
    post_break_contact_reward_weight = 2.0
    post_break_support_reward_weight = 0.8
    post_break_missing_digit_penalty_weight = 1.5
    center_distance_penalty_weight = 0.8
    relative_speed_penalty_weight = 0.25
    # A finite reaction window models the fact that tactile feedback is
    # sampled at the policy rate.  It delays *termination* after detachment,
    # not the strict success gate: a fruit still outside the final retention
    # envelope remains a failure once this window expires.
    post_break_grace_steps = 0

    # Command and servo limits (same conservative bounds as the sphere task).
    command_rate_limit_rad_s = 0.45
    # Keep the target command clamped, but make the post-solve pose projection
    # independently switchable.  A real PhysX joint-limit solver should
    # resolve contact-induced excursions; overwriting the measured pose at
    # every policy step can inject an impulse exactly when the fruit is
    # released.  The default remains enabled for the historical safety path;
    # evaluator/training ablations may disable it explicitly.
    joint_projection_enabled = True
    joint_projection_after_break = True
    # PhysX training must not teleport DOF positions at every substep.  That
    # shortcut injects energy when a moving finger first overlaps the fruit
    # (the reset probe measured a 0.78 m/s lateral launch).  Use the bounded
    # torque PD path so finger/object contacts are advanced by the solver.
    position_servo_enabled = False
    position_servo_rate_rad_s = 0.65
    # Diagnostic only: retain the palm insert but disable every imported hand
    # collider.  The probe uses this to distinguish reset/support faults from
    # initial finger/object overlap; training must leave it False.
    debug_support_only_collision = False
    debug_collision_digit = ""
    # Audit switch: use the vendor base-link collision without the flat
    # replacement.  This remains off until its rest/contact probe passes.
    use_original_palm_collision = False
    residual_span_fraction = 0.12
    # The four independent distal residuals need enough physical travel to
    # clear the 0.03 rad participation gate.  Keeping them at the legacy
    # 0.12 span made ordinary policy outputs move only a few hundredths of a
    # radian and recreated the proximal-only local optimum.  This wider
    # residual is still clamped by the measured safe joint limits.
    distal_residual_span_fraction = 0.25
    # New palm-supported enclosure environments opt into the strict geometry
    # gate and 14D action adapter.  The archived 10D environment keeps this
    # disabled so its old checkpoints and contract remain unchanged.
    strict_enveloping_enabled = False
    strict_enveloping_require_stem_break = False
    strict_palm_up_cos_min = 0.984807753
    strict_angular_coverage_min_deg = 200.0
    strict_thumb_opposition_cos_max = -0.35
    strict_same_hemisphere_score_max = 0.85
    strict_palm_load_share_min = 0.05
    strict_palm_load_share_max = 0.70
    strict_palm_support_streak_steps = 5
    strict_palm_first_margin_steps = 3
    strict_hold_steps = 10
    strict_distal_participation_delta_rad = 0.03
    strict_proximal_dominance_delta_rad = 0.15
    strict_reward_weight = 1.5
    strict_distal_reward_weight = 1.2

    # Per-episode simulation priors; stage 1 keeps these narrow, stage 4
    # widens them (plan §8).
    friction_range = (0.55, 0.95)
    restitution_range = (0.0, 0.06)
    stiffness_range_n_m = (15000.0, 22000.0)
    tactile_gain_range = (0.92, 1.08)
    tactile_bias_range_n = (-0.02, 0.02)
    sensor_delay_steps_max = 2
    sensor_dropout_probability = 0.02
    sensor_stale_probability = 0.02
    # Episode-fixed actuator and harvesting-trajectory uncertainty.  Neutral
    # defaults preserve stages 0--3; stage 4 widens them progressively.
    actuator_response_scale_range = (1.0, 1.0)
    actuator_bias_range_rad = (0.0, 0.0)
    actuator_deadband_range_rad = (0.0, 0.0)
    actuator_backlash_range_rad = (0.0, 0.0)
    pick_speed_range_m_s = (0.02, 0.02)
    pick_direction_tilt_max_rad = 0.0

    policy_dt_s = 1.0 / POLICY_RATE_HZ


class PhysxTomatoGraspEnv(DirectRLEnv):
    """Vectorized PhysX palm-supported tomato enclosure with stem pick."""

    cfg: PhysxTomatoGraspEnvCfg

    def __init__(self, cfg: PhysxTomatoGraspEnvCfg, render_mode: str | None = None, **kwargs):
        self.active_joint_names = tuple(name.removeprefix("left_hand_") for name in LEFT_ACTIVE_JOINT_ORDER)
        super().__init__(cfg, render_mode, **kwargs)
        if not hasattr(self, "_tomato_axes_m"):
            raise RuntimeError("tomato geometry prestartup event did not run")
        self.tomato_axes = self._tomato_axes_m.to(self.device)
        self._tomato_density_kg_m3 = self._tomato_density_kg_m3.to(self.device)
        self.active_joint_ids = [self.robot.joint_names.index(name) for name in self.active_joint_names]
        self.tip_ids = [
            self.robot.body_names.index(name)
            for name in (
                "thumb_link5",
                "index_link4",
                "middle_link4",
                "ring_link4",
                "little_link4",
            )
        ]
        self.base_id = self.robot.body_names.index("base_link")
        with torch.inference_mode(False):
            self.action_dim = int(self.cfg.action_space)
            if self.action_dim not in (ACTION_DIM, 14):
                raise ValueError("PhysxTomatoGraspEnv supports only the legacy 10D or enclosure 14D action contract")
            self.actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
            self.action_delta = torch.zeros_like(self.actions)
            self.prev_targets = torch.zeros((self.num_envs, 16), device=self.device)
            self.post_break_targets = torch.zeros((self.num_envs, 16), device=self.device)
            self.post_break_targets_latched = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.contact_forces = torch.zeros((self.num_envs, 5), device=self.device)
            self.previous_contact_forces = torch.zeros_like(self.contact_forces)
            self.palm_force = torch.zeros(self.num_envs, device=self.device)
            self.contact_counts = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
            self.peak_force = torch.zeros_like(self.contact_forces)
            self.force_exposure = torch.zeros_like(self.contact_forces)
            self.impact_exposure = torch.zeros_like(self.contact_forces)
            self.damage_score = torch.zeros(self.num_envs, device=self.device)
            self.stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.post_break_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.prepick_stable_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.prepick_best_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # Diagnostic code from the most recent done evaluation:
            # 0 running, 1 success, 2 drop, 3 severe overload, 4 damage,
            # 5 timeout.  It is deliberately not cleared by auto-reset so a
            # batch feasibility runner can classify the terminal transition.
            self.last_termination_code = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.last_drop_before_pick = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.last_drop_after_pick = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.last_drop_after_break = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.secure_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.enclosure_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.grasp_center = torch.zeros((self.num_envs, 3), device=self.device)
            task_xy = tuple(float(value) for value in self.cfg.tomato_rest_xy)
            local_support_point = torch.tensor(
                (*task_xy, self.cfg.palm_top_z_m), device=self.device
            )
            # Actor observations are palm/task-frame quantities.  Every
            # cloned environment has a different world origin, so a single
            # world-space (0, 0, z) reference silently saturated fruit-center
            # observations for envs 1..N.  Keep one world-space palm support
            # reference per clone while retaining the same local task frame.
            self.palm_support_point = (
                self.scene.env_origins.to(self.device) + local_support_point
            )
            self.position_jitter = torch.tensor(cfg.position_jitter, device=self.device)
            self._lower = torch.tensor(LEFT_SAFE_LOWER_RAD, device=self.device)
            self._upper = torch.tensor(LEFT_SAFE_UPPER_RAD, device=self.device)
            self._velocity = torch.tensor(LEFT_SAFE_VELOCITY_RAD_S, device=self.device)
            self._torque_kp = torch.tensor(
                (0.020, 0.016, 0.016, 0.016, 0.016, 0.012, 0.008, 0.008,
                 0.008, 0.008, 0.020, 0.020, 0.014, 0.014, 0.014, 0.014),
                device=self.device,
            )
            self._torque_kd = torch.tensor(
                (0.0020, 0.0015, 0.0015, 0.0015, 0.0015, 0.0010, 0.0008,
                 0.0008, 0.0008, 0.0008, 0.0020, 0.0020, 0.0012, 0.0012,
                 0.0012, 0.0012),
                device=self.device,
            )
            self._torque_limit = torch.tensor(
                (0.006, 0.006, 0.006, 0.006, 0.006, 0.004, 0.003, 0.003,
                 0.003, 0.003, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006),
                device=self.device,
            )
            self._sensor_groups = (
                ("thumb_contact", "thumb_contact_link4"),
                ("index_contact", "index_contact_link3"),
                ("middle_contact", "middle_contact_link3"),
                ("ring_contact", "ring_contact_link3"),
                ("little_contact", "little_contact_link3"),
            )
            # ContactSensor point tracking is enabled in the scene config.  A
            # per-digit cache selects the point from the link carrying the
            # larger filtered force; when a contact point is unavailable, the
            # geometry verifier falls back to the corresponding fingertip
            # body proxy instead of inventing a point.
            self.contact_points_w = self._tip_positions().detach().clone()
            self.contact_points_valid = torch.zeros(
                (self.num_envs, 5), dtype=torch.bool, device=self.device
            )
            self.palm_contact_point_w = self.palm_support_point.detach().clone()
            self.palm_contact_point_valid = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            # At least two entries so the policy-step force-rate feature has a
            # previous sample even when the delay range is reduced to zero.
            self.force_history = torch.zeros(
                (self.num_envs, max(2, cfg.sensor_delay_steps_max + 2), 5),
                device=self.device,
            )
            self.sensor_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.sensor_dropout = torch.zeros((self.num_envs, 5), dtype=torch.bool, device=self.device)
            self.sensor_stale = torch.zeros((self.num_envs, 5), dtype=torch.bool, device=self.device)
            self.tactile_gain = torch.ones((self.num_envs, 5), device=self.device)
            self.tactile_bias = torch.zeros((self.num_envs, 5), device=self.device)
            self.retention_force_ema = torch.zeros((self.num_envs, 5), device=self.device)
            self.actuator_response_scale = torch.ones((self.num_envs, 16), device=self.device)
            self.actuator_bias = torch.zeros((self.num_envs, 16), device=self.device)
            self.actuator_deadband = torch.zeros((self.num_envs, 16), device=self.device)
            self.actuator_backlash = torch.zeros((self.num_envs, 16), device=self.device)
            self.last_target_direction = torch.zeros((self.num_envs, 16), device=self.device)
            self.pick_speed = torch.full((self.num_envs,), cfg.pick_speed_m_s, device=self.device)
            self.pick_direction_xy = torch.zeros((self.num_envs, 2), device=self.device)
            self.object_mass = torch.full((self.num_envs,), 0.075, device=self.device)
            self.object_friction = torch.full((self.num_envs,), 0.70, device=self.device)
            self.object_stiffness = torch.full((self.num_envs,), 18000.0, device=self.device)
            self.anchor_rest_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.anchor_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.anchor_rest_quat = torch.zeros((self.num_envs, 4), device=self.device)
            self.anchor_rest_quat[:, 0] = 1.0
            self.anchor_yaw = torch.zeros(self.num_envs, device=self.device)
            self.stem_load_ratios = {
                name: torch.zeros(self.num_envs, device=self.device)
                for name in ("tension", "shear", "bending", "torsion", "combined")
            }
            self.stem_damage = torch.zeros(self.num_envs, device=self.device)
            self.stem_attached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.stem_broke = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.break_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_break_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.stem_limits = {
                name: torch.empty(self.num_envs, device=self.device)
                for name in ("tensile", "shear", "bending", "torsional")
            }
            self.stem_critical_duration = torch.empty(self.num_envs, device=self.device)
            self.pick_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.pick_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.pick_progress_m = torch.zeros(self.num_envs, device=self.device)
            self.break_opposition_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.secure_before_pick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.slipped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.overloaded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # Reward progress trackers ("best-ever progress" per plan §7).
            self.best_proximity = torch.zeros(self.num_envs, device=self.device)
            self.best_fingertip_proximity = torch.zeros(self.num_envs, device=self.device)
            self.best_contact_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            # Episode metric latches.
            self.metric_approach = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.metric_multi_digit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.metric_thumb_opposition = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.metric_palm_support = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.metric_secure_enclosure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.joint_safety_projection = torch.zeros(self.num_envs, device=self.device)
        self._cached_actor: torch.Tensor | None = None
        self._cached_critic: torch.Tensor | None = None
        self._tomato_material_template = self.tomato.root_physx_view.get_material_properties().clone()
        self._ceiling = torch.tensor(cfg.force_ceiling_n, device=self.device)
        if hasattr(self.robot.root_physx_view, "set_dof_stiffnesses"):
            stiffness = self.robot.root_physx_view.get_dof_stiffnesses().clone()
            damping = self.robot.root_physx_view.get_dof_dampings().clone()
            stiffness[:, self.active_joint_ids] = 0.0
            damping[:, self.active_joint_ids] = 0.0
            physx_env_ids = self.robot._ALL_INDICES.detach().cpu()
            self.robot.root_physx_view.set_dof_stiffnesses(stiffness.detach().cpu(), physx_env_ids)
            self.robot.root_physx_view.set_dof_dampings(damping.detach().cpu(), physx_env_ids)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.tomato = RigidObject(self.cfg.tomato_cfg)
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        for pattern in (
            self.cfg.tomato_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0"),
            *(
                f"/World/envs/env_0/Robot/{name}"
                for name in (
                    "thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4",
                    "thumb_link4", "index_link3", "middle_link3", "ring_link3", "little_link3",
                    "base_link",
                )
            ),
        ):
            for prim in sim_utils.find_matching_prims(pattern):
                if prim.IsValid() and not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim)
        # The imported palm collision and the flat support insert must not
        # both solve the same contact.  Their overlapping surfaces generated
        # unequal impulses and launched the ellipsoid several centimetres
        # sideways on reset.  Disable only the original base-link collision;
        # finger/link collisions remain physical and the child support cube
        # below becomes the single palm contact surface.
        palm_collision = self.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/base_link/collisions")
        if palm_collision.IsValid() and not self.cfg.use_original_palm_collision:
            UsdPhysics.CollisionAPI(palm_collision).GetCollisionEnabledAttr().Set(False)
        support_path = "/World/envs/env_0/Robot/base_link/palm_support_surface"
        support_prim = None
        if not self.cfg.use_original_palm_collision:
            support_prim = UsdGeom.Cube.Define(self.sim.stage, support_path).GetPrim()
            support_cube = UsdGeom.Cube(support_prim)
            support_cube.GetSizeAttr().Set(1.0)
            support_xform = UsdGeom.Xformable(support_prim)
            support_center_x, support_center_y = self.cfg.tomato_rest_xy
            support_xform.AddTranslateOp().Set(
                Gf.Vec3d(
                    float(support_center_x),
                    float(support_center_y),
                    self.cfg.palm_top_z_m - 0.5 * self.cfg.palm_support_surface_size_m[2],
                )
            )
            support_xform.AddScaleOp().Set(Gf.Vec3d(*self.cfg.palm_support_surface_size_m))
            UsdPhysics.CollisionAPI.Apply(support_prim)
            UsdPhysics.CollisionAPI(support_prim).GetCollisionEnabledAttr().Set(True)
            # A primitive cube does not need MeshCollisionAPI.  More importantly,
            # PhysX's inherited/default contact margin is large relative to the
            # tomato's 1 mm reset clearance.  That made the very first simulation
            # step resolve a speculative contact with a large, asymmetric impulse.
            # Give this purpose-built support surface explicit metre-scale offsets.
            support_physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(support_prim)
            support_physx_collision.CreateContactOffsetAttr().Set(
                float(self.cfg.palm_support_contact_offset_m)
            )
            support_physx_collision.CreateRestOffsetAttr().Set(
                float(self.cfg.palm_support_rest_offset_m)
            )
            rim_half_x, rim_half_y = self.cfg.palm_cup_inner_half_extent_m
            rim_thickness = self.cfg.palm_cup_rim_thickness_m
            rim_height = self.cfg.palm_cup_rim_height_m
            rim_specs = (
                ("rim_pos_x", (support_center_x + rim_half_x + 0.5 * rim_thickness, support_center_y), (rim_thickness, 2.0 * (rim_half_y + rim_thickness), rim_height)),
                ("rim_neg_x", (support_center_x - rim_half_x - 0.5 * rim_thickness, support_center_y), (rim_thickness, 2.0 * (rim_half_y + rim_thickness), rim_height)),
                ("rim_pos_y", (support_center_x, support_center_y + rim_half_y + 0.5 * rim_thickness), (2.0 * (rim_half_x + rim_thickness), rim_thickness, rim_height)),
                ("rim_neg_y", (support_center_x, support_center_y - rim_half_y - 0.5 * rim_thickness), (2.0 * (rim_half_x + rim_thickness), rim_thickness, rim_height)),
            )
            for rim_name, (rim_x, rim_y), rim_scale in rim_specs:
                rim_prim = UsdGeom.Cube.Define(
                    self.sim.stage, f"{support_path}_{rim_name}"
                ).GetPrim()
                UsdGeom.Cube(rim_prim).GetSizeAttr().Set(1.0)
                rim_xform = UsdGeom.Xformable(rim_prim)
                rim_xform.AddTranslateOp().Set(
                    Gf.Vec3d(
                        float(rim_x),
                        float(rim_y),
                        self.cfg.palm_top_z_m + 0.5 * rim_height,
                    )
                )
                rim_xform.AddScaleOp().Set(Gf.Vec3d(*rim_scale))
                UsdPhysics.CollisionAPI.Apply(rim_prim)
                rim_physx = PhysxSchema.PhysxCollisionAPI.Apply(rim_prim)
                rim_physx.CreateContactOffsetAttr().Set(
                    float(self.cfg.palm_support_contact_offset_m)
                )
                rim_physx.CreateRestOffsetAttr().Set(
                    float(self.cfg.palm_support_rest_offset_m)
                )
                UsdGeom.Imageable(rim_prim).MakeInvisible()
        if self.cfg.debug_support_only_collision or bool(getattr(self.cfg, "proxy_only_collision", False)):
            robot_root = self.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")
            for prim in self.sim.stage.Traverse():
                path = str(prim.GetPath())
                is_palm_support = path == support_path or path.startswith(support_path + "_")
                if (
                    path.startswith(str(robot_root.GetPath()) + "/")
                    and not is_palm_support
                    and prim.HasAPI(UsdPhysics.CollisionAPI)
                ):
                    # Proxy-only isolation intentionally leaves all imported
                    # hand colliders disabled; the v2 proxy authoring event
                    # re-enables only its explicit fingertip spheres.  This
                    # avoids conflating a vendor link4 collision box with the
                    # calibrated-in-progress proxy during geometry audits.
                    if bool(getattr(self.cfg, "proxy_only_collision", False)):
                        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(
                            path.endswith("/v2_fingertip_proxy")
                        )
                        continue
                    # The single-digit contact audit must not retain that
                    # finger's link3/proximal collider: it would alter the
                    # object trajectory before link4 can be measured.  Keep
                    # only the selected link4 collision group; the v2 proxy
                    # authoring applies the same filter to its sphere.
                    # The straight fingers use link4, but the thumb's
                    # physical fingertip is link5 in the imported L20 USD.
                    # Keep thumb_link5 in the simulation-only support audit;
                    # otherwise ``debug_collision_digit=all`` silently
                    # disables the thumb and makes five-finger contact
                    # impossible.  The default (debug flag off) path is
                    # unchanged and archived v1 artifacts are not modified.
                    keep_link4 = any(
                        f"/{name}_link4" in path
                        for name in ("index", "middle", "ring", "little")
                    ) or "/thumb_link5" in path
                    keep_digit = keep_link4 and (
                        self.cfg.debug_collision_digit == "all" or
                        (bool(self.cfg.debug_collision_digit) and
                         f"/{self.cfg.debug_collision_digit}_link4" in path)
                    )
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(keep_digit)
        # Keep the insert out of rendered observations; it is a conservative
        # collision representation of the palm surface, not a capture assist.
        if support_prim is not None:
            UsdGeom.Imageable(support_prim).MakeInvisible()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["tomato"] = self.tomato
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    @torch.inference_mode()
    def _read_contact_forces(self) -> torch.Tensor:
        forces = []
        counts = []
        selected_points = []
        selected_valid = []
        for names in self._sensor_groups:
            channel = []
            points = []
            point_valid = []
            for name in names:
                sensor = self.scene.sensors[name]
                force_vector = sensor.data.net_forces_w[:, 0]
                force = torch.linalg.vector_norm(force_vector, dim=-1)
                channel.append(force.nan_to_num(0.0).clamp_min(0.0))
                contact_pos = getattr(sensor.data, "contact_pos_w", None)
                if contact_pos is None:
                    point = torch.zeros((self.num_envs, 3), device=self.device)
                    valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                else:
                    # The tomato is the sole filtered shape for these
                    # sensors.  A contact manifold can contain several
                    # points; averaging all finite points is less sensitive
                    # to PhysX's arbitrary manifold ordering than taking
                    # ``[..., 0]`` and gives the geometry verifier a stable
                    # measured point while preserving the force channel.
                    raw_points = contact_pos[:, 0].to(self.device)
                    finite = torch.isfinite(raw_points).all(-1)
                    count = finite.sum(-1).clamp_min(1).unsqueeze(-1)
                    point = torch.where(finite.unsqueeze(-1), raw_points, torch.zeros_like(raw_points)).sum(-2) / count
                    valid = (finite.any(-1)) & (force > self.cfg.contact_threshold_n)
                    point = torch.nan_to_num(point, nan=0.0)
                points.append(point)
                point_valid.append(valid)
            channel = torch.stack(channel, dim=-1)
            channel_force, max_idx = channel.max(dim=-1)
            point_stack = torch.stack(points, dim=1)
            valid_stack = torch.stack(point_valid, dim=1)
            chosen_point = point_stack[torch.arange(self.num_envs, device=self.device), max_idx]
            chosen_valid = valid_stack[torch.arange(self.num_envs, device=self.device), max_idx]
            selected_points.append(chosen_point)
            selected_valid.append(chosen_valid & (channel_force > self.cfg.contact_threshold_n))
            forces.append(channel_force)
            counts.append((channel > self.cfg.contact_threshold_n).float().sum(dim=-1))
        forces = torch.stack(forces, dim=-1)
        self.contact_points_w = torch.stack(selected_points, dim=1)
        self.contact_points_valid = torch.stack(selected_valid, dim=-1)
        palm = self.scene.sensors["palm_contact"]
        palm_force = torch.linalg.vector_norm(palm.data.net_forces_w[:, 0], dim=-1)
        palm_force = palm_force.nan_to_num(0.0).clamp_min(0.0)
        palm_point = getattr(palm.data, "contact_pos_w", None)
        if palm_point is not None:
            raw_palm_points = palm_point[:, 0].to(self.device)
            finite = torch.isfinite(raw_palm_points).all(-1)
            count = finite.sum(-1).clamp_min(1).unsqueeze(-1)
            palm_point = torch.where(
                finite.unsqueeze(-1), raw_palm_points, torch.zeros_like(raw_palm_points)
            ).sum(-2) / count
            palm_valid = finite.any(-1) & (palm_force > self.cfg.contact_threshold_n)
            self.palm_contact_point_w = torch.nan_to_num(palm_point, nan=0.0)
            self.palm_contact_point_valid = palm_valid
        else:
            self.palm_contact_point_valid.zero_()
        self.previous_contact_forces = self.contact_forces
        self.contact_forces = forces
        self.palm_force = palm_force
        self.contact_counts = torch.cat((torch.stack(counts, dim=-1), (palm_force > self.cfg.contact_threshold_n).float().unsqueeze(-1)), dim=-1)
        self.force_history = torch.roll(self.force_history, shifts=1, dims=1)
        self.force_history[:, 0] = forces
        return forces

    def _geometry_contact_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return measured contact points with an explicit fingertip fallback."""
        tips = self._tip_positions()
        valid = self.contact_points_valid & (self.contact_forces >= self.cfg.contact_threshold_n)
        points = torch.where(valid.unsqueeze(-1), self.contact_points_w, tips)
        return points, valid

    def _delayed_tactile_force(self) -> torch.Tensor:
        rows = torch.arange(self.num_envs, device=self.device)
        delayed = self.force_history[rows, self.sensor_delay]
        stale_index = (self.sensor_delay + 1).clamp_max(self.force_history.shape[1] - 1)
        stale_value = self.force_history[rows, stale_index]
        delayed = torch.where(self.sensor_stale, stale_value, delayed)
        delayed = delayed * self.tactile_gain + self.tactile_bias
        delayed = delayed.clamp_min(0.0)
        delayed = torch.where(self.sensor_dropout, torch.zeros_like(delayed), delayed)
        return delayed

    def _opposition_mask(self, forces: torch.Tensor) -> torch.Tensor:
        """Thumb opposition plus at least two other digits above support force."""
        active = forces >= self.cfg.support_force_n
        return active[:, 0] & (active[:, 1:].sum(dim=-1) >= 2)

    @torch.inference_mode()
    def _apply_stem_wrench(self) -> None:
        """Advance the pick trajectory and apply the stem spring-damper wrench.

        Called every PhysX substep.  The anchor rises at a fixed slow speed
        while the enclosure is secure and the stem is still attached, then
        stops; the policy continues to command the fingers throughout.
        """

        if self.cfg.training_stage < 2:
            # Stage 0/1 isolates contact and enclosure learning.  Keeping the
            # virtual stem dormant avoids an unobserved tether torque during
            # settling and makes the stage-1 success signal depend only on the
            # palm and five digit contacts.  Stage 2 enables the same wrench
            # and pick trajectory without changing the actor/critic contract.
            self.anchor_pos = self.anchor_rest_pos.clone()
            for values in self.stem_load_ratios.values():
                values.zero_()
            self.stem_damage.zero_()
            return

        tomato_pos = self.tomato.data.root_pos_w
        tomato_quat = self.tomato.data.root_quat_w
        tomato_lin_vel = getattr(self.tomato.data, "root_lin_vel_w", torch.zeros_like(tomato_pos))
        tomato_ang_vel = getattr(self.tomato.data, "root_ang_vel_w", torch.zeros_like(tomato_pos))
        picking = (
            (self.cfg.training_stage >= 2)
            & self.pick_started
            & self.stem_attached
            & ~self.pick_done
        )
        if picking.any():
            progress = torch.where(
                picking,
                (self.pick_progress_m + self.pick_speed * self.physics_dt).clamp_max(self.cfg.pick_displacement_m),
                self.pick_progress_m,
            )
            self.pick_done |= picking & (progress >= self.cfg.pick_displacement_m)
            self.pick_progress_m = progress
            self.anchor_yaw += torch.where(
                picking,
                self.cfg.pick_twist_rate_rad_s * self.physics_dt,
                torch.zeros_like(self.anchor_yaw),
            )
            self.anchor_yaw.clamp_max_(self.cfg.pick_twist_angle_rad)
        self.anchor_pos = self.anchor_rest_pos + torch.cat(
            (
                self.pick_direction_xy * self.pick_progress_m.unsqueeze(-1),
                self.cfg.pick_direction_z_sign * self.pick_progress_m.unsqueeze(-1),
            ),
            dim=-1,
        )
        yaw_quat = torch.stack(
            (
                torch.cos(self.anchor_yaw / 2.0),
                torch.zeros_like(self.anchor_yaw),
                torch.zeros_like(self.anchor_yaw),
                torch.sin(self.anchor_yaw / 2.0),
            ),
            dim=-1,
        )
        anchor_quat = quat_mul_wxyz(yaw_quat, self.anchor_rest_quat)

        inertia = (
            0.4 * self.object_mass * torch.square(self.tomato_axes).mean(dim=-1)
        ).clamp_min(1.0e-8)
        gravity_w = tomato_pos.new_tensor(tuple(self.cfg.sim.gravity)).expand_as(tomato_pos)
        if not self.cfg.stem_gravity_compensation:
            gravity_w = torch.zeros_like(gravity_w)
        force, torque, attachment_pos, pull_axis = stem_attachment_wrench(
            mass_kg=self.object_mass,
            inertia_kg_m2=inertia,
            axes_m=self.tomato_axes,
            tomato_pos_w=tomato_pos,
            tomato_quat_wxyz=tomato_quat,
            tomato_lin_vel_w=tomato_lin_vel,
            tomato_ang_vel_w=tomato_ang_vel,
            anchor_pos_w=self.anchor_pos,
            anchor_quat_wxyz=anchor_quat,
            omega_linear_rad_s=self.cfg.stem_linear_natural_frequency_rad_s,
            zeta_linear=self.cfg.stem_linear_damping_ratio,
            omega_angular_rad_s=self.cfg.stem_angular_natural_frequency_rad_s,
            zeta_angular=self.cfg.stem_angular_damping_ratio,
            maximum_force_n=self.cfg.maximum_stem_force_n,
            maximum_torque_nm=self.cfg.maximum_stem_torque_nm,
            gravity_w=gravity_w,
        )
        # Before picking the real fruit is still attached to the plant.  Use a
        # very weak compliant tether to prevent free-sphere roll-off without
        # letting the stem carry the palm load.  At the verified pick trigger
        # the current attachment frame is re-latched at zero strain and the
        # full stem wrench becomes active.
        wrench_scale = torch.where(
            self.pick_started,
            torch.ones_like(self.stem_damage),
            torch.full_like(self.stem_damage, self.cfg.pregrasp_stem_wrench_scale),
        )
        wrench_scale = wrench_scale * self.stem_attached.float()
        force = force * wrench_scale.unsqueeze(-1)
        torque = torque * wrench_scale.unsqueeze(-1)
        self.tomato.root_physx_view.apply_forces_and_torques_at_position(
            force_data=force,
            torque_data=torque,
            position_data=None,
            indices=self.tomato._ALL_INDICES,
            is_global=True,
        )

        limits = self.stem_limits
        components = stem_load_components(
            force,
            torque,
            pull_axis,
            limits["tensile"],
            limits["shear"],
            limits["bending"],
            limits["torsional"],
        )
        for name, values in components.items():
            self.stem_load_ratios[name].copy_(values)
        # Contact transients while the fruit is settling are attachment
        # reaction loads, not picking damage.  Damage accumulation starts only
        # after the verifier has latched a stable enclosure and the pick phase
        # has begun; otherwise a free-fall/impact spike breaks the stem before
        # the hand ever gets a chance to close.
        damage_load = torch.where(
            self.pick_started,
            components["combined"],
            torch.zeros_like(components["combined"]),
        )
        damage, break_now = update_stem_damage(
            self.stem_damage,
            damage_load,
            dt_s=self.physics_dt,
            critical_duration_s=self.stem_critical_duration,
            onset_ratio=self.cfg.stem_damage_onset_ratio,
            recovery_time_s=self.cfg.stem_damage_recovery_time_s,
            attached=self.stem_attached,
        )
        self.stem_damage = damage
        if bool(break_now.any()):
            self.stem_broke |= break_now
            self.break_event |= break_now
            self.stem_attached &= ~break_now
            self.break_opposition_ok[break_now] = self._opposition_mask(
                torch.maximum(self.contact_forces, self.force_history.amax(dim=1))
            )[break_now]
            self.secure_before_pick[break_now] = self.pick_started[break_now] & self.secure_latched[break_now]

    def _sample_stem_limits(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        tensile = torch.empty(n, device=self.device).uniform_(
            *self.cfg.stem_tensile_limit_n
        )
        self.stem_limits["tensile"][env_ids] = tensile
        self.stem_limits["shear"][env_ids] = tensile * self.cfg.stem_shear_to_tension_ratio
        self.stem_limits["bending"][env_ids] = tensile * self.cfg.stem_bending_lever_m
        self.stem_limits["torsional"][env_ids] = (
            tensile * self.cfg.stem_bending_lever_m * self.cfg.stem_torsion_to_bending_ratio
        )
        self.stem_critical_duration[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.stem_critical_duration_s
        )

    @torch.inference_mode()
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        bounded = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self.action_delta = bounded - self.actions
        self.actions = bounded
        settle_steps = int(getattr(self.cfg, "pregrasp_settle_steps", 0))
        if settle_steps > 0:
            settle = self.episode_length_buf < settle_steps
            if bool(settle.any()):
                ids = torch.nonzero(settle, as_tuple=False).flatten()
                transforms = self.tomato.root_physx_view.get_transforms().clone()
                transform_ids = ids.to(device=transforms.device)
                target_z = (
                    self.scene.env_origins[ids, 2]
                    + float(self.cfg.palm_top_z_m)
                    + self.tomato_axes[ids, 2]
                    + float(self.cfg.palm_support_rest_offset_m)
                )
                transforms[transform_ids, 2] = target_z.to(device=transforms.device)
                self.tomato.root_physx_view.set_transforms(transforms, indices=transform_ids)
                velocities = self.tomato.root_physx_view.get_velocities().clone()
                velocities[transform_ids] = 0.0
                self.tomato.root_physx_view.set_velocities(velocities, indices=transform_ids)

    @torch.inference_mode()
    def _apply_action(self) -> None:
        if self.action_dim == ACTION_DIM:
            desired = hybrid_action_to_joint_targets(
                self.actions,
                residual_span_fraction=float(getattr(self.cfg, "residual_span_fraction", 0.12)),
            )
        else:
            desired = enveloping_action_to_joint_targets(
                self.actions,
                residual_span_fraction=float(getattr(self.cfg, "residual_span_fraction", 0.12)),
                distal_residual_span_fraction=float(
                    getattr(self.cfg, "distal_residual_span_fraction", 0.25)
                ),
                per_digit_root_residual_span_fraction=(
                    None
                    if getattr(self.cfg, "per_digit_root_residual_span_fraction", None) is None
                    else float(self.cfg.per_digit_root_residual_span_fraction)
                ),
            )
        if self.cfg.tactile_reflex_enabled and self.action_dim == 14:
            # Use the latest available tactile sample.  The correction is
            # bounded in joint space and applied only after the pick has
            # started, so approach contacts cannot pull a finger prematurely.
            force = self._delayed_tactile_force()
            target_force = force.new_tensor(self.cfg.tactile_reflex_target_force_n)
            high_force = force.new_tensor(self.cfg.tactile_reflex_high_force_n)
            error = (target_force - force).clamp(-2.5, 2.5)
            correction = (self.cfg.tactile_reflex_gain_rad_per_n * error).clamp(
                -self.cfg.tactile_reflex_max_correction_rad,
                self.cfg.tactile_reflex_max_correction_rad,
            )
            correction = correction * self.stem_broke.float().unsqueeze(-1)
            # A positive correction curls each fingertip; force above the
            # high threshold receives the same signed negative correction.
            correction = torch.where(
                force > high_force,
                -self.cfg.tactile_reflex_gain_rad_per_n
                * (force - high_force).clamp(0.0, 2.5),
                correction,
            ).clamp(
                -self.cfg.tactile_reflex_max_correction_rad,
                self.cfg.tactile_reflex_max_correction_rad,
            )
            for digit_index, joint_index in enumerate((11, 12, 13, 14, 15)):
                desired[:, joint_index] = desired[:, joint_index] + correction[:, digit_index]
        if self.cfg.adaptive_retention_reflex_enabled and self.action_dim == 14:
            # This correction is deliberately applied before the ordinary
            # rate-limited target path.  It cannot teleport a joint or bypass
            # the calibrated limits, and all five digits use their own force
            # channel rather than a shared pinch command.
            sensed = self._delayed_tactile_force()
            alpha = float(self.cfg.adaptive_retention_filter_alpha)
            self.retention_force_ema = (
                alpha * self.retention_force_ema + (1.0 - alpha) * sensed
            )
            center_distance = torch.linalg.vector_norm(
                self.tomato.data.root_pos_w - self.grasp_center, dim=-1
            )
            in_hand = center_distance <= self.cfg.max_center_distance_m
            palm_support = self.palm_force >= self.cfg.palm_support_force_n
            correction = adaptive_retention_correction(
                filtered_force_n=self.retention_force_ema,
                target_force_n=sensed.new_tensor(self.cfg.adaptive_retention_target_force_n),
                high_force_n=sensed.new_tensor(self.cfg.adaptive_retention_high_force_n),
                relative_speed_m_s=self._relative_speed(),
                in_hand=in_hand,
                palm_support=palm_support,
                contact_threshold_n=self.cfg.contact_threshold_n,
                force_gain_rad_per_n=float(self.cfg.adaptive_retention_force_gain_rad_per_n),
                slip_gain_rad_per_m_s=float(self.cfg.adaptive_retention_slip_gain_rad_per_m_s),
                slip_threshold_m_s=float(self.cfg.adaptive_retention_slip_threshold_m_s),
                max_correction_rad=float(self.cfg.adaptive_retention_max_correction_rad),
            )
            correction = correction * self.stem_broke.float().unsqueeze(-1)
            for digit_index, joint_index in enumerate((11, 12, 13, 14, 15)):
                desired[:, joint_index] = desired[:, joint_index] + correction[:, digit_index]
        if self.cfg.post_break_target_latch:
            new_break = self.stem_broke & ~self.post_break_targets_latched
            if bool(new_break.any()):
                self.post_break_targets[new_break] = self.prev_targets[new_break]
                self.post_break_targets_latched[new_break] = True
            post_break = self.stem_broke & self.post_break_targets_latched
            if bool(post_break.any()):
                mix = float(self.cfg.post_break_command_mix)
                desired = torch.where(
                    post_break.unsqueeze(-1),
                    self.post_break_targets * (1.0 - mix) + desired * mix,
                    desired,
                )
        desired = desired + self.actuator_bias
        requested = desired - self.prev_targets
        direction = requested.sign()
        reversing = (direction != 0.0) & (self.last_target_direction != 0.0) & (direction != self.last_target_direction)
        # Backlash is represented as lost target travel on direction reversal;
        # deadband removes sub-resolution commands. Both are episode-fixed.
        requested = torch.where(
            reversing,
            direction * (requested.abs() - self.actuator_backlash).clamp_min(0.0),
            requested,
        )
        requested = torch.where(requested.abs() < self.actuator_deadband, torch.zeros_like(requested), requested)
        self.last_target_direction = torch.where(direction != 0.0, direction, self.last_target_direction)
        max_delta = (
            self._velocity.clamp_max(self.cfg.command_rate_limit_rad_s)
            * self.cfg.policy_dt_s
            * self.actuator_response_scale
        )
        delta = requested.clamp(-max_delta, max_delta)
        targets = (self.prev_targets + delta).maximum(self._lower).minimum(self._upper)
        self.prev_targets = targets
        dof_pos = self.robot.root_physx_view.get_dof_positions()[:, self.active_joint_ids].to(self.device)
        dof_vel = self.robot.root_physx_view.get_dof_velocities()[:, self.active_joint_ids].to(self.device)
        invalid = (dof_pos < self._lower) | (dof_pos > self._upper)
        if not self.cfg.joint_projection_after_break:
            invalid = invalid & (~self.stem_broke).unsqueeze(-1)
        if self.cfg.joint_projection_enabled and invalid.any():
            full_pos = self.robot.root_physx_view.get_dof_positions().clone()
            full_vel = self.robot.root_physx_view.get_dof_velocities().clone()
            corrected = dof_pos.maximum(self._lower).minimum(self._upper)
            full_pos[:, self.active_joint_ids] = corrected.to(full_pos.device)
            full_vel[:, self.active_joint_ids] = torch.where(
                invalid, torch.zeros_like(dof_vel), dof_vel
            ).to(full_vel.device)
            self.robot.root_physx_view.set_dof_positions(full_pos, self.robot._ALL_INDICES)
            self.robot.root_physx_view.set_dof_velocities(full_vel, self.robot._ALL_INDICES)
            self.joint_safety_projection += invalid.float().sum(dim=-1)
            dof_pos = corrected
            dof_vel = full_vel[:, self.active_joint_ids].to(self.device)
        if self.cfg.position_servo_enabled:
            servo_step = self._velocity.clamp_max(self.cfg.position_servo_rate_rad_s) * self.physics_dt
            next_pos = dof_pos + (targets - dof_pos).clamp(-servo_step, servo_step)
            full_pos = self.robot.root_physx_view.get_dof_positions().clone()
            full_vel = self.robot.root_physx_view.get_dof_velocities().clone()
            full_pos[:, self.active_joint_ids] = next_pos.to(full_pos.device)
            full_vel[:, self.active_joint_ids] = ((next_pos - dof_pos) / self.physics_dt).to(full_vel.device)
            self.robot.root_physx_view.set_dof_positions(full_pos, self.robot._ALL_INDICES)
            self.robot.root_physx_view.set_dof_velocities(full_vel, self.robot._ALL_INDICES)
            self.robot.root_physx_view.set_dof_position_targets(full_pos, self.robot._ALL_INDICES)
            self.robot.set_joint_effort_target(torch.zeros_like(targets), joint_ids=self.active_joint_ids)
        else:
            self.robot.set_joint_position_target(targets, joint_ids=self.active_joint_ids)
        self._apply_stem_wrench()

    def _tip_positions(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.tip_ids]

    @torch.inference_mode()
    def _sync_servo_pose_after_physics(self) -> None:
        if not self.cfg.position_servo_enabled:
            return
        # Do not teleport the articulation after a PhysX solve.  The old
        # implementation overwrote measured positions with ``prev_targets``
        # here, which could place a fingertip inside the tomato at the end of
        # every policy step and create artificial 20--50 N impulses.  The
        # smooth position/velocity servo in ``_apply_action`` already advances
        # the joints on every physics substep; this hook only refreshes the
        # target buffer and leaves the solved pose untouched.
        full_pos = self.robot.root_physx_view.get_dof_positions().clone()
        full_pos[:, self.active_joint_ids] = self.prev_targets.to(full_pos.device)
        # Position targets are written without overwriting the current pose;
        # the next substep's bounded servo performs the actual motion.
        self.robot.root_physx_view.set_dof_position_targets(full_pos, self.robot._ALL_INDICES)
        self.robot.update(dt=1.0e-6)

    @torch.inference_mode()
    def _get_observations(self) -> dict[str, torch.Tensor]:
        tomato_pos = self.tomato.data.root_pos_w
        tomato_quat = self.tomato.data.root_quat_w
        tomato_lin_vel = getattr(self.tomato.data, "root_lin_vel_w", torch.zeros_like(tomato_pos))
        tomato_ang_vel = getattr(self.tomato.data, "root_ang_vel_w", torch.zeros_like(tomato_pos))
        center = ((tomato_pos - self.palm_support_point) / self.cfg.grasp_reference_scale_m).clamp(-1.0, 1.0)
        axes = ((self.tomato_axes - 0.025) / 0.012).clamp(-1.0, 1.0)
        orientation = quat_to_rot_matrix_6d(tomato_quat)
        twist = torch.cat(((tomato_lin_vel / 0.20).clamp(-1.0, 1.0), (tomato_ang_vel / 4.0).clamp(-1.0, 1.0)), dim=-1)
        q = self.robot.data.joint_pos[:, self.active_joint_ids]
        qd = self.robot.data.joint_vel[:, self.active_joint_ids]
        q_norm = (2.0 * (q - self._lower) / (self._upper - self._lower) - 1.0).clamp(-1.0, 1.0)
        qd_norm = (qd / self._velocity.clamp_min(1.0e-3)).clamp(-1.0, 1.0)
        force = self._delayed_tactile_force()
        force_rate = ((force - self.force_history[:, 1]) / self.cfg.policy_dt_s).clamp(-20.0, 20.0)
        finger_index = torch.arange(5, device=self.device, dtype=force.dtype).unsqueeze(0).expand(self.num_envs, -1)
        total = (force / 6.0).clamp(0.0, 1.0)
        tactile = torch.stack(
            (
                total,
                total,
                (total / 0.15).clamp(0.0, 1.0),
                torch.full_like(total, 0.5),
                ((finger_index + 0.5) / 5.0),
                torch.full_like(total, 0.20),
                torch.full_like(total, 0.20),
                (force_rate / 20.0).clamp(-1.0, 1.0),
                (force > 0.05).float(),
                (self.sensor_stale | self.sensor_dropout).float(),
            ),
            dim=-1,
        ).reshape(self.num_envs, -1)
        # Stage-2 harvesting needs an explicit phase signal: after the stem
        # starts moving, the same tactile pattern can mean either “continue
        # closing” or “preserve the enclosure while the fruit is lifted”.
        # The legacy tactile contract contains four simulator placeholder
        # channels (centroid/spread proxies) that carried constants in this
        # scene.  Reuse those channels only for stage 2, preserving the 110D
        # shape and the stage-1/legacy checkpoint contract.
        if self.cfg.training_stage >= 2:
            phase = torch.stack(
                (
                    (self.prepick_stable_steps.float() / max(float(self.cfg.prepick_hold_steps), 1.0)).clamp(0.0, 1.0),
                    self.pick_started.float(),
                    self.stem_broke.float(),
                    (self.pick_progress_m / max(float(self.cfg.pick_displacement_m), 1.0e-6)).clamp(0.0, 1.0),
                ),
                dim=-1,
            )
            phase_digit = phase.unsqueeze(1).expand(-1, 5, -1)
            tactile_matrix = tactile.reshape(self.num_envs, 5, 10)
            tactile_matrix[:, :, 3:7] = phase_digit
            tactile = tactile_matrix.reshape(self.num_envs, -1)
        actor = pack_actor_observation(
            fruit_center_in_palm=center,
            fruit_semi_axes=axes,
            fruit_orientation_6d=orientation,
            fruit_twist_in_palm=twist,
            hand_joint_position=q_norm,
            hand_joint_velocity=qd_norm,
            tactile_matrix_features=tactile,
            # The deployable observation contract remains 110D.  For the new
            # 14D policy the four distal commands are observable through the
            # measured joint state/tactile channels; the legacy 10D prefix is
            # retained here to keep the actor layout explicit and compact.
            previous_action=self.actions[..., :ACTION_DIM],
        )
        if not torch.isfinite(actor).all():
            print("[physx-tomato] non-finite actor sample masked", flush=True)
        actor = torch.nan_to_num(actor, nan=0.0, posinf=1.0, neginf=-1.0)
        self._cached_actor = actor
        return {"policy": actor}

    @torch.inference_mode()
    def _get_states(self) -> torch.Tensor:
        """Return the 163D tomato Critic state with simulator truth."""
        if self._cached_actor is None:
            self._get_observations()
        tomato_pos = self.tomato.data.root_pos_w
        tomato_vel = getattr(self.tomato.data, "root_lin_vel_w", torch.zeros_like(tomato_pos))
        tips, _ = self._geometry_contact_points()
        contact_points = tips - tomato_pos.unsqueeze(1)
        palm_contact_point = torch.zeros((self.num_envs, 3), device=self.device)
        palm_sensor = self.scene.sensors["palm_contact"]
        if self.palm_contact_point_valid.any():
            point = self.palm_contact_point_w - tomato_pos
            palm_contact_point = torch.where(
                self.palm_contact_point_valid.unsqueeze(-1), point, palm_contact_point
            ).nan_to_num(0.0)
        peak_indentation = (self.contact_forces / self.object_stiffness.unsqueeze(-1)).amax(dim=-1)
        mean_exposure = self.force_exposure.mean(dim=-1)
        mean_impact = self.impact_exposure.mean(dim=-1)
        critic = pack_tomato_critic_state(
            self._cached_actor,
            physical_fingertip_force=(self.contact_forces / 6.0).clamp(0.0, 2.0),
            physical_palm_force=(self.palm_force / 8.0).clamp(0.0, 2.0).unsqueeze(-1),
            contact_points_in_palm=(contact_points / 0.08).clamp(-2.0, 2.0).flatten(-2),
            palm_contact_point_in_palm=(palm_contact_point / 0.08).clamp(-2.0, 2.0),
            contact_counts=(self.contact_counts / 2.0).clamp(0.0, 2.0),
            object_mass=(self.object_mass / 0.30).clamp(0.0, 2.0).unsqueeze(-1),
            static_friction=self.object_friction.clamp(0.0, 2.0).unsqueeze(-1),
            effective_stiffness=(self.object_stiffness / 9000.0).clamp(0.0, 2.0).unsqueeze(-1),
            peak_indentation=(peak_indentation / 0.010).clamp(0.0, 2.0).unsqueeze(-1),
            force_exposure=(mean_exposure / 3.0).clamp(0.0, 2.0).unsqueeze(-1),
            impact_exposure=(mean_impact / 10.0).clamp(0.0, 2.0).unsqueeze(-1),
            damage_score=self.damage_score.clamp(0.0, 1.0).unsqueeze(-1),
            stem_loads=torch.stack(
                tuple(self.stem_load_ratios[name] for name in ("tension", "shear", "bending", "torsion")),
                dim=-1,
            ).clamp(0.0, 2.0),
            stem_damage=self.stem_damage.clamp(0.0, 1.0).unsqueeze(-1),
            slip_speed=(self._relative_speed() / 0.50).clamp(0.0, 2.0).unsqueeze(-1),
            tactile_latent_gain=((self.tactile_gain - 1.0) / 0.18).clamp(-1.0, 1.0),
            tactile_latent_bias=(self.tactile_bias / 0.03).clamp(-1.0, 1.0),
        )
        if not torch.isfinite(critic).all():
            print("[physx-tomato] non-finite critic sample masked", flush=True)
        self._cached_critic = torch.nan_to_num(critic, nan=0.0, posinf=1.0, neginf=-1.0)
        return self._cached_critic

    def _relative_speed(self) -> torch.Tensor:
        tomato_vel = getattr(self.tomato.data, "root_lin_vel_w", torch.zeros((self.num_envs, 3), device=self.device))
        palm_vel = getattr(self.robot.data, "body_lin_vel_w", torch.zeros((self.num_envs, 3), device=self.device))[:, self.base_id]
        return torch.linalg.vector_norm(tomato_vel - palm_vel, dim=-1)

    @torch.inference_mode()
    def _get_rewards(self) -> torch.Tensor:
        # Read exactly once after the final physics substep.  Reading in
        # ``_apply_action`` and again in observations used to advance the
        # force history two or three times per policy step, corrupting tactile
        # derivatives and delayed channels.
        self._read_contact_forces()
        self._sync_servo_pose_after_physics()
        tomato_pos = self.tomato.data.root_pos_w
        center_distance = torch.linalg.vector_norm(tomato_pos - self.grasp_center, dim=-1)
        speed = self._relative_speed()
        contact = self.contact_forces >= self.cfg.contact_threshold_n
        window_forces = torch.maximum(self.contact_forces, self.force_history.amax(dim=1))
        window_contact = window_forces >= self.cfg.contact_threshold_n
        secure_now = window_contact[:, 0] & (window_contact[:, 1:].sum(dim=-1) >= 2)
        self.secure_latched |= secure_now
        digit_count = window_contact.sum(dim=-1)
        palm_support = self.palm_force >= self.cfg.palm_support_force_n
        enclosure_now = (digit_count >= int(self.cfg.minimum_enclosing_digits)) & palm_support
        self.enclosure_latched |= enclosure_now
        # Damage diagnostics (per-digit peak/exposure/impact, mirroring the
        # sphere task).
        self.peak_force = torch.maximum(self.peak_force, self.contact_forces)
        self.force_exposure += (self.contact_forces - 0.65 * self._ceiling).clamp_min(0.0) * self.step_dt
        force_rate = ((self.contact_forces - self.previous_contact_forces) / self.step_dt).clamp_min(0.0)
        self.impact_exposure += (force_rate - 30.0).clamp_min(0.0) * self.step_dt
        peak_ratio = ((self.peak_force - 0.65 * self._ceiling) / (0.60 * self._ceiling)).clamp(0.0, 1.0)
        exposure_ratio = (self.force_exposure / (0.50 * self._ceiling)).clamp(0.0, 1.0)
        impact_ratio = (self.impact_exposure / (1.50 * self._ceiling)).clamp(0.0, 1.0)
        self.damage_score = torch.maximum(
            self.damage_score,
            (0.50 * peak_ratio + 0.35 * exposure_ratio + 0.15 * impact_ratio).amax(dim=-1),
        )

        force_safe = (self.contact_forces <= self.cfg.safe_force_n).all(dim=-1) & (self.palm_force <= 1.25 * self.cfg.safe_force_n)
        self.overloaded |= ~force_safe
        in_hand = center_distance <= self.cfg.max_center_distance_m
        support = self._opposition_mask(window_forces)
        post_break_support = window_contact[:, 0] & (
            (window_contact[:, 1:].sum(dim=-1) >= 2) | palm_support
        )
        retained_support = torch.where(self.stem_attached, support, post_break_support)
        retained = (
            self.stem_broke
            & in_hand
            & (speed <= self.cfg.max_relative_speed_m_s)
            & retained_support
            & force_safe
            & (self.damage_score <= self.cfg.damage_limit)
        )
        stage1_retained = (
            (digit_count >= int(self.cfg.minimum_enclosing_digits))
            & palm_support
            & in_hand
            & (speed <= self.cfg.max_relative_speed_m_s)
            & ~self.dropped
            & force_safe
            & (self.damage_score <= self.cfg.damage_limit)
        )
        if self.cfg.training_stage == 1:
            retained = stage1_retained
        self.stable_steps = torch.where(retained, self.stable_steps + 1, torch.zeros_like(self.stable_steps))
        self.post_break_steps = torch.where(
            self.stem_broke,
            self.post_break_steps + 1,
            torch.zeros_like(self.post_break_steps),
        )

        # Require a sustained, safe five-digit enclosure before starting the
        # fixed harvest trajectory.  A one-frame contact transient must not
        # pull the plant anchor while a finger is still approaching.
        prepick_ok = stage1_retained & self.stem_attached & ~self.pick_started
        self.prepick_stable_steps = torch.where(
            prepick_ok,
            self.prepick_stable_steps + 1,
            torch.where(
                self.pick_started,
                self.prepick_stable_steps,
                torch.zeros_like(self.prepick_stable_steps),
            ),
        )
        self.prepick_best_steps = torch.maximum(
            self.prepick_best_steps, self.prepick_stable_steps
        )
        delayed_recovery_ready = (
            (self.episode_length_buf >= int(self.cfg.prepick_recovery_after_steps))
            & (self.prepick_best_steps >= int(self.cfg.prepick_recovery_min_steps))
            & stage1_retained
        )
        ready_to_pick = (
            (self.cfg.training_stage >= 2)
            & (
                (self.prepick_stable_steps >= int(self.cfg.prepick_hold_steps))
                | delayed_recovery_ready
            )
            & ~self.pick_started
        )
        if bool(ready_to_pick.any()):
            tomato_pos = self.tomato.data.root_pos_w
            tomato_quat = self.tomato.data.root_quat_w
            attachment_offset_local = torch.zeros_like(self.tomato_axes)
            attachment_offset_local[:, 2] = self.tomato_axes[:, 2]
            attachment_pos = tomato_pos + quat_rotate_vector_wxyz(
                tomato_quat, attachment_offset_local
            )
            self.anchor_rest_pos[ready_to_pick] = attachment_pos[ready_to_pick]
            self.anchor_pos[ready_to_pick] = attachment_pos[ready_to_pick]
            self.anchor_rest_quat[ready_to_pick] = tomato_quat[ready_to_pick]
            self.anchor_yaw[ready_to_pick] = 0.0
            self.pick_progress_m[ready_to_pick] = 0.0
            self.pick_started |= ready_to_pick
            self.secure_before_pick[ready_to_pick] = True

        # Verifier conditions (plan §6) and success.
        post_break_grace = self.stem_broke & (
            self.post_break_steps < int(self.cfg.post_break_grace_steps)
        )
        drop_candidate = (tomato_pos[:, 2] < self.cfg.drop_height_m) | (
            ~in_hand & ((self.stem_broke) | (self.cfg.training_stage == 1))
        )
        grace_steps = int(getattr(self.cfg, "pregrasp_drop_grace_steps", 0))
        if grace_steps > 0:
            pregrasp_grace = (self.episode_length_buf < grace_steps) & ~self.pick_started
            drop_candidate &= ~pregrasp_grace
        self.dropped |= drop_candidate & ~post_break_grace
        self.slipped |= self.stem_broke & (speed > self.cfg.max_relative_speed_m_s)
        conditions = evaluate_success_conditions(
            secure_before_pick=self.secure_before_pick,
            break_opposition_ok=self.break_opposition_ok,
            stem_broken=self.stem_broke,
            hold_streak=self.stable_steps,
            center_distance=center_distance,
            relative_speed=speed,
            dropped=self.dropped,
            force_safe=force_safe & ~self.overloaded,
            damage_ok=self.damage_score <= self.cfg.damage_limit,
            maximum_center_distance_m=self.cfg.max_center_distance_m,
            maximum_relative_speed_m_s=self.cfg.max_relative_speed_m_s,
            success_hold_steps=self.cfg.success_hold_steps,
        )
        stage1_conditions = evaluate_stage1_hold_conditions(
            enclosure_contact=window_contact,
            palm_support=palm_support,
            hold_streak=self.stable_steps,
            center_distance=center_distance,
            relative_speed=speed,
            dropped=self.dropped,
            force_safe=force_safe & ~self.overloaded,
            damage_ok=self.damage_score <= self.cfg.damage_limit,
            maximum_center_distance_m=self.cfg.max_center_distance_m,
            maximum_relative_speed_m_s=self.cfg.max_relative_speed_m_s,
            success_hold_steps=self.cfg.success_hold_steps,
        )
        success_now = stage1_conditions["success"] if self.cfg.training_stage == 1 else conditions["success"]
        newly_success = success_now & ~self.success
        self.success |= success_now

        # Reward chain (plan §7, causal order).
        proximity = torch.exp(-center_distance / 0.035)
        progress = (proximity - self.best_proximity).clamp_min(0.0)
        geometry_points, _ = self._geometry_contact_points()
        fingertip_distance = torch.linalg.vector_norm(geometry_points - tomato_pos.unsqueeze(1), dim=-1)
        fingertip_proximity = torch.exp(-fingertip_distance / 0.045).mean(dim=-1)
        fingertip_progress = (fingertip_proximity - self.best_fingertip_proximity).clamp_min(0.0)
        self.best_proximity = torch.maximum(self.best_proximity, proximity)
        self.best_fingertip_proximity = torch.maximum(self.best_fingertip_proximity, fingertip_proximity)
        contact_count = window_contact.float().sum(dim=-1)
        new_digits = (contact_count - self.best_contact_count.float()).clamp_min(0.0)
        self.best_contact_count = torch.maximum(self.best_contact_count, contact_count.long())

        balance = torch.zeros_like(self.damage_score)
        active_digits = self.contact_forces > self.cfg.contact_threshold_n
        mean_force = self.contact_forces.mean(dim=-1)
        non_zero = (active_digits.sum(dim=-1) >= 2) & (mean_force > 0.05)
        if non_zero.any():
            balance = torch.where(
                non_zero,
                (self.contact_forces - mean_force.unsqueeze(-1)).abs().mean(dim=-1) / mean_force.clamp_min(1.0e-3),
                balance,
            )
        concentration = (self.contact_forces.amax(dim=-1) / mean_force.clamp_min(1.0e-3)) - 1.0

        # Stage 1 must not receive a permanent reward after merely touching
        # the fruit with three fingers once.  ``secure_latched`` is retained
        # for stem diagnostics, but using it as a dense reward allowed PPO to
        # abandon all contacts while continuing to collect 2.5 reward/step.
        # Reward the exact stage objective instead: simultaneous five-digit
        # contact with palm support, followed by a sustained safe hold.
        secure_reward_signal = enclosure_now if self.cfg.training_stage == 1 else secure_now
        hold_progress = (
            self.stable_steps.float() / float(self.cfg.success_hold_steps)
        ).clamp(0.0, 1.0)
        lost_enclosure = self.enclosure_latched & ~enclosure_now

        reward = (
            self.cfg.proximity_reward_weight * progress
            + self.cfg.fingertip_proximity_reward_weight * fingertip_progress
            + self.cfg.contact_reward_weight * contact_count / 5.0
            + self.cfg.new_digit_reward_weight * new_digits
            + self.cfg.thumb_opposition_reward_weight * window_contact[:, 0].float()
            + self.cfg.palm_support_reward_weight * (palm_support & in_hand).float()
            + self.cfg.secure_reward_weight * secure_reward_signal.float()
            + self.cfg.stable_hold_reward_weight * hold_progress
        )
        post_break = self.stem_broke.float()
        post_break_contact_fraction = contact.float().mean(dim=-1)
        post_break_missing_fraction = (5.0 - contact.float().sum(dim=-1)).clamp_min(0.0) / 5.0
        # This term is intentionally independent of the strict success latch:
        # it teaches the actor to react to a lost fingertip after pick rather
        # than treating the pre-pick pose as a frozen endpoint.
        reward += post_break * (
            self.cfg.post_break_contact_reward_weight * post_break_contact_fraction
            + self.cfg.post_break_support_reward_weight * post_break_support.float()
            - self.cfg.post_break_missing_digit_penalty_weight * post_break_missing_fraction
        )
        reward -= self.cfg.balance_penalty_weight * balance
        # Continuous retention shaping closes the gap between a transient
        # five-contact frame and a fruit that remains centred in the palm.
        # The former reward only penalized speed/offset at hard thresholds,
        # so PPO could collect contact reward while pushing the fruit toward
        # the palm rim and then suffering a terminal drop.
        reward -= self.cfg.center_distance_penalty_weight * (
            center_distance / max(float(self.cfg.max_center_distance_m), 1.0e-6)
        ).clamp(0.0, 1.0)
        reward -= self.cfg.relative_speed_penalty_weight * (
            speed / max(float(self.cfg.max_relative_speed_m_s), 1.0e-6)
        ).clamp(0.0, 1.0)
        reward -= self.cfg.speed_penalty_weight * (speed > self.cfg.max_relative_speed_m_s).float()
        reward -= self.cfg.wait_penalty_weight * (contact_count < 1.0).float()
        reward -= self.cfg.jitter_penalty_weight * self.action_delta.abs().mean(dim=-1)
        reward -= self.cfg.concentration_penalty_weight * concentration.clamp(0.0, 1.0)
        reward -= self.cfg.overload_penalty_weight * (~force_safe).float()
        # A terminal drop must be worse than intentionally ending an episode
        # early and collecting fresh approach/contact rewards after reset.
        # Without this one-shot terminal cost, PPO learned exactly that reset
        # shortcut instead of maintaining the enclosure.
        reward -= self.cfg.drop_penalty_weight * self.dropped.float()
        reward -= self.cfg.impact_penalty_weight * impact_ratio.amax(dim=-1)
        reward -= self.cfg.damage_penalty_weight * self.damage_score
        # ``joint_safety_projection`` counts corrections since the preceding
        # reward call.  Penalizing the cumulative episode count every frame
        # made returns grow quadratically negative and destabilized the value
        # function even after the offending state had been corrected.
        reward -= self.cfg.joint_penalty_weight * self.joint_safety_projection
        self.joint_safety_projection.zero_()
        reward -= self.cfg.saturation_penalty_weight * (self.actions.abs() > 0.99).float().mean(dim=-1)
        reward -= self.cfg.unstable_pick_penalty_weight * (self.pick_started & self.stem_attached & ~secure_now).float()
        if self.cfg.training_stage == 1:
            reward -= self.cfg.lost_enclosure_penalty_weight * lost_enclosure.float()
        newly_broken = self.break_event
        self.last_break_event.copy_(newly_broken)
        self.break_event.zero_()
        reward += self.cfg.break_bonus * newly_broken.float()
        reward += self.cfg.success_bonus * newly_success.float()

        # Episode metrics (plan §11).
        self.metric_approach |= contact.any(dim=-1)
        self.metric_multi_digit |= contact_count >= 2.0
        self.metric_thumb_opposition |= window_contact[:, 0] & (window_contact[:, 1:].sum(dim=-1) >= 1)
        self.metric_palm_support |= palm_support
        self.metric_secure_enclosure |= secure_now
        log = self.extras.setdefault("log", {})
        log["tomato_success_rate"] = self.success.float().mean()
        log["stage1_enclosure_hold_rate"] = stage1_conditions["success"].float().mean()
        log["five_digit_enclosure_rate"] = stage1_conditions["five_digit_enclosure"].float().mean()
        log["approach_rate"] = self.metric_approach.float().mean()
        log["multi_digit_contact_rate"] = self.metric_multi_digit.float().mean()
        log["thumb_opposition_rate"] = self.metric_thumb_opposition.float().mean()
        log["palm_support_rate"] = self.metric_palm_support.float().mean()
        log["secure_enclosure_rate"] = self.metric_secure_enclosure.float().mean()
        log["pick_triggered_rate"] = self.pick_started.float().mean()
        log["stem_break_rate"] = self.stem_broke.float().mean()
        log["post_break_hold_rate"] = (self.stem_broke & (self.stable_steps >= self.cfg.success_hold_steps)).float().mean()
        log["drop_rate"] = self.dropped.float().mean()
        log["slip_rate"] = self.slipped.float().mean()
        log["overload_rate"] = self.overloaded.float().mean()
        log["damage_score"] = self.damage_score.mean()
        log["palm_force_mean_n"] = self.palm_force.mean()
        log["contact_force_max_n"] = self.contact_forces.max()
        log["stem_tension_ratio"] = self.stem_load_ratios["tension"].mean()
        log["stem_combined_ratio"] = self.stem_load_ratios["combined"].mean()
        log["action_saturation_rate"] = (self.actions.abs() > 0.99).float().mean()
        log["joint_safety_projection_mean"] = self.joint_safety_projection.mean()
        return reward

    @torch.inference_mode()
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        tomato_pos = self.tomato.data.root_pos_w
        # A newly detached fruit is allowed a short, explicitly bounded
        # tactile-reaction window.  Keep this gate on the physical height
        # signal used by the historical termination path; the reward-side
        # sticky ``self.dropped`` flag also tracks center-distance violations,
        # but folding that diagnostic into termination would change the
        # baseline episode timing and confound the ablation.
        post_break_grace = self.stem_broke & (
            self.post_break_steps < int(self.cfg.post_break_grace_steps)
        )
        dropped_now = (tomato_pos[:, 2] < self.cfg.drop_height_m) & ~post_break_grace
        grace_steps = int(getattr(self.cfg, "pregrasp_drop_grace_steps", 0))
        if grace_steps > 0:
            pregrasp_grace = (self.episode_length_buf < grace_steps) & ~self.pick_started
            dropped_now &= ~pregrasp_grace
        self.last_drop_before_pick.copy_(dropped_now & ~self.pick_started)
        self.last_drop_after_pick.copy_(dropped_now & self.pick_started)
        self.last_drop_after_break.copy_(dropped_now & self.stem_broke)
        severe_overload = (
            self.contact_forces >= self.cfg.severe_overload_scale * self._ceiling
        ).any(dim=-1) | (self.palm_force >= 1.25 * self.cfg.safe_force_n)
        terminated = (
            self.success
            | dropped_now
            | severe_overload
            | (self.damage_score >= 1.0)
        )
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        code = torch.zeros_like(self.last_termination_code)
        code = torch.where(self.success, torch.ones_like(code), code)
        code = torch.where(dropped_now, torch.full_like(code, 2), code)
        code = torch.where(severe_overload, torch.full_like(code, 3), code)
        code = torch.where(self.damage_score >= 1.0, torch.full_like(code, 4), code)
        code = torch.where(timeout & ~terminated, torch.full_like(code, 5), code)
        self.last_termination_code.copy_(code)
        return terminated, timeout

    @torch.inference_mode()
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        axes = self.tomato_axes[env_ids]
        rest_xy = torch.tensor(self.cfg.tomato_rest_xy, device=self.device).expand(n, -1).clone()
        jitter = (2.0 * torch.rand((n, 3), device=self.device) - 1.0) * self.position_jitter
        # Keep only a 1 mm clearance above the palm surface.  A larger drop
        # gives the small ellipsoid enough lateral impulse to roll away from
        # the intended palm-center grasp region before the first policy step.
        # The remaining clearance still avoids initial mesh penetration.
        rest_pos = torch.cat(
            (
                rest_xy,
                (
                    self.cfg.palm_top_z_m
                    + axes[:, 2:3]
                    + float(self.cfg.tomato_reset_clearance_m)
                ).clamp_max(0.24),
            ),
            dim=-1,
        ) + jitter
        tomato_state = torch.empty(
            (n, self.tomato.data.default_root_state.shape[-1]),
            device=self.device,
        )
        tomato_state.copy_(self.tomato.data.default_root_state[env_ids])
        tomato_state[:, :3] = rest_pos + self.scene.env_origins[env_ids]
        tomato_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
        env_cpu = env_ids.detach().cpu()
        transforms = self.tomato.root_physx_view.get_transforms().clone()
        transform_ids = env_ids.to(device=transforms.device)
        transforms[transform_ids, :3] = tomato_state[:, :3].to(device=transforms.device)
        transforms[transform_ids, 3:7] = math_utils.convert_quat(
            tomato_state[:, 3:7], to="xyzw"
        ).to(device=transforms.device)
        self.tomato.root_physx_view.set_transforms(transforms, indices=transform_ids)
        velocities = self.tomato.root_physx_view.get_velocities().clone()
        velocities[transform_ids] = 0.0
        self.tomato.root_physx_view.set_velocities(velocities, indices=transform_ids)

        # Per-episode object and sensor randomization (stage-1 narrow priors).
        self.object_mass[env_ids] = (
            4.0 / 3.0 * math.pi * axes.prod(dim=-1) * self._tomato_density_kg_m3[env_ids]
        )
        self.object_friction[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.friction_range)
        self.object_stiffness[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.stiffness_range_n_m)
        self.sensor_delay[env_ids] = torch.randint(0, self.cfg.sensor_delay_steps_max + 1, (n,), device=self.device)
        self.sensor_dropout[env_ids] = torch.rand((n, 5), device=self.device) < self.cfg.sensor_dropout_probability
        self.sensor_stale[env_ids] = torch.rand((n, 5), device=self.device) < self.cfg.sensor_stale_probability
        self.tactile_gain[env_ids] = torch.empty((n, 5), device=self.device).uniform_(*self.cfg.tactile_gain_range)
        self.tactile_bias[env_ids] = torch.empty((n, 5), device=self.device).uniform_(*self.cfg.tactile_bias_range_n)
        self.retention_force_ema[env_ids] = 0.0
        self.actuator_response_scale[env_ids] = torch.empty((n, 16), device=self.device).uniform_(*self.cfg.actuator_response_scale_range)
        self.actuator_bias[env_ids] = torch.empty((n, 16), device=self.device).uniform_(*self.cfg.actuator_bias_range_rad)
        self.actuator_deadband[env_ids] = torch.empty((n, 16), device=self.device).uniform_(*self.cfg.actuator_deadband_range_rad)
        self.actuator_backlash[env_ids] = torch.empty((n, 16), device=self.device).uniform_(*self.cfg.actuator_backlash_range_rad)
        self.last_target_direction[env_ids] = 0.0
        self.pick_speed[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.pick_speed_range_m_s)
        tilt = self.cfg.pick_direction_tilt_max_rad
        self.pick_direction_xy[env_ids] = torch.empty((n, 2), device=self.device).uniform_(-math.tan(tilt), math.tan(tilt))
        masses = self.tomato.root_physx_view.get_masses()
        masses[env_cpu, 0] = self.object_mass[env_ids].detach().cpu()
        self.tomato.root_physx_view.set_masses(masses, env_cpu)
        materials = self.tomato.root_physx_view.get_material_properties()
        static = self.object_friction[env_ids].detach().cpu()
        dynamic = (0.85 * self.object_friction[env_ids]).detach().cpu()
        restitution = torch.empty(n, device="cpu").uniform_(*self.cfg.restitution_range)
        materials[env_cpu, :, 0] = static[:, None]
        materials[env_cpu, :, 1] = dynamic[:, None]
        materials[env_cpu, :, 2] = restitution[:, None]
        self.tomato.root_physx_view.set_material_properties(materials, env_cpu)

        joint_pos = torch.empty(
            (n, self.robot.data.default_joint_pos.shape[-1]), device=self.device
        )
        joint_pos.copy_(self.robot.data.default_joint_pos[env_ids])
        # Reset the controlled chain to a *valid open manifold pose* before
        # the fruit is allowed to advance in PhysX.  The vendor USD defaults
        # put several joints just outside our calibrated safe interval.  The
        # first action previously projected those DOFs by overwriting their
        # positions while the fruit was already present, so four finger
        # colliders resolved the resulting overlap by launching the tomato.
        open_action = torch.zeros((n, self.action_dim), device=self.device)
        open_action[:, :ACTION_DIM] = torch.tensor(
            (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
            device=self.device,
        )
        if self.action_dim == ACTION_DIM:
            open_joint_pos = hybrid_action_to_joint_targets(
                open_action,
                residual_span_fraction=float(getattr(self.cfg, "residual_span_fraction", 0.12)),
            )
        else:
            open_joint_pos = enveloping_action_to_joint_targets(
                open_action,
                residual_span_fraction=float(getattr(self.cfg, "residual_span_fraction", 0.12)),
                per_digit_root_residual_span_fraction=(
                    None
                    if getattr(self.cfg, "per_digit_root_residual_span_fraction", None) is None
                    else float(self.cfg.per_digit_root_residual_span_fraction)
                ),
            )
        joint_pos[:, self.active_joint_ids] = open_joint_pos
        if self.cfg.retention_only:
            seed_bank = getattr(self.cfg, "retention_seed_joint_positions", None)
            if seed_bank is not None:
                if not isinstance(seed_bank, torch.Tensor) or seed_bank.ndim != 2:
                    raise ValueError("retention_seed_joint_positions must have shape (K, 16)")
                if seed_bank.shape[-1] not in (len(self.active_joint_ids), self.robot.data.default_joint_pos.shape[-1]) or seed_bank.shape[0] <= 0:
                    raise ValueError("retention_seed_joint_positions must contain K x active-joint or full-joint states")
                seed_bank = seed_bank.to(device=self.device, dtype=joint_pos.dtype)
                seed_indices = torch.randint(seed_bank.shape[0], (n,), device=self.device)
                selected = seed_bank[seed_indices]
                seed_active = selected if selected.shape[-1] == len(self.active_joint_ids) else selected[:, self.active_joint_ids]
                # Replay states can contain a small post-solve excursion past
                # the calibrated envelope.  Retention resets must stay inside
                # the same safe interval as ordinary resets; otherwise the
                # first PhysX projection can generate an artificial impulse.
                seed_active = seed_active.maximum(self._lower).minimum(self._upper)
                joint_pos[:, self.active_joint_ids] = seed_active
            else:
                start_action = torch.tensor(
                    self.cfg.retention_start_action, device=self.device
                ).expand(n, -1)
                start_joint_pos = enveloping_action_to_joint_targets(
                    start_action,
                    residual_span_fraction=float(getattr(self.cfg, "residual_span_fraction", 0.12)),
                    distal_residual_span_fraction=float(
                        getattr(self.cfg, "distal_residual_span_fraction", 0.25)
                    ),
                    per_digit_root_residual_span_fraction=(
                        None
                        if getattr(self.cfg, "per_digit_root_residual_span_fraction", None) is None
                        else float(self.cfg.per_digit_root_residual_span_fraction)
                    ),
                )
                joint_pos[:, self.active_joint_ids] = start_joint_pos
        dof_positions = self.robot.root_physx_view.get_dof_positions().clone()
        robot_ids = env_ids.to(device=dof_positions.device)
        dof_positions[robot_ids] = joint_pos.to(device=dof_positions.device)
        self.robot.root_physx_view.set_dof_positions(dof_positions, indices=robot_ids)
        dof_velocities = self.robot.root_physx_view.get_dof_velocities().clone()
        dof_velocities[robot_ids] = 0.0
        self.robot.root_physx_view.set_dof_velocities(dof_velocities, indices=robot_ids)
        self.robot.update(dt=1.0e-6)
        self.tomato.update(dt=1.0e-6)

        # Stem / pick / verifier state.  The tomato transforms are global, so
        # the anchor and the retention reference must include the env origin.
        global_rest = rest_pos + self.scene.env_origins[env_ids]
        self.grasp_center[env_ids] = global_rest
        # The observation reference is the nominal palm support point, not
        # the randomized fruit reset.  This preserves the intended ±jitter
        # signal while correctly accounting for each clone's world origin.
        nominal_support = torch.tensor(
            (*tuple(float(value) for value in self.cfg.tomato_rest_xy), self.cfg.palm_top_z_m),
            device=self.device,
        )
        self.palm_support_point[env_ids] = self.scene.env_origins[env_ids] + nominal_support
        # Anchor sits exactly at the attachment point at reset (zero stem
        # strain).  The previous version placed the anchor another 4 mm above
        # the attachment, which applied a continuous upward stem force before
        # the palm could support the tomato and made the rest probe fail.
        self.anchor_rest_pos[env_ids] = global_rest + torch.cat(
            (torch.zeros((n, 2), device=self.device), axes[:, 2:3]), dim=-1
        )
        self.anchor_pos[env_ids] = self.anchor_rest_pos[env_ids]
        self.anchor_rest_quat[env_ids] = tomato_state[:, 3:7]
        self.anchor_yaw[env_ids] = 0.0
        self.pick_progress_m[env_ids] = 0.0
        self.stem_damage[env_ids] = 0.0
        self.stem_attached[env_ids] = True
        self.stem_broke[env_ids] = False
        self.break_event[env_ids] = False
        self.last_break_event[env_ids] = False
        self._sample_stem_limits(env_ids)
        self.pick_started[env_ids] = False
        self.pick_done[env_ids] = False
        self.break_opposition_ok[env_ids] = False
        self.secure_before_pick[env_ids] = False
        self.dropped[env_ids] = False
        self.slipped[env_ids] = False
        self.overloaded[env_ids] = False
        if self.cfg.retention_only:
            # Retention curriculum starts immediately after a successful
            # release.  The object remains dynamic on the palm; no kinematic
            # capture or stem force is added.  The policy must preserve the
            # seeded five-finger enclosure against gravity and disturbances.
            self.pick_started[env_ids] = True
            self.pick_done[env_ids] = True
            self.stem_attached[env_ids] = False
            self.stem_broke[env_ids] = True
            self.secure_latched[env_ids] = True
            self.secure_before_pick[env_ids] = True
            self.break_opposition_ok[env_ids] = True
        for values in self.stem_load_ratios.values():
            values[env_ids] = 0.0

        self.actions[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        physx_reset_pos = self.robot.root_physx_view.get_dof_positions()[robot_ids]
        self.prev_targets[env_ids] = physx_reset_pos[:, self.active_joint_ids].to(self.device)
        self.post_break_targets[env_ids] = self.prev_targets[env_ids]
        self.post_break_targets_latched[env_ids] = False
        self.contact_forces[env_ids] = 0.0
        self.previous_contact_forces[env_ids] = 0.0
        self.palm_force[env_ids] = 0.0
        self.contact_counts[env_ids] = 0.0
        self.force_history[env_ids] = 0.0
        self.peak_force[env_ids] = 0.0
        self.force_exposure[env_ids] = 0.0
        self.impact_exposure[env_ids] = 0.0
        self.damage_score[env_ids] = 0.0
        self.joint_safety_projection[env_ids] = 0.0
        self._cached_actor = None
        self._cached_critic = None
        self.stable_steps[env_ids] = 0
        self.post_break_steps[env_ids] = 0
        self.prepick_stable_steps[env_ids] = 0
        self.prepick_best_steps[env_ids] = 0
        self.success[env_ids] = False
        self.secure_latched[env_ids] = False
        self.enclosure_latched[env_ids] = False
        if self.cfg.retention_only:
            self.secure_latched[env_ids] = True
            self.pick_started[env_ids] = True
            self.pick_done[env_ids] = True
            self.stem_attached[env_ids] = False
            self.stem_broke[env_ids] = True
            self.secure_before_pick[env_ids] = True
            self.break_opposition_ok[env_ids] = True
        self.best_proximity[env_ids] = 0.0
        self.best_fingertip_proximity[env_ids] = 0.0
        self.best_contact_count[env_ids] = 0
        self.metric_approach[env_ids] = False
        self.metric_multi_digit[env_ids] = False
        self.metric_thumb_opposition[env_ids] = False
        self.metric_palm_support[env_ids] = False
        self.metric_secure_enclosure[env_ids] = False


__all__ = ["PhysxTomatoGraspEnv", "PhysxTomatoGraspEnvCfg", "author_tomato_ellipsoid_geometry"]
