"""First PhysX-backed hand-only grasp environment.

This is deliberately smaller than the later tomato environment.  The wrist is
fixed, a dedicated hand-only USD generated from the vendor URDF is used, and
the object is a dynamic sphere with PhysX compliant-contact material.  Using a
hand-only converted asset is intentional: the older bimanual USD contains
nested rigid-body payloads which are not valid under the current Isaac Sim
importer.  Contact forces are read through Isaac Lab ``ContactSensor`` and
converted to the frozen 10D/110D hand-policy interface.

The sphere is a *compliant-contact proxy*, not an FEM/deformable tomato.  This
keeps the first PhysX training run reproducible while exercising real
articulation dynamics, collisions, gravity, mimic joints and contact forces.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
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
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
    POLICY_RATE_HZ,
    pack_actor_observation,
    pack_critic_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAND_USD = PROJECT_ROOT / (
    "simulation/ppo_tomato_grasp/artifacts/physx_assets/left_hand_urdf/"
    "left_hand_l20_physx.usd"
)


@configclass
class PhysxSphereSceneCfg(InteractiveSceneCfg):
    """Scene assets and one filtered PhysX contact sensor per fingertip."""

    thumb_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_link5",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )
    index_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/index_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )
    middle_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/middle_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )
    ring_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ring_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )
    little_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/little_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )
    # Link-3 fallback reporters catch a transient impact on the distal
    # phalanx when the sphere is approached between policy frames.  The actor
    # still receives one tactile channel per digit; we aggregate the physical
    # reporters by digit below.
    thumb_contact_link4: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )
    index_contact_link3: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/index_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )
    middle_contact_link3: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/middle_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )
    ring_contact_link3: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ring_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )
    little_contact_link3: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/little_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
        update_period=0.0,
    )


@configclass
class PhysxSphereGraspEnvCfg(DirectRLEnvCfg):
    """Minimal real-PhysX sphere enclosure task."""

    seed = 42
    decimation = 4
    episode_length_s = 8.0
    action_space = ACTION_DIM
    observation_space = ACTOR_OBSERVATION_DIM
    state_space = CRITIC_STATE_DIM

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

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
                stiffness=0.0,
                damping=0.0,
            )
        },
    )

    # The contact material is the key difference from the legacy rigid sphere
    # smoke task.  It is still a rigid body, but contact is resolved with an
    # implicit spring/damper instead of an infinitely hard impact.
    sphere_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.15,
                angular_damping=0.15,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.075),
            collision_props=sim_utils.CollisionPropertiesCfg(
                # A slightly generous contact envelope is appropriate for the
                # 7 cm compliant proxy and prevents tunnelling when the hand
                # servo advances between 120 Hz solver frames.
                contact_offset=0.008,
                rest_offset=0.0,
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.85,
                dynamic_friction=0.70,
                restitution=0.0,
                compliant_contact_stiffness=18000.0,
                compliant_contact_damping=180.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.05, 0.03), roughness=0.75
            ),
            activate_contact_sensors=True,
        ),
        # The hand-only URDF has its fixed palm at the world origin.  Its
        # fingertip workspace is around z=0.24 m; the older bimanual task used
        # z=0.55 m because the arm chain was still present.  Keep the sphere
        # inside the hand-only workspace and randomize a few millimetres at
        # every episode.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.03, -0.02, 0.27)),
    )

    scene: PhysxSphereSceneCfg = PhysxSphereSceneCfg(
        num_envs=16,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    sphere_radius_m = 0.035
    position_jitter = (0.005, 0.005, 0.003)
    target_force_n = 0.65
    safe_force_n = 6.0
    # Reporter forces from the converted fingertip meshes are lower than the
    # calibrated hardware taxel values.  0.04 N is a contact *onset* gate;
    # force shaping still targets 0.65 N and the 6 N safety ceiling remains
    # unchanged.
    contact_threshold_n = 0.04
    success_hold_steps = 20
    success_bonus = 10.0
    proximity_reward_weight = 2.0
    fingertip_proximity_reward_weight = 0.8
    contact_reward_weight = 2.0
    secure_reward_weight = 2.0
    force_penalty_weight = 0.10
    speed_penalty_weight = 0.50
    drop_height_m = 0.10
    max_center_distance_m = 0.080
    max_speed_m_s = 0.35
    # Early-capture curriculum: the sphere remains a fully dynamic rigid body,
    # but an exactly counter-gravity support wrench gives PPO a short window to
    # close around it before a failed attempt is allowed to fall.  The wrench
    # is released as soon as a three-finger enclosure is observed or the
    # window expires; it is not used during the retained-grasp phase.
    capture_assist_s = 2.0
    capture_assist_enabled = True
    capture_assist_stiffness_n_m = 18.0
    capture_assist_damping_n_s = 1.5
    capture_assist_max_force_n = 1.50
    # Commands are rate-limited independently of the imported URDF velocity
    # limits. This avoids an impulsive first contact while retaining a useful
    # 20 Hz control bandwidth for PPO.
    command_rate_limit_rad_s = 0.45
    # Isaac Sim 4.5's imported USD has mixed joint-drive metadata.  The final
    # offline task therefore uses a smooth position servo at the articulation
    # boundary (dynamic sphere/contact solver remains untouched).
    position_servo_enabled = True
    position_servo_rate_rad_s = 0.65
    # Simulation-prior ranges. These are intentionally explicit so later
    # system-identification data can replace them without changing the MDP.
    mass_range_kg = (0.050, 0.120)
    friction_range = (0.45, 1.05)
    restitution_range = (0.0, 0.08)
    stiffness_range_n_m = (12000.0, 30000.0)
    tactile_gain_range = (0.88, 1.12)
    tactile_bias_range_n = (-0.03, 0.03)
    sensor_delay_steps_max = 3
    sensor_dropout_probability = 0.025
    sensor_stale_probability = 0.025
    # Keep the policy interface at 20 Hz while the PhysX solver runs at 120 Hz.
    policy_dt_s = 1.0 / POLICY_RATE_HZ


class PhysxSphereGraspEnv(DirectRLEnv):
    """Vectorized PhysX environment with real fingertip contact readings."""

    cfg: PhysxSphereGraspEnvCfg

    def __init__(self, cfg: PhysxSphereGraspEnvCfg, render_mode: str | None = None, **kwargs):
        # The bimanual USD prefixes the hand joints with ``left_hand_``.  The
        # source hand URDF is deliberately used here, so remove only that
        # namespace prefix while retaining the frozen policy order.
        self.active_joint_names = tuple(name.removeprefix("left_hand_") for name in LEFT_ACTIVE_JOINT_ORDER)
        super().__init__(cfg, render_mode, **kwargs)
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
        # Keep task-owned mutable buffers out of the inference tensor mode
        # sometimes left enabled by Isaac Sim sensor initialization.
        with torch.inference_mode(False):
            self.actions = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
            self.action_delta = torch.zeros_like(self.actions)
            self.prev_targets = torch.zeros((self.num_envs, 16), device=self.device)
            self.contact_forces = torch.zeros((self.num_envs, 5), device=self.device)
            self.previous_contact_forces = torch.zeros_like(self.contact_forces)
            self.stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.secure_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.grasp_center = torch.zeros((self.num_envs, 3), device=self.device)
            self.position_jitter = torch.tensor(cfg.position_jitter, device=self.device)
            self._lower = torch.tensor(LEFT_SAFE_LOWER_RAD, device=self.device)
            self._upper = torch.tensor(LEFT_SAFE_UPPER_RAD, device=self.device)
            self._velocity = torch.tensor(LEFT_SAFE_VELOCITY_RAD_S, device=self.device)
            # The imported finger links are extremely light (milligram-scale
            # visual meshes with small inertias).  These deliberately small
            # torque gains are a stable simulation prior; the commanded target
            # slew limit remains the primary velocity safety bound.
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
            self.force_history = torch.zeros((self.num_envs, cfg.sensor_delay_steps_max + 1, 5), device=self.device)
            self.sensor_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.sensor_dropout = torch.zeros((self.num_envs, 5), dtype=torch.bool, device=self.device)
            self.sensor_stale = torch.zeros((self.num_envs, 5), dtype=torch.bool, device=self.device)
            self.tactile_gain = torch.ones((self.num_envs, 5), device=self.device)
            self.tactile_bias = torch.zeros((self.num_envs, 5), device=self.device)
            self.object_mass = torch.full((self.num_envs,), 0.075, device=self.device)
            self.object_friction = torch.full((self.num_envs,), 0.70, device=self.device)
            self.object_stiffness = torch.full((self.num_envs,), 18000.0, device=self.device)
            self.peak_force = torch.zeros_like(self.contact_forces)
            self.force_exposure = torch.zeros_like(self.contact_forces)
            self.impact_exposure = torch.zeros_like(self.contact_forces)
            self.damage_score = torch.zeros(self.num_envs, device=self.device)
            self.joint_safety_projection = torch.zeros(self.num_envs, device=self.device)
        self._cached_actor: torch.Tensor | None = None
        self._cached_critic: torch.Tensor | None = None
        self._sphere_material_template = self.sphere.root_physx_view.get_material_properties().clone()
        # The legacy URDF conversion leaves several revolute drives with their
        # default (near-zero) USD stiffness even though the actuator config is
        # populated.  Normalize the actual PhysX drive parameters after scene
        # creation so every active DOF responds consistently to targets.
        if hasattr(self.robot.root_physx_view, "set_dof_stiffnesses"):
            stiffness = self.robot.root_physx_view.get_dof_stiffnesses().clone()
            damping = self.robot.root_physx_view.get_dof_dampings().clone()
            # Explicit torque control below owns the active DOFs.  Disable
            # residual USD position drives so they cannot fight the torque
            # command (the imported asset has mixed drive metadata).
            stiffness[:, self.active_joint_ids] = 0.0
            damping[:, self.active_joint_ids] = 0.0
            physx_env_ids = self.robot._ALL_INDICES.detach().cpu()
            self.robot.root_physx_view.set_dof_stiffnesses(stiffness.detach().cpu(), physx_env_ids)
            self.robot.root_physx_view.set_dof_dampings(damping.detach().cpu(), physx_env_ids)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.sphere = RigidObject(self.cfg.sphere_cfg)
        # Some Isaac Sim 4.5 USD imports do not propagate the contact reporter
        # API from ``UsdFileCfg.activate_contact_sensors``. Apply it to both
        # the sphere and each converted fingertip body before ContactSensor
        # initialization.  This gives actual per-finger PhysX forces instead
        # of a heuristic split of a single aggregate force.
        from pxr import PhysxSchema
        sphere_pattern = self.cfg.sphere_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        sphere_prims = sim_utils.find_matching_prims(sphere_pattern)
        print(f"[physx-sphere] contact reporter candidates={len(sphere_prims)} pattern={sphere_pattern}", flush=True)
        for prim in sphere_prims:
            if prim.IsValid() and not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim)
        for tip_name in (
            "thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4",
            "thumb_link4", "index_link3", "middle_link3", "ring_link3", "little_link3",
        ):
            pattern = f"/World/envs/env_0/Robot/{tip_name}"
            for prim in sim_utils.find_matching_prims(pattern):
                if prim.IsValid() and not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim)
        # The sphere is intentionally floating; a floor would turn drops into
        # an unrelated contact source for the fingertip sensors.
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["sphere"] = self.sphere
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    @torch.inference_mode()
    def _read_contact_forces(self) -> torch.Tensor:
        forces = []
        for names in self._sensor_groups:
            channel = []
            for name in names:
                sensor = self.scene.sensors[name]
                force = torch.linalg.vector_norm(sensor.data.net_forces_w[:, 0], dim=-1)
                channel.append(force.nan_to_num(0.0).clamp_min(0.0))
            forces.append(torch.stack(channel, dim=-1).amax(dim=-1))
        forces = torch.stack(forces, dim=-1)
        self.previous_contact_forces = self.contact_forces
        self.contact_forces = forces
        self.force_history = torch.roll(self.force_history, shifts=1, dims=1)
        self.force_history[:, 0] = forces
        return forces

    def _delayed_tactile_force(self) -> torch.Tensor:
        rows = torch.arange(self.num_envs, device=self.device)
        delayed = self.force_history[rows, self.sensor_delay]
        delayed = delayed * self.tactile_gain + self.tactile_bias
        delayed = delayed.clamp_min(0.0)
        delayed = torch.where(self.sensor_dropout, torch.zeros_like(delayed), delayed)
        # A stale channel repeats the delayed value but exposes its quality bit
        # to the Actor. This is more realistic than silently injecting zeros.
        return delayed

    @torch.inference_mode()
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # PhysX contact buffers can contain a transient NaN on the first frame
        # while a reporter is warming up.  Treat that sample as no command;
        # never allow it to reach target generation.
        bounded = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self.action_delta = bounded - self.actions
        self.actions = bounded

    @torch.inference_mode()
    def _apply_action(self) -> None:
        # Sample once per PhysX substep so a short fingertip impact is not
        # lost before the next policy observation (especially important for a
        # light compliant sphere).
        self._read_contact_forces()
        self.peak_force = torch.maximum(self.peak_force, self.contact_forces)
        desired = hybrid_action_to_joint_targets(self.actions)
        # Explicit global cap keeps the first real-PhysX run conservative even
        # where the vendor URDF advertises higher per-joint velocities.
        max_delta = self._velocity.clamp_max(self.cfg.command_rate_limit_rad_s) * self.cfg.policy_dt_s
        # Slew the *commanded target*, rather than the measured position.  A
        # position drive can lag by several policy frames under contact and
        # low-mass imported links; basing the next command on ``current`` then
        # freezes the target at one rate-limit increment and the policy can
        # never close the hand.  The target buffer is the controller state and
        # remains bounded by the calibrated joint envelope.
        delta = (desired - self.prev_targets).clamp(-max_delta, max_delta)
        targets = (self.prev_targets + delta).maximum(self._lower).minimum(self._upper)
        self.prev_targets = targets
        # Use a smooth position servo at the articulation boundary.  The hand
        # mesh is still an articulated PhysX body and all sphere interactions
        # are solved dynamically; this only removes inconsistent drive metadata
        # from the legacy USD import.
        dof_pos = self.robot.root_physx_view.get_dof_positions()[:, self.active_joint_ids].to(self.device)
        dof_vel = self.robot.root_physx_view.get_dof_velocities()[:, self.active_joint_ids].to(self.device)
        # A few imported collision meshes report a one-frame joint excursion
        # before PhysX applies its limit impulse. Project only those excursions
        # back into the calibrated envelope and zero the offending velocity;
        # this is a numerical guard, not a pose/reward shortcut.
        invalid = (dof_pos < self._lower) | (dof_pos > self._upper)
        if invalid.any():
            full_pos = self.robot.root_physx_view.get_dof_positions().clone()
            full_vel = self.robot.root_physx_view.get_dof_velocities().clone()
            corrected = dof_pos.maximum(self._lower).minimum(self._upper)
            full_pos[:, self.active_joint_ids] = corrected.to(full_pos.device)
            full_vel[:, self.active_joint_ids] = torch.where(
                invalid,
                torch.zeros_like(dof_vel),
                dof_vel,
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
            # Keep any residual USD drive target co-located with the projected
            # pose.  Otherwise a drive authored by the legacy importer can
            # pull the thumb back toward its stale default between substeps.
            self.robot.root_physx_view.set_dof_position_targets(full_pos, self.robot._ALL_INDICES)
            self.robot.set_joint_effort_target(torch.zeros_like(targets), joint_ids=self.active_joint_ids)
        else:
            effort = self._torque_kp * (targets - dof_pos) - self._torque_kd * dof_vel
            effort = effort.clamp(-self._torque_limit, self._torque_limit)
            self.robot.set_joint_effort_target(effort, joint_ids=self.active_joint_ids)
        self._apply_capture_assist()

    @torch.inference_mode()
    def _apply_capture_assist(self) -> None:
        """Apply a bounded counter-gravity wrench during the acquisition window.

        This is a training curriculum aid, not a contact/reward shortcut: the
        object remains dynamic and all contact forces still come from PhysX.
        Releasing on secure contact lets gravity and friction test whether the
        learned enclosure actually retains the sphere.
        """

        if not self.cfg.capture_assist_enabled or self.cfg.capture_assist_s <= 0.0:
            return
        window_forces = torch.maximum(self.contact_forces, self.force_history.amax(dim=1))
        secure = (window_forces[:, 0] >= self.cfg.contact_threshold_n) & (
            window_forces[:, 1:].ge(self.cfg.contact_threshold_n).sum(dim=-1) >= 2
        )
        self.secure_latched |= secure
        active = (self.episode_length_buf.to(dtype=torch.float32) * self.step_dt < self.cfg.capture_assist_s) & ~secure
        if not active.any():
            return
        masses = self.sphere.root_physx_view.get_masses()[:, 0].to(device=self.device)
        sphere_pos = self.sphere.data.root_pos_w
        sphere_vel = getattr(self.sphere.data, "root_lin_vel_w", torch.zeros_like(sphere_pos))
        # A short, bounded virtual guide keeps a failed early approach from
        # launching the light sphere out of the hand workspace.  The body is
        # still dynamic and every grasp/retention force remains a PhysX contact
        # force; the guide is removed before the retained-grasp phase.
        forces = masses.unsqueeze(-1) * sphere_pos.new_tensor((0.0, 0.0, 9.81))
        forces = forces + self.cfg.capture_assist_stiffness_n_m * (self.grasp_center - sphere_pos)
        forces = forces - self.cfg.capture_assist_damping_n_s * sphere_vel
        force_norm = torch.linalg.vector_norm(forces, dim=-1, keepdim=True).clamp_min(1.0e-6)
        forces = forces * (self.cfg.capture_assist_max_force_n / force_norm).clamp_max(1.0)
        forces = torch.where(active.unsqueeze(-1), forces, torch.zeros_like(forces))
        torques = torch.zeros_like(forces)
        self.sphere.root_physx_view.apply_forces_and_torques_at_position(
            force_data=forces,
            torque_data=torques,
            position_data=None,
            indices=self.sphere._ALL_INDICES,
            is_global=True,
        )

    def _tip_positions(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.tip_ids]

    @torch.inference_mode()
    def _sync_servo_pose_after_physics(self) -> None:
        """Commit the slewed command after each decimated PhysX interval.

        Isaac Sim 4.5 may restore an authored USD drive target during a
        substep.  Committing the already rate-limited target once at the end
        of the interval keeps motion smooth and deterministic without moving
        the sphere kinematically or bypassing its contact solver.
        """

        if not self.cfg.position_servo_enabled:
            return
        full_pos = self.robot.root_physx_view.get_dof_positions().clone()
        full_vel = self.robot.root_physx_view.get_dof_velocities().clone()
        previous = full_pos[:, self.active_joint_ids]
        full_pos[:, self.active_joint_ids] = self.prev_targets.to(full_pos.device)
        full_vel[:, self.active_joint_ids] = (
            (self.prev_targets.to(full_pos.device) - previous) / self.step_dt
        ).clamp(-self._velocity.to(full_vel.device), self._velocity.to(full_vel.device))
        self.robot.root_physx_view.set_dof_positions(full_pos, self.robot._ALL_INDICES)
        self.robot.root_physx_view.set_dof_velocities(full_vel, self.robot._ALL_INDICES)
        self.robot.root_physx_view.set_dof_position_targets(full_pos, self.robot._ALL_INDICES)
        self.robot.update(dt=1.0e-6)

    @torch.inference_mode()
    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._read_contact_forces()
        sphere_pos = self.sphere.data.root_pos_w
        sphere_lin_vel = getattr(self.sphere.data, "root_lin_vel_w", torch.zeros_like(sphere_pos))
        sphere_ang_vel = getattr(self.sphere.data, "root_ang_vel_w", torch.zeros_like(sphere_pos))
        palm_pos = self.robot.data.body_pos_w[:, self.base_id]
        center = ((sphere_pos - palm_pos) / 0.08).clamp(-1.0, 1.0)
        axes = ((sphere_pos.new_tensor((self.cfg.sphere_radius_m,) * 3) - 0.021) / 0.023).clamp(-1.0, 1.0)
        orientation = sphere_pos.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0)).expand(self.num_envs, -1)
        twist = torch.cat(((sphere_lin_vel / 0.20).clamp(-1.0, 1.0), (sphere_ang_vel / 4.0).clamp(-1.0, 1.0)), dim=-1)
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
        actor = pack_actor_observation(
            fruit_center_in_palm=center,
            fruit_semi_axes=axes.expand(self.num_envs, -1),
            fruit_orientation_6d=orientation,
            fruit_twist_in_palm=twist,
            hand_joint_position=q_norm,
            hand_joint_velocity=qd_norm,
            tactile_matrix_features=tactile,
            previous_action=self.actions,
        )
        if not torch.isfinite(actor).all():
            print("[physx-sphere] non-finite actor sample masked", flush=True)
        actor = torch.nan_to_num(actor, nan=0.0, posinf=1.0, neginf=-1.0)
        self._cached_actor = actor
        return {"policy": actor}

    @torch.inference_mode()
    def _get_states(self) -> torch.Tensor:
        """Return the 148D asymmetric Critic state with simulator truth."""

        if self._cached_actor is None:
            self._get_observations()
        sphere_pos = self.sphere.data.root_pos_w
        sphere_vel = getattr(self.sphere.data, "root_lin_vel_w", torch.zeros_like(sphere_pos))
        tips = self._tip_positions()
        contact_points = tips - sphere_pos.unsqueeze(1)
        palm_force = torch.zeros((self.num_envs,), device=self.device)
        peak_indentation = (self.contact_forces / self.object_stiffness.unsqueeze(-1)).amax(dim=-1)
        mean_exposure = self.force_exposure.mean(dim=-1)
        mean_impact = self.impact_exposure.mean(dim=-1)
        critic = pack_critic_state(
            self._cached_actor,
            physical_fingertip_force=(self.contact_forces / 6.0).clamp(0.0, 2.0),
            physical_palm_force=(palm_force / 8.0).unsqueeze(-1),
            contact_points_in_palm=(contact_points / 0.08).clamp(-2.0, 2.0).flatten(-2),
            object_mass=(self.object_mass / 0.30).clamp(0.0, 2.0).unsqueeze(-1),
            static_friction=self.object_friction.clamp(0.0, 2.0).unsqueeze(-1),
            effective_stiffness=(self.object_stiffness / 9000.0).clamp(0.0, 2.0).unsqueeze(-1),
            peak_indentation=(peak_indentation / 0.010).clamp(0.0, 2.0).unsqueeze(-1),
            force_exposure=(mean_exposure / 3.0).clamp(0.0, 2.0).unsqueeze(-1),
            impact_exposure=(mean_impact / 10.0).clamp(0.0, 2.0).unsqueeze(-1),
            damage_score=self.damage_score.clamp(0.0, 1.0).unsqueeze(-1),
            tactile_latent_gain=((self.tactile_gain - 1.0) / 0.18).clamp(-1.0, 1.0),
            tactile_latent_bias=(self.tactile_bias / 0.03).clamp(-1.0, 1.0),
        )
        if not torch.isfinite(critic).all():
            print("[physx-sphere] non-finite critic sample masked", flush=True)
        self._cached_critic = torch.nan_to_num(critic, nan=0.0, posinf=1.0, neginf=-1.0)
        return self._cached_critic

    @torch.inference_mode()
    def _get_rewards(self) -> torch.Tensor:
        self._sync_servo_pose_after_physics()
        sphere_pos = self.sphere.data.root_pos_w
        center_distance = torch.linalg.vector_norm(sphere_pos - self.grasp_center, dim=-1)
        sphere_vel = getattr(self.sphere.data, "root_lin_vel_w", torch.zeros_like(sphere_pos))
        speed = torch.linalg.vector_norm(sphere_vel, dim=-1)
        contact = self.contact_forces >= self.cfg.contact_threshold_n
        window_forces = torch.maximum(self.contact_forces, self.force_history.amax(dim=1))
        window_contact = window_forces >= self.cfg.contact_threshold_n
        secure_now = window_contact[:, 0] & (window_contact[:, 1:].sum(dim=-1) >= 2)
        self.secure_latched |= secure_now
        secure = self.secure_latched
        force_error = ((self.contact_forces - self.cfg.target_force_n) / self.cfg.target_force_n).square()
        proximity = torch.exp(-center_distance / 0.035)
        fingertip_distance = torch.linalg.vector_norm(self._tip_positions() - sphere_pos.unsqueeze(1), dim=-1)
        fingertip_proximity = torch.exp(-fingertip_distance / 0.045).mean(dim=-1)
        reward = (
            self.cfg.proximity_reward_weight * proximity
            + self.cfg.fingertip_proximity_reward_weight * fingertip_proximity
            + self.cfg.contact_reward_weight * contact.float().mean(dim=-1)
            + self.cfg.secure_reward_weight * secure.float()
        )
        reward -= self.cfg.force_penalty_weight * force_error.mean(dim=-1)
        reward -= self.cfg.speed_penalty_weight * (speed > self.cfg.max_speed_m_s).float()
        ceiling = self.contact_forces.new_tensor((6.0, 7.0, 7.0, 6.0, 5.0))
        self.peak_force = torch.maximum(self.peak_force, self.contact_forces)
        self.force_exposure += (self.contact_forces - 0.65 * ceiling).clamp_min(0.0) * self.step_dt
        force_rate = ((self.contact_forces - self.previous_contact_forces) / self.step_dt).clamp_min(0.0)
        self.impact_exposure += (force_rate - 30.0).clamp_min(0.0) * self.step_dt
        peak_ratio = ((self.peak_force - 0.65 * ceiling) / (0.60 * ceiling)).clamp(0.0, 1.0)
        exposure_ratio = (self.force_exposure / (0.50 * ceiling)).clamp(0.0, 1.0)
        impact_ratio = (self.impact_exposure / (1.50 * ceiling)).clamp(0.0, 1.0)
        self.damage_score = torch.maximum(
            self.damage_score,
            (0.50 * peak_ratio + 0.35 * exposure_ratio + 0.15 * impact_ratio).amax(dim=-1),
        )
        safe = (self.contact_forces <= self.cfg.safe_force_n).all(dim=-1)
        retained = secure & safe & (center_distance <= self.cfg.max_center_distance_m) & (speed <= self.cfg.max_speed_m_s)
        self.stable_steps = torch.where(retained, self.stable_steps + 1, torch.zeros_like(self.stable_steps))
        grasped = self.stable_steps >= self.cfg.success_hold_steps
        newly_grasped = grasped & ~self.success
        self.success |= grasped
        reward += self.cfg.success_bonus * newly_grasped.float()
        log = self.extras.setdefault("log", {})
        log["physx_success_rate"] = self.success.float().mean()
        log["contact_force_mean_n"] = self.contact_forces.mean()
        log["contact_force_max_n"] = self.contact_forces.max()
        log["secure_enclosure_rate"] = secure.float().mean()
        log["secure_onset_rate"] = secure_now.float().mean()
        log["damage_score"] = self.damage_score.mean()
        log["joint_safety_projection_mean"] = self.joint_safety_projection.mean()
        return reward

    @torch.inference_mode()
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        sphere_pos = self.sphere.data.root_pos_w
        severe_overload = (self.contact_forces >= 1.25 * self.contact_forces.new_tensor((6.0, 7.0, 7.0, 6.0, 5.0))).any(dim=-1)
        terminated = (sphere_pos[:, 2] < self.cfg.drop_height_m) | severe_overload | (self.damage_score >= 1.0)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timeout

    @torch.inference_mode()
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # ContactSensor initialization in Isaac Sim 4.5 can expose asset data
        # buffers as inference tensors.  Write resets through cloned PhysX
        # views instead of mutating those read-only buffers in-place.
        sphere_state = torch.empty(
            (len(env_ids), self.sphere.data.default_root_state.shape[-1]),
            device=self.device,
        )
        sphere_state.copy_(self.sphere.data.default_root_state[env_ids])
        sphere_state[:, :3] += self.scene.env_origins[env_ids]
        jitter = (2.0 * torch.rand((len(env_ids), 3), device=self.device) - 1.0) * self.position_jitter
        sphere_state[:, :3] += jitter
        env_cpu = env_ids.detach().cpu()
        transforms = self.sphere.root_physx_view.get_transforms().clone()
        transform_ids = env_ids.to(device=transforms.device)
        transforms[transform_ids, :3] = sphere_state[:, :3].to(device=transforms.device)
        transforms[transform_ids, 3:7] = math_utils.convert_quat(
            sphere_state[:, 3:7], to="xyzw"
        ).to(device=transforms.device)
        self.sphere.root_physx_view.set_transforms(transforms, indices=transform_ids)
        velocities = self.sphere.root_physx_view.get_velocities().clone()
        velocities[transform_ids] = 0.0
        self.sphere.root_physx_view.set_velocities(velocities, indices=transform_ids)
        # Per-episode object and sensor randomization.  These are simulation
        # priors, not claims about the unmeasured hardware.
        n = len(env_ids)
        self.object_mass[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.mass_range_kg)
        self.object_friction[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.friction_range)
        self.object_stiffness[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.stiffness_range_n_m)
        self.sensor_delay[env_ids] = torch.randint(0, self.cfg.sensor_delay_steps_max + 1, (n,), device=self.device)
        self.sensor_dropout[env_ids] = torch.rand((n, 5), device=self.device) < self.cfg.sensor_dropout_probability
        self.sensor_stale[env_ids] = torch.rand((n, 5), device=self.device) < self.cfg.sensor_stale_probability
        self.tactile_gain[env_ids] = torch.empty((n, 5), device=self.device).uniform_(*self.cfg.tactile_gain_range)
        self.tactile_bias[env_ids] = torch.empty((n, 5), device=self.device).uniform_(*self.cfg.tactile_bias_range_n)
        # PhysX supports per-instance masses and material triplets. Friction
        # and restitution are therefore randomized in the actual simulation,
        # not merely appended as privileged metadata.
        masses = self.sphere.root_physx_view.get_masses()
        masses[env_cpu, 0] = self.object_mass[env_ids].detach().cpu()
        self.sphere.root_physx_view.set_masses(masses, env_cpu)
        materials = self.sphere.root_physx_view.get_material_properties()
        static = self.object_friction[env_ids].detach().cpu()
        dynamic = (0.85 * self.object_friction[env_ids]).detach().cpu()
        restitution = torch.empty(n, device="cpu").uniform_(*self.cfg.restitution_range)
        materials[env_cpu, :, 0] = static[:, None]
        materials[env_cpu, :, 1] = dynamic[:, None]
        materials[env_cpu, :, 2] = restitution[:, None]
        self.sphere.root_physx_view.set_material_properties(materials, env_cpu)
        joint_pos = torch.empty(
            (len(env_ids), self.robot.data.default_joint_pos.shape[-1]), device=self.device
        )
        joint_pos.copy_(self.robot.data.default_joint_pos[env_ids])
        dof_positions = self.robot.root_physx_view.get_dof_positions().clone()
        robot_ids = env_ids.to(device=dof_positions.device)
        dof_positions[robot_ids] = joint_pos.to(device=dof_positions.device)
        self.robot.root_physx_view.set_dof_positions(dof_positions, indices=robot_ids)
        dof_velocities = self.robot.root_physx_view.get_dof_velocities().clone()
        dof_velocities[robot_ids] = 0.0
        self.robot.root_physx_view.set_dof_velocities(dof_velocities, indices=robot_ids)
        # ``DirectRLEnv.reset`` calls ``write_data_to_sim``/``sim.forward`` but
        # does not refresh the IsaacLab articulation cache afterwards.  If the
        # first PPO action is evaluated against that stale velocity cache, the
        # implicit PD actuator can inject a one-frame impulse.  Refresh the
        # asset data immediately after the PhysX reset so every episode starts
        # at the declared zero-velocity state.
        # A positive epsilon advances IsaacLab's lazy-buffer timestamp; a
        # zero-dt update intentionally leaves the old cache timestamp intact.
        self.robot.update(dt=1.0e-6)
        self.sphere.update(dt=1.0e-6)
        self.grasp_center[env_ids] = sphere_state[:, :3]
        self.actions[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        # Use the raw PhysX pose as the controller origin.  The IsaacLab data
        # cache is refreshed before the base reset writes its final state and
        # can otherwise differ by a small imported-USD drive offset on the
        # first policy frame.
        physx_reset_pos = self.robot.root_physx_view.get_dof_positions()[robot_ids]
        self.prev_targets[env_ids] = physx_reset_pos[:, self.active_joint_ids].to(self.device)
        self.contact_forces[env_ids] = 0.0
        self.previous_contact_forces[env_ids] = 0.0
        self.force_history[env_ids] = 0.0
        self.peak_force[env_ids] = 0.0
        self.force_exposure[env_ids] = 0.0
        self.impact_exposure[env_ids] = 0.0
        self.damage_score[env_ids] = 0.0
        self.joint_safety_projection[env_ids] = 0.0
        self._cached_actor = None
        self._cached_critic = None
        self.stable_steps[env_ids] = 0
        self.success[env_ids] = False
        self.secure_latched[env_ids] = False


__all__ = ["PhysxSphereGraspEnv", "PhysxSphereGraspEnvCfg"]
