"""Legacy Isaac Stage-0 smoke task (6D/58D; not the frozen hand PPO v1).

Keep this module for Isaac asset and dependency bring-up only.  Its normalized
geometry loads and six-action policy are intentionally incompatible with the
CPU-gated 10D Actor-110/Critic-148 contract.  Do not resume a v1 hand-policy
checkpoint into this environment.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from simulation.ppo_tomato_grasp.stage0_logic import (
    enclosure_reward,
    normalized_contact_loads,
    proxy_tip_positions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The mass-corrected training asset is the safer default for PhysX.  The
# standard asset remains available in the same directory for later comparison.
HAND_USD = PROJECT_ROOT / "urdf/02_project_derived/isaac_assets/linkbot_bimanual_sdk_training_mass1kg/linkbot_bimanual_sdk_training_mass1kg.usda"


@configclass
class TomatoGraspEnvCfg(DirectRLEnvCfg):
    """Configuration for the local, low-dimensional grasping milestone."""

    seed = 42
    decimation = 4
    episode_length_s = 8.0
    action_space = 6
    # 16 independent joint positions + 16 velocities + 15 fingertip offsets
    # + 5 normalized contact loads + 6 previous actions.
    observation_space = 58
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(HAND_USD),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            # The arm and right side are held by the USD defaults.  Only the
            # left-hand joints are exposed to PPO in this first milestone.
            "left_hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    r"left_hand_thumb_joint[0-3]",
                    r"left_hand_(index|middle|ring|little)_joint[0-2]",
                ],
                effort_limit_sim=8.0,
                velocity_limit_sim=3.0,
                stiffness=80.0,
                damping=4.0,
            )
        },
    )

    tomato_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tomato",
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.075),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.05, 0.03), roughness=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.06, 0.55)),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=1.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    # Keep the kinematic proxy as the default for dependency/asset smoke
    # tests. Pass --linkbot to exercise the real USD articulation once its
    # PhysX conversion has been validated in the local installation.
    use_linkbot_asset = False

    # Six actions: thumb opposition, thumb flex, then independent index,
    # middle, ring and little flexion. Finger spread stays at the URDF default.
    thumb_opposition_low = -0.25
    thumb_opposition_high = 0.35
    target_smoothing = 0.35
    tomato_radius = 0.035
    object_position_jitter = (0.004, 0.004, 0.002)
    contact_band = 0.008
    target_contact_load = 0.35
    safe_contact_load = 0.90
    stable_steps_required = 8
    success_bonus = 8.0
    drop_penalty = 2.0


class _KinematicProxyHand:
    """Tiny tensor-only hand used to validate the RL data path.

    It deliberately has no physics or collision claims.  The real Linkbot
    articulation is selected explicitly with ``use_linkbot_asset=True``.
    """

    def __init__(self, num_envs: int, device: str):
        self.device = device
        self.joint_names = [
            "left_hand_thumb_joint0", "left_hand_thumb_joint1", "left_hand_thumb_joint2", "left_hand_thumb_joint3",
            "left_hand_index_joint0", "left_hand_index_joint1", "left_hand_index_joint2",
            "left_hand_middle_joint0", "left_hand_middle_joint1", "left_hand_middle_joint2",
            "left_hand_ring_joint0", "left_hand_ring_joint1", "left_hand_ring_joint2",
            "left_hand_little_joint0", "left_hand_little_joint1", "left_hand_little_joint2",
        ]
        self.body_names = [
            "left_hand_base_link", "left_hand_thumb_link5", "left_hand_index_link4", "left_hand_middle_link4",
            "left_hand_ring_link4", "left_hand_little_link4",
        ]
        zeros = torch.zeros((num_envs, len(self.joint_names)), device=device)
        self.data = SimpleNamespace(
            default_joint_pos=zeros.clone(),
            joint_pos=zeros.clone(),
            joint_vel=zeros.clone(),
            body_pos_w=torch.zeros((num_envs, len(self.body_names), 3), device=device),
        )
        self._ALL_INDICES = torch.arange(num_envs, device=device)

    def set_joint_position_target(self, target: torch.Tensor, joint_ids):
        self.data.joint_vel[:, joint_ids] = target - self.data.joint_pos[:, joint_ids]
        self.data.joint_pos[:, joint_ids] = target

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
        if env_ids is None:
            env_ids = self._ALL_INDICES
        self.data.joint_pos[env_ids] = joint_pos
        self.data.joint_vel[env_ids] = joint_vel


class _KinematicProxyObject:
    """Tensor-only sphere state used by the default smoke/training task."""

    def __init__(self, num_envs: int, device: str, initial_pos: tuple[float, float, float]):
        default_root_state = torch.zeros((num_envs, 13), device=device)
        default_root_state[:, :3] = torch.tensor(initial_pos, device=device)
        default_root_state[:, 3] = 1.0
        self.data = SimpleNamespace(
            default_root_state=default_root_state,
            root_pos_w=default_root_state[:, :3].clone(),
        )

    def write_root_pose_to_sim(self, pose: torch.Tensor, env_ids) -> None:
        self.data.root_pos_w[env_ids] = pose[:, :3]

    def write_root_velocity_to_sim(self, _velocity: torch.Tensor, _env_ids) -> None:
        return


class TomatoGraspEnv(DirectRLEnv):
    """PPO environment with a simple geometry/contact proxy reward."""

    cfg: TomatoGraspEnvCfg

    def __init__(self, cfg: TomatoGraspEnvCfg, render_mode: str | None = None, **kwargs):
        self.use_linkbot_asset = cfg.use_linkbot_asset
        if not self.use_linkbot_asset:
            # The tensor-only proxy has no collision bodies, so keep its sphere
            # fixed. Real Linkbot runs retain gravity and rigid-body dynamics.
            cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
            cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.action_delta = torch.zeros_like(self.actions)
        self.prev_hand_targets = self.robot.data.default_joint_pos.clone()

        # Resolve names once against the imported USD articulation.  We use
        # real fingertip links for the reward and keep all other joints fixed.
        self.hand_joint_names = [
            "left_hand_thumb_joint0", "left_hand_thumb_joint1", "left_hand_thumb_joint2", "left_hand_thumb_joint3",
            "left_hand_index_joint0", "left_hand_index_joint1", "left_hand_index_joint2",
            "left_hand_middle_joint0", "left_hand_middle_joint1", "left_hand_middle_joint2",
            "left_hand_ring_joint0", "left_hand_ring_joint1", "left_hand_ring_joint2",
            "left_hand_little_joint0", "left_hand_little_joint1", "left_hand_little_joint2",
        ]
        self.hand_joint_ids = [self.robot.joint_names.index(name) for name in self.hand_joint_names]
        self.thumb_opposition_ids = [self.robot.joint_names.index("left_hand_thumb_joint0")]
        self.thumb_flex_ids = [self.robot.joint_names.index(f"left_hand_thumb_joint{i}") for i in range(1, 4)]
        self.finger_flex_ids = {
            finger: [self.robot.joint_names.index(f"left_hand_{finger}_joint{i}") for i in (1, 2)]
            for finger in ("index", "middle", "ring", "little")
        }
        self.controlled_joint_ids = self.thumb_opposition_ids + self.thumb_flex_ids + [
            joint_id for finger in ("index", "middle", "ring", "little") for joint_id in self.finger_flex_ids[finger]
        ]
        self.tip_ids = [
            self.robot.body_names.index(name)
            for name in ("left_hand_thumb_link5", "left_hand_index_link4", "left_hand_middle_link4",
                         "left_hand_ring_link4", "left_hand_little_link4")
        ]
        self.contact_loads = torch.zeros((self.num_envs, 5), device=self.device)
        self.surface_gap = torch.full((self.num_envs, 5), 1.0, device=self.device)
        self.stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_center = torch.zeros((self.num_envs, 3), device=self.device)
        self.position_jitter = torch.tensor(self.cfg.object_position_jitter, device=self.device)

    def set_position_jitter(self, xy_m: float, z_m: float) -> None:
        """Update reset randomization for a curriculum transition."""

        if xy_m < 0.0 or z_m < 0.0:
            raise ValueError("position jitter cannot be negative")
        self.position_jitter = torch.tensor((xy_m, xy_m, z_m), device=self.device)

    def _setup_scene(self):
        if self.use_linkbot_asset:
            self.robot = Articulation(self.cfg.robot_cfg)
        else:
            self.robot = _KinematicProxyHand(self.cfg.scene.num_envs, self.device)
        if self.use_linkbot_asset:
            self.tomato = RigidObject(self.cfg.tomato_cfg)
        else:
            self.tomato = _KinematicProxyObject(
                self.cfg.scene.num_envs,
                self.device,
                self.cfg.tomato_cfg.init_state.pos,
            )
        # The proxy sphere is kinematic and needs no floor. Skipping the floor
        # also keeps this smoke test independent from the remote Isaac asset
        # server used by GroundPlaneCfg in Isaac Sim 4.5.
        if self.use_linkbot_asset:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.use_linkbot_asset:
            self.scene.articulations["robot"] = self.robot
        if self.use_linkbot_asset:
            self.scene.rigid_objects["tomato"] = self.tomato
            light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
            light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        bounded = actions.clamp(-1.0, 1.0)
        self.action_delta = bounded - self.actions
        self.actions = bounded

    def _apply_action(self) -> None:
        # Convert six bounded synergies into independent-joint targets. Mimic
        # joints are intentionally not targeted a second time.
        a = self.actions
        desired = self.prev_hand_targets.clone()
        opposition_01 = 0.5 * (a[:, 0:1] + 1.0)
        desired[:, self.thumb_opposition_ids] = self.cfg.thumb_opposition_low + opposition_01 * (
            self.cfg.thumb_opposition_high - self.cfg.thumb_opposition_low
        )
        thumb_flex_01 = 0.5 * (a[:, 1:2] + 1.0)
        desired[:, self.thumb_flex_ids] = thumb_flex_01 * torch.tensor(
            (1.10, 0.72, 1.05), device=self.device
        )
        for action_col, finger in enumerate(("index", "middle", "ring", "little"), start=2):
            flex_01 = 0.5 * (a[:, action_col:action_col + 1] + 1.0)
            desired[:, self.finger_flex_ids[finger]] = flex_01 * torch.tensor((1.25, 1.00), device=self.device)
        desired = self.cfg.target_smoothing * desired + (1.0 - self.cfg.target_smoothing) * self.prev_hand_targets
        self.prev_hand_targets = desired
        self.robot.set_joint_position_target(desired[:, self.controlled_joint_ids], joint_ids=self.controlled_joint_ids)
        if not self.use_linkbot_asset:
            tomato_pos = self.tomato.data.root_pos_w
            tip = proxy_tip_positions(self.grasp_center, self.actions, self.cfg.tomato_radius)
            self.robot.data.body_pos_w[:, self.tip_ids] = tip
            base_id = self.robot.body_names.index("left_hand_base_link")
            self.robot.data.body_pos_w[:, base_id] = tomato_pos + torch.tensor(
                (0.0, -0.045, -0.010), device=self.device
            )

        tomato_pos = self.tomato.data.root_pos_w
        tip_pos = self.robot.data.body_pos_w[:, self.tip_ids]
        self.contact_loads, self.surface_gap = normalized_contact_loads(
            tip_pos, tomato_pos, self.cfg.tomato_radius, self.cfg.contact_band
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        tomato_pos = self.tomato.data.root_pos_w
        tip_pos = self.robot.data.body_pos_w[:, self.tip_ids]
        rel_tip = (tip_pos - tomato_pos.unsqueeze(1)).reshape(self.num_envs, -1)
        obs = torch.cat((
            self.robot.data.joint_pos[:, self.hand_joint_ids],
            self.robot.data.joint_vel[:, self.hand_joint_ids].clamp(-10.0, 10.0) * 0.1,
            rel_tip,
            self.contact_loads,
            self.actions,
        ), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        reward, metrics = enclosure_reward(
            self.contact_loads,
            self.surface_gap,
            self.action_delta,
            target_load=self.cfg.target_contact_load,
            safe_load=self.cfg.safe_contact_load,
        )
        self.stable_steps = torch.where(metrics["stable"], self.stable_steps + 1, 0)
        grasped = self.stable_steps >= self.cfg.stable_steps_required
        newly_grasped = grasped & ~self.success
        self.success |= grasped
        reward += self.cfg.success_bonus * newly_grasped.float()
        tomato_pos = self.tomato.data.root_pos_w
        reward -= self.cfg.drop_penalty * (tomato_pos[:, 2] < 0.20).float()
        log = self.extras.setdefault("log", {})
        log["grasp_success_rate"] = self.success.float().mean()
        log["contact_coverage"] = metrics["coverage"].mean()
        log["mean_contact_load"] = metrics["mean_load"].mean()
        log["max_contact_load"] = metrics["max_load"].mean()
        log["overload_rate"] = (metrics["overload"] > 0.0).float().mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        tomato_pos = self.tomato.data.root_pos_w
        terminated = tomato_pos[:, 2] < 0.15
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        tomato_state = self.tomato.data.default_root_state[env_ids].clone()
        tomato_state[:, :3] += self.scene.env_origins[env_ids]
        self.grasp_center[env_ids] = tomato_state[:, :3]
        jitter = (2.0 * torch.rand((len(env_ids), 3), device=self.device) - 1.0) * self.position_jitter
        tomato_state[:, :3] += jitter
        self.tomato.write_root_pose_to_sim(tomato_state[:, :7], env_ids)
        if self.use_linkbot_asset:
            self.tomato.write_root_velocity_to_sim(torch.zeros_like(tomato_state[:, 7:]), env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        if not self.use_linkbot_asset:
            self.robot.data.body_pos_w[env_ids] = tomato_state[:, :3].unsqueeze(1)
        self.prev_hand_targets[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.actions[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        self.contact_loads[env_ids] = 0.0
        self.surface_gap[env_ids] = 1.0
        self.stable_steps[env_ids] = 0
        self.success[env_ids] = False
