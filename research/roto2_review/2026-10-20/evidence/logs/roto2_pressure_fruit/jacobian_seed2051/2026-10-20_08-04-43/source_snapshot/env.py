"""Single-hand, continuous in-hand fruit rotation (Isaac Lab DirectRLEnv).

No arm, scripted phase switching, reward from pressure decreases, or dense
pressure leakage to the actor. Simulation-only dense load is a critic target.
"""
from __future__ import annotations

import torch
import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.utils.stage import get_current_stage

from module.skills.tomato_pick.hand_synergy import (
    PREGRASP_SYNERGY, g20_synergy_to_joint_targets,
)
from simulation.tomato_pick.isaac.assets import activate_nested_contact_reporters
from .core import (
    ACTIVE_JOINTS, MIMICS, FINGERS, POLICY_DIM,
    PRESSURE_REF_PA, surface_sites, pressure_from_contacts,
    fingertip_taxels, pressure_risk, rotation_reward,
    quat_inv_rotate, quat_rotate,
)
from .pressure_track import PressureJacobianTrack, hotspot_cost
from .env_cfg import FruitRotationEnvCfg
from .reward_geometry import pose_heading, heading_delta, pad_contact, reward_v2, REWARD_VERSION, pad_approach_quality, valid_rotation_step

CONTACT_BODIES = {
    "thumb": "thumb_link4",
    "index": "index_link3",
    "middle": "middle_link3",
    "ring": "ring_link3",
    "little": "little_link3",
    "palm": "base_link",
}


class FruitRotationEnv(DirectRLEnv):
    cfg: FruitRotationEnvCfg

    def __init__(self, cfg: FruitRotationEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._ids = torch.arange(self.num_envs, device=self.device)
        self._active_ids, found = self._robot.find_joints(
            list(ACTIVE_JOINTS), preserve_order=True
        )
        if tuple(found) != ACTIVE_JOINTS:
            raise RuntimeError(f"Active L20 joint mapping differs: {found}")
        self._follower_ids, follower_names = self._robot.find_joints(
            list(MIMICS), preserve_order=True
        )
        if len(self._follower_ids) != 5:
            raise RuntimeError(f"Expected five passive follower joints: {follower_names}")
        self._all_joint_ids = list(range(self._robot.num_joints))
        self._active_limits = self._robot.data.joint_pos_limits.torch[:, self._active_ids].clone()
        for follower, (driver, ratio) in MIMICS.items():
            fid = self._robot.find_joints(follower)[0][0]
            did = ACTIVE_JOINTS.index(driver)
            limits = self._robot.data.joint_pos_limits.torch[:, fid] / ratio
            self._active_limits[:, did, 0] = torch.maximum(self._active_limits[:, did, 0], limits[:, 0])
            self._active_limits[:, did, 1] = torch.minimum(self._active_limits[:, did, 1], limits[:, 1])
        self._contact_body_ids = [
            self._robot.find_bodies(CONTACT_BODIES[name])[0][0]
            for name in (*FINGERS, "palm")
        ]
        self._q_target = self._robot.data.default_joint_pos.torch[:, self._active_ids].clone()
        self._action = torch.zeros((self.num_envs, 16), device=self.device)
        self._previous_action = self._action.clone()
        self._goal = torch.ones(self.num_envs, device=self.device) * 0.6
        self._angle = torch.zeros(self.num_envs, device=self.device)
        self._heading_prev = self._angle.clone()
        self._pose_delta = self._angle.clone()
        self._rotation_highwater = self._angle.clone()
        self._new_rotation_progress = self._angle.clone()
        self._previous_upright = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._previous_pad_support = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pressure_prev = torch.zeros((self.num_envs, 24), device=self.device)
        self._panel_valid = torch.ones((self.num_envs, 6), device=self.device)
        self._overpressure_streak = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._center_reference_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._contact_centroid_prev = torch.zeros((self.num_envs, 3), device=self.device)
        self._track = PressureJacobianTrack(
            self.num_envs, self.device,
            enabled=(cfg.pressure_variant == "jacobian"),
        )
        self._previous_risk = torch.zeros(self.num_envs, device=self.device)
        self._pressure_relief = torch.zeros(self.num_envs, device=self.device)
        self._executed = torch.zeros((self.num_envs, 16), device=self.device)
        # Source 2026 multimodal_rl expects the task to expose this contract.
        self.obs_stack = cfg.obs_stack
        self.num_actions = cfg.num_actions
        self.num_eval_envs = cfg.num_eval_envs
        self.debug = cfg.debug

        self._axes_m = torch.as_tensor(
            self._prestartup_fruit_axes_m, device=self.device, dtype=torch.float32
        )
        self._dense_sites, self._sparse_sites = surface_sites(self.device)
        self._forces = torch.zeros((self.num_envs, 6), device=self.device)
        self._pad_forces = torch.zeros((self.num_envs, 5), device=self.device)
        self._pad_quality = self._pad_forces.clone()
        self._pad_valid = self._pad_forces.bool()
        self._force_vectors = torch.zeros((self.num_envs, 6, 3), device=self.device)
        self._contact_positions = torch.zeros((self.num_envs, 6, 3), device=self.device)
        self._dense = torch.zeros((self.num_envs, 96), device=self.device)
        self._sparse = torch.zeros((self.num_envs, 24), device=self.device)
        self._taxels = torch.zeros((self.num_envs, 5, 12, 6), device=self.device)
        # Warp data exposes xyzw in this project's Isaac 3 beta. The branch
        # below also supports a wxyz default pose; order is fixed at reset.
        q0 = self._fruit.data.default_root_pose.torch[0, 3:7]
        self._native_xyzw = bool(q0[3].abs() > q0[0].abs())
        self._reset_grasps = None
        if cfg.reset_grasps_file:
            self._reset_grasps = json.loads(Path(cfg.reset_grasps_file).read_text())

    def _wxyz(self, native_quat: torch.Tensor) -> torch.Tensor:
        return torch.cat((native_quat[..., 3:], native_quat[..., :3]), -1) if self._native_xyzw else native_quat

    def _setup_scene(self) -> None:
        from pxr import Usd, UsdPhysics
        self._robot = Articulation(self.cfg.robot)
        self._fruit = RigidObject(self.cfg.fruit)
        activated = activate_nested_contact_reporters(
            "/World/envs/env_0/Robot", tuple(CONTACT_BODIES.values())
        )
        stage = get_current_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        body_paths = {}
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in CONTACT_BODIES.values():
                body_paths[prim.GetName()] = str(prim.GetPath())
        if len(activated) != 6 or len(body_paths) != 6:
            raise RuntimeError(f"Right hand lost tip/palm reporters: {body_paths}")
        self._sensors = {}
        for name, body in CONTACT_BODIES.items():
            body_path = body_paths[body].replace("/env_0/", "/env_.*/")
            sensor_cfg = ContactSensorCfg(
                prim_path=body_path,
                update_period=self.cfg.sim.dt,
                history_length=0,
                filter_prim_paths_expr=[f"{self.scene.env_regex_ns}/Fruit"],
                track_pose=False,
                track_contact_points=True,
                track_friction_forces=False,
                max_contact_data_count_per_prim=64,
            )
            self._sensors[name] = ContactSensor(sensor_cfg)
            self.scene.sensors[f"fruit_{name}"] = self._sensors[name]
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["fruit"] = self._fruit
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=1800.0)
        light.func("/World/Light", light)

    def _measure(self) -> None:
        pose = self._fruit.data.root_pose_w.torch
        for i, name in enumerate((*FINGERS, "palm")):
            data = self._sensors[name].data
            matrix = data.force_matrix_w
            location = data.contact_pos_w
            if matrix is None or location is None:
                raise RuntimeError("Missing filtered physical contacts; never substitute fake sensor readings")
            force = matrix.torch[:, 0, 0]
            position = location.torch[:, 0, 0]
            # PhysX explicitly uses NaN for an absent contact point. A zero
            # resultant has no pressure contribution; its position is unused.
            # Reject nonfinite measurements if an actual force is present.
            has_force = torch.linalg.vector_norm(force, dim=-1) > 0
            if not torch.isfinite(force).all() or (
                has_force & ~torch.isfinite(position).all(-1)
            ).any():
                raise RuntimeError(f"Invalid physical contact measurement: {name}")
            self._force_vectors[:, i] = force
            self._contact_positions[:, i, :] = torch.where(
                has_force[:, None], position, pose[:, :3]
            )
        quat = self._wxyz(pose[:, 3:7])
        # Project measured contact resultant onto the local ellipsoid normal.
        # Force magnitude alone conflates normal compression with tangential
        # friction and overestimates film pressure during rotation.
        contact_local = quat_inv_rotate(
            quat[:, None, :].expand(-1, 6, -1),
            self._contact_positions - pose[:, None, :3],
        )
        ellipsoid_normal_local = torch.nn.functional.normalize(
            contact_local / self._axes_m[:, None, :].square().clamp_min(1e-7),
            dim=-1, eps=1e-6,
        )
        normal_w = quat_rotate(
            quat[:, None, :].expand(-1, 6, -1), ellipsoid_normal_local
        )
        self._forces.copy_(
            (self._force_vectors * normal_w).sum(-1).abs().clamp(0, 25)
        )
        # Remaining approximation: one effective contact location per link;
        # multiple simultaneous contact patches and fruit tissue are not solved.
        self._dense = pressure_from_contacts(
            self._forces, self._contact_positions, pose[:, :3],
            quat, self._axes_m, self._dense_sites
        )
        self._sparse = pressure_from_contacts(
            self._forces, self._contact_positions, pose[:, :3],
            quat, self._axes_m, self._sparse_sites
        )
        tip_body_pos = self._robot.data.body_pos_w.torch[:, self._contact_body_ids[:5]]
        tip_body_quat = self._wxyz(
            self._robot.data.body_quat_w.torch[:, self._contact_body_ids[:5]]
        )
        contact_local = quat_inv_rotate(
            tip_body_quat, self._contact_positions[:, :5] - tip_body_pos
        )
        contact_xy, self._pad_forces, self._pad_quality, self._pad_valid = pad_contact(
            contact_local, self._forces[:, :5])
        self._taxels = fingertip_taxels(self._pad_forces, contact_xy)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != self._action.shape:
            raise ValueError(f"Expected {self._action.shape} normalized actions")
        proposed = actions.clamp(-1.0, 1.0)
        self._previous_action.copy_(self._action)
        valid = self._panel_valid.repeat_interleave(4, dim=-1)
        grip_ok = (
            (self._pad_forces[:, 0] > self.cfg.minimum_support_force_n)
            & ((self._pad_forces[:, 1:5] > self.cfg.minimum_support_force_n).sum(-1) >= 2)
        )
        applied = self._track.correct(
            proposed, (self._sparse / PRESSURE_REF_PA).clamp(0, 4),
            valid, grip_ok
        )
        self._action.copy_(applied)
        self._executed.copy_(applied)
        q = self._robot.data.joint_pos.torch[:, self._active_ids]
        bounds = self._active_limits
        lead = self.cfg.maximum_joint_target_lead_rad
        proposal = self._q_target + applied * self.cfg.maximum_joint_target_delta_rad
        proposal = torch.maximum(q - lead, torch.minimum(q + lead, proposal))
        self._q_target = torch.maximum(
            bounds[..., 0], torch.minimum(bounds[..., 1], proposal)
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(
            target=self._q_target, joint_ids=self._active_ids
        )

    def _slip_speed(self) -> torch.Tensor:
        fruit_pos = self._fruit.data.root_pos_w.torch
        fruit_v = self._fruit.data.root_lin_vel_w.torch
        fruit_w = self._fruit.data.root_ang_vel_w.torch
        tips = self._robot.data.body_pos_w.torch[:, self._contact_body_ids[:5]]
        tip_v = self._robot.data.body_lin_vel_w.torch[:, self._contact_body_ids[:5]]
        arm = tips - fruit_pos[:, None]
        relative = tip_v - fruit_v[:, None] - torch.cross(
            fruit_w[:, None, :].expand_as(arm), arm, dim=-1
        )
        radial = torch.nn.functional.normalize(arm, dim=-1, eps=1e-6)
        tangential = relative - (relative * radial).sum(-1, keepdim=True) * radial
        active = self._forces[:, :5] > 0.15
        return (torch.linalg.vector_norm(tangential, dim=-1) * active).sum(-1) / active.sum(-1).clamp_min(1)

    def _get_dones(self):
        self._measure()
        current_heading = pose_heading(self._wxyz(self._fruit.data.root_pose_w.torch[:, 3:7]))
        self._pose_delta.copy_(heading_delta(self._heading_prev, current_heading))
        self._heading_prev.copy_(current_heading)
        self._angle += self._pose_delta
        directed = self._angle * self._goal.sign()
        self._new_rotation_progress.copy_((directed - self._rotation_highwater).clamp_min(0))
        self._rotation_highwater.copy_(torch.maximum(directed, self._rotation_highwater))
        self._track.observe(
            self._executed,
            (self._sparse / PRESSURE_REF_PA).clamp(0, 4),
            self._panel_valid.repeat_interleave(4, dim=-1),
            (self._pose_delta / (self.step_dt * 6.0)).clamp(-2, 2),
        )
        pressure_peak = self._dense.amax(-1)
        self._overpressure_streak = torch.where(
            pressure_peak > self.cfg.maximum_pressure_proxy_pa,
            self._overpressure_streak + 1,
            torch.zeros_like(self._overpressure_streak),
        )
        obj_pos = self._fruit.data.root_pos_w.torch
        displacement = torch.linalg.vector_norm(obj_pos - self._center_reference_w, dim=-1)
        failure = (
            (displacement > self.cfg.maximum_fruit_center_displacement_m)
            | (self._forces[:, :5].amax(-1) > self.cfg.maximum_contact_force_n)
            | (self._overpressure_streak >= self.cfg.overpressure_termination_steps)
            | (obj_pos[:, 2] < 0.78)
            | ~torch.isfinite(obj_pos).all(-1)
        )
        timeout = self.episode_length_buf >= (self.max_episode_length - 1)
        return failure, timeout

    def _get_rewards(self):
        dt = self.cfg.sim.dt * self.cfg.decimation
        signed_omega = self._pose_delta / dt
        fruit_pose = self._fruit.data.root_pose_w.torch
        quat = self._wxyz(fruit_pose[:, 3:7])
        z_axis = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, -1)
        local_z = quat_rotate(quat, z_axis)
        tilt = 1.0 - local_z[:, 2].clamp(-1.0, 1.0)
        center_error = torch.linalg.vector_norm(
            fruit_pose[:, :3] - self._center_reference_w, dim=-1
        )
        approach = pad_approach_quality(
            self._robot.data.body_pos_w.torch[:, self._contact_body_ids[:5]],
            self._wxyz(self._robot.data.body_quat_w.torch[:, self._contact_body_ids[:5]]),
            fruit_pose[:, :3], quat, self._axes_m)
        pose_step_valid = valid_rotation_step(self._previous_upright, tilt, self._pose_delta)
        self._previous_upright.copy_(tilt < .5)
        reward, terms, secure_support = reward_v2(
            self._pose_delta, self._goal, self._pad_forces, self._pad_quality,
            self._new_rotation_progress * pose_step_valid, center_error, tilt,
            self._action, self._previous_action, self._previous_pad_support, dt, approach, self.cfg.reward_stage)
        self._previous_pad_support.copy_(secure_support)
        # DirectRLEnv computes this exact failure mask before rewards. Keep
        # over-force / low-height failures subject to the same -60 penalty;
        # time-limit truncation is not a physical failure.
        terminated = self.reset_terminated
        # Pressure reduction is never itself a positive reward. This closes
        # the release/regrasp cycle loophole while keeping the Jacobian intact.
        valid = self._panel_valid.repeat_interleave(4, dim=-1)
        measured_risk = hotspot_cost((self._sparse / PRESSURE_REF_PA).clamp(0, 4), valid)
        self._pressure_relief.zero_()
        self._previous_risk.copy_(measured_risk.detach())
        if self.cfg.pressure_variant != "plain":
            reward -= 0.05 * measured_risk
        reward -= (40.0 if self.cfg.reward_stage == "contact" else 60.0) * terminated.float()
        self.extras["step"] = {
            "rotation_rate": signed_omega.detach().clone(),
            "pose_delta_rad": self._pose_delta.detach().clone(),
            "pose_step_valid": pose_step_valid.detach().clone(),
            "fruit_pose_native": fruit_pose.detach().clone(),
            "reported_angular_velocity_z": self._fruit.data.root_ang_vel_w.torch[:, 2].detach().clone(),
            "pad_contact": (self._pad_forces > .15).detach().clone(),
            "pad_quality": self._pad_quality.detach().clone(),
            "pad_approach_quality": approach.detach().clone(),
            "reward_rotation": terms["rotation"].detach().clone(),
            "pressure_risk": pressure_risk((self._dense / PRESSURE_REF_PA).clamp(0, 4)).detach().clone(),
            "peak_pressure_pa": self._dense.amax(-1).detach().clone(),
            "unwrapped_rotation_rad": self._angle.detach().clone(),
            "secure_support": secure_support.detach().clone(),
            "jacobian_accepted": self._track.last_accepted.detach().clone(),
            "slip_speed": (self._slip_speed() / 0.25).detach().clone(),
        }
        self.extras["log"] = {
            "rotation/signed_speed_rad_s": (signed_omega * self._goal.sign()).detach().clone(),
            "rotation/secure_grasp": secure_support.float().detach().clone(),
            "pressure/sparse_hotspot_risk": measured_risk.detach().clone(),
            "pressure/peak_pressure_proxy_pa": self._dense.amax(-1).detach().clone(),
            "reward/rotation": terms["rotation"].detach().clone(),
            "contact/pad_approach": approach.mean(-1).detach().clone(),
            "contact/pad_quality": self._pad_quality.mean(-1).detach().clone(),
            "contact/pad_count": (self._pad_forces > .15).sum(-1).float().detach().clone(),
            "pressure/gradient_track_acceptance": self._track.last_accepted.float().detach().clone(),
        }
        return reward

    def _get_observations(self):
        self._measure()
        q = self._robot.data.joint_pos.torch[:, self._active_ids]
        qd = self._robot.data.joint_vel.torch[:, self._active_ids]
        limits = self._robot.data.joint_pos_limits.torch[:, self._active_ids]
        qnorm = (2 * (q - limits[..., 0]) / (
            limits[..., 1] - limits[..., 0]).clamp_min(1e-5) - 1).clamp(-1, 1)
        # Actor observations are intentionally computed ONLY from fingertip
        # taxels, attached sparse pressure panels, FK and encoder velocity.
        # Ground-truth fruit pose / angular velocity stay privileged for critic.
        pose = self._fruit.data.root_pose_w.torch
        qbase = self._wxyz(self._robot.data.root_pose_w.torch[:, 3:7])
        base_pos = self._robot.data.root_pos_w.torch
        gravity_world = torch.tensor(
            (0., 0., -1.), device=self.device
        ).expand(self.num_envs, -1)
        rotation_axis_world = -gravity_world
        gravity = quat_inv_rotate(qbase, gravity_world)
        axis = quat_inv_rotate(qbase, rotation_axis_world)
        valid = self._panel_valid.repeat_interleave(4, dim=1)
        sparse = (self._sparse / PRESSURE_REF_PA).clamp(0., 4.)
        sparse_noisy = (
            sparse + torch.randn_like(sparse) * self.cfg.pressure_noise_std_fraction
        ).clamp(0., 4.) * valid
        delta = (sparse - self._pressure_prev).clamp(-4., 4.) * valid
        self._pressure_prev.copy_(sparse.detach())
        tactile = (self._taxels + torch.randn_like(self._taxels) *
                   self.cfg.tactile_noise_std_fraction).clamp(0., 4.)
        finger_weights = tactile.reshape(self.num_envs, 5, 72).sum(-1)
        tips = self._robot.data.body_pos_w.torch[
            :, self._contact_body_ids[:5]
        ]
        weighted_center = (
            (tips * finger_weights[..., None]).sum(dim=1)
            / finger_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        )
        centroid = torch.where(
            (finger_weights.sum(dim=1) > 1e-5)[:, None],
            weighted_center, tips.mean(dim=1),
        )
        relative_center = quat_inv_rotate(qbase, centroid - base_pos)
        centroid_velocity = quat_inv_rotate(
            qbase, (centroid - self._contact_centroid_prev)
            / (self.cfg.sim.dt * self.cfg.decimation)
        )
        self._contact_centroid_prev.copy_(centroid.detach())
        face_moments = sparse_noisy.reshape(self.num_envs, 6, 4).mean(-1)
        delta_faces = delta.reshape(self.num_envs, 6, 4).mean(-1)
        face_flux = torch.stack((
            delta_faces[:, 1] - delta_faces[:, 0],
            delta_faces[:, 3] - delta_faces[:, 2],
            delta_faces[:, 5] - delta_faces[:, 4],
        ), dim=-1)
        parts = (
            tactile.reshape(self.num_envs, -1), sparse_noisy, delta, valid,
            qnorm, (qd / 4.0).clamp(-3, 3),
            (relative_center / 0.20).clamp(-3, 3), face_moments,
            (centroid_velocity / 0.5).clamp(-3, 3),
            face_flux.clamp(-3, 3),
            gravity, axis, self._goal[:, None], self._action,
        )
        policy = torch.cat(parts, dim=1)
        if policy.shape[-1] != POLICY_DIM:
            raise RuntimeError(f"Policy layout drift: {policy.shape[-1]}")
        # 2026 RoTO 2.0 data contract: 4-frame tactile and proprioception.
        # The policy never gets privileged exact fruit state or dense oracle.
        self._last_observations = {"policy": {
            "tactile": policy[:, :432],
            "prop": policy[:, 432:],
        }}
        return self._last_observations

    def get_observations(self, env_ids=None):
        if env_ids is None:
            return self._last_observations
        # Evaluation resets must not consume another temporal delta/noise
        # sample or replace the already-stacked training observation.
        previous = self._last_observations
        pressure = self._pressure_prev.clone()
        centroid = self._contact_centroid_prev.clone()
        fresh = self._get_observations()
        keep = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        keep[env_ids] = False
        self._pressure_prev[keep] = pressure[keep]
        self._contact_centroid_prev[keep] = centroid[keep]
        for key in fresh["policy"]:
            fresh["policy"][key][keep] = previous["policy"][key][keep]
        return fresh

    def set_spaces(self, single_obs, obs, single_action, action):
        # Used by RoTO 2.0 common_utils.make_env without copying PPO code.
        self.single_observation_space = single_obs
        self.observation_space = obs
        self.single_action_space = single_action
        self.action_space = action

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._ids if hasattr(self, "_ids") else torch.arange(
                self.num_envs, device=self.device
            )
        # Keep full per-environment metrics for the upstream train/eval split.
        self.extras["counters"] = {
            "episode_rotation_abs_turns": self._angle.abs().clone() / (2 * torch.pi)
        }
        super()._reset_idx(env_ids)
        n = len(env_ids)
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        pregrasp = torch.tensor(
            g20_synergy_to_joint_targets(PREGRASP_SYNERGY),
            device=self.device, dtype=joint_pos.dtype,
        )
        if pregrasp.numel() != 16:
            raise RuntimeError("Unexpected LinkAgent calibrated pregrasp manifold")
        active_ids = self._active_ids
        active_limits = self._active_limits[env_ids]
        noise = torch.randn((n, 16), device=self.device) * self.cfg.reset_joint_noise_rad
        active = (pregrasp + noise).clamp(
            min=active_limits[..., 0], max=active_limits[..., 1]
        )
        if self._reset_grasps is not None:
            candidates = self._reset_grasps
            chosen = [candidates[int(i) % len(candidates)] for i in env_ids.cpu()]
            active = torch.tensor([c["joint_positions"] for c in chosen],
                                  device=self.device, dtype=joint_pos.dtype)
            active = (active + noise).clamp(min=active_limits[..., 0], max=active_limits[..., 1])
        joint_pos[:, active_ids] = active
        for follower_name, (driver_name, ratio) in MIMICS.items():
            fid = self._robot.find_joints(follower_name)[0][0]
            did = self._robot.find_joints(driver_name)[0][0]
            joint_pos[:, fid] = ratio * joint_pos[:, did]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_position_to_sim_index(
            position=joint_pos, joint_ids=self._all_joint_ids, env_ids=env_ids
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel, joint_ids=self._all_joint_ids, env_ids=env_ids
        )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        tips = self._robot.data.body_pos_w.torch[
            env_ids[:, None], self._contact_body_ids[:5]
        ]
        palm = self._robot.data.body_pos_w.torch[
            env_ids, self._contact_body_ids[-1]
        ]
        finger_mean = tips[:, 1:4].mean(1)
        direction = torch.nn.functional.normalize(
            finger_mean - palm, dim=-1, eps=1e-6
        )
        initial = finger_mean + direction * 0.034
        if self._reset_grasps is not None:
            local = torch.tensor([c["fruit_position_hand"] for c in chosen],
                                 device=self.device, dtype=joint_pos.dtype)
            base = self._robot.data.root_pose_w.torch[env_ids]
            initial = base[:, :3] + quat_rotate(self._wxyz(base[:, 3:]), local)
        initial += torch.randn_like(initial) * self.cfg.reset_position_noise_m
        root_pose = self._fruit.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] = initial
        self._fruit.write_root_pose_to_sim_index(
            root_pose=root_pose, env_ids=env_ids
        )
        self._fruit.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((n, 6), device=self.device), env_ids=env_ids
        )
        self._q_target[env_ids] = active
        self._action[env_ids] = 0
        self._previous_action[env_ids] = 0
        self._goal[env_ids] = (
            torch.where(torch.rand(n, device=self.device) > 0.5, 1., -1.)
            * torch.empty(n, device=self.device).uniform_(
                *self.cfg.target_angular_speed_rad_s
            )
        )
        self._angle[env_ids] = 0
        self._heading_prev[env_ids] = pose_heading(self._wxyz(root_pose[:, 3:7]))
        self._pose_delta[env_ids] = 0
        self._rotation_highwater[env_ids] = 0
        self._new_rotation_progress[env_ids] = 0
        self._previous_pad_support[env_ids] = False
        self._previous_upright[env_ids] = True
        self._pad_forces[env_ids] = 0
        self._pad_quality[env_ids] = 0
        self._pad_valid[env_ids] = False
        self._pressure_prev[env_ids] = 0
        self._center_reference_w[env_ids] = initial
        self._contact_centroid_prev[env_ids] = tips.mean(dim=1)
        self._overpressure_streak[env_ids] = 0
        self._track.reset(env_ids)
        self._previous_risk[env_ids] = 0
        self._pressure_relief[env_ids] = 0
        self._executed[env_ids] = 0
        self._forces[env_ids] = 0
        self._sparse[env_ids] = 0
        self._dense[env_ids] = 0
        self._panel_valid[env_ids] = (
            torch.rand((n, 6), device=self.device)
            > self.cfg.sparse_panel_dropout_probability
        ).float()
