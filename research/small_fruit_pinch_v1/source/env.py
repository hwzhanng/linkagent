"""左手小果三指捏取 PhysX 环境。

该环境只复用历史工作区的左手 USD、接触传感器读数和低层关节安全范围。
它不复用历史大果掌托成功条件、五指包络奖励、采摘状态机或 checkpoint。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from .contract import EPISODE_STEPS, FORCE_BANDS, POLICY_DT_S
from .legacy_bridge import load_legacy_physx
from .temporal import CriticalDampedActionFilter, TemporalLimits
from .verifier import TimStrictVerifier, VerifierConfig


_LegacyEnv, _LegacyCfg = load_legacy_physx()


@configclass
class SmallFruitPinchEnvCfg(_LegacyCfg):
    """三指捏取的独立配置，禁止掌面支撑作为成功捷径。"""

    seed = 20260831
    episode_length_s = EPISODE_STEPS * POLICY_DT_S
    training_stage = 0
    # 果实依然由物理果梗支撑，便于固定腕部的手部策略学习；它不落在掌面。
    pregrasp_stem_wrench_scale = 1.0
    pinch_vertical_offset_m = 0.030
    # B0 标称回归域：64--70 mm 直径。
    axis_range_m = (0.032, 0.035)
    density_range_kg_m3 = (700.0, 1050.0)
    accepted_mass_kg = (0.050, 0.180)
    position_jitter = (0.0015, 0.0015, 0.0010)
    friction_range = (0.55, 0.80)
    restitution_range = (0.0, 0.02)
    stiffness_range_n_m = (10000.0, 26000.0)
    tactile_gain_range = (0.90, 1.10)
    tactile_bias_range_n = (-0.02, 0.02)
    sensor_delay_steps_max = 1
    sensor_dropout_probability = 0.005
    sensor_stale_probability = 0.005
    actuator_response_scale_range = (0.96, 1.04)
    actuator_bias_range_rad = (-0.002, 0.002)
    actuator_deadband_range_rad = (0.0, 0.0015)
    actuator_backlash_range_rad = (0.0, 0.002)
    command_rate_limit_rad_s = 0.45
    # 不开启历史的掌托拾取、断梗或 post-break reflex。
    palm_support_force_n = 1.0e6
    minimum_enclosing_digits = 3
    success_hold_steps = 80
    safe_force_n = 3.50
    force_ceiling_n = tuple(band.hard_high_n for band in FORCE_BANDS)
    severe_overload_scale = 1.0
    drop_height_m = 0.10
    # 新奖励由子类完全定义；这些字段仅让历史低层环境保持稳定。
    success_bonus = 0.0
    contact_reward_weight = 0.0
    palm_support_reward_weight = 0.0
    secure_reward_weight = 0.0
    stable_hold_reward_weight = 0.0


class SmallFruitPinchEnv(_LegacyEnv):
    """连续接触—持稳—扰动恢复的 10D 左手捏取环境。"""

    cfg: SmallFruitPinchEnvCfg

    def __init__(self, cfg: SmallFruitPinchEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._raw_actions = torch.zeros_like(self.actions)
        self._filter = CriticalDampedActionFilter(
            self.num_envs,
            self.cfg.action_space,
            dt_s=float(self.cfg.policy_dt_s),
            device=self.device,
            limits=TemporalLimits(),
        )
        self._verifier = TimStrictVerifier(self.num_envs, device=self.device, cfg=VerifierConfig())
        self._pinch_best_reach = torch.zeros(self.num_envs, device=self.device)
        self._pinch_hard_overload = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pinch_drop = torch.zeros_like(self._pinch_hard_overload)
        self._pinch_self_collision = torch.zeros_like(self._pinch_hard_overload)
        self._disturbance_applied = torch.zeros_like(self._pinch_hard_overload)
        direction = torch.tensor(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), device=self.device)
        self._disturbance_directions = direction[torch.randint(0, 4, (self.num_envs,), device=self.device)]
        self._filter.reset()
        self._verifier.reset()

    @torch.inference_mode()
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_filter"):
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # 果实相对掌面抬高，仅由真实、可见的物理果梗支撑；不使用 capture
        # assist、吸附或位置锁定。该高度会在 P1 几何扫描中校验。
        transforms = self.tomato.root_physx_view.get_transforms().clone()
        physx_ids = ids.to(device=transforms.device)
        transforms[physx_ids, 2] += float(self.cfg.pinch_vertical_offset_m)
        self.tomato.root_physx_view.set_transforms(transforms, indices=physx_ids)
        velocities = self.tomato.root_physx_view.get_velocities().clone()
        velocities[physx_ids] = 0.0
        self.tomato.root_physx_view.set_velocities(velocities, indices=physx_ids)
        offset = torch.tensor((0.0, 0.0, float(self.cfg.pinch_vertical_offset_m)), device=self.device)
        self.grasp_center[ids] += offset
        self.palm_support_point[ids] += offset
        self.anchor_rest_pos[ids] += offset
        self.anchor_pos[ids] += offset
        self._filter.reset(ids)
        self._verifier.reset(ids)
        self._raw_actions[ids] = 0.0
        self._pinch_best_reach[ids] = 0.0
        self._pinch_hard_overload[ids] = False
        self._pinch_drop[ids] = False
        self._pinch_self_collision[ids] = False
        self._disturbance_applied[ids] = False
        directions = torch.tensor(
            ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
            device=self.device,
        )
        self._disturbance_directions[ids] = directions[
            torch.randint(0, 4, (ids.numel(),), device=self.device)
        ]

    @torch.inference_mode()
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0))
        hold_phase = self._verifier.hold_age > 0
        executed = self._filter.step(self._raw_actions, hold_phase=hold_phase)
        super()._pre_physics_step(executed)

    @torch.inference_mode()
    def _tip_self_collision_proxy(self) -> torch.Tensor:
        """保守的指尖间距监测；P1 将以接触几何扫描补充实体自碰撞审计。"""

        tips = self._tip_positions()
        distances = torch.cdist(tips, tips)
        eye = torch.eye(distances.shape[-1], dtype=torch.bool, device=self.device).unsqueeze(0)
        distances = torch.where(eye, torch.full_like(distances, float("inf")), distances)
        return distances.amin(dim=(-2, -1)) < 0.004

    @torch.inference_mode()
    def _apply_standard_disturbance(self, due: torch.Tensor) -> None:
        due = due & ~self._disturbance_applied
        if not bool(due.any()):
            return
        masses = self.tomato.root_physx_view.get_masses()[:, 0].to(self.device)
        force = masses.unsqueeze(-1) * (0.25 * 9.81) * self._disturbance_directions
        force = torch.where(due.unsqueeze(-1), force, torch.zeros_like(force))
        self.tomato.root_physx_view.apply_forces_and_torques_at_position(
            force_data=force,
            torque_data=torch.zeros_like(force),
            position_data=None,
            indices=self.tomato._ALL_INDICES,
            is_global=True,
        )
        self._disturbance_applied |= due

    @torch.inference_mode()
    def _get_rewards(self) -> torch.Tensor:
        self._read_contact_forces()
        tomato_pos = self.tomato.data.root_pos_w
        center_drift = torch.linalg.vector_norm(tomato_pos - self.grasp_center, dim=-1)
        relative_speed = self._relative_speed()
        palm_contact = self.palm_force >= self.cfg.contact_threshold_n
        self._pinch_self_collision |= self._tip_self_collision_proxy()
        status = self._verifier.update(
            force_n=self.contact_forces,
            palm_contact=palm_contact,
            center_drift_m=center_drift,
            relative_speed_m_s=relative_speed,
            safety_projection=self.joint_safety_projection > 0.0,
            self_collision=self._pinch_self_collision,
        )
        self._pinch_hard_overload |= status["hard_overload"]
        tips = self._tip_positions()[:, :3]
        tip_distance = torch.linalg.vector_norm(tips - tomato_pos.unsqueeze(1), dim=-1)
        radius = self.tomato_axes.mean(dim=-1, keepdim=True)
        reach = torch.exp(-((tip_distance - radius).abs() / 0.035)).mean(dim=-1)
        reach_progress = (reach - self._pinch_best_reach).clamp_min(0.0)
        self._pinch_best_reach = torch.maximum(self._pinch_best_reach, reach)
        force = self.contact_forces[:, :3]
        lows = force.new_tensor([band.soft_low_n for band in FORCE_BANDS[:3]])
        highs = force.new_tensor([band.soft_high_n for band in FORCE_BANDS[:3]])
        force_band_score = (
            1.0 - ((force - (lows + highs) * 0.5) / ((highs - lows) * 0.5)).abs()
        ).clamp_min(0.0).mean(dim=-1)
        filter_stats = self._filter.last_stats
        action_delta = filter_stats.get("action_delta", torch.zeros_like(self.actions)).mean(dim=-1)
        acceleration = filter_stats.get("acceleration", torch.zeros_like(self.actions)).mean(dim=-1)
        jerk = filter_stats.get("jerk", torch.zeros_like(self.actions)).mean(dim=-1)
        tri = status["tim_contact"].float()
        hold_progress = (status["hold_age"].float() / float(self._verifier.cfg.hold_steps)).clamp(0.0, 1.0)
        new_success = status["success_event"]
        reward = (
            1.2 * reach_progress
            + 0.25 * tri
            + 0.40 * force_band_score * tri
            + 0.35 * hold_progress * tri
            + 8.0 * new_success.float()
            - 0.08 * action_delta
            - 0.010 * acceleration
            - 0.002 * jerk
            - 0.40 * palm_contact.float()
            - 2.0 * status["hard_overload"].float()
            - 1.0 * self._pinch_self_collision.float()
        )
        self.success |= status["final_success"]
        self._apply_standard_disturbance(status["disturbance_due"])
        log = self.extras.setdefault("log", {})
        log.update({
            "pinch/tim_contact_rate": status["tim_contact"].float().mean(),
            "pinch/hold_step_mean": status["hold_age"].float().mean(),
            "pinch/strict_success_rate": self.success.float().mean(),
            "pinch/contact_ratio": status["contact_ratio"].mean(),
            "pinch/force_band_ratio": status["band_ratio"].mean(),
            "pinch/palm_contact_ratio": status["palm_ratio"].mean(),
            "pinch/action_delta_mean": action_delta.mean(),
            "pinch/action_accel_mean": acceleration.mean(),
            "pinch/action_jerk_mean": jerk.mean(),
            "pinch/disturbance_applied_rate": self._disturbance_applied.float().mean(),
        })
        return reward

    @torch.inference_mode()
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        tomato_pos = self.tomato.data.root_pos_w
        center_drift = torch.linalg.vector_norm(tomato_pos - self.grasp_center, dim=-1)
        dropped_now = (tomato_pos[:, 2] < self.cfg.drop_height_m) | (center_drift > 0.12)
        self._pinch_drop |= dropped_now
        terminated = self.success | dropped_now | self._pinch_hard_overload | self._pinch_self_collision
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        code = torch.zeros_like(self.last_termination_code)
        code = torch.where(self.success, torch.ones_like(code), code)
        code = torch.where(dropped_now, torch.full_like(code, 2), code)
        code = torch.where(self._pinch_hard_overload, torch.full_like(code, 3), code)
        code = torch.where(self._pinch_self_collision, torch.full_like(code, 4), code)
        code = torch.where(timeout & ~terminated, torch.full_like(code, 5), code)
        self.last_termination_code.copy_(code)
        return terminated, timeout


__all__ = ["SmallFruitPinchEnv", "SmallFruitPinchEnvCfg"]
