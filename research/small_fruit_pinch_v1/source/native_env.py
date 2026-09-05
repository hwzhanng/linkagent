"""小果捏取 v1 的当前 Isaac Lab 原生左手 PhysX 环境。

这不是历史 89% 任务的复训：手臂保持在确定的左臂初始位姿，策略只输出
历史可迁移的 10D 手部流形动作。水果、五个指尖接触、茎弹簧和真实 PhysX
仍由当前项目的原生 ``TomatoPickEnv`` 提供。
"""

from __future__ import annotations

from copy import deepcopy
import re

import torch

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

from .contract import ACTOR_OBSERVATION_DIM, ACTION_DIM, CRITIC_STATE_DIM, EPISODE_STEPS
from .pinch_action_manifold import pinch_action_to_joint_targets
from .temporal import CriticalDampedActionFilter
from .verifier import TimStrictVerifier


def _left(value: str) -> str:
    """把当前原生任务的右手命名空间映射到同构的左手命名空间。"""

    return re.sub(
        r"/R([1-8]_Link)", lambda match: f"/L{match.group(1)}", value.replace("right_hand", "left_hand")
    )


def configure_native_module_for_left() -> tuple[object, object]:
    """在本进程内配置原生环境为左手；不修改项目源文件。"""

    from simulation.tomato_pick.isaac import env as native
    from simulation.tomato_pick.isaac import env_cfg as cfg_module

    native.ACTION_DIM = ACTION_DIM
    native.RIGHT_HAND_CONTROL_JOINT_NAMES = tuple(
        name.replace("right_hand_", "left_hand_", 1)
        for name in native.RIGHT_HAND_CONTROL_JOINT_NAMES
    )
    native.RIGHT_FINGERTIP_FRAME_BODY_NAMES = {
        name: _left(value) for name, value in native.RIGHT_FINGERTIP_FRAME_BODY_NAMES.items()
    }
    native.RIGHT_FINGERTIP_CONTACT_BODY_NAMES = {
        name: _left(value) for name, value in native.RIGHT_FINGERTIP_CONTACT_BODY_NAMES.items()
    }
    native.RIGHT_HAND_BODY_NAMES = tuple(_left(value) for value in native.RIGHT_HAND_BODY_NAMES)
    cfg_module.RIGHT_FINGERTIP_PATHS = {
        name: _left(value) for name, value in cfg_module.RIGHT_FINGERTIP_PATHS.items()
    }
    return native, cfg_module


native, native_cfg_module = configure_native_module_for_left()


@configclass
class NativeSmallFruitPinchCfg(native_cfg_module.TomatoPickEnvCfg):
    """固定手臂、14D 策略、114D/175D 合同的训练配置。"""

    episode_length_s = EPISODE_STEPS / 20.0
    action_space = ACTION_DIM
    observation_space = ACTOR_OBSERVATION_DIM
    state_space = CRITIC_STATE_DIM
    # 训练阶段不让策略寻找水果；只在闭合工作空间内做尺寸和横向扰动。
    # 左腕零位下的实测闭合工作区。右手任务的 (0.148,-0.0607,0.0535)
    # 不能镜像复用：它会把果实放在左手指尖约 12 cm 之外。先以无接近
    # 平移的局部工作区训练手部闭合；后续 P1 只在此小区域随机化。
    closed_grasp_fruit_center_m = (0.007, -0.066, 0.160)
    pregrasp_distance_range_m = (0.0, 0.0)
    pregrasp_lateral_offset_range_m = (0.0, 0.0)
    pregrasp_axial_noise_m = 0.0
    pregrasp_arm_joint_noise_deg = 0.0
    minimum_initial_clearance_m = 0.002
    # P1 联合扫描得到的无穿透三指共同可达位移；它相对父环境已经通过
    # 全碰撞体 clearance 修正后的初始位姿表达，不是硬编码最终关节姿态。
    pinch_working_offset_m = (0.018, 0.030, 0.024)
    action_delay_steps = (0, 0)
    # 父类在定义期就创建了右手传感器字典；这里深复制后改路径，不能只改
    # 模块常量，否则 Isaac Lab 会继续注册右手指尖。
    fingertip_contacts = deepcopy(native_cfg_module.TomatoPickEnvCfg().fingertip_contacts)
    for _sensor_cfg in fingertip_contacts.values():
        _sensor_cfg.prim_path = _left(_sensor_cfg.prim_path)
    auxiliary_contacts = {
        "palm": ContactSensorCfg(
            prim_path=_left(native_cfg_module.RIGHT_HAND_BODY_ROOT),
            update_period=1.0 / 120.0,
            history_length=0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
            track_pose=True,
            track_contact_points=False,
            track_friction_forces=False,
        )
    }
    # B0 的物理域按计划固定为果径 64–70 mm（椭球半轴 32–35 mm）。父任务
    # 默认是 50–70 mm；如果不在这里覆写，训练日志会把小于 B0 的样本误算
    # 成标称域，导致阶段门和回归比较失真。后续 B1–B3 通过显式配置分支
    # 扩大该范围，不修改历史检查点的含义。
    events = deepcopy(native_cfg_module.TomatoPickEventCfg())
    events.fruit_geometry.params["axis_ranges_m"] = {
        "x": (0.032, 0.035),
        "y": (0.032, 0.035),
        "z": (0.032, 0.035),
    }


class NativeSmallFruitPinchEnv(native.TomatoPickEnv):
    """静态左手小果捏取：连续闭合、低力持稳、无掌托成功捷径。"""

    cfg: NativeSmallFruitPinchCfg
    # 位置驱动的目标默认在每个策略步保持不变。显式开启后，目标会在
    # 一个 policy step 的 PhysX 子步内做线性微轨迹，避免 20 Hz 的位置
    # 目标跳变被隐式 PD 解释成瞬时大力。默认 1 保持历史合同不变。
    position_target_interpolation_steps = 1
    # PhysX backends that fold decimation into one ``sim.step`` call cannot
    # receive a new position target on every physics substep.  The default
    # keeps the historical path untouched; the structural smoothing branch
    # explicitly enables manual decimation so a continuous reference can be
    # sent at the native 120 Hz physics rate.
    manual_decimation_control = False

    def __init__(self, cfg: NativeSmallFruitPinchCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self._manual_decimation_active = bool(
            self.manual_decimation_control
            and not self.sim.physics_manager.handles_decimation()
        )
        if self._manual_decimation_active:
            # DirectRLEnv.step() will now execute the configured decimation
            # loop itself.  This is safe for the PhysX manager (which steps a
            # single dt); if a backend owns decimation, the opt-in remains a
            # no-op instead of silently advancing the simulation twice.
            self._physics_handles_decimation = False
        # 基类把右腕名称写在构造函数中，不能仅靠全局指尖路径映射替换。
        # 这里在任何 reset 前显式切换为左腕参考帧，确保果实采样位姿与
        # 左手而不是右手绑定。
        self._ee_body_id = self._robot.find_bodies("left_hand_base_link")[0][0]
        self._ee_jacobian_id = (
            self._ee_body_id - 1 if self._robot.is_fixed_base else self._ee_body_id
        )
        self._filter = CriticalDampedActionFilter(
            self.num_envs, ACTION_DIM, dt_s=1.0 / 20.0, device=self.device
        )
        self._verifier = TimStrictVerifier(self.num_envs, device=self.device)
        self._raw_actions = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        self._executed_actions = torch.zeros_like(self._raw_actions)
        self._previous_executed_actions = torch.zeros_like(self._raw_actions)
        self._previous_force_n = torch.zeros((self.num_envs, 5), device=self.device)
        self._force_limiter_scale = torch.ones(self.num_envs, device=self.device)
        self._pinch_reference_center_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._pinch_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pinch_drop = torch.zeros_like(self._pinch_success)
        self._pinch_overload = torch.zeros_like(self._pinch_success)
        self._pinch_self_collision = torch.zeros_like(self._pinch_success)
        self._last_palm_force_n = torch.zeros(self.num_envs, device=self.device)
        self._last_verification: dict[str, torch.Tensor] = {}
        # hand_control_ids 同时包含 16 个 canonical 驱动轴和 10 个 mimic
        # 跟随轴；插值必须覆盖二者，否则跟随轴仍会在首个子步产生跳变。
        self._hand_control_ids = list(
            dict.fromkeys(self._hand_joint_ids + self._mimic_follower_ids)
        )
        self._position_command_current = self._joint_targets[:, self._hand_control_ids].clone()
        self._position_command_start = self._position_command_current.clone()
        self._position_command_goal = self._position_command_current.clone()
        # 插值进度按 environment 保存，而不是一个全局整数。DirectRLEnv
        # 在一个 batch 中可能同时 reset 部分环境；若用全局计数器，单个
        # 环境的 reset 会把其它仍在运行的环境的子步进度清零，形成新的
        # 目标斜率跳变。每个策略步开始时统一清零，随后各 env 独立推进。
        self._position_interpolation_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"actions 必须为 {tuple(self._raw_actions.shape)}，得到 {tuple(actions.shape)}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions 包含 NaN 或 Inf")
        self._previous_executed_actions.copy_(self._executed_actions)
        self._raw_actions.copy_(actions.clamp(-1.0, 1.0))
        hold_phase = self._verifier.hold_age > 0
        # 结构化 residual 环境可以在三指 Verifier latch 之前就请求慢速
        # 接触相位；普通 Native 环境没有该属性，保持原有合同和速率。
        force_slow_phase = getattr(self, "_force_slow_phase", None)
        if force_slow_phase is not None:
            hold_phase = hold_phase | force_slow_phase.to(dtype=torch.bool)
        filtered = self._filter.step(
            self._raw_actions,
            hold_phase=hold_phase,
            speed_scale=getattr(self, "_filter_speed_scale", None),
        )
        # 只在上一物理步已经接近硬压力上限时缩短下一步动作的行程。
        # 这是面向真实执行器的软安全层，不是成功判据，也不锁定动作；
        # 策略仍会收到实时压力观测并学习避开该区域。
        hard_high = self._last_physical_force_n.new_tensor((4.0, 3.5, 3.5))
        required = self._last_physical_force_n[:, :3]
        near_hard = torch.relu(required / (0.70 * hard_high) - 1.0).amax(dim=1)
        scale = (1.0 - 0.45 * near_hard).clamp(0.45, 1.0)
        self._force_limiter_scale.copy_(scale)
        self._executed_actions.copy_(
            self._previous_executed_actions
            + scale.unsqueeze(1) * (filtered - self._previous_executed_actions)
        )
        targets = pinch_action_to_joint_targets(self._executed_actions)
        targets = self._postprocess_joint_targets(targets)
        self._joint_targets[:, self._hand_joint_ids] = targets
        self._update_mimic_targets(torch.arange(self.num_envs, device=self.device))
        # GuidedResidual 可能在 super() 返回后继续改写 canonical 目标；这里
        # 只重置子步计数，真正的 start/goal 捕获延迟到第一次 _apply_action，
        # 因而总能看到该策略步的最终目标。
        self._position_interpolation_step.zero_()
        self._stem_break_event.zero_()

    def _postprocess_joint_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Hook for simulation-only structured controllers.

        The base 14D environment keeps the exact action contract.  A subclass
        may apply a stateful, force-aware projection after the temporal filter
        and before mimic expansion; the projection must return canonical 16D
        left-hand targets and must not issue hardware commands.
        """

        return targets

    def _apply_action(self) -> None:
        # 不调用原任务的 IK：手臂不由 PPO 控制，避免把手部策略问题混入臂手协同。
        interpolation_steps = max(1, int(self.position_target_interpolation_steps))
        if interpolation_steps > 1 and not self._physics_handles_decimation:
            initial = self._position_interpolation_step == 0
            if bool(initial.any()):
                self._position_command_start[initial] = self._position_command_current[initial]
                self._position_command_goal[initial] = self._joint_targets[initial][
                    :, self._hand_control_ids
                ]
            self._position_interpolation_step.add_(1)
            fraction = (
                self._position_interpolation_step.clamp_max(interpolation_steps)
                .to(dtype=self._joint_targets.dtype)
                .div(float(interpolation_steps))
                .unsqueeze(1)
            )
            command = self._position_command_start + fraction * (
                self._position_command_goal - self._position_command_start
            )
            self._position_command_current.copy_(command)
            position_targets = self._joint_targets[:, self._actuated_joint_ids].clone()
            position_targets[:, self._hand_control_ids] = command
        else:
            self._position_command_current.copy_(
                self._joint_targets[:, self._hand_control_ids]
            )
            position_targets = self._joint_targets[:, self._actuated_joint_ids]
        hold_effort = None
        hold_effort_hook = getattr(self, "_hold_effort_action", None)
        if hold_effort_hook is not None:
            position_targets, hold_effort = hold_effort_hook(position_targets)
        self._robot.set_joint_position_target_index(
            target=position_targets,
            joint_ids=self._actuated_joint_ids,
        )
        if hold_effort is not None:
            self._robot.set_joint_velocity_target_index(
                target=torch.zeros_like(hold_effort),
                joint_ids=self._hand_joint_ids,
            )
            self._robot.set_joint_effort_target_index(
                target=hold_effort,
                joint_ids=self._hand_joint_ids,
            )
        # 可选的真实速度阻尼前馈：在安全保持段将实测关节速度转换成
        # 有界反向努力，叠加在隐式位置驱动上。它不截断状态，也不改变
        # 接触/成功判据；压力越过软带立即撤掉该项，保留回撤能力。
        velocity_damping = getattr(self, "hold_velocity_damping_effort", None)
        if velocity_damping is not None:
            velocity_damping = float(velocity_damping)
            if velocity_damping < 0.0:
                raise ValueError("hold_velocity_damping_effort 不能为负数")
            force = self._last_physical_force_n[:, :3]
            soft_high = force.new_tensor(
                tuple(float(value) for value in getattr(self, "soft_high_n", (1.8, 1.5, 1.3)))
            )
            safe = self._verifier.hold_age > 0
            force_slow_phase = getattr(self, "_force_slow_phase", None)
            if force_slow_phase is not None:
                safe = safe | force_slow_phase.to(dtype=torch.bool)
            safe &= (force <= soft_high).all(dim=1)
            velocity = self._robot.data.joint_vel.torch[:, self._hand_joint_ids]
            effort = (-velocity_damping * velocity).clamp(-1.0, 1.0)
            effort = torch.where(safe.unsqueeze(1), effort, torch.zeros_like(effort))
            self._robot.set_joint_effort_target_index(
                target=effort,
                joint_ids=self._hand_joint_ids,
            )
        # 可选的保持段速度参考。位置目标仍是主控制量；该速度参考只在
        # 三指保持/接触减速阶段生效，用于降低隐式 PD 在目标附近的过冲，
        # 不会限制接近阶段，也不改变 Verifier 的接触、压力或掉落判据。
        zero_hold_velocity = bool(getattr(self, "hold_zero_velocity_target", False))
        velocity_limit = getattr(self, "hold_joint_velocity_target_rad_s", None)
        if zero_hold_velocity or velocity_limit is not None:
            if zero_hold_velocity:
                velocity_target = torch.zeros(
                    (self.num_envs, len(self._hand_joint_ids)),
                    device=self.device,
                    dtype=position_targets.dtype,
                )
            else:
                velocity_limit = float(velocity_limit)
                if velocity_limit <= 0.0:
                    raise ValueError("hold_joint_velocity_target_rad_s 必须为正数")
                current = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
                target = self._joint_targets[:, self._hand_joint_ids]
                desired_velocity = ((target - current) / (1.0 / 20.0)).clamp(
                    -velocity_limit, velocity_limit
                )
                velocity_target = desired_velocity
            hold = self._verifier.hold_age > 0
            force_slow_phase = getattr(self, "_force_slow_phase", None)
            if force_slow_phase is not None:
                hold = hold | force_slow_phase.to(dtype=torch.bool)
            velocity_target = torch.where(hold.unsqueeze(1), velocity_target, torch.zeros_like(velocity_target))
            self._robot.set_joint_velocity_target_index(
                target=velocity_target,
                joint_ids=self._hand_joint_ids,
            )
        arm_ids = self._arm_joint_ids + self._left_arm_joint_ids
        gravity_columns = [index + self._robot.num_base_dofs for index in arm_ids]
        gravity_effort = self._robot.data.gravity_compensation_forces.torch[
            :, gravity_columns
        ]
        self._robot.set_joint_effort_target_index(target=gravity_effort, joint_ids=arm_ids)
        self._apply_attachment_wrench()

    def _palm_force(self) -> torch.Tensor:
        sensor = self._aux_contact_sensors.get("palm")
        if sensor is None or sensor.data.force_matrix_w is None:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.linalg.vector_norm(sensor.data.force_matrix_w.torch[:, 0, 0], dim=-1)

    def _self_collision_proxy(self) -> torch.Tensor:
        # 指尖彼此的极小距离是保守代理；它绝不作为“成功”证据。
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        distance = torch.cdist(tips, tips)
        eye = torch.eye(5, dtype=torch.bool, device=self.device).unsqueeze(0)
        return distance.masked_fill(eye, 1.0).amin(dim=(1, 2)) < 0.004

    def _verification_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        force = self._physical_finger_force()
        self._last_physical_force_n.copy_(force)
        self._last_palm_force_n.copy_(self._palm_force())
        center_drift = torch.linalg.vector_norm(
            self._tomato.data.root_pos_w.torch - self._pinch_reference_center_w, dim=1
        )
        ee_velocity = self._robot.data.body_lin_vel_w.torch[:, self._ee_body_id]
        relative_speed = torch.linalg.vector_norm(
            self._tomato.data.root_lin_vel_w.torch - ee_velocity, dim=1
        )
        self._pinch_self_collision = self._self_collision_proxy()
        return force, center_drift, relative_speed, self._pinch_self_collision

    def _refresh_verifier(self) -> None:
        force, drift, relative_speed, self_collision = self._verification_inputs()
        self._last_verification = self._verifier.update(
            force_n=force,
            palm_contact=self._last_palm_force_n >= 0.08,
            center_drift_m=drift,
            relative_speed_m_s=relative_speed,
            safety_projection=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            self_collision=self_collision,
        )
        # `success` 是 80 步严格持稳的阶段 latch；只有追加 40 步连续保持
        # 后的 `final_success` 才允许终止并计入独立评估，保证抓住后仍有
        # 动态微调窗口。
        self._pinch_success.copy_(self._last_verification["final_success"])
        hard_high = force.new_tensor([4.0, 3.5, 3.5, 3.0, 3.0])
        self._pinch_overload = (force > hard_high).any(dim=1)
        self._pinch_drop = drift > 0.095

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_verifier()
        terminated = self._pinch_success | self._pinch_drop | self._pinch_overload | self._pinch_self_collision
        truncated = (self.episode_length_buf >= self.max_episode_length - 1) & ~terminated
        self.extras["step"] = {
            "success_count": int(self._pinch_success.sum().item()),
            "drop_count": int(self._pinch_drop.sum().item()),
            "overload_count": int(self._pinch_overload.sum().item()),
            "overload_by_finger": (
                self._last_physical_force_n
                > self._last_physical_force_n.new_tensor((4.0, 3.5, 3.5, 3.0, 3.0))
            ).sum(dim=0).detach().cpu().tolist(),
            "self_collision_count": int(self._pinch_self_collision.sum().item()),
        }
        # GuidedResidual 环境可选地保存终止快照，供离线审计识别“哪种果径/
        # 受力组合失败”。这里不改变返回值、奖励或任何控制行为。
        if hasattr(self, "_last_terminal_axes_m"):
            terminal = self._pinch_success | self._pinch_drop | self._pinch_overload | self._pinch_self_collision
            self._last_terminal_axes_m.copy_(self._fruit_axes_m)
            self._last_terminal_force_n.copy_(self._last_physical_force_n)
            self._last_terminal_hold_age.copy_(self._verifier.hold_age)
            self._last_terminal_contact_ratio.copy_(
                self._last_verification.get(
                    "contact_ratio", torch.zeros(self.num_envs, device=self.device)
                )
            )
            self._last_terminal_band_ratio.copy_(
                self._last_verification.get(
                    "band_ratio", torch.zeros(self.num_envs, device=self.device)
                )
            )
            self._last_terminal_reason.zero_()
            self._last_terminal_reason[self._pinch_success] = 1
            self._last_terminal_reason[self._pinch_drop] = 2
            self._last_terminal_reason[self._pinch_overload] = 3
            self._last_terminal_reason[self._pinch_self_collision] = 4
            self.extras["step"]["terminal_snapshot_count"] = int(terminal.sum().item())
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # _get_dones 在 Isaac Lab 中先调用；仍防御性刷新，保证单步调用也合法。
        if not self._last_verification:
            self._refresh_verifier()
        force = self._last_physical_force_n
        required_force = force[:, :3]
        low = force.new_tensor((0.20, 0.15, 0.12))
        high = force.new_tensor((1.80, 1.50, 1.30))
        contact = (required_force >= 0.08).float().mean(dim=1)
        in_band = ((required_force >= low) & (required_force <= high)).float().mean(dim=1)
        overload = torch.relu(required_force - high).sum(dim=1)
        force_barrier = torch.relu(
            required_force / force.new_tensor((2.80, 2.45, 2.45)) - 1.0
        ).square().sum(dim=1)
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        fruit = self._tomato.data.root_pos_w.torch
        surface_error = (
            torch.linalg.vector_norm(tips[:, :3] - fruit.unsqueeze(1), dim=2)
            - self._fruit_axes_m.mean(dim=1, keepdim=True)
        ).abs().mean(dim=1)
        geometry = torch.exp(-surface_error / 0.020)
        temporal = self._filter.last_stats
        smoothness = temporal["action_delta"].mean(dim=1) + 0.05 * temporal["jerk"].mean(dim=1)
        hold = (self._verifier.hold_age.float() / 80.0).clamp(0.0, 1.0)
        palm_penalty = (self._last_palm_force_n / 0.25).clamp(0.0, 2.0)
        reward = (
            0.45 * geometry
            + 0.85 * contact
            + 1.25 * in_band
            + 1.50 * hold
            # 过载以连续代价抑制，但不让其淹没持稳链路；奖励变更时只允许
            # 迁移 Actor，不得迁移 Critic、优化器动量或探索方差。
            - 0.65 * overload
            - 0.55 * force_barrier
            - 0.12 * smoothness
            - 0.50 * palm_penalty
            - 2.75 * self._pinch_overload.float()
            - 1.5 * self._pinch_drop.float()
            + 5.0 * self._pinch_success.float()
        )
        logs = self.extras.setdefault("log", {})
        logs.update(
            {
                "SmallPinch/contact_ratio": contact.mean().item(),
                "SmallPinch/in_band_ratio": in_band.mean().item(),
                "SmallPinch/hold_steps": self._verifier.hold_age.float().mean().item(),
                "SmallPinch/palm_force_n": self._last_palm_force_n.mean().item(),
                "SmallPinch/action_delta": temporal["action_delta"].mean().item(),
                "SmallPinch/action_jerk": temporal["jerk"].mean().item(),
                "SmallPinch/force_limiter_scale": self._force_limiter_scale.mean().item(),
            }
        )
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        fruit_pose = self._tomato.data.root_pose_w.torch
        ee_pose = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        relative = quat_apply_inverse(ee_pose[:, 3:7], fruit_pose[:, :3] - ee_pose[:, :3]) / 0.10
        axes = (self._fruit_axes_m / 0.040).clamp(0.0, 1.5)
        # 使用果实局部坐标轴前两列，避免四元数正负号导致策略观测跳变。
        quat = fruit_pose[:, 3:7]
        w, x, y, z = quat.unbind(dim=1)
        rotation = torch.stack(
            (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
             2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=1
        )
        fruit_twist = torch.cat(
            (quat_apply_inverse(ee_pose[:, 3:7], self._tomato.data.root_lin_vel_w.torch) / 0.5,
             quat_apply_inverse(ee_pose[:, 3:7], self._tomato.data.root_ang_vel_w.torch) / 2.0), dim=1
        )
        joint_pos = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
        joint_vel = self._robot.data.joint_vel.torch[:, self._hand_joint_ids]
        limits = self._robot.data.joint_pos_limits.torch[:, self._hand_joint_ids]
        joint_pos_n = (2.0 * (joint_pos - limits[..., 0]) / (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6) - 1.0)
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        tip_velocity = self._robot.data.body_lin_vel_w.torch[:, self._fingertip_contact_body_ids]
        tip_relative = quat_apply_inverse(
            fruit_pose[:, None, 3:7].expand(-1, 5, -1).reshape(-1, 4),
            (tips - fruit_pose[:, None, :3]).reshape(-1, 3),
        ).reshape(self.num_envs, 5, 3) / 0.08
        surface_error = (
            torch.linalg.vector_norm(tips - fruit_pose[:, None, :3], dim=2)
            - self._fruit_axes_m.mean(dim=1, keepdim=True)
        ) / 0.020
        low = self._last_physical_force_n.new_tensor((0.20, 0.15, 0.12, 0.08, 0.05))
        high = self._last_physical_force_n.new_tensor((1.80, 1.50, 1.30, 1.20, 1.00))
        in_band = ((self._last_physical_force_n >= low) & (self._last_physical_force_n <= high)).float()
        # 每指十维由 PhysX 指尖接触力、相对几何和速度派生；它是仿真触觉
        # 代理，不会被标记成尚未标定的实机传感器矩阵。
        tactile = torch.cat(
            (
                (self._last_physical_force_n / 4.0).unsqueeze(-1),
                ((self._last_physical_force_n - self._previous_force_n) / 2.0).unsqueeze(-1),
                (self._last_physical_force_n >= 0.08).float().unsqueeze(-1),
                in_band.unsqueeze(-1),
                surface_error.unsqueeze(-1),
                tip_relative,
                (torch.linalg.vector_norm(tip_velocity, dim=2) / 0.5).unsqueeze(-1),
                (self._verifier.enter_age.float() / 4.0).unsqueeze(-1),
            ),
            dim=2,
        ).reshape(self.num_envs, 50)
        policy = torch.cat((relative, axes, rotation, fruit_twist, joint_pos_n, joint_vel / 2.0, tactile, self._previous_executed_actions), dim=1).clamp(-5.0, 5.0)
        if policy.shape[1] != ACTOR_OBSERVATION_DIM:
            raise RuntimeError(f"Actor observation drifted to {policy.shape[1]}D")
        privileged = torch.cat(
            (self._last_physical_force_n / 4.0, self._last_palm_force_n.unsqueeze(1) / 2.0,
             (tips - fruit_pose[:, None, :3]).reshape(self.num_envs, -1) / 0.10,
             self._tomato.data.root_lin_vel_w.torch / 0.5,
             self._tomato.data.root_ang_vel_w.torch / 2.0,
             self._fruit_mass_kg.unsqueeze(1) / 0.25,
             self._executed_actions, self._raw_actions,
             self._verifier.hold_age.float().unsqueeze(1) / 80.0,
             self._last_verification.get("contact_ratio", torch.zeros(self.num_envs, device=self.device)).unsqueeze(1),
             self._last_verification.get("band_ratio", torch.zeros(self.num_envs, device=self.device)).unsqueeze(1),
             self._last_verification.get("palm_ratio", torch.zeros(self.num_envs, device=self.device)).unsqueeze(1),
             self._pinch_self_collision.float().unsqueeze(1)), dim=1
        )
        if privileged.shape[1] != 61:
            raise RuntimeError(f"Critic privileged observation drifted to {privileged.shape[1]}D")
        critic = torch.cat((policy, privileged), dim=1)
        self._previous_force_n.copy_(self._last_physical_force_n)
        return {"policy": policy, "critic": critic}

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if not hasattr(self, "_filter"):
            return
        offset_g = torch.tensor(self.cfg.pinch_working_offset_m, device=self.device)
        offset_w = torch.bmm(
            self._grasp_basis_w[env_ids], offset_g.expand(len(env_ids), -1).unsqueeze(2)
        ).squeeze(2)
        fruit_pose = self._tomato.data.root_pose_w.torch[env_ids].clone()
        fruit_pose[:, :3] += offset_w
        clearance = self._hand_collision_clearance_for(
            fruit_pose[:, :3], self._fruit_axes_m[env_ids], env_ids
        )
        if bool((clearance < self.cfg.minimum_initial_clearance_m).any()):
            raise RuntimeError("P1 工作区位移破坏了无初始穿透合同")
        self._tomato.write_root_pose_to_sim_index(root_pose=fruit_pose, env_ids=env_ids)
        self._tomato.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self._stem_rest_attachment_pos_w[env_ids] += offset_w
        self._stem_plant_anchor_pos_w[env_ids] += offset_w
        self._filter.reset(env_ids)
        self._verifier.reset(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._executed_actions[env_ids] = 0.0
        self._previous_executed_actions[env_ids] = 0.0
        self._previous_force_n[env_ids] = 0.0
        self._force_limiter_scale[env_ids] = 1.0
        self._pinch_reference_center_w[env_ids] = fruit_pose[:, :3]
        self._pinch_success[env_ids] = False
        self._pinch_drop[env_ids] = False
        self._pinch_overload[env_ids] = False
        self._pinch_self_collision[env_ids] = False
        self._last_palm_force_n[env_ids] = 0.0
        self._last_verification = {}
        if hasattr(self, "_position_command_current"):
            self._position_command_current[env_ids] = self._joint_targets[env_ids][
                :, self._hand_control_ids
            ]
            self._position_command_start[env_ids] = self._position_command_current[env_ids]
            self._position_command_goal[env_ids] = self._position_command_current[env_ids]
            self._position_interpolation_step[env_ids] = 0


__all__ = ["NativeSmallFruitPinchCfg", "NativeSmallFruitPinchEnv", "configure_native_module_for_left"]
