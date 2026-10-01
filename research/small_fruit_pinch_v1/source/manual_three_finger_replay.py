#!/usr/bin/env python3
"""固定果心的左手三指候选回放（仅人工验收，不训练、不连接实机）。"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=360)
parser.add_argument("--delay-ms", type=float, default=80.0)
parser.add_argument("--center-offset-mm", type=float, default=0.0,
                    help="沿抓取基 approach 轴把果心向掌外偏移（mm）")
parser.add_argument("--action-scale", type=float, default=1.0,
                    help="候选动作幅度缩放，便于低压人工验收")
parser.add_argument("--root-scale", type=float, default=0.35,
                    help="末段根部闭合幅度；默认限幅为 35%%，降低近端先接触风险")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="可选：从权重 infos.small_fruit_teacher.action 读取 14D 候选动作")
parser.add_argument("--thumb-opposition-scale", type=float, default=1.0,
                    help="仅缩放第 5 个协同量（拇指对向），用于离线安全扫描")
parser.add_argument("--force-servo", action="store_true",
                    help="离线回放启用逐指小步力反馈，接触后自动回撤/补偿")
parser.add_argument("--contact-hold", action="store_true",
                    help="三指同时进入压力带后冻结目标，避免继续闭合过冲")
parser.add_argument("--stop-on-overload", action="store_true",
                    help="任一必需指超过软上限时立即停止回放并保留当前姿态")
parser.add_argument("--stop-on-proximal", action="store_true",
                    help="近端碰撞力超过阈值时立即停止，避免连续轨迹穿透")
parser.add_argument("--debug-first", action="store_true",
                    help="打印首步的目标/实际关节，定位轨迹插值碰撞")
parser.add_argument("--adaptive-collision", action="store_true",
                    help="近端碰撞时仅回撤对应手指目标，保留其余手指继续运动")
parser.add_argument("--kinematic-preview", action="store_true",
                    help="仅离线可视化：直接写入关节目标以复现 CEM 姿态，不代表动力学执行")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import time
import torch
from pathlib import Path
from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv
from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets

ACTION = (0.57710, 0.01111, -0.46491, 0.70339, 0.60163, -0.32956,
          0.05951, 0.74682, -0.07004, -0.20360, 0.19861, 0.40589,
          -0.50958, -0.15034)
CENTER_MM = (64.869, -30.414, 187.324)
if args.checkpoint is not None:
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    meta = (payload.get("infos") or {}).get("small_fruit_teacher") or {}
    if len(meta.get("action", ())) == 14:
        ACTION = tuple(float(value) for value in meta["action"])
    elif len(meta.get("direct_joint_targets", ())) != 16:
        raise ValueError("checkpoint 未包含 14D action 或 16D direct_joint_targets")
    contract = (payload.get("infos") or {}).get("manual_inspection_contract") or {}
    if len(contract.get("center_local_mm", ())) == 3:
        CENTER_MM = tuple(float(value) for value in contract["center_local_mm"])

cfg = NativeSmallFruitPinchCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device
class ReplayEnv(NativeSmallFruitPinchEnv):
    def _get_dones(self):
        return (torch.zeros(1, dtype=torch.bool, device=self.device),
                torch.zeros(1, dtype=torch.bool, device=self.device))
    def _apply_action(self):
        target = getattr(self, "_probe_joint_targets", None)
        if target is None:
            return super()._apply_action()
        if args.kinematic_preview:
            self._joint_targets[:, self._hand_joint_ids] = target
            self._update_mimic_targets(torch.zeros(1, dtype=torch.long, device=self.device))
            full = self._robot.data.default_joint_pos.torch.clone()
            full[:, self._hand_joint_ids] = target
            ids = torch.zeros(1, dtype=torch.long, device=self.device)
            self._robot.write_joint_position_to_sim_index(position=full, joint_ids=None, env_ids=ids)
            self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(full), joint_ids=None, env_ids=ids)
            self._tomato.write_root_pose_to_sim_index(root_pose=self._probe_fruit_pose, env_ids=ids)
            self._tomato.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.device), env_ids=ids)
            return
        self._joint_targets[:, self._hand_joint_ids] = target
        self._update_mimic_targets(torch.zeros(1, dtype=torch.long, device=self.device))
        full = self._robot.data.default_joint_pos.torch.clone()
        full[:, self._actuated_joint_ids] = self._joint_targets
        self._robot.write_joint_position_to_sim_index(position=full, joint_ids=None, env_ids=torch.zeros(1, dtype=torch.long, device=self.device))
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(full), joint_ids=None, env_ids=torch.zeros(1, dtype=torch.long, device=self.device))
        self._tomato.write_root_pose_to_sim_index(root_pose=self._probe_fruit_pose, env_ids=torch.zeros(1, dtype=torch.long, device=self.device))
        self._tomato.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.device), env_ids=torch.zeros(1, dtype=torch.long, device=self.device))

env = ReplayEnv(cfg=cfg, render_mode="human")
try:
    env.reset(seed=20260910)
    dev = env.device; ids = torch.zeros(1, dtype=torch.long, device=dev)
    pose = env._tomato.data.root_pose_w.torch.clone()
    pose[:, :3] = env.scene.env_origins + torch.tensor(CENTER_MM, device=dev).view(1, 3) / 1000.0
    pose[:, 3:7] = pose.new_tensor((0., 0., 0., 1.))
    env._probe_fruit_pose = pose
    # 先把果实放在掌前约 60 mm，待指垫预置后沿抓取轴缓慢送入，
    # 避免手指从球体内部闭合而产生近端假接触。
    approach_axis = env._grasp_basis_w[:, :, 0]
    pose[:, :3] = pose[:, :3] + approach_axis * (args.center_offset_mm / 1000.0)
    pose_start = pose.clone(); pose_start[:, :3] = pose[:, :3] + approach_axis * 0.060
    if args.kinematic_preview:
        # CEM 候选是在固定果心下评分的；运动学预演保持同一果心，避免把
        # “接近过程”与候选姿态验收混在一起。动力学回放仍使用原有接近轨迹。
        pose_start = pose.clone()
    action = torch.tensor(ACTION, device=dev).mul(float(args.action_scale)).clamp(-1.0, 1.0)
    action[4] = action[4] * max(0.0, min(1.0, float(args.thumb_opposition_scale)))
    # reset 后 articulation 的 joint_pos 可能仍是全零，而 USD 的安全张开姿态
    # 存在于 default_joint_pos；若直接从全零插值，第一步会触发非目标关节
    # 自碰撞并产生几十牛顿的伪近端力。
    # 动力学回放采用 14D 零动作安全模板；运动学预演则复现 dynamic CEM
    # 的 reset 起点（当前 articulation 状态），保证两者的轨迹定义一致。
    opened = (env._robot.data.joint_pos.torch[:, env._hand_joint_ids].clone()
              if args.kinematic_preview else
              pinch_action_to_joint_targets(torch.zeros((1, 14), device=dev)))
    full_open = env._robot.data.default_joint_pos.torch.clone()
    full_open[:, env._hand_joint_ids] = opened
    env._robot.write_joint_position_to_sim_index(position=full_open, joint_ids=None,
                                                 env_ids=torch.zeros(1, dtype=torch.long, device=dev))
    env._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(full_open), joint_ids=None,
                                                 env_ids=torch.zeros(1, dtype=torch.long, device=dev))
    env.scene.write_data_to_sim(); env.sim.forward()
    # 先让张开姿态完成一个短暂物理稳定段，避免把写入姿态产生的瞬时
    # 自碰撞冲量错误归因于抓取轨迹。
    for _ in range(5):
        env.sim.step(render=False); env.scene.update(dt=cfg.sim.dt)
    # 稳定段仅用于清除初始冲量，不计入抓取轨迹的碰撞统计。
    env._forbidden_proximal_force_n.zero_()
    env._forbidden_contact_force_n.zero_()
    env._forbidden_first_contact.zero_()
    direct = meta.get("direct_joint_targets") if args.checkpoint is not None else None
    if direct is not None:
        target = torch.tensor(direct, dtype=torch.float32, device=dev).view(1, 16)
        limits = env._robot.data.joint_pos_limits.torch[:, env._hand_joint_ids]
        target = torch.maximum(limits[..., 0], torch.minimum(limits[..., 1], target))
    else:
        target = pinch_action_to_joint_targets(action.view(1, 14))
    # 根部只执行候选目标的一小部分，远端/对向阶段仍完整执行；这是安全的
    # 人工验收轨迹限幅，不改变训练环境或历史权重。
    # 仅限幅三指根部：食指0、中指2、拇指4，以及对应第一屈曲轴5/7/9。
    # 旧版把小指/无名指根部和远端轴一并缩放，会造成非目标手指自碰撞，
    # 并掩盖拇指对向轨迹的真实效果。
    root_indices = (0, 1, 2, 5, 6, 7, 10, 11)
    root_scale = max(0.0, min(1.0, float(args.root_scale)))
    target[:, root_indices] = opened[:, root_indices] + root_scale * (
        target[:, root_indices] - opened[:, root_indices]
    )
    # 指尖优先：先卷曲拇指/食指/中指远端，再做拇指对向，最后才闭合根部。
    # 这样可避免根部碰撞壳先扫过果实。
    # 与 dynamic_contact_cem 完全一致的三阶段：食/中/拇根部预对向 →
    # 拇指横摆轴到位 → 全部目标轴远端闭合。旧版阶段索引错误会驱动
    # 无名/小指远端，造成“拇指单独包络”与侧摆碰撞。
    phases = ((0, 100, (5, 6, 7)), (100, 210, (0, 10, 11)),
              (210, max(211, args.steps), tuple(range(16))))
    contact_latched = False
    hold_target = None
    names = [str(env._robot.joint_names[i]) for i in env._hand_joint_ids]
    digit_ids = {
        "thumb": [i for i, n in enumerate(names) if "thumb" in n],
        "index": [i for i, n in enumerate(names) if "index" in n],
        "middle": [i for i, n in enumerate(names) if "middle" in n],
    }
    finger_hold = [False, False, False]
    finger_hold_target = [None, None, None]
    collision_retract = torch.zeros((1, 3), device=dev)
    for step in range(max(1, args.steps)):
        stage_target = opened.clone()
        for start, end, indices in phases:
            if step >= start:
                u = min(1.0, max(0.0, (step - start + 1) / max(1, end - start)))
                u = u * u * (3.0 - 2.0 * u)
                stage_target[:, indices] = opened[:, indices] + u * (target[:, indices] - opened[:, indices])
        if contact_latched and hold_target is not None:
            stage_target = hold_target.clone()
        for digit, key in enumerate(("thumb", "index", "middle")):
            if finger_hold[digit] and finger_hold_target[digit] is not None:
                stage_target[:, digit_ids[key]] = finger_hold_target[digit]
            if args.adaptive_collision and collision_retract[0, digit] > 0:
                # 只回撤该指的屈曲轴，不改变拇指对向或其他手指阶段目标。
                stage_target[:, digit_ids[key]] -= collision_retract[0, digit]
        env._probe_joint_targets = stage_target
        fruit_u = min(1.0, max(0.0, (step - 180) / 150.0))
        fruit_u = fruit_u * fruit_u * (3.0 - 2.0 * fruit_u)
        env._probe_fruit_pose = pose_start + fruit_u * (pose - pose_start)
        env.step(torch.zeros((1, 14), device=dev))
        env._tomato.write_root_pose_to_sim_index(root_pose=env._probe_fruit_pose, env_ids=ids)
        env._tomato.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev), env_ids=ids)
        env.scene.write_data_to_sim(); env.sim.forward()
        if args.debug_first and step == 0:
            print("manual_replay_first", {
                "opened": [round(float(x), 5) for x in opened[0].detach().cpu()],
                "stage_target": [round(float(x), 5) for x in stage_target[0].detach().cpu()],
                "actual": [round(float(x), 5) for x in env._robot.data.joint_pos.torch[0, env._hand_joint_ids].detach().cpu()],
                "proximal_force_n": round(float(env._forbidden_proximal_force_n[0].sum()), 4),
            }, flush=True)
        if args.contact_hold:
            force_now = env._raw_finger_force_n[0, :3]
            in_band = (force_now >= 0.08) & (force_now <= force_now.new_tensor((1.80, 1.50, 1.30)))
            for digit, key in enumerate(("thumb", "index", "middle")):
                if bool(in_band[digit]) and not finger_hold[digit]:
                    finger_hold[digit] = True
                    finger_hold_target[digit] = stage_target[:, digit_ids[key]].clone()
            if all(finger_hold):
                contact_latched = True
                hold_target = stage_target.clone()
        if args.adaptive_collision:
            proximal = env._forbidden_proximal_force_n[0]
            # USD 近端碰撞统计按 thumb/index/middle 三个目标组聚合；
            # 若资产返回更短向量，则使用总量保守回撤全部必需指。
            if proximal.numel() >= 3:
                groups = (proximal[0], proximal[1], proximal[2])
            else:
                groups = (proximal.sum(), proximal.sum(), proximal.sum())
            for digit, value in enumerate(groups):
                if float(value) > 1.5:
                    collision_retract[0, digit] = (collision_retract[0, digit] + 0.02).clamp_max(0.12)
            # 同一状态机同时处理指尖过载；回撤量持久化到下一步的 stage_target，
            # 不会被后续阶段插值覆盖。这样拇指先接触时，食/中指仍可继续靠近。
            finger_force = env._raw_finger_force_n[0, :3]
            soft_high = finger_force.new_tensor((1.80, 1.50, 1.30))
            for digit, value in enumerate(finger_force):
                if float(value) > float(soft_high[digit]):
                    collision_retract[0, digit] = (collision_retract[0, digit] + 0.015).clamp_max(0.16)
        if args.force_servo:
            # 只在接触后做极小的逐指修正；用 joint name 建立映射，避免
            # 依赖 articulation 的物理排列顺序。正向闭合为增加 canonical
            # 目标，过力时反向回撤，修正量每步不超过 0.01 rad。
            names = [str(env._robot.joint_names[i]) for i in env._hand_joint_ids]
            digit_ids = {
                "thumb": [i for i, n in enumerate(names) if "thumb" in n],
                "index": [i for i, n in enumerate(names) if "index" in n],
                "middle": [i for i, n in enumerate(names) if "middle" in n],
            }
            force_now = env._raw_finger_force_n[0, :3]
            soft_high = (1.80, 1.50, 1.30)
            soft_low = (0.20, 0.15, 0.12)
            for digit, key in enumerate(("thumb", "index", "middle")):
                if not digit_ids[key]:
                    continue
                if float(force_now[digit]) > soft_high[digit]:
                    delta = -0.050
                elif 0.02 <= float(force_now[digit]) < soft_low[digit] and step > 170:
                    delta = 0.004
                else:
                    delta = 0.0
                if delta:
                    target[:, digit_ids[key]] += delta
                    limits = env._robot.data.joint_pos_limits.torch[:, env._hand_joint_ids]
                    target[:] = torch.maximum(limits[..., 0], torch.minimum(limits[..., 1], target))
        if args.stop_on_overload:
            # 安全停止同时参考有效指尖力与原始指垫力；原始力即使因轴向区域
            # 尚未锁存而未计入 effective，也代表真实碰撞风险，不能被过滤掩盖。
            force_now = torch.maximum(
                env._physical_finger_force()[0, :3], env._raw_finger_force_n[0, :3]
            )
            if bool((force_now > force_now.new_tensor((1.80, 1.50, 1.30))).any()):
                print("manual_replay_stop_on_overload", {"step": step, "force_n": [round(float(x), 4) for x in force_now.detach().cpu()]}, flush=True)
                break
        if args.stop_on_proximal:
            proximal_now = float(env._forbidden_proximal_force_n[0].sum())
            if proximal_now > 2.0:
                print("manual_replay_stop_on_proximal", {"step": step, "proximal_force_n": round(proximal_now, 4)}, flush=True)
                break
        if args.delay_ms > 0: time.sleep(args.delay_ms / 1000.0)
    tips = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids[:3]]
    fruit = env._tomato.data.root_pos_w.torch
    print(
        "manual_replay_final",
        {
            "effective_force_n": [round(float(x), 4) for x in env._physical_finger_force()[0, :3].detach().cpu()],
            "raw_force_n": [round(float(x), 4) for x in env._raw_finger_force_n[0, :3].detach().cpu()],
            "axial_progress": [round(float(x), 4) for x in env._contact_region_axial_progress[0, :3].detach().cpu()],
            "proximal_force_n": round(float(env._forbidden_proximal_force()[0].sum().detach().cpu()), 4),
            "tip_distance_mm": [round(float(x * 1000.0), 2) for x in torch.linalg.vector_norm(tips[0] - fruit[0], dim=1).detach().cpu()],
        },
        flush=True,
    )
finally:
    env.close()
