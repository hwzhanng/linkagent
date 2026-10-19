#!/usr/bin/env python3
"""批量动态教师搜索：验证候选目标经过真实驱动执行后仍能三指有效接触。

仅左手、离线 PhysX；不连接实机、不训练 PPO。候选直接搜索拇指/食指/中指
十个 canonical 关节和果实中心，轨迹按根部预对向、拇指引导、远端闭合、保持执行。
"""
from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=8)
parser.add_argument("--steps", type=int, default=420)
parser.add_argument("--coarse-steps", type=int, default=55)
parser.add_argument("--thumb-steps", type=int, default=75)
parser.add_argument("--distal-steps", type=int, default=85)
parser.add_argument("--diameter-mm", type=float, default=70.0)
parser.add_argument("--seed", type=int, default=20260940)
parser.add_argument("--initial-control", nargs=10, type=float, default=None)
parser.add_argument("--initial-distal", nargs=4, type=float, default=None,
                    help="14D 动作的四个远端残差，仅用于教师验收")
parser.add_argument("--initial-center-mm", nargs=3, type=float, default=None)
parser.add_argument("--center-std-mm", type=float, default=8.0)
parser.add_argument("--auto-center", action="store_true", help="用候选最终三指 FK 中心初始化果心")
parser.add_argument("--hand-stiffness", type=float, default=12.0)
parser.add_argument("--hand-damping", type=float, default=0.6)
parser.add_argument("--hand-effort", type=float, default=0.2)
parser.add_argument("--control-std", type=float, default=0.13)
parser.add_argument("--palm-fruit-filter", action="store_true", help="仅过滤粗糙掌部 AABB（诊断候选，不用于发布）")
parser.add_argument("--kinematic-teacher", action="store_true", help="限速显式位置教师，仅用于离线物理验收")
parser.add_argument("--debug-geometry", action="store_true")
parser.add_argument("--direct-joints", action="store_true",
                    help="直接搜索 10 个必需 canonical 关节，跳过 14D 流形耦合")
parser.add_argument("--root-fraction", type=float, default=0.65,
                    help="连续轨迹末段根部闭合比例；0–1，默认 0.65")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv
from research.small_fruit_pinch_v1.source.legacy_bridge import load_legacy_hand_manifold
from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets

class DynamicProbeEnv(NativeSmallFruitPinchEnv):
    def _postprocess_joint_targets(self, targets: torch.Tensor) -> torch.Tensor:
        override = getattr(self, "_probe_joint_targets", None)
        return targets if override is None else override

    def _apply_action(self) -> None:
        target = getattr(self, "_probe_joint_targets", None)
        if target is None or not bool(getattr(self, "_kinematic_teacher", False)):
            return super()._apply_action()
        self._joint_targets[:, self._hand_joint_ids] = target
        ids = torch.arange(self.num_envs, device=self.device)
        self._update_mimic_targets(ids)
        joint_pos = self._robot.data.default_joint_pos.torch.clone()
        joint_pos[:, self._actuated_joint_ids] = self._joint_targets
        self._robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=None, env_ids=ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), joint_ids=None, env_ids=ids)
        # 教师验收阶段把果心固定在候选中心，避免未接触前果实受重力/茎部
        # 约束漂移而把“指尖是否到位”与“果实是否掉落”混为一谈。
        fruit_pose = getattr(self, "_probe_fruit_pose", None)
        if fruit_pose is not None:
            self._tomato.write_root_pose_to_sim_index(root_pose=fruit_pose, env_ids=ids)
            self._tomato.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((self.num_envs, 6), device=self.device), env_ids=ids
            )
        else:
            self._apply_attachment_wrench()

    def _get_dones(self):
        # 搜索阶段不让 DirectRLEnv 自动 reset，保留失败前的接触/力峰值；
        # 安全事件仍由父类更新到状态，评分时显式惩罚 forbidden/overload。
        super()._get_dones()
        return (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

    def step(self, action):
        """执行一步后恢复教师果心，避免接触反力把候选球推离工作区。

        训练环境仍保留真实刚体动力学；这里只在离线教师搜索中锁定
        ``_probe_fruit_pose``，这样首接触、指尖力和轴向进度都在同一果心
        参考下累计，不会因单步反力漂移而把轨迹误判为失败。
        """
        result = super().step(action)
        pose = getattr(self, "_probe_fruit_pose", None)
        if pose is not None:
            ids = torch.arange(self.num_envs, device=self.device)
            self._tomato.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            self._tomato.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((self.num_envs, 6), device=self.device),
                env_ids=ids,
            )
            self.scene.write_data_to_sim()
            self.sim.forward()
        return result

def smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def main() -> None:
    if args.steps <= args.coarse_steps + args.thumb_steps + args.distal_steps:
        raise ValueError("steps 必须大于三个闭合阶段之和")
    if not 0.0 < args.root_fraction <= 1.0:
        raise ValueError("root-fraction 必须位于 (0,1]")
    if not 64.0 <= args.diameter_mm <= 70.0:
        raise ValueError("diameter-mm 必须位于 64–70 mm")
    cfg = NativeSmallFruitPinchCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device
    cfg.palm_fruit_collision_filter_enabled = bool(args.palm_fruit_filter)
    for actuator_name in ("g20_independent", "g20_followers"):
        actuator = cfg.robot.actuators[actuator_name]
        actuator.stiffness = args.hand_stiffness; actuator.damping = args.hand_damping
        actuator.effort_limit = args.hand_effort; actuator.effort_limit_sim = args.hand_effort
    cfg.seed = args.seed; cfg.episode_length_s = max(30.0, args.steps / 20.0 + 5.0)
    radius = args.diameter_mm / 2000.0
    cfg.events.fruit_geometry.params["axis_ranges_m"] = {a: (radius, radius) for a in ("x", "y", "z")}
    env = DynamicProbeEnv(cfg=cfg)
    env._kinematic_teacher = bool(args.kinematic_teacher)
    try:
        env.reset(seed=args.seed); dev = env.device; n = env.num_envs
        ids = torch.arange(n, device=dev)
        legacy = load_legacy_hand_manifold(); lo = torch.tensor(legacy.LEFT_SAFE_LOWER_RAD, device=dev); hi = torch.tensor(legacy.LEFT_SAFE_UPPER_RAD, device=dev)
        # SDK/资产实际 canonical 顺序：thumb2,index1,middle1,ring1,little1,
        # thumb0,index0,middle0,ring0,little0,thumb1,thumb3,index2,middle2,...。
        # 三指必需轴因此是拇指 0/5/10/11、食指 1/6/12、中指 2/7/13。
        idx = torch.tensor((0,1,2,5,6,7,10,11,12,13), device=dev)
        base = torch.tensor([-0.93,-0.23,0.21,0.64,0.25,0.36,-0.24,0.59,0.91,0.88, 0.074,-0.040,0.163], device=dev)
        distal = torch.zeros(4, device=dev)
        if args.initial_control is not None: base[:10] = torch.tensor(args.initial_control, device=dev)
        if args.initial_distal is not None: distal[:] = torch.tensor(args.initial_distal, device=dev)
        if args.initial_center_mm is not None: base[10:] = torch.tensor(args.initial_center_mm, device=dev)/1000.0
        mean = base.clone(); std = torch.tensor([args.control_std]*10+[args.center_std_mm/1000.0]*3, device=dev)
        lower = torch.tensor([-1.0]*10+[0.015,-0.095,0.080], device=dev); upper = torch.tensor([1.0]*10+[0.105,0.020,0.235], device=dev)
        opened = env._robot.data.default_joint_pos.torch[:, env._hand_joint_ids].clone()
        template = pinch_action_to_joint_targets(torch.zeros((1,14),device=dev))[0]
        best = None
        for it in range(args.iterations):
            samples = (mean + std*torch.randn((n,13),device=dev)).clamp(lower,upper)
            controls = samples[:,:10]; centers = samples[:,10:]
            action14 = torch.zeros((n, 14), device=dev)
            action14[:, :10] = controls
            action14[:, 10:] = distal
            if args.direct_joints:
                targets = template.expand(n, -1).clone()
                normalized = controls
                targets[:, idx] = lo[idx] + (normalized + 1.0) * 0.5 * (hi[idx] - lo[idx])
            else:
                targets = pinch_action_to_joint_targets(action14)
            # 末段只执行根部目标的一部分；远端指垫和拇指对向保持完整，
            # 让 CEM 搜索“接触后小幅收拢”而不是一次性压入果实。
            # 只对三指根部/掌指轴做限幅，远端和拇指横摆保持完整目标。
            root_idx = torch.tensor((0, 1, 2, 5, 6, 7, 10, 11), device=dev)
            targets[:, root_idx] = opened[:, root_idx] + args.root_fraction * (targets[:, root_idx] - opened[:, root_idx])
            env.reset(seed=args.seed+it+1)
            if args.auto_center:
                # 仅用于离线几何初始化：暂时写入候选关节求 FK，读取目标
                # 三指接触壳的中心，再恢复张开姿态；不把该姿态直接作为轨迹。
                pose_joint = env._robot.data.default_joint_pos.torch.clone()
                pose_joint[:, env._hand_joint_ids] = targets
                env._robot.write_joint_position_to_sim_index(position=pose_joint, joint_ids=None, env_ids=ids)
                env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=cfg.sim.dt)
                tips = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids[:3]]
                centers = tips.mean(dim=1) - env.scene.env_origins
                env._robot.write_joint_position_to_sim_index(position=env._robot.data.default_joint_pos.torch, joint_ids=None, env_ids=ids)
                env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=cfg.sim.dt)
            old = env._tomato.data.root_pos_w.torch.clone(); pose = env._tomato.data.default_root_pose.torch.clone(); pose[:,:3] = env.scene.env_origins + centers; pose[:,3:7] = pose.new_tensor((0.,0.,0.,1.))
            env._probe_fruit_pose = pose
            env._tomato.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids); env._tomato.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n,6),device=dev),env_ids=ids)
            delta=pose[:,:3]-old; env._stem_rest_attachment_pos_w += delta; env._stem_plant_anchor_pos_w += delta; env._pinch_reference_center_w.copy_(pose[:,:3])
            start = env._robot.data.joint_pos.torch[:,env._hand_joint_ids].clone(); pre=start.clone()
            if args.debug_geometry:
                debug_pose = env._robot.data.default_joint_pos.torch.clone()
                debug_pose[:, env._hand_joint_ids] = targets
                env._robot.write_joint_position_to_sim_index(position=debug_pose, joint_ids=None, env_ids=ids)
                env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=cfg.sim.dt)
                probe_tips = env._robot.data.body_pos_w.torch[:,env._fingertip_contact_body_ids[:3]] - env.scene.env_origins[:,None,:]
                print("DEBUG target_tips_mm", (probe_tips * 1000.0).detach().cpu().tolist(), "fruit_mm", (pose[:,:3]-env.scene.env_origins).detach().cpu().tolist(), flush=True)
            # 预对向阶段使用 canonical 16D 手部索引：
            # index0/little0/middle0/ring0/thumb0 = 0..4，
            # index1/little1/middle1/ring1/thumb1 = 5..9，
            # index2/little2/middle2/ring2/thumb2 = 10..14，thumb3 = 15。
            # 旧版误把 (5,10) 当作“根部”并把 11 当作拇指，实际会驱动食指
            # 中节和小指关节，导致拇指横摆未执行、指间距异常。现在先让拇指
            # 横摆/根部与食指、中指第一屈曲关节到位，再进行末端闭合。
            coarse_ids = torch.tensor((5, 6, 7), device=dev)
            pre[:, coarse_ids] = targets[:, coarse_ids]
            thumb=pre.clone()
            thumb[:, (0, 10, 11)] = targets[:, (0, 10, 11)]
            peak=torch.zeros((n,3),device=dev); late=torch.zeros((n,3),device=dev); failed=torch.zeros(n,dtype=torch.bool,device=dev)
            for step in range(args.steps):
                if step<args.coarse_steps: a=smoothstep(torch.tensor((step+1)/args.coarse_steps,device=dev)); tgt=start+a*(pre-start)
                elif step<args.coarse_steps+args.thumb_steps: a=smoothstep(torch.tensor((step-args.coarse_steps+1)/args.thumb_steps,device=dev)); tgt=pre+a*(thumb-pre)
                elif step<args.coarse_steps+args.thumb_steps+args.distal_steps: a=smoothstep(torch.tensor((step-args.coarse_steps-args.thumb_steps+1)/args.distal_steps,device=dev)); tgt=thumb+a*(targets-thumb)
                else: tgt=targets
                env._probe_joint_targets=tgt; _,_,term,trunc,_=env.step(torch.zeros((n,14),device=dev)); force=env._last_physical_force_n[:,:3]; peak=torch.maximum(peak,force)
                if step>=args.steps-50: late=torch.maximum(late,force)
                failed |= term | trunc | env._forbidden_first_contact
            axial=torch.nan_to_num(env._contact_region_axial_progress[:,:3], nan=0.0, posinf=0.0, neginf=0.0); prox=env._forbidden_proximal_force_n; margin=env._thumb_index_surface_margin_m
            tip_dist=torch.linalg.vector_norm(env._robot.data.body_pos_w.torch[:,env._fingertip_contact_body_ids[:3]]-env._tomato.data.root_pos_w.torch[:,None,:],dim=2)
            target_error=torch.abs(env._robot.data.joint_pos.torch[:,env._hand_joint_ids]-targets).amax(dim=1)
            valid=(late>=0.02).sum(1); missing=torch.relu(0.02-late).sum(1)
            # 连续教师搜索不能只看“是否有力”：此前 loss 在 7 N 以上的
            # 过载轨迹上也会得到 0，导致 CEM 偏向猛夹。按逐指软上限加入
            # 峰值积分惩罚，同时保留近端/首接触硬门。
            soft_high = late.new_tensor((1.80, 1.50, 1.30))
            force_excess = torch.relu(torch.maximum(peak, late) - soft_high).square().sum(1)
            loss=5*(3-valid.float())+30*missing+1.5*force_excess+2*prox.sum(1)+20*failed.float()+4*torch.relu((0.003-margin)/0.003)+0.5*torch.relu(env.fingertip_axial_progress_min-axial).sum(1)
            k=torch.argmin(loss); print(f"iter={it:02d} best_loss={float(loss[k]):.4f} late={[round(float(x),4) for x in late[k]]} peak={[round(float(x),4) for x in peak[k]]} axial={[round(float(x),4) for x in axial[k]]} tip_dist_mm={[round(float(x*1000),2) for x in tip_dist[k]]} target_error_rad={float(target_error[k]):.4f} prox={float(prox[k].sum()):.4f} margin_mm={float(margin[k]*1000):.2f} failed={bool(failed[k])} control={[round(float(x),4) for x in samples[k,:10]]} center_mm={[round(float(x*1000),3) for x in centers[k]]}",flush=True)
            elite=torch.topk(loss,min(n,max(1,n//8)),largest=False).indices; mean=0.25*mean+0.75*samples[elite].mean(0); std=(0.85*std+0.15*samples[elite].std(0, unbiased=False).clamp_min(0.004)).clamp_max(torch.tensor([0.35]*10+[0.025]*3,device=dev))
            if best is None or float(loss[k])<best[0]: best=(float(loss[k]),samples[k].detach().cpu())
        if best: print("BEST",best[0], [round(float(x),6) for x in best[1]])
    finally: env.close()
if __name__ == "__main__": main()
