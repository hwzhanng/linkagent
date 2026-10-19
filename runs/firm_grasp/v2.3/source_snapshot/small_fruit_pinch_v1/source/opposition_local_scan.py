#!/usr/bin/env python3
"""围绕一个 FK 对向候选做局部动作扰动，检查是否能移出掌部碰撞区。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--batches", type=int, default=60)
parser.add_argument("--top-k", type=int, default=20)
parser.add_argument("--radius-mm", type=float, default=33.5)
parser.add_argument("--noise", type=float, default=0.30)
parser.add_argument("--seed", type=int, default=20260914)
parser.add_argument("--base-action", nargs=14, type=float, default=None)
parser.add_argument("--clearance-mm", type=float, default=1.0)
parser.add_argument("--maximum-opposition-angle-deg", type=float, default=28.0)
parser.add_argument("--maximum-chord-ratio", type=float, default=0.35)
parser.add_argument("--minimum-support-span-mm", type=float, default=20.0)
parser.add_argument("--minimum-forbidden-clearance-mm", type=float, default=1.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv
from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets


BASE = torch.tensor((
    0.5563, 0.3046, -0.4787, 0.1579, 0.3133, 0.6795, 0.1310,
    -0.1900, 0.2070, 0.5084, -0.2590, -0.6060, -0.1727, 0.0229,
), dtype=torch.float32)

LEFT_TRUE_TIP_NAMES = ("thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4")


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        generator = torch.Generator(device=env.device)
        generator.manual_seed(args.seed)
        ids = torch.arange(env.num_envs, device=env.device)
        origins = env.scene.env_origins
        base = (
            torch.tensor(args.base_action, dtype=torch.float32, device=env.device)
            if args.base_action is not None
            else BASE.to(env.device)
        )
        radius = args.radius_mm / 1000.0
        threshold = args.clearance_mm / 1000.0
        minimum_support_span = args.minimum_support_span_mm / 1000.0
        minimum_forbidden_clearance = args.minimum_forbidden_clearance_mm / 1000.0
        body_names = tuple(
            env._robot.body_names[index] for index in env._hand_collision_body_ids
        )
        true_tip_names = set(LEFT_TRUE_TIP_NAMES)
        forbidden_mask = torch.tensor(
            [
                name not in true_tip_names
                and name.startswith(("thumb_", "index_", "middle_"))
                for name in body_names
            ],
            dtype=torch.bool,
            device=env.device,
        )
        forbidden_names = [
            name for name, selected in zip(body_names, forbidden_mask.cpu().tolist()) if selected
        ]
        records = []
        for _ in range(args.batches):
            action = base.unsqueeze(0).expand(env.num_envs, -1).clone()
            action += torch.randn(action.shape, device=env.device, generator=generator) * float(args.noise)
            action.clamp_(-1.0, 1.0)
            # retain a closed workspace proposal instead of spending all samples
            # at the affine manifold's fully open end.
            action[:, 0] = action[:, 0].clamp(0.35, 1.0)
            targets = pinch_action_to_joint_targets(action)
            env._joint_targets[:, env._hand_joint_ids] = targets
            env._update_mimic_targets(ids)
            env._robot.write_joint_position_to_sim_index(position=env._joint_targets, joint_ids=env._actuated_joint_ids, env_ids=ids)
            env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(dt=0.0)
            points = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids][:, :3] - origins[:, None, :]
            a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
            n = torch.linalg.cross(a, b, dim=1)
            n2 = torch.sum(n*n, dim=1)
            a2, b2 = torch.sum(a*a, dim=1), torch.sum(b*b, dim=1)
            off = (a2.unsqueeze(1)*torch.linalg.cross(b,n,dim=1) + b2.unsqueeze(1)*torch.linalg.cross(n,a,dim=1)) / (2*n2).clamp_min(1e-12).unsqueeze(1)
            circum = points[:,0] + off
            h = torch.sqrt(torch.relu(radius*radius - torch.sum(off*off,dim=1)))
            un = n / torch.sqrt(n2).clamp_min(1e-8).unsqueeze(1)
            centers = torch.stack((circum + h.unsqueeze(1)*un, circum - h.unsqueeze(1)*un), dim=1)
            for side in range(2):
                c = centers[:,side]
                rel = points-c[:,None,:]
                d = torch.linalg.vector_norm(rel,dim=2)
                u = rel/d.clamp_min(1e-8).unsqueeze(-1)
                im = torch.nn.functional.normalize(u[:,1:3].mean(dim=1),dim=1)
                theta = torch.rad2deg(torch.acos(torch.clamp((u[:,0]*-im).sum(dim=1),-1,1)))
                mid = rel[:,1:3].mean(dim=1); line = mid-rel[:,0]; l2=torch.sum(line*line,dim=1).clamp_min(1e-12)
                t=torch.clamp(torch.sum(-rel[:,0]*line,dim=1)/l2,0,1); closest=rel[:,0]+t[:,None]*line
                chord=torch.linalg.vector_norm(closest,dim=1)/radius
                support_span=torch.linalg.vector_norm(points[:,1]-points[:,2],dim=1)
                clearance_matrix=env._hand_collision_clearance_matrix_for(
                    origins+c,env._fruit_axes_m,ids
                )
                clear=clearance_matrix.amin(dim=1)
                forbidden_clearance=clearance_matrix[:,forbidden_mask].amin(dim=1)
                forbidden_worst=clearance_matrix[:,forbidden_mask].argmin(dim=1)
                valid=(forbidden_clearance>=minimum_forbidden_clearance)&(theta<args.maximum_opposition_angle_deg)&(chord<args.maximum_chord_ratio)&(support_span>=minimum_support_span)&(d[:,:3].sub(radius).abs().amax(dim=1)<0.010)
                loss=(theta/20+3*torch.relu(chord-.25)+4*torch.relu((minimum_support_span-support_span)/minimum_support_span)+10*torch.relu(d[:,:3].sub(radius).abs().mean(dim=1)/radius))
                loss=torch.where(valid,loss,torch.full_like(loss,1e4))
                for i in torch.argsort(loss)[:args.top_k].tolist():
                    records.append((float(loss[i]),c[i].detach().cpu(),action[i].detach().cpu(),theta[i].detach().cpu(),chord[i].detach().cpu(),support_span[i].detach().cpu(),forbidden_clearance[i].detach().cpu(),forbidden_names[int(forbidden_worst[i])],clear[i].detach().cpu()))
        records.sort(key=lambda x:x[0])
        print(f"opposition_local_scan records={len(records)} noise={args.noise}",flush=True)
        for rank,(loss,c,a,t,ch,span,forbidden_cl,forbidden_body,cl) in enumerate(records[:args.top_k],1):
            print({"rank":rank,"loss":round(loss,5),"center_mm":[round(float(v*1000),3) for v in c],"minimum_clearance_mm":round(float(cl*1000),3),"forbidden_clearance_mm":round(float(forbidden_cl*1000),3),"forbidden_worst_body":forbidden_body,"theta_opp_deg":round(float(t),3),"chord_ratio":round(float(ch),4),"index_middle_span_mm":round(float(span*1000),3),"action":[round(float(v),4) for v in a]},flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
