"""Inspect strict-gate terms for a fixed v2 action (diagnostic only)."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser(); p.add_argument('--action',required=True); p.add_argument('--profile',default='v1_nominal_reliable'); p.add_argument('--steps',type=int,default=250); p.add_argument('--seed',type=int,default=20261027); p.add_argument('--output',type=Path,required=True); AppLauncher.add_app_launcher_args(p); args=p.parse_args(); app=AppLauncher(args).app
import torch
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv,V2EnvelopingPhysxTomatoGraspEnvCfg
@torch.inference_mode()
def main():
    vals=[float(v.strip()) for v in args.action.split(',') if v.strip()]
    if len(vals)!=14: raise ValueError('expected 14 action values')
    cfg=V2EnvelopingPhysxTomatoGraspEnvCfg(); cfg.sim.device=args.device; cfg.training_stage=1; cfg.pregrasp_drop_grace_steps=180; cfg.coarse_action_smoothing_alpha=1.0; cfg=apply_profile(cfg,args.profile,num_envs=1,seed=args.seed)
    env=V2EnvelopingPhysxTomatoGraspEnv(cfg); env.reset(seed=args.seed)
    open_action=torch.zeros((1,14),device=env.device); open_action[:,:10]=torch.tensor((-1,0,0,0,-1,-1,-1,-1,-1,-1),device=env.device)
    target=torch.tensor(vals,device=env.device).unsqueeze(0); best={}; best_score=-1e9
    for step in range(args.steps):
        f=min(1.0,(step+1)/150.0); env.step(open_action+f*(target-open_action))
        center=env.tomato.data.root_pos_w; tip,valid=env._geometry_contact_points(); active=env.contact_forces>=env.cfg.contact_threshold_n; palm=env.palm_force>=env.cfg.palm_support_force_n
        coverage=env.strict_coverage_current.item(); opposition=env.strict_opposition_current.item(); hemisphere=env.strict_hemisphere_current.item(); load=(env.palm_force/(env.palm_force+env.contact_forces.sum(-1)).clamp_min(1e-6)).item(); force_safe=bool((env.contact_forces<=env._ceiling).all() and (env.palm_force<=1.25*env.cfg.safe_force_n)); score=int(active.sum())*10+float(palm)*5+coverage/100
        if score>best_score:
            best_score=score; best={'step':step+1,'active':active[0].tolist(),'contact_forces':env.contact_forces[0].tolist(),'palm_force':env.palm_force.item(),'palm':bool(palm.item()),'coverage_deg':coverage,'opposition':opposition,'hemisphere':hemisphere,'load_share':load,'force_safe':force_safe,'dropped':bool(env.dropped.item()),'in_hand':float(torch.linalg.vector_norm(center-env.grasp_center,dim=-1).item()),'height':center[0,2].item(),'contact_geometry_valid':bool(valid.all().item()),'strict_hold':env.strict_hold_streak.item()}
    out={'version':'v2_strict_probe_v1','profile':args.profile,'seed':args.seed,'best':best,'hardware_command_allowed':False}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2),flush=True); env.close(); os._exit(0)
if __name__=='__main__': main()
