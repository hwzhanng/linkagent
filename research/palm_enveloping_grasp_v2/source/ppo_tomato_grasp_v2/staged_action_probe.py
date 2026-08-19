"""Probe a root-first then distal-finetune action schedule (diagnostic only)."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser(); p.add_argument('--root-action',required=True); p.add_argument('--distal-action',required=True); p.add_argument('--profile',default='v1_nominal_reliable'); p.add_argument('--num-envs',type=int,default=256); p.add_argument('--root-steps',type=int,default=150); p.add_argument('--distal-steps',type=int,default=180); p.add_argument('--seed',type=int,default=20260927); p.add_argument('--physical-fingertip-proxy',action='store_true'); p.add_argument('--disable-virtual-distal-contact',action='store_true'); p.add_argument('--output',type=Path,required=True); AppLauncher.add_app_launcher_args(p); args=p.parse_args(); app=AppLauncher(args).app
import torch
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv,V2EnvelopingPhysxTomatoGraspEnvCfg
def vals(s,n):
    x=[float(v.strip()) for v in s.split(',') if v.strip()]
    if len(x)!=n: raise ValueError(f'expected {n} values')
    return x
@torch.inference_mode()
def main():
    root=vals(args.root_action,14); distal=vals(args.distal_action,4)
    cfg=V2EnvelopingPhysxTomatoGraspEnvCfg(); cfg.sim.device=args.device; cfg.training_stage=1; cfg.episode_length_s=18.0; cfg=apply_profile(cfg,args.profile,num_envs=args.num_envs,seed=args.seed); cfg.fingertip_collision_proxy_physical=bool(args.physical_fingertip_proxy); cfg.virtual_distal_contact_enabled=not args.disable_virtual_distal_contact
    env=V2EnvelopingPhysxTomatoGraspEnv(cfg); env.reset(seed=args.seed)
    open_action=torch.zeros((args.num_envs,14),device=env.device); open_action[:,:10]=torch.tensor((-1,0,0,0,-1,-1,-1,-1,-1,-1),device=env.device)
    root_t=torch.tensor(root,device=env.device).expand(args.num_envs,-1); distal_t=torch.tensor(distal,device=env.device).expand(args.num_envs,-1)
    done_count=success=drop=overload=distal_all=distal_contact_all=0; hold=coverage=palm=0.0; codes={}
    for step in range(args.root_steps+args.distal_steps):
        if step<args.root_steps:
            f=(step+1)/max(args.root_steps,1); action=open_action+f*(root_t-open_action); action[:,10:]=0
        else:
            f=(step-args.root_steps+1)/max(args.distal_steps,1); action=root_t.clone(); action[:,10:]=f*distal_t
        _,_,term,trunc,_=env.step(action); done=term|trunc
        if bool(done.any()):
            snap=env.consume_terminal_snapshots(done); n=int(done.sum()); code=snap['termination_code']; done_count+=n; success+=int((code==1).sum()); drop+=int((code==2).sum()); overload+=int((code==3).sum()); distal_all+=int(snap['strict_distal_participation'].all(-1).sum()); distal_contact_all+=int(snap['distal_contact_observed'].all(-1).sum()); hold+=float(snap['strict_hold_max'].float().sum()); coverage+=float(snap['strict_coverage_max'].float().sum()); palm+=float((snap['palm_force']>=env.cfg.palm_support_force_n).float().sum());
            for c in code.cpu().tolist(): codes[str(int(c))]=codes.get(str(int(c)),0)+1
        if done_count>=args.num_envs: break
    den=max(done_count,1); out={'version':'staged_action_probe_v1','profile':args.profile,'seed':args.seed,'root_action':root,'distal_action':distal,'physical_fingertip_proxy':bool(args.physical_fingertip_proxy),'virtual_distal_contact_enabled':not args.disable_virtual_distal_contact,'completed_episodes':done_count,'success_rate':success/den,'drop_rate':drop/den,'overload_rate':overload/den,'distal_all_rate':distal_all/den,'distal_contact_all_rate':distal_contact_all/den,'strict_hold_max_mean':hold/den,'strict_coverage_max_mean_deg':coverage/den,'palm_support_rate':palm/den,'termination_code_counts':codes,'hardware_command_allowed':False}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2),flush=True); env.close(); os._exit(0)
if __name__=='__main__': main()
