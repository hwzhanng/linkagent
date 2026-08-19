"""Probe a frozen v1 actor in the v2 strict scene; diagnostic only."""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--profile", default="v1_nominal_reliable")
p.add_argument("--num-envs", type=int, default=128)
p.add_argument("--steps", type=int, default=280)
p.add_argument("--seed", type=int, default=20260923)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch
from torch import nn
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(110,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU())
        self.mean = nn.Linear(128,14)
    def forward(self, x): return torch.tanh(self.mean(self.body(x)))

def to_v1(env, obs):
    axes = ((env.tomato_axes - 0.025) / 0.012).clamp(-1.0, 1.0)
    tactile = obs[:,50:85].reshape(env.num_envs,5,7)
    phase = obs[:,100:104].unsqueeze(1).expand(-1,5,-1)
    force = tactile[:,:,0]
    old = torch.cat((force.unsqueeze(-1), force.unsqueeze(-1), (force/0.15).clamp(0,1).unsqueeze(-1), phase,
                     tactile[:,:,1].unsqueeze(-1), tactile[:,:,2].unsqueeze(-1), (1-tactile[:,:,5]).unsqueeze(-1)), -1).reshape(env.num_envs,50)
    return torch.cat((obs[:,:3], axes, obs[:,6:50], old, obs[:,108:118]), -1).clamp(-1,1)

@torch.inference_mode()
def main():
    ck = torch.load(args.checkpoint.expanduser().resolve(), map_location=args.device, weights_only=False)
    actor = Actor().to(args.device)
    m=ck["model"]
    actor.load_state_dict({"body.0.weight":m["actor_body.0.weight"],"body.0.bias":m["actor_body.0.bias"],"body.2.weight":m["actor_body.2.weight"],"body.2.bias":m["actor_body.2.bias"],"body.4.weight":m["actor_body.4.weight"],"body.4.bias":m["actor_body.4.bias"],"mean.weight":m["actor_mean.weight"],"mean.bias":m["actor_mean.bias"]})
    cfg=V2EnvelopingPhysxTomatoGraspEnvCfg(); cfg.sim.device=args.device; cfg.training_stage=1; cfg.episode_length_s=16.0
    cfg=apply_profile(cfg,args.profile,num_envs=args.num_envs,seed=args.seed)
    env=V2EnvelopingPhysxTomatoGraspEnv(cfg); obs,_=env.reset(seed=args.seed)
    counts={k:0 for k in ("success","drop","overload","five","palm","distal","distal_contact","palm_first")}; completed=0
    for _ in range(args.steps):
        action=actor(to_v1(env,obs["policy"])); obs,_,term,trunc,_=env.step(action); done=term|trunc
        if bool(done.any()):
            snap=env.consume_terminal_snapshots(done); code=snap["termination_code"]; completed+=int(done.sum())
            counts["success"]+=int((code==1).sum()); counts["drop"]+=int((code==2).sum()); counts["overload"]+=int((code==3).sum())
            counts["five"]+=int((snap["contact_forces"]>=env.cfg.contact_threshold_n).all(-1).sum()); counts["palm"]+=int((snap["palm_force"]>=env.cfg.palm_support_force_n).sum()); counts["distal"]+=int(snap["strict_distal_participation"].all(-1).sum()); counts["distal_contact"]+=int(snap["distal_contact_observed"].all(-1).sum()); counts["palm_first"]+=int(((snap["strict_first_palm"]>=0)&(snap["strict_first_three"]>=0)&(snap["strict_first_palm"]+env.cfg.strict_palm_first_margin_steps<=snap["strict_first_three"])).sum())
        if completed>=args.num_envs: break
    den=max(completed,1); out={"version":"v1_transfer_probe_v1","checkpoint":str(args.checkpoint),"profile":args.profile,"completed_episodes":completed,**{f"{k}_rate":v/den for k,v in counts.items()},"hardware_command_allowed":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2),flush=True); env.close(); os._exit(0)
if __name__=="__main__": main()
