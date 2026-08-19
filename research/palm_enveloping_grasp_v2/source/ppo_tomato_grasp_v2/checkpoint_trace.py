"""Trace one deterministic checkpoint rollout for failure diagnosis only."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--profile", default="v1_nominal_reliable")
p.add_argument("--steps", type=int, default=100)
p.add_argument("--seed", type=int, default=20260830)
p.add_argument("--physical-fingertip-proxy", action="store_true")
p.add_argument("--fingertip-proxy-radius", type=float, default=0.006)
p.add_argument("--disable-virtual-distal-contact", action="store_true")
p.add_argument("--palm-first-gate", action="store_true")
p.add_argument("--palm-first-coarse-scale", type=float, default=0.35)
p.add_argument("--palm-first-coarse-max-scale", type=float, default=1.0)
p.add_argument("--palm-first-ramp-steps", type=int, default=60)
p.add_argument("--smoothing-alpha", type=float, default=1.0)
p.add_argument("--assist-gain", type=float, default=None)
p.add_argument("--per-digit-distal-gate", action="store_true")
p.add_argument("--per-digit-link3-threshold", type=float, default=0.04)
p.add_argument("--per-digit-link3-stable-steps", type=int, default=4)
p.add_argument("--per-digit-near-surface-gate", action="store_true")
p.add_argument("--per-digit-near-surface-distance", type=float, default=0.043)
p.add_argument("--per-digit-near-surface-min-episode-step", type=int, default=20)
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); app = AppLauncher(args).app
import torch
from torch import nn
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.contract import V2_ACTOR_OBSERVATION_DIM, V2_ACTION_DIM
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(V2_ACTOR_OBSERVATION_DIM,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,128),nn.Tanh())
        self.mean = nn.Linear(128,V2_ACTION_DIM)
    def forward(self, obs): return torch.tanh(self.mean(self.body(obs)))

def js(x):
    raw = x.detach().cpu().tolist()
    def c(v): return [c(z) for z in v] if isinstance(v,list) else float(v)
    return c(raw)

@torch.inference_mode()
def main():
    ck = torch.load(args.checkpoint.expanduser().resolve(), map_location=args.device, weights_only=False)
    actor = Actor().to(args.device)
    m = ck["model"]
    actor.load_state_dict({"body.0.weight":m["actor_body.0.weight"],"body.0.bias":m["actor_body.0.bias"],"body.2.weight":m["actor_body.2.weight"],"body.2.bias":m["actor_body.2.bias"],"body.4.weight":m["actor_body.4.weight"],"body.4.bias":m["actor_body.4.bias"],"mean.weight":m["actor_mean.weight"],"mean.bias":m["actor_mean.bias"]})
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg(); cfg.sim.device=args.device; cfg.training_stage=1; cfg.episode_length_s=12.0; cfg.strict_enveloping_enabled=True
    cfg.safe_force_n=12.0; cfg.force_ceiling_n=(12.0,9.75,9.75,8.25,8.25)
    cfg.fingertip_collision_proxy_physical=bool(args.physical_fingertip_proxy); cfg.fingertip_collision_proxy_radius_m=float(args.fingertip_proxy_radius)
    if args.disable_virtual_distal_contact: cfg.virtual_distal_contact_enabled=False
    cfg.strict_require_measured_distal_contact=True; cfg.palm_first_distal_gate=bool(args.palm_first_gate); cfg.palm_first_coarse_scale=float(args.palm_first_coarse_scale); cfg.palm_first_coarse_max_scale=float(args.palm_first_coarse_max_scale); cfg.palm_first_ramp_steps=int(args.palm_first_ramp_steps); cfg.coarse_action_smoothing_alpha=float(args.smoothing_alpha)
    if args.assist_gain is not None:
        cfg.distal_proximal_reach_assist_gain=float(args.assist_gain); cfg.distal_proximal_reach_assist_gain_per_digit=None
    cfg.per_digit_distal_gate=bool(args.per_digit_distal_gate); cfg.per_digit_distal_link3_threshold_n=float(args.per_digit_link3_threshold); cfg.per_digit_distal_link3_stable_steps=int(args.per_digit_link3_stable_steps); cfg.per_digit_distal_near_surface_enabled=bool(args.per_digit_near_surface_gate); cfg.per_digit_distal_near_surface_distance_m=float(args.per_digit_near_surface_distance); cfg.per_digit_distal_near_surface_min_episode_step=int(args.per_digit_near_surface_min_episode_step)
    cfg=apply_profile(cfg,args.profile,num_envs=1,seed=args.seed)
    env=V2EnvelopingPhysxTomatoGraspEnv(cfg); obs,_=env.reset(seed=args.seed); rows=[]
    for step in range(args.steps):
        action=actor(obs["policy"])
        obs,reward,term,trunc,_=env.step(action)
        tips = env._tip_positions()[:, 1:]
        distances = torch.linalg.vector_norm(tips - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1)
        rows.append({"step":step,"action":js(action),"filtered_distal_action":js(env._v2_distal_action),"tip_distance_m":js(distances),"reward":js(reward),"tomato_pos_w":js(env.tomato.data.root_pos_w),"palm_force_n":js(env.palm_force),"measured_palm_force_n":js(env._v2_measured_palm_force),"contact_forces_n":js(env.contact_forces),"measured_link4_force_n":js(env._v2_measured_link4_force),"link3_force_n":js(env._v2_link3_force),"palm_age":js(env._v2_palm_support_age),"tip_positions_w":js(env._tip_positions()),"dropped":js(env.dropped),"overloaded":js(env.overloaded),"terminated":js(term),"truncated":js(trunc)})
        if bool((term|trunc).any()): break
    out={"version":"v2_checkpoint_trace_v1","checkpoint":str(args.checkpoint),"profile":args.profile,"seed":args.seed,"rows":rows,"hardware_command_allowed":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2),flush=True); env.close(); os._exit(0)
if __name__=="__main__": main()
