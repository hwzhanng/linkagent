"""Pure existing-network recovery test; no phase, waypoint or teacher action input."""
from collections import deque
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from multimodal_rl.models.encoder import Encoder
from multimodal_rl.rl.policy_value import GaussianPolicy
from .calibrate_redundancy import step_physics,save


@torch.no_grad()
def evaluate_recovery_policy(env,output,checkpoint,agent_cfg):
    raw=env.unwrapped;output=Path(output);assert not output.exists()
    spaces={k:gym.spaces.Box(-np.inf,np.inf,(d,),dtype=np.float32) for k,d in [('prop',280),('tactile',1728)]}
    actions=gym.spaces.Box(-np.inf,np.inf,(16,),dtype=np.float32)
    encoder=Encoder(spaces,actions,raw.cfg,agent_cfg,raw.device)
    policy=GaussianPolicy(encoder.num_outputs,spaces,actions,raw.device,**agent_cfg['policy'])
    state=torch.load(checkpoint,map_location=raw.device,weights_only=True)
    encoder.load_state_dict(state['encoder']);policy.load_state_dict(state['policy'])
    encoder.state_preprocessor.load_state_dict(state['state_preprocessor'])
    encoder.eval();policy.eval();env.reset()
    history={k:deque([raw.get_observations()['policy'][k].clone() for _ in range(4)],maxlen=4) for k in spaces}
    q0=raw._robot.data.joint_pos.torch[:,raw._active_ids].clone()
    alive=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device);stable=alive.clone()
    free_streak=torch.zeros(raw.num_envs,device=raw.device);longest=free_streak.clone()
    rows={k:[] for k in ('pose','pad','stable','failed','whole_finger_force','joint','action','peak_pressure','displacement')}
    for t in range(200):
        obs={k:torch.cat(list(h),dim=-1) for k,h in history.items()}
        action=policy.act(encoder(obs),deterministic=True)[0]
        # step_physics uses the unmodified action/actuator path and avoids reset.
        d=step_physics(raw,raw._q_target+action.clamp(-1,1)*raw.cfg.maximum_joint_target_delta_rad)
        alive &= ~d['failed']
        if t>=20:stable &= d['stable'] & alive
        free=d['whole_finger_force'][:,0]<.05
        free_streak=torch.where(free,free_streak+1,torch.zeros_like(free_streak));longest=torch.maximum(longest,free_streak)
        d['joint']=raw._robot.data.joint_pos.torch[:,raw._active_ids];d['action']=action
        for k in rows:rows[k].append(d[k].cpu().numpy().copy())
        fresh=raw._get_observations()['policy']
        for k,h in history.items():h.append(fresh[k].clone())
    data={k:np.asarray(v) for k,v in rows.items()};initial=q0.cpu().numpy()
    pad=data['pad'][160:,:,0].mean(0)
    reserve=((data['joint'][160:,:,0]-initial[None,:,0]>=.1)&(data['joint'][160:,:,0]>=.08)).mean(0)
    maxpressure=data['peak_pressure'].max(0)
    passed=stable.cpu().numpy()&alive.cpu().numpy()&(longest.cpu().numpy()>=10)&(pad>=.8)&(reserve>=.8)&(maxpressure<=raw.cfg.maximum_pressure_proxy_pa)
    np.savez_compressed(output.with_suffix('.npz'),**data,initial_joint=initial)
    result=[dict(state_id=r['source_state_id'],split=r['demonstration_split'],passed=bool(passed[i]),
        failed=not bool(alive[i]),stable=bool(stable[i]),final_pad_fraction=float(pad[i]),
        reserve_fraction=float(reserve[i]),final_joint2=float(data['joint'][160:,i,0].mean()),
        peak_pressure_pa=float(maxpressure[i]),maximum_unloaded_steps=float(longest[i])) for i,r in enumerate(raw._reset_grasps)]
    save(output,dict(completed=True,checkpoint=str(checkpoint),pure_policy=True,teacher_actions_used=False,
        actor_features='Unmodified tactile+prop, four frames; no waypoint, clock, phase or privileged pose',
        initial_settling_teacher_actions=False,seconds=200*raw.step_dt,total=raw.num_envs,
        passed=int(passed.sum()),results=result,goal_complete=False,
        scope='Local recovery only. Full rotation acceptance remains a separate unchanged evaluation.'))
    print(f'Pure policy recovery: {int(passed.sum())}/{raw.num_envs}',flush=True)
