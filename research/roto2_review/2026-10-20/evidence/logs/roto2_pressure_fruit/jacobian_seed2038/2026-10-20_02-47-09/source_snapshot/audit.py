"""Diagnostic sampling at every physics substep; never used for policy updates."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .core import quat_inv_rotate

@torch.no_grad()
def audit_checkpoint(trainer, wrapped, checkpoint, output):
    agent=trainer.agent
    saved=torch.load(checkpoint,map_location=wrapped.device,weights_only=True)
    for name in ('encoder','policy'):
        getattr(agent,name).load_state_dict(saved[name]);getattr(agent,name).eval()
    agent.encoder.state_preprocessor.load_state_dict(saved['state_preprocessor'])
    states,_=wrapped.reset(hard=True);raw=wrapped._unwrapped
    n=min(8,raw.num_envs);sub=[];contacts=[];control=[];done=[]
    original=raw.scene.update
    def sample(dt,*args,**kwargs):
        result=original(dt,*args,**kwargs)
        if dt>0:
            sub.append(torch.cat((raw._fruit.data.root_pose_w.torch[:n],
                                  raw._fruit.data.root_ang_vel_w.torch[:n],
                                  raw._fruit.data.root_com_vel_w.torch[:n]),-1).cpu().numpy().copy())
        return result
    raw.scene.update=sample
    try:
        for step in range(200):
            action,_,_=agent.policy.act(agent.encoder(states),deterministic=True)
            states,_,term,trunc,info=wrapped.step(action)
            q=raw._wxyz(raw._robot.data.body_quat_w.torch[:n,raw._contact_body_ids[:5]])
            p=raw._robot.data.body_pos_w.torch[:n,raw._contact_body_ids[:5]]
            local=quat_inv_rotate(q,raw._contact_positions[:n,:5]-p)
            contacts.append(torch.cat((local,raw._forces[:n,:5,None]),-1).cpu().numpy().copy())
            control.append(info['step']['rotation_rate'][:n].cpu().numpy().copy())
            done.append((term|trunc).reshape(-1)[:n].cpu().numpy().copy())
    finally: raw.scene.update=original
    sub=np.array(sub);contacts=np.array(contacts);done=np.array(done)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'),substeps=sub,contacts=contacts,control_omega=control,done=done)
    summaries=[]
    for i in range(n):
        end=int(np.flatnonzero(done[:,i])[0]) if done[:,i].any() else 200
        end=max(1,end);a=sub[:end*raw.cfg.decimation,i];q=Rotation.from_quat(a[:,3:7]);rv=(q[1:]*q[:-1].inv()).as_rotvec();w=a[:,7:10]
        summaries.append({'env':i,'control_steps':end,'pose_z_turns':float(rv[:,2].sum()/(2*np.pi)),
                          'physics_omega_integral_turns':float(w[:,2].sum()*raw.cfg.sim.dt/(2*np.pi)),
                          'control_omega_integral_turns':float(np.array(control)[:end,i].sum()*raw.step_dt/(2*np.pi)),
                          'substep_omega_pose_rate_rmse':float(np.sqrt(np.mean((rv/raw.cfg.sim.dt-w[1:])**2))),
                          'finger_contact_fractions':(contacts[:end,i,:,3]>.15).mean(0).tolist(),
                          'positive_contact_local_xyz_m':[[float(x) for x in np.median(contacts[:end,i,j,:3][contacts[:end,i,j,3]>.15],axis=0)] if (contacts[:end,i,j,3]>.15).any() else None for j in range(5)]})
    result={'checkpoint':str(checkpoint),'physics_dt':raw.cfg.sim.dt,'decimation':raw.cfg.decimation,'substeps':len(sub),'per_env':summaries}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
