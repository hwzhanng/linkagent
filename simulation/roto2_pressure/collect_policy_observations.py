"""Read-only first-episode source-policy observations, using the actual wrapper."""
from pathlib import Path
import numpy as np
import torch
from .calibrate_redundancy import save


@torch.no_grad()
def collect_policy_observations(trainer,wrapped,checkpoint,output):
    output=Path(output);assert not output.exists()
    agent=trainer.agent;raw=wrapped._unwrapped
    modules=torch.load(checkpoint,map_location=raw.device,weights_only=True)
    for key in ('encoder','policy'):
        getattr(agent,key).load_state_dict(modules[key]);getattr(agent,key).eval()
    agent.encoder.state_preprocessor.load_state_dict(modules['state_preprocessor'])
    states,_=wrapped.reset(hard=True)
    active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool)
    data={k:[] for k in ('prop','tactile','action','active','pose','pad')}
    for t in range(160):
        for k in ('prop','tactile'):data[k].append(states['policy'][k][:].cpu().numpy().copy())
        action,_,_=agent.policy.act(agent.encoder(states),deterministic=True)
        data['action'].append(action.cpu().numpy().copy());data['active'].append(active.cpu().numpy().copy())
        states,_,term,trunc,info=wrapped.step(action)
        data['pose'].append(info['step']['fruit_pose_native'].cpu().numpy().copy())
        data['pad'].append(info['step']['pad_contact'].cpu().numpy().copy())
        active &= ~(term|trunc).reshape(-1)
    np.savez_compressed(output.with_suffix('.npz'),**{k:np.asarray(v) for k,v in data.items()})
    save(output,dict(completed=True,checkpoint=str(checkpoint),environments=raw.num_envs,steps=160,
        seconds=160*raw.step_dt,valid_samples=int(np.asarray(data['active']).sum()),
        actual_upstream_frame_stack_used=True,no_training=True,pre_action_observations=True))
