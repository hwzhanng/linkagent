"""First-episode, pre-action observation/value diagnostic; no training."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from .critic_features import critic_features,FEATURE_NAMES

@torch.no_grad()
def collect_value_diagnostic(trainer,wrapped,checkpoint,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    assert not (output/'COMPLETE.json').exists()
    agent=trainer.agent;raw=wrapped._unwrapped
    modules=torch.load(checkpoint,map_location=raw.device,weights_only=True)
    for key in ('encoder','policy','value'):
        getattr(agent,key).load_state_dict(modules[key]);getattr(agent,key).eval()
    agent.encoder.state_preprocessor.load_state_dict(modules['state_preprocessor'])
    agent._value_preprocessor.load_state_dict(modules['value_preprocessor'])
    states,_=wrapped.reset(hard=True)
    initial_pose=raw._fruit.data.root_pose_w.torch.clone().cpu().numpy()
    active=torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool)
    initial_goal=raw._goal.clone()
    direction=initial_goal.sign()
    T=int(raw.max_episode_length+2);N=raw.num_envs
    arrays={}
    spec={'prop':(280,), 'tactile':(1728,), 'critic':(64,), 'latent':(256,),
          'action':(16,), 'value':(), 'reward':(), 'pose_delta':(), 'active':(),
          'terminated':(), 'truncated':(), 'pose_valid':(), 'pad':(5,), 'pose':(7,)}
    for k,shape in spec.items():
        dtype=np.bool_ if k in ('active','terminated','truncated','pose_valid','pad') else np.float32
        arrays[k]=np.lib.format.open_memmap(output/f'{k}.npy',mode='w+',dtype=dtype,shape=(T,N,*shape))
    start=time.monotonic()
    for t in range(T):
        if not active.any():break
        for k in ('prop','tactile'):
            arrays[k][t]=states['policy'][k][:].cpu().numpy()
        # Source observations already measured; this pure extractor cannot alter actor inputs.
        arrays['critic'][t]=critic_features(raw).cpu().numpy()
        z=agent.encoder(states)
        action,_,_=agent.policy.act(z,deterministic=True)
        v=agent._value_preprocessor(agent.value.compute_value(z),inverse=True)
        for k,val in [('latent',z),('action',action),('value',v.reshape(-1)),('active',active)]:
            arrays[k][t]=val.cpu().numpy()
        states,reward,term,trunc,info=wrapped.step(action)
        ev=info['step']
        post={'reward':reward.reshape(-1),'terminated':term.reshape(-1),'truncated':trunc.reshape(-1),
              'pose_delta':ev['pose_delta_rad']*direction,'pose_valid':ev['pose_step_valid'],
              'pad':ev['pad_contact'],'pose':ev['fruit_pose_native']}
        for k,val in post.items():arrays[k][t]=val.cpu().numpy()
        active &= ~(term|trunc).reshape(-1)
        if t%200==0:print(json.dumps({'step':t,'active':int(active.sum()),'seconds':time.monotonic()-start}),flush=True)
    else:raise RuntimeError('First episodes did not finish within horizon')
    steps=t
    for v in arrays.values():v.flush()
    active_np=np.asarray(arrays['active'][:steps])
    summary={'completed':True,'steps':steps,'allocated_steps':T,'environments':N,'dt':float(raw.step_dt),
        'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'actual_actor_observations':True,'pre_action_features_only':True,'no_training':True,
        'policy_input_keys':['prop','tactile'],'critic_features':FEATURE_NAMES,
        'valid_samples':int(active_np.sum()),'failures':int((arrays['terminated'][:steps]&active_np).any(0).sum()),
        'domain':raw.cfg.domain_profile,'initial_pose':initial_pose.tolist(),
        'goal':initial_goal.cpu().tolist(),'goal_capture':'initial_reset_before_first_action',
        'elapsed_seconds':time.monotonic()-start}
    (output/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('initial_pose','critic_features','goal')},indent=2),flush=True)
