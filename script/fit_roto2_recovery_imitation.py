"""Offline action pretraining of the EXISTING encoder/policy; PPO is untouched.

Uses only real pre-action actor observations, with whole-source-state holdout.
This is an exploratory initialization. Success requires independent pure-policy
physical rollouts; validation MSE is never reported as task success.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
import yaml
from multimodal_rl.models.encoder import Encoder
from multimodal_rl.rl.policy_value import GaussianPolicy


def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    a.output.mkdir(exist_ok=False);torch.manual_seed(9722);np.random.seed(9722);torch.set_num_threads(8)
    device='cuda:0';root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'simulation/roto2_pressure/agent.yaml').read_text())
    spaces={k:gym.spaces.Box(-np.inf,np.inf,(size,),dtype=np.float32) for k,size in [('prop',280),('tactile',1728)]}
    action_space=gym.spaces.Box(-np.inf,np.inf,(16,),dtype=np.float32)
    encoder=Encoder(spaces,action_space,SimpleNamespace(),cfg,device)
    policy=GaussianPolicy(encoder.num_outputs,spaces,action_space,device,**cfg['policy'])
    source=torch.load(a.checkpoint,map_location=device,weights_only=True)
    encoder.load_state_dict(source['encoder']);policy.load_state_dict(source['policy'])
    encoder.state_preprocessor.load_state_dict(source['state_preprocessor'])
    encoder.eval();policy.eval();policy.log_std_parameter.requires_grad_(False)
    teacher=np.load(a.data/'TEACHER.npz');meta=json.loads((a.data/'TEACHER.json').read_text())
    preservation=np.load(a.data/'PRESERVATION.npz')
    split=np.array([r['split']=='validation' for r in meta['rows']])
    tids,eids=np.nonzero(teacher['eligible'])
    data={}
    for is_val,name in [(False,'train'),(True,'validation')]:
        ids=np.flatnonzero(split[eids]==is_val);t=tids[ids];e=eids[ids]
        assert len(ids)>0
        data[name]={k:torch.tensor(np.concatenate([teacher['obs_'+k][t-offset,e] for offset in (3,2,1,0)],axis=-1),device=device) for k in spaces}
        data[name]['action']=torch.tensor(teacher['teacher_action'][t,e],device=device)
        # Whole episodes, not adjacent frames, separate preservation validation.
        mask=preservation['active'].copy();mask[:20]=False
        mask &= (np.arange(mask.shape[1])%4==0 if is_val else np.arange(mask.shape[1])%4!=0)[None]
        t,e=np.nonzero(mask)
        data['preserve_'+name]={k:torch.tensor(preservation[k][t,e],device=device) for k in (*spaces,'action')}
    train_states={r['state_id'] for r in meta['rows'] if r['split']=='train'}
    val_states={r['state_id'] for r in meta['rows'] if r['split']=='validation'}
    assert not train_states & val_states
    assert all(torch.isfinite(v).all() for group in data.values() for v in group.values())
    manifest=dict(steps=a.steps,batch_size_each=256,learning_rate=.0001,source_checkpoint=str(a.checkpoint.resolve()),
        source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),architecture_changed=False,
        ppo_changed=False,scalers_frozen=True,log_std_frozen=True,value_weights_frozen=True,
        training_method='One offline supervised warm start: recovery MSE + equal-weight source-action preservation MSE',
        selection='Lowest sum of whole-state recovery holdout MSE and whole-episode preservation holdout MSE',
        no_privileged_actor_inputs=True,ancestral_grasp_families=1,
        source_state_overlap=0,sample_counts={k:len(v['action']) for k,v in data.items()},
        teacher_sha256=hashlib.sha256((a.data/'TEACHER.npz').read_bytes()).hexdigest(),
        preservation_sha256=hashlib.sha256((a.data/'PRESERVATION.npz').read_bytes()).hexdigest(),
        frame_stack='Oldest to latest, pre-action[t-3:t+1]; matches upstream LazyFrames concatenation')
    save(a.output/'PLAN.json',manifest)
    def mean_action(group):return policy.act(encoder({k:group[k] for k in spaces}),deterministic=True)[0]
    @torch.no_grad()
    def metrics():
        result={}
        for name,group in data.items():
            squared=0.;count=0;direction=0;direction_count=0
            for start in range(0,len(group['action']),512):
                mini={k:v[start:start+512] for k,v in group.items()};pred=mean_action(mini)
                squared+=float((pred-mini['action']).square().sum());count+=pred.numel()
                nonzero=mini['action'].abs()>.1
                direction+=int(((pred.sign()==mini['action'].sign())&nonzero).sum());direction_count+=int(nonzero.sum())
            result[name+'_mse']=squared/count
            result[name+'_moving_action_sign_accuracy']=direction/max(1,direction_count)
        return result
    initial=metrics();history=[dict(step=0,**initial)]
    best_score=initial['validation_mse']+initial['preserve_validation_mse'];best_step=0
    best_encoder=copy.deepcopy(encoder.state_dict());best_policy=copy.deepcopy(policy.state_dict())
    params=[v for v in list(encoder.parameters())+list(policy.parameters()) if v.requires_grad]
    optimizer=torch.optim.Adam(params,lr=.0001)
    started=time.monotonic()
    for step in range(1,a.steps+1):
        optimizer.zero_grad(set_to_none=True);loss=0
        for name in ('train','preserve_train'):
            group=data[name];ids=torch.randint(len(group['action']),(256,),device=device)
            mini={k:v[ids] for k,v in group.items()};loss=loss+(mean_action(mini)-mini['action']).square().mean()
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if step%100==0 or step==a.steps:
            m=metrics();history.append(dict(step=step,**m));score=m['validation_mse']+m['preserve_validation_mse']
            if score<best_score:
                best_score=score;best_step=step;best_encoder=copy.deepcopy(encoder.state_dict());best_policy=copy.deepcopy(policy.state_dict())
            save(a.output/'HISTORY.json',history)
            print(f'Imitation {step}/{a.steps}: holdout recovery MSE {m["validation_mse"]:.5f}; preservation {m["preserve_validation_mse"]:.5f}',flush=True)
    encoder.load_state_dict(best_encoder);policy.load_state_dict(best_policy)
    result=copy.deepcopy(source);result['encoder']=best_encoder;result['policy']=best_policy
    for key in ('encoder_optimiser','policy_optimiser'):result[key]['state']={}
    assert set(result['encoder'])==set(source['encoder']) and set(result['policy'])==set(source['policy'])
    for module in ('encoder','policy'):
        assert all(result[module][k].shape==source[module][k].shape and torch.isfinite(result[module][k]).all() for k in source[module])
    assert all(torch.equal(result['encoder'][k],source['encoder'][k]) for k in source['encoder'] if k.startswith('state_preprocessor.'))
    assert torch.equal(result['policy']['log_std_parameter'],source['policy']['log_std_parameter'])
    target=a.output/'imitation_agent.pt';torch.save(result,target)
    save(a.output/'RESULT.json',dict(completed=True,training_method='offline action imitation; no PPO updates',
        selected_step=best_step,initial=initial,selected=metrics(),seconds=time.monotonic()-started,
        model_schema_unchanged=True,checkpoint=str(target.resolve()),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        optimizer_states_reset=['encoder_optimiser','policy_optimiser'],physical_policy_validation_pending=True,
        goal_complete=False))


if __name__=='__main__':main()
