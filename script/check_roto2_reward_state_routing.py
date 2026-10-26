"""Real-checkpoint tests for actor isolation, value migration and memory alignment."""
import json
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
import yaml
from multimodal_rl.models.encoder import Encoder
from multimodal_rl.models.running_standard_scaler import RunningStandardScaler
from multimodal_rl.rl.policy_value import GaussianPolicy,DeterministicValue
from multimodal_rl.rl.memories import Memory
from simulation.roto2_pressure.critic_reward_state import (
    RewardStateEncoder,ActorOnlyPolicy,load_reward_state_checkpoint,
)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v17/routing'
SOURCE=ROOT/'logs/roto2_pressure_fruit/plain_seed7401/2026-10-21_16-16-13/checkpoints/final_agent.pt'


def make(mode):
    cfg=yaml.safe_load((ROOT/'simulation/roto2_pressure/agent.yaml').read_text())
    obs=gym.spaces.Dict({k:gym.spaces.Box(-np.inf,np.inf,(dim,),np.float32)
                         for k,dim in [('prop',280),('tactile',1728)]})
    action=gym.spaces.Box(-np.inf,np.inf,(16,),np.float32)
    encoder=(Encoder(obs,action,None,cfg,'cpu') if mode=='off' else
             RewardStateEncoder(obs,action,None,cfg,'cpu',critic_mode=mode))
    policy=(GaussianPolicy(256,obs,action,'cpu',**cfg['policy']) if mode=='off' else
            ActorOnlyPolicy(256,obs,action,'cpu',actor_latent_dim=256,**cfg['policy']))
    value=DeterministicValue(encoder.num_outputs,obs,action,'cpu',**cfg['value'])
    agent=SimpleNamespace(device='cpu',encoder=encoder,policy=policy,value=value,
                          _value_preprocessor=RunningStandardScaler(1,device='cpu',dtype=torch.float32))
    for key in ['encoder','policy','value']:
        setattr(agent,key+'_optimiser',torch.optim.Adam(getattr(agent,key).parameters(),lr=.000202))
    return agent


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    source=torch.load(SOURCE,map_location='cpu',weights_only=True)
    reference=make('off')
    for k in ['encoder','policy','value']:getattr(reference,k).load_state_dict(source[k])
    reference.encoder.state_preprocessor.load_state_dict(source['state_preprocessor'])
    data=ROOT/'artifacts/simulation/rotation_v15/critic_information/collect_seed9801'
    obs={k:torch.as_tensor(np.load(data/f'{k}.npy',mmap_mode='r')[50].copy()) for k in ['prop','tactile']}
    actor_input={'policy':obs}
    with torch.no_grad():
        z0=reference.encoder(actor_input);a0=reference.policy.act(z0,deterministic=True)[0]
        v0=reference.value.compute_value(z0)
        log0=reference.policy.act(z0,taken_actions=a0)[1]
    reports=[]
    for mode in ['masked','validity']:
        agent=make(mode);migration=load_reward_state_checkpoint(agent,SOURCE)
        zs=[];values=[]
        for bit in [0.,1.]:
            inputs={'policy':{**obs,'gt':torch.full((len(a0),1),bit)}}
            z=agent.encoder(inputs);z.retain_grad();zs.append(z)
            mean=agent.policy.act(z,deterministic=True)[0]
            log=agent.policy.act(z,taken_actions=a0)[1]
            assert torch.equal(z[:,:256],z0) and torch.equal(mean,a0) and torch.equal(log,log0)
            value=agent.value.compute_value(z);values.append(value.detach())
            assert torch.allclose(value,v0,atol=5e-6,rtol=1e-6)
            mean.sum().backward(retain_graph=True)
            assert torch.count_nonzero(z.grad[:,-1])==0
        first_name,first=next(iter(agent.value.named_parameters()))
        state=agent.value_optimiser.state[first]
        old_id=source['value_optimiser']['param_groups'][0]['params'][0]
        old_state=source['value_optimiser']['state'][old_id]
        for key in ['exp_avg','exp_avg_sq']:
            assert torch.equal(state[key][:,:-1],old_state[key])
            assert torch.count_nonzero(state[key][:,-1])==0
        assert torch.equal(state['step'],old_state['step'])
        agent.value_optimiser.zero_grad();loss=agent.value.compute_value(zs[1]).square().mean();loss.backward()
        extra_gradient=float(first.grad[:,-1].abs().sum())
        assert (extra_gradient==0) if mode=='masked' else (extra_gradient>0)
        # A learned critic bit may affect value but must still never affect actor.
        with torch.no_grad():first[:,-1]=.1
        with torch.no_grad():
            new_values=[agent.value.compute_value(z) for z in zs]
            if mode=='validity':assert not torch.equal(*new_values)
            assert torch.equal(agent.policy.act(zs[0],deterministic=True)[0],agent.policy.act(zs[1],deterministic=True)[0])
        reports.append(dict(mode=mode,migration=migration,actor_latent_mean_logprob_bitwise_equal=True,
                            actor_gradient_to_extra_input_zero=True,new_value_column_gradient=extra_gradient,
                            source_value_max_abs_error=max(float((v-v0).abs().max()) for v in values),
                            source_value_adam_moments_and_step_preserved=True))
    memory=Memory(2,3,device='cpu')
    memory.create_tensor('prop',size=(2,),dtype=torch.float32)
    memory.create_tensor('gt',size=(1,),dtype=torch.float32)
    for t in range(2):
        markers=torch.arange(3,dtype=torch.float32)+3*t
        memory.add_samples('parallel',states={'policy':{'prop':torch.stack([markers,markers+10],-1),'gt':(markers%2)[:,None]}})
    batches=memory.sample_all(['prop','gt'],mini_batches=2)
    for batch in batches:assert torch.equal(batch[0]['gt'][:,0],batch[0]['prop'][:,0]%2)
    result=dict(passed=True,real_checkpoint=str(SOURCE),arms=reports,upstream_memory_sample_alignment_passed=True,
                physical_integration_pending=True)
    (OUT/'MODEL_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
