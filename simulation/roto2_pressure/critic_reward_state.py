"""One persistent reward-state bit for critic only; upstream PPO stays intact.

Actor observations/weights remain exactly the original prop+tactile network.
The bit is 1 after the task's existing positive-rotation reward latch is lost,
0 otherwise. It neither changes that latch nor exposes object state to actor.
"""
from collections import deque
from copy import deepcopy
import gymnasium as gym
import torch
from multimodal_rl.models.encoder import Encoder
from multimodal_rl.rl.policy_value import GaussianPolicy
from multimodal_rl.wrappers.frame_stack import FrameStack


def with_reward_state(raw,obs):
    bit=(~raw._reward_path_valid).to(torch.float32).unsqueeze(-1)
    return {**obs,'policy':{**obs['policy'],'gt':bit}}


class RewardStateObservation(gym.ObservationWrapper):
    def observation(self,obs):
        return with_reward_state(self.unwrapped,obs)


class RewardStateFrameStack(FrameStack):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.frames['gt']=deque(maxlen=1)


class RewardStateEncoder(Encoder):
    def __init__(self,*args,critic_mode,**kwargs):
        super().__init__(*args,**kwargs)
        assert critic_mode in ('masked','validity')
        self.critic_mode=critic_mode
        self.actor_latent_dim=self.num_outputs
        self.num_outputs+=1

    def forward(self,obs_dict,detach=False,train=False):
        obs=obs_dict.get('policy',obs_dict)
        z=super().forward({'policy':{k:obs[k] for k in ('prop','tactile')}},detach=detach,train=train)
        bit=obs['gt'][:].detach()
        if bit.shape[-1]!=1:raise ValueError('Reward-state bit must be current, not stacked')
        if self.critic_mode=='masked':bit=torch.zeros_like(bit)
        return torch.cat((z,bit),-1)


class ActorOnlyPolicy(GaussianPolicy):
    def __init__(self,*args,actor_latent_dim,**kwargs):
        self.actor_latent_dim=actor_latent_dim
        super().__init__(*args,**kwargs)

    def act(self,z,taken_actions=None,deterministic=False):
        return super().act(z[...,:self.actor_latent_dim],taken_actions=taken_actions,deterministic=deterministic)


def load_reward_state_checkpoint(agent,path):
    """Preserve old parameters and Adam moments; zero-pad the one new column."""
    source=torch.load(path,map_location=agent.device,weights_only=True)
    for name in ('encoder','policy'):
        getattr(agent,name).load_state_dict(source[name],strict=True)
    agent.encoder.state_preprocessor.load_state_dict(source['state_preprocessor'])
    agent._value_preprocessor.load_state_dict(source['value_preprocessor'])
    target=agent.value.state_dict();expanded=[]
    assert target.keys()==source['value'].keys()
    for key,src in source['value'].items():
        dst=target[key]
        if dst.shape==src.shape:dst.copy_(src)
        elif dst.ndim==2 and dst.shape==(src.shape[0],src.shape[1]+1):
            dst.zero_();dst[:,:-1].copy_(src);expanded.append(key)
        else:raise ValueError(f'Unexpected value migration: {key}')
    assert len(expanded) in (0,1)
    agent.value.load_state_dict(target,strict=True)
    for name in ('encoder_optimiser','policy_optimiser'):
        getattr(agent,name).load_state_dict(source[name])
    optimizer=deepcopy(source['value_optimiser'])
    assert len(optimizer['param_groups'])==1
    ids=optimizer['param_groups'][0]['params'];params=list(agent.value.parameters())
    assert len(ids)==len(params)
    for index,param in zip(ids,params):
        for name,value in optimizer['state'].get(index,{}).items():
            if isinstance(value,torch.Tensor) and value.ndim>0 and value.shape!=param.shape:
                assert name in ('exp_avg','exp_avg_sq','max_exp_avg_sq')
                assert param.ndim==2 and param.shape==(value.shape[0],value.shape[1]+1)
                padded=torch.zeros_like(param);padded[:,:-1].copy_(value)
                optimizer['state'][index][name]=padded
    agent.value_optimiser.load_state_dict(optimizer)
    return dict(source=str(path),expanded_parameters=expanded,
                actor_and_optimizers_preserved=True,value_existing_adam_preserved=True,
                new_value_input_and_adam_columns_zero=bool(expanded),
                critic_feature='persistent_reward_path_invalid',critic_dimensions=1,
                ppo_implementation_unchanged=True)
