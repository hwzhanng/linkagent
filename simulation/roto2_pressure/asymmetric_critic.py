"""UNWIRED EXPERIMENT DRAFT: V15 information gate failed; not runtime validated.

Opt-in critic information route; upstream PPO and actor modules unchanged.

The upstream memory supports the `gt` observation slot. It carries a SINGLE
current-state critic vector, not actor input. The encoder adapter strips it
before calling the original encoder; the policy adapter strips it before the
original GaussianPolicy. Checkpoint actor keys/shapes remain unchanged.
"""
from collections import deque
import torch
import gymnasium as gym
from multimodal_rl.models.encoder import Encoder
from multimodal_rl.rl.policy_value import GaussianPolicy
from multimodal_rl.wrappers.frame_stack import FrameStack
from .critic_features import critic_features,CRITIC_FEATURE_DIM

class CriticObservationWrapper(gym.ObservationWrapper):
    def observation(self,obs):
        return {'policy':{**obs['policy'],'gt':critic_features(self.unwrapped)}}

class CriticFrameStack(FrameStack):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Actor remains four-frame. Current privileged state is not stacked.
        self.frames['gt']=deque(maxlen=1)

class CriticRoutingEncoder(Encoder):
    def __init__(self,*args,critic_mode,**kwargs):
        super().__init__(*args,**kwargs)
        assert critic_mode in ('masked','privileged')
        self.critic_mode=critic_mode
        self.actor_latent_dim=self.num_outputs
        self.num_outputs+=CRITIC_FEATURE_DIM
    def forward(self,obs_dict,detach=False,train=False):
        obs=obs_dict.get('policy',obs_dict)
        actor={k:obs[k] for k in ('prop','tactile')}
        z=super().forward({'policy':actor},detach=detach,train=train)
        gt=obs['gt'][:].detach()
        if gt.shape[-1]!=CRITIC_FEATURE_DIM:raise ValueError('Critic shape drift')
        if self.critic_mode=='masked':gt=torch.zeros_like(gt)
        return torch.cat((z,gt),-1)

class ActorInputOnlyPolicy(GaussianPolicy):
    def __init__(self,*args,actor_latent_dim,**kwargs):
        self.actor_latent_dim=actor_latent_dim
        super().__init__(*args,**kwargs)
    def act(self,z,taken_actions=None,deterministic=False):
        return super().act(z[...,:self.actor_latent_dim],taken_actions=taken_actions,deterministic=deterministic)

def load_source(agent,path):
    """Identical warm start in both arms; only new critic columns start at zero.

Value Adam state is reset in BOTH arms because its parameter shape changed.
Actor encoder/policy weights, normalizers and their optimizer states are kept.
"""
    source=torch.load(path,map_location=agent.device,weights_only=True)
    for name in ('encoder','policy'):
        getattr(agent,name).load_state_dict(source[name],strict=True)
    if 'state_preprocessor' in source:
        agent.encoder.state_preprocessor.load_state_dict(source['state_preprocessor'])
    agent._value_preprocessor.load_state_dict(source['value_preprocessor'])
    target=agent.value.state_dict();expanded=[]
    for key,src in source['value'].items():
        dst=target[key]
        if dst.shape==src.shape:dst.copy_(src)
        elif dst.ndim==2 and dst.shape[0]==src.shape[0] and dst.shape[1]==src.shape[1]+CRITIC_FEATURE_DIM:
            dst.zero_();dst[:,:src.shape[1]].copy_(src);expanded.append(key)
        else:raise ValueError(f'Unexpected critic migration shape: {key} {src.shape} -> {dst.shape}')
    if set(target)!=set(source['value']) or len(expanded)!=1:raise ValueError('Expected exactly one extended critic input matrix')
    agent.value.load_state_dict(target,strict=True)
    for name in ('encoder_optimiser','policy_optimiser'):
        getattr(agent,name).load_state_dict(source[name])
    agent.value_optimiser.state.clear()
    return {'source':str(path),'actor_weights_and_optimizer_preserved':True,
            'value_new_input_columns_zero':True,'value_optimizer_reset_in_both_arms':True,
            'expanded_parameter':expanded[0],'critic_features':CRITIC_FEATURE_DIM,
            'ppo_implementation_unchanged':True}
