"""Exploratory isolation of clock from reward history on preserved V15 episodes."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from script import audit_roto2_critic_information as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v17/clock_information'
ARMS=['actor_only','actor_clock','actor_reward_history','actor_clock_and_history']
SEEDS=[1701,1702,1703]
MODE='clock'


class ClockProbes(base.MatchedProbes):
    def __init__(self,dim):
        super().__init__(dim)
        self.mask.fill_(1)
        self.mask[:,:,-4:]=0
        if MODE=='clock':
            self.mask[1,:,-4]=1
            self.mask[2,:,-3:]=1
            self.mask[3,:,-4:]=1
        else:
            self.mask[1,:,-3]=1
            self.mask[2,:,-2:]=1
            self.mask[3,:,-3:]=1


def main():
    assert (OUT/'PLAN.json').exists() and not (OUT/'RESULT.json').exists()
    base.OUT=OUT
    data=base.load_data()
    train=data['split']==0;val=data['split']==1;test=data['split']==2
    ym=data['y'][train,:2].mean(0);ys=data['y'][train,:2].std(0).clip(.001)
    y=data['y'].copy();y[:,:2]=(y[:,:2]-ym)/ys
    weight=float(np.clip((1-y[train,2].mean())/max(y[train,2].mean(),1e-6),1,20))
    Y=torch.as_tensor(y,device='cuda');ids=torch.as_tensor(np.flatnonzero(train),device='cuda')
    result=dict(completed=False,policy_training=False,exploratory_reuse_of_v15_heldout_episodes=True,
                arms=ARMS,mode=MODE,fit_seeds=SEEDS,rows=[],plan=str(OUT/'PLAN.json'))
    predictions={}
    for family in ['latent','raw']:
        features=np.concatenate([data[family],data['critic'][:,-4:]],-1)
        mean=features[train].mean(0);std=features[train].std(0).clip(.001)
        X=torch.as_tensor(((features-mean)/std).clip(-10,10),device='cuda');del features
        for seed in SEEDS:
            torch.manual_seed(seed);model=ClockProbes(X.shape[-1]).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
            best=np.full(4,np.inf);snapshot=None;beststep=[0]*4
            for step in range(1,1201):
                idx=ids[torch.randint(len(ids),(512,),device='cuda')]
                loss=base.loss_by_arm(model(X[idx]),Y[idx],torch.tensor(weight,device='cuda')).sum()
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
                if step%100==0:
                    vp=base.predict(model,X[val]);vy=y[val]
                    reg=((vp[:,:,:2]-vy[None,:,:2])**2).mean((1,2))
                    logits=vp[:,:,2]
                    bce=((1-vy[None,:,2])*np.logaddexp(0,logits)+weight*vy[None,:,2]*np.logaddexp(0,-logits)).mean(1)
                    losses=reg+bce
                    if snapshot is None:snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    for arm in range(4):
                        if losses[arm]<best[arm]:
                            best[arm]=losses[arm];beststep[arm]=step
                            for k,v in model.state_dict().items():snapshot[k][arm]=v[arm].detach().cpu()
                    print(json.dumps(dict(family=family,seed=seed,step=step,val_loss=losses.tolist())),flush=True)
            model.load_state_dict(snapshot)
            pred=base.predict(model,X[test]);pred[:,:,:2]=pred[:,:,:2]*ys+ym
            pred[:,:,2]=1/(1+np.exp(-np.clip(pred[:,:,2],-40,40)))
            predictions[f'{family}_{seed}']=pred
            for arm,name in enumerate(ARMS):
                mse=((pred[arm,:,:2]-data['y'][test,:2])**2).mean(0)
                result['rows'].append(dict(family=family,seed=seed,arm=name,best_step=beststep[arm],
                                          return_mse=float(mse[0]),rotation5s_mse=float(mse[1]),
                                          failure_brier=float(((pred[arm,:,2]-data['y'][test,2])**2).mean())))
            (OUT/'PROGRESS.json').write_text(json.dumps(result,indent=2)+'\n')
        del X
    result['comparisons']={}
    for family in ['latent','raw']:
        pp=np.stack([predictions[f'{family}_{s}'] for s in SEEDS])
        errors=(pp[:,:,:,0]-data['y'][test,0][None,None])**2
        episodes=data['episode'][test]
        ee=np.stack([errors[:,:,episodes==e].mean((0,2)) for e in np.unique(episodes)])
        rng=np.random.default_rng(17123);draw=rng.integers(len(ee),size=(2000,len(ee)));boot=ee[draw].mean(1)
        comparisons={}
        for a,b,name in [(1,0,'focal_state_vs_actor'),(2,0,'other_state_vs_actor'),(3,1,'other_state_increment_over_focal')]:
            gain=1-errors[:,a].mean(1)/errors[:,b].mean(1)
            interval=np.quantile(1-boot[:,a]/boot[:,b],[.025,.975])
            comparisons[name]=dict(gain_per_fit_seed=gain.tolist(),median_gain=float(np.median(gain)),
                                   episode_bootstrap_95pct=interval.tolist())
        c=comparisons['focal_state_vs_actor']
        comparisons['bounded_pair_supported']=bool(c['median_gain']>=.15 and min(c['gain_per_fit_seed'])>=-.05 and c['episode_bootstrap_95pct'][0]>0)
        result['comparisons'][family]=comparisons
    result['completed']=True
    result['bounded_pair_supported']=all(v['bounded_pair_supported'] for v in result['comparisons'].values())
    result['limitations']=['Adaptive exploratory reuse of V15 episode split, not independent confirmation.',
                          'One deterministic policy and one ancestral grasp family.',
                          'Predictive benefit is not PPO improvement or sustained rotation.',
                          'Temporal boundary mechanism requires separate online validation.']
    np.savez_compressed(OUT/'PREDICTIONS.npz',**predictions,targets=data['y'][test],episode=data['episode'][test])
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--probe',choices=['clock','reward_path'],default='clock')
    MODE=parser.parse_args().probe
    if MODE=='reward_path':
        OUT=ROOT/'artifacts/simulation/rotation_v17/reward_path_information'
        ARMS=['actor_only','actor_reward_path_valid','actor_streak_and_upright','actor_all_history']
    main()
