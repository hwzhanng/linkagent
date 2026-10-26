"""Matched-capacity causal prediction probes with whole-episode holdout.

Never modifies or exports a policy. Evaluation labels are future outcomes;
all model inputs are stored BEFORE the associated action.
"""
from pathlib import Path
import json,time,copy,hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v15/critic_information'
ARMS=['actor_only','actor_bookkeeping','actor_physics','actor_all']
SEEDS=[1501,1502,1503]
STEPS=1200
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

class MatchedProbes(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.weights=nn.ParameterList();self.biases=nn.ParameterList()
        for a,b in zip([dim,128,64],[128,64,3]):
            layer=nn.Linear(a,b)
            self.weights.append(nn.Parameter(layer.weight.detach().T[None].repeat(4,1,1)))
            self.biases.append(nn.Parameter(layer.bias.detach()[None,None].repeat(4,1,1)))
        mask=torch.ones((4,1,dim))
        mask[0,:,-64:]=0
        mask[1,:,-64:-4]=0
        mask[2,:,-4:]=0
        self.register_buffer('mask',mask)
    def forward(self,x):
        z=x[None]*self.mask
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            z=torch.bmm(z,w)+b
            if i<2:z=F.elu(z)
        return z

def load_data():
    acc={k:[] for k in ['raw','latent','critic','y','episode','split','old_value','time_s']}
    split_rows=[]
    plan=json.loads((OUT/'COLLECTION_PLAN.json').read_text())
    for run,row in enumerate(plan['commands']):
        p=Path(row['output']);meta=json.loads((p/'COMPLETE.json').read_text())
        T=meta['steps'];N=meta['environments'];dt=meta['dt'];h=round(5/dt)
        get=lambda k:np.load(p/f'{k}.npy',mmap_mode='r')[:T]
        active=np.asarray(get('active'));term=np.asarray(get('terminated'))&active
        r=np.asarray(get('reward'))*active
        returns=np.zeros_like(r);future=np.zeros(N,dtype=np.float32)
        for t in range(T-1,-1,-1):
            future=(r[t]+.99*future)*active[t];returns[t]=future
        delta=np.asarray(get('pose_delta')).copy()
        delta[(~active)|(np.abs(delta)>=.5)]=0
        prefix=np.concatenate([np.zeros((1,N)),np.cumsum(delta,axis=0)])
        failure_prefix=np.concatenate([np.zeros((1,N)),np.cumsum(term,axis=0)])
        ts=np.arange(round(1/dt),T-h+1,4)
        mask=active[ts]
        rotation=prefix[ts+h]-prefix[ts]
        failure=(failure_prefix[ts+h]-failure_prefix[ts]>0).astype(np.float32)
        y=np.stack([returns[ts],rotation,failure],-1)[mask]
        env=np.broadcast_to(np.arange(N),(len(ts),N))[mask]
        # Fixed episode assignment selected before seeing probe results.
        order=np.random.default_rng(15000+run).permutation(N)
        split=np.zeros(N,dtype=np.int64);split[order[32:48]]=1;split[order[48:]]=2
        for e in range(N):split_rows.append(dict(run=run,seed=row['seed'],env=e,episode=run*N+e,split=['train','validation','test'][split[e]],failed=bool(term[:,e].any())))
        acc['raw'].append(np.concatenate([get('prop')[ts][mask],get('tactile')[ts][mask]],-1))
        for k in ['latent','critic']:acc[k].append(np.asarray(get(k)[ts][mask]))
        acc['y'].append(y.astype(np.float32));acc['episode'].append(run*N+env)
        acc['split'].append(split[env]);acc['old_value'].append(get('value')[ts][mask])
        acc['time_s'].append(np.broadcast_to(ts[:,None]*dt,mask.shape)[mask])
    d={k:np.concatenate(v) for k,v in acc.items()}
    (OUT/'EPISODE_SPLIT.json').write_text(json.dumps(split_rows,indent=2)+'\n')
    assert all(not set(d['episode'][d['split']==a])&set(d['episode'][d['split']==b]) for a,b in [(0,1),(0,2),(1,2)])
    assert all(np.isfinite(v).all() for v in d.values())
    return d

def loss_by_arm(pred,y,pos_weight):
    regression=(pred[:,:,:2]-y[None,:,:2]).square().mean((1,2))
    binary=F.binary_cross_entropy_with_logits(pred[:,:,2],y[None,:,2].expand(4,-1),pos_weight=pos_weight,reduction='none').mean(1)
    return regression+binary

@torch.no_grad()
def predict(model,x):
    return torch.cat([model(x[i:i+2048]).cpu() for i in range(0,len(x),2048)],1).numpy()

def main():
    d=load_data();tr=d['split']==0;va=d['split']==1;te=d['split']==2
    scale=d['y'][tr,:2].std(0).clip(.001);ym=d['y'][tr,:2].mean(0)
    y=d['y'].copy();y[:,:2]=(y[:,:2]-ym)/scale
    pos_weight=float(np.clip((1-y[tr,2].mean())/max(y[tr,2].mean(),1e-6),1,20))
    Y=torch.as_tensor(y,device='cuda');train_idx=torch.as_tensor(np.flatnonzero(tr),device='cuda')
    result={'completed':False,'policy_training':False,'arms':ARMS,'fit_seeds':SEEDS,'steps_per_seed':STEPS,
       'target_names':['discounted_remaining_return_gamma0.99','next5s_net_radians','next5s_physical_failure'],
       'target_train_mean':ym.tolist(),'target_train_std':scale.tolist(),'failure_pos_weight':pos_weight,
       'split_counts':{n:{'frames':int(m.sum()),'episodes':len(np.unique(d['episode'][m])),'positive_failure_frames':int(d['y'][m,2].sum())} for n,m in [('train',tr),('validation',va),('test',te)]},
       'pre_action_features_only':True,'normalization_fit_on_train_only':True,'episode_sets_disjoint':True,
       'rows':[],'diagnostic_gate':'Proceed to bounded PPO pair only with >=15% median heldout return MSE improvement of actor_all over actor_bookkeeping, no seed >5% worse; bootstrap lower bound >0. Secondary outcomes reported, not substituted.'}
    allpred={}
    for family in ['latent','raw']:
        features=np.concatenate([d[family],d['critic']],-1)
        mu=features[tr].mean(0);sd=features[tr].std(0).clip(.001)
        X=torch.as_tensor(((features-mu)/sd).clip(-10,10),device='cuda')
        del features
        for seed in SEEDS:
            torch.manual_seed(seed);model=MatchedProbes(X.shape[-1]).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
            best=np.full(4,np.inf);snap=None;best_step=[0]*4
            for step in range(1,STEPS+1):
                idx=train_idx[torch.randint(len(train_idx),(512,),device='cuda')]
                pred=model(X[idx]);loss=loss_by_arm(pred,Y[idx],torch.tensor(pos_weight,device='cuda')).sum()
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
                if step%100==0:
                    vp=predict(model,X[va]);vy=y[va]
                    reg=((vp[:,:,:2]-vy[None,:,:2])**2).mean((1,2))
                    logits=vp[:,:,2];bce=((1-vy[None,:,2])*np.logaddexp(0,logits)+pos_weight*vy[None,:,2]*np.logaddexp(0,-logits)).mean(1)
                    vl=reg+bce
                    if snap is None:snap={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    for a in range(4):
                        if vl[a]<best[a]:
                            best[a]=vl[a];best_step[a]=step
                            for k,v in model.state_dict().items():snap[k][a]=v[a].detach().cpu()
                    print(json.dumps({'family':family,'seed':seed,'step':step,'val_loss':vl.tolist()}),flush=True)
            model.load_state_dict(snap)
            pred=predict(model,X[te]);pred[:,:,:2]=pred[:,:,:2]*scale+ym
            pred[:,:,2]=1/(1+np.exp(-np.clip(pred[:,:,2],-40,40)))
            allpred[f'{family}_{seed}']=pred
            for a,arm in enumerate(ARMS):
                mse=((pred[a,:,:2]-d['y'][te,:2])**2).mean(0)
                result['rows'].append({'family':family,'seed':seed,'arm':arm,'best_step':best_step[a],
                    'return_mse':float(mse[0]),'rotation5s_mse':float(mse[1]),
                    'failure_brier':float(((pred[a,:,2]-d['y'][te,2])**2).mean()),
                    'return_nmse_train_variance':float(mse[0]/scale[0]**2),'rotation_nmse_train_variance':float(mse[1]/scale[1]**2)})
            (OUT/'PROBE_PROGRESS.json').write_text(json.dumps(result,indent=2)+'\n')
        del X
    np.savez_compressed(OUT/'HELDOUT_PREDICTIONS.npz',**allpred,targets=d['y'][te],episode=d['episode'][te],time_s=d['time_s'][te],old_value=d['old_value'][te])
    summaries={}
    for family in ['latent','raw']:
        predictions=np.asarray([allpred[f'{family}_{s}'] for s in SEEDS])
        errors=(predictions[:,:,:,:2]-d['y'][te][None,None,:,:2])**2
        # Average losses across fit seeds, then bootstrap complete heldout episodes.
        episode_ids=np.unique(d['episode'][te]);ee=np.stack([errors[:,:,d['episode'][te]==e].mean((0,2)) for e in episode_ids])
        # shape episodes,arms,2; paired resampling preserves each arm's evaluation states.
        rng=np.random.default_rng(15123);draw=rng.integers(len(ee),size=(2000,len(ee)))
        boot=ee[draw].mean(1);gain=1-boot[:,3,0]/boot[:,1,0]
        byseed=1-errors[:,3,:,0].mean(1)/errors[:,1,:,0].mean(1)
        summaries[family]={'relative_return_mse_reduction_vs_bookkeeping_per_seed':byseed.tolist(),
            'median_reduction':float(np.median(byseed)),'episode_bootstrap_95pct_gain_interval':np.quantile(gain,[.025,.975]).tolist(),
            'gate_pass':bool(np.median(byseed)>=.15 and (byseed>=-.05).all() and np.quantile(gain,.025)>0)}
    result['summaries']=summaries;result['completed']=True
    result['bounded_ppo_pair_justified']=all(s['gate_pass'] for s in summaries.values())
    result['limitations']=['One fixed mature deterministic policy; does not prove on-policy improvement or physical feasibility.','One ancestral grasp family; whole-episode holdout is not new-family generalization.','Physical features include actuator target lead and reset reference as well as object state.','Bookkeeping-only control isolates time and hidden reward-history information from physical-state gain.','Bootstrap intervals describe this episode sample, not independent training-seed generalization.']
    (OUT/'PROBE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Critic 信息离线对照','',f"256 回合，128/64/64 按整回合划分训练/验证/测试；当前固定策略，非 PPO 成果。",'','|输入表示|输入条件|回报 MSE（3拟合种子均值）|未来5秒转角 MSE|失败 Brier|','|---|---|---:|---:|---:|']
    for family in ['latent','raw']:
        for arm in ARMS:
            rows=[r for r in result['rows'] if r['family']==family and r['arm']==arm]
            lines.append(f"|{family}|{arm}|{np.mean([r['return_mse'] for r in rows]):.4f}|{np.mean([r['rotation5s_mse'] for r in rows]):.5f}|{np.mean([r['failure_brier'] for r in rows]):.5f}|")
    lines+=['','物理真值相对仅增加时钟/奖励历史的控制组：',json.dumps(summaries,ensure_ascii=False,indent=2),'',f"是否支持开展有界 PPO 配对试验：{result['bounded_ppo_pair_justified']}。这不是策略成功判定。"]
    (OUT/'PROBE_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':True,'summaries':summaries,'bounded_ppo_pair_justified':result['bounded_ppo_pair_justified']},indent=2),flush=True)

if __name__=='__main__':main()
