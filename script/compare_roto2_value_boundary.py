"""Compare the diagnostic horizon contrast without changing success criteria."""
import json
from pathlib import Path
import torch
from script.summarize_roto2_value_gradient import summarize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v16'


def main():
    records={}
    initial=[]
    for name in ['value_gradient','horizon48']:
        p=OUT/name
        plan=json.loads((p/'PLAN.json').read_text())
        result=json.loads((p/'RESULT.json').read_text())
        rows=[json.loads(l) for l in (p/'observed/minibatches.jsonl').read_text().splitlines()]
        records[name]={'episode_seconds':plan['episode_seconds'],
                       'windows':{f'{lo}-{hi}':summarize([r for r in rows if lo<=r['update']<=hi])
                                  for lo,hi in [(1,8),(9,29),(30,34),(35,59),(60,64)]},
                       'per_update':result['per_update']}
        candidates=[]
        for run in (ROOT/'logs/roto2_pressure_fruit/plain_seed9811').iterdir():
            config=run/'run_config.json'
            if config.exists():
                c=json.loads(config.read_text())
                if c['episode_length_seconds']==plan['episode_seconds'] and c['million_steps']==8.372225:
                    candidates.append(run)
        assert len(candidates)==1, candidates
        run=candidates[0]
        done=json.loads((run/'training_result.json').read_text())
        assert done['training_completed'] and done['training_transitions']==8372224 and done['finite_parameters']
        initial.append(torch.load(run/'checkpoints/initial_agent.pt',map_location='cpu',weights_only=True))
        records[name]['run']=str(run)
        records[name]['training_result']=done
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    assert equal(*initial), 'Initial model/scaler/optimizer state mismatch'
    rollout_path=OUT/'horizon48/observed/rollouts.jsonl'
    rollouts=[json.loads(l) for l in rollout_path.read_text().splitlines()]
    records['initial_model_optimizer_scaler_bitwise_equal']=True
    records['horizon48_timeout_updates']=[r for r in rollouts if r['truncated']>0]
    (OUT/'BOUNDARY_COMPARISON.json').write_text(json.dumps(records,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for name,color,label in [('value_gradient','#d97706','24 s horizon'),('horizon48','#2563eb','48 s horizon')]:
        rr=records[name]['per_update'];xx=[r['update'] for r in rr]
        axes[0].plot(xx,[100*r['clipped_fraction'] for r in rr],color=color,label=label,lw=1.8)
        axes[1].plot(xx,[r['unclipped_mse'] for r in rr],color=color,label=label,lw=1.8)
    for ax in axes:
        ax.axvline(30,color='#d97706',ls=':',alpha=.7)
        ax.axvline(60,color='#777777',ls=':',alpha=.7)
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Clipped value predictions (%)')
    axes[0].legend(frameon=False)
    axes[1].set_ylabel('Value MSE before clipping')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('PPO update (16 control steps per update)')
    axes[0].set_title('Value-learning shocks near episode boundaries\nSame initial model / optimizer / seed; one diagnostic pair',loc='left')
    fig.savefig(OUT/'boundary_comparison.png',dpi=180)
    plt.close(fig)
    print(json.dumps({k:{'windows':v['windows']} for k,v in records.items() if k in ['value_gradient','horizon48']},indent=2))
    print(json.dumps({'horizon48_timeout_updates':records['horizon48_timeout_updates'],
                      'initial_model_optimizer_scaler_bitwise_equal':True},indent=2))


if __name__=='__main__':main()
