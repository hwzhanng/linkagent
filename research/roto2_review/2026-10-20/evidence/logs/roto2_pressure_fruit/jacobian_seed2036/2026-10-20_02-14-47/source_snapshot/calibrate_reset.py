"""Bounded real-physics search for a reset grasp; never a policy controller.

Candidates are saved separately. This module never overwrites the task's reset
file, changes architecture/physics, or reports a grasp as learned rotation.
"""
import json
from pathlib import Path
import torch

@torch.no_grad()
def search_pad_resets(env, output, rounds=24, horizon=48):
    raw=env.unwrapped;device=raw.device;n=raw.num_envs
    if n%8:raise ValueError('Use a multiple of 8 environments')
    root=Path(__file__).resolve().parents[2];base=root/'simulation/.runtime/setup/reward_v2'
    seeds=json.loads((root/'simulation/roto2_pressure/reset_grasps.json').read_text())
    for filename in ['antipodal_pad_candidates.json','surface_pad_candidates.json','pad_grasp_separated_candidates.json']:
        seeds.extend(x for x in json.loads((base/filename).read_text()) if x['cost']<1)
    seeds=seeds[:8]
    while len(seeds)<8:seeds.append(seeds[-1])
    means=torch.tensor([x['joint_positions']+x['fruit_position_hand'] for x in seeds],device=device)
    sigma=torch.full_like(means,.07);sigma[:,16:]=.004
    floor=sigma*.22;per=n//8;elite=max(8,per//8)
    bounds=raw._active_limits[0]
    lo=torch.cat((bounds[:,0]+.001,torch.tensor([.01,-.04,.12],device=device)))
    hi=torch.cat((bounds[:,1]-.001,torch.tensor([.12,.04,.235],device=device)))
    zero=torch.zeros_like(raw._action);archive=[];output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for iteration in range(rounds):
        x=means[:,None]+torch.randn(8,per,19,device=device)*sigma[:,None]
        x[:,0]=means;x=x.clamp(lo,hi);flat=x.reshape(n,19)
        rows=flat.cpu().tolist()
        raw._reset_grasps=[{'joint_positions':r[:16],'fruit_position_hand':r[16:]} for r in rows]
        env.reset();pending=torch.ones(n,device=device,dtype=torch.bool)
        full=torch.zeros(n,device=device);coverage=full.clone();age=full.clone();failed=pending.clone().fill_(False);pad_sum=torch.zeros(n,5,device=device)
        for t in range(horizon):
            _,_,term,trunc,info=env.step(zero)
            p=info['step']['pad_contact'].float();active=pending.float()
            complete=info['step']['secure_support'].float()
            full+=active*complete;pad_sum+=active[:,None]*p
            coverage+=active*(4*p[:,0]+2*p[:,1:].sum(-1).clamp(max=2))
            age+=active;done=term|trunc;failed|=pending&term;pending&=~done
        score=30*full/horizon+coverage/horizon+2*age/horizon-10*failed.float()
        score=score.reshape(8,per);ids=score.topk(elite,dim=1).indices
        selected=torch.gather(x,1,ids[...,None].expand(-1,-1,19))
        means=.25*means+.75*selected.mean(1)
        sigma=torch.maximum(floor,.3*sigma+.7*selected.std(1))
        best=score.reshape(-1).topk(8).indices
        for i in best.cpu().tolist():
            archive.append({'id':f'physx_pad_cem_{iteration}_{i}',
                'joint_positions':rows[i][:16],'fruit_position_hand':rows[i][16:],
                'score':float(score.reshape(-1)[i]),'pad_support_fraction':float(full[i]/horizon),
                'per_finger_pad_fraction':(pad_sum[i]/horizon).cpu().tolist(),
                'failed':bool(failed[i]),'steps_survived':int(age[i]),'nominal_sphere_only':True})
        archive.sort(key=lambda r:r['score'],reverse=True);archive=archive[:32]
        output.write_text(json.dumps(archive,indent=2)+'\n')
        print(json.dumps({'round':iteration+1,'best':archive[0],'max_support_this_round':float(full.max()/horizon)}),flush=True)
    return archive
