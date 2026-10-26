"""Settle and physically screen intermediate rotation states for recovery curriculum."""
from pathlib import Path
import numpy as np
import torch
from .core import quat_inv_rotate
from .calibrate_redundancy import step_physics, save


@torch.no_grad()
def validate_recovery_states(env,output,steps=120):
    raw=env.unwrapped;output=Path(output);candidates=list(raw._reset_grasps)
    if len(candidates)!=raw.num_envs:raise ValueError('Exactly one proposed state per environment required')
    env.reset();target=raw._q_target.clone()
    alive=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
    traces={k:[] for k in ('pose','pad','stable','failed','peak_pressure','whole_finger_force','displacement')}
    for t in range(steps):
        d=step_physics(raw,target);alive&=~d['failed'];d['stable']&=alive
        for k in traces:traces[k].append(d[k].cpu().numpy().copy())
    data={k:np.asarray(v) for k,v in traces.items()}
    np.savez_compressed(output.with_suffix('.npz'),**data)
    root=raw._robot.data.root_pose_w.torch
    local=quat_inv_rotate(raw._wxyz(root[:,3:]),d['pose'][:,:3]-root[:,:3]).cpu().numpy()
    joint=raw._robot.data.joint_pos.torch[:,raw._active_ids].cpu().numpy()
    target=raw._q_target.cpu().numpy()
    quat=(d['pose'][:,3:] if raw._native_xyzw else d['pose'][:,[4,5,6,3]]).cpu().numpy()
    results=[];accepted=[]
    for i,row in enumerate(candidates):
        safe=bool(data['stable'][20:,i].all() and not data['failed'][:,i].any())
        pad_fraction=(data['pad'][20:,i].sum(-1)>=1).mean()
        passed=bool(safe and pad_fraction>=.8 and data['pad'][-1,i].any())
        results.append(dict(id=row['id'],split=row['split'],source_seconds=row['source_seconds'],
            passed=passed,stable=safe,any_effective_pad_fraction=float(pad_fraction),
            peak_pressure_pa=float(data['peak_pressure'][:,i].max())))
        if passed:
            accepted.append({**row,'source_proposed_joint_positions':row['joint_positions'],
                'joint_positions':joint[i].tolist(),'joint_targets':target[i].tolist(),
                'fruit_position_hand':local[i].tolist(),'fruit_quaternion_xyzw':quat[i].tolist(),
                'physical_validation':True,'proposal_only':False,
                'validation_scope':'Six-second safe settling with >=1 effective pad, not an N-1 anchor or learned gait',
                'validation_file':str(output),'reset_schema':'settled_recovery_curriculum_v1'})
    save(output,dict(validation_completed=True,steps=steps,seconds=steps*raw.step_dt,
         proposed_count=len(candidates),accepted_count=len(accepted),results=results,
         gate='No physical failure; original stable-object conditions after settling; >=1 pad80% of measured steps',
         n_minus_one_anchor_claim=False))
    save(output.with_name(output.stem+'_accepted.json'),accepted)
    print(f'Recovery states accepted {len(accepted)}/{len(candidates)}',flush=True)
