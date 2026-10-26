"""Physical periodic trajectory search, separate from network/PPO task rewards.

Signed quaternion heading increments measure NET rotation per complete cycle.
First-cycle settling cannot pass the late-cycle gate. Search is not a policy.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import ACTIVE_JOINTS
from .reward_geometry import pose_heading, heading_delta
from .calibrate_redundancy import step_physics, save

JOINT_NAMES = ['thumb_joint0','thumb_joint1','thumb_joint2','thumb_joint3',
               'index_joint0','index_joint1','index_joint2',
               'middle_joint0','middle_joint1','middle_joint2']
PER_STATE=128
PHASE_STEPS=30
CYCLE_STEPS=4*PHASE_STEPS


@torch.no_grad()
def run_cycles(env, waypoints, cycles=3, capture=False, phase_steps=PHASE_STEPS, target_slew=.015,
               joint_names=None, late_window=None):
    raw=env.unwrapped;env.reset();base=raw._q_target.clone()
    ids=[ACTIVE_JOINTS.index(j) for j in (joint_names or JOINT_NAMES)]
    assert waypoints.shape==(raw.num_envs,4,len(ids))
    alive=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device)
    stable=alive.clone();valid=alive.clone()
    peak=torch.zeros(raw.num_envs,device=raw.device)
    angle=peak.clone();pad_count=peak.clone();stable_count=peak.clone()
    angle_rows=[];pad_rows=[];stable_rows=[];joint_rows=[]
    previous=None
    records={k:[] for k in ('pose','joint','target','pad','stable','failed','peak_pressure',
                           'whole_finger_force','displacement','heading_delta_rad')} if capture else None
    cycle_steps=4*phase_steps
    for t in range(40+cycle_steps*cycles):
        target=base.clone()
        if t>=40:target[:,ids]=waypoints[:,((t-40)//phase_steps)%4]
        target=target.clamp(raw._active_limits[...,0],raw._active_limits[...,1])
        target=raw._q_target+(target-raw._q_target).clamp(-target_slew,target_slew)
        d=step_physics(raw,target);alive &= ~d['failed']
        if t>=20:stable &= d['stable'] & alive
        peak=torch.maximum(peak,d['peak_pressure'])
        heading=pose_heading(raw._wxyz(d['pose'][:,3:]))
        delta=torch.zeros_like(heading) if previous is None else heading_delta(previous,heading)
        previous=heading
        if t>=40:
            valid &= torch.isfinite(delta)&(delta.abs()<.5)&(d['tilt']<.5)
            angle+=torch.nan_to_num(delta)
            pad_count+=d['pad'].any(-1).float()
            stable_count+=(d['stable']&alive).float()
            if (t-40+1)%cycle_steps==0:
                angle_rows.append(angle.clone());pad_rows.append(pad_count/cycle_steps)
                stable_rows.append(stable_count/cycle_steps)
                joint_rows.append(raw._robot.data.joint_pos.torch[:,raw._active_ids].clone())
                angle.zero_();pad_count.zero_();stable_count.zero_()
        if capture:
            d['joint']=raw._robot.data.joint_pos.torch[:,raw._active_ids]
            d['target']=raw._q_target;d['heading_delta_rad']=delta
            for k in records:records[k].append(d[k].cpu().numpy().copy())
    angles=torch.stack(angle_rows,1);pads=torch.stack(pad_rows,1)
    safe=alive & stable & valid & (peak<=raw.cfg.maximum_pressure_proxy_pa)
    # A candidate screen, not the unchanged final PPO success definition.
    gate_slice=slice(-late_window,None) if late_window is not None else slice(1,None)
    passed=safe & (angles[:,gate_slice]>=torch.pi/9).all(1) & (pads[:,gate_slice]>=.25).all(1)
    return dict(passed=passed,safe=safe,alive=alive,stable=stable,valid_pose=valid,
        peak_pressure=peak,cycle_net_rad=angles,cycle_any_pad_fraction=pads,
        cycle_stable_fraction=torch.stack(stable_rows,1),cycle_end_joint=torch.stack(joint_rows,1)),records


@torch.no_grad()
def search_cycles(env,output,rounds=12,particles=PER_STATE,cycles_per_trial=3,all_joints=False,warm_start=None,
                  contact_feasible_first=False):
    raw=env.unwrapped;output=Path(output);assert not output.exists()
    rows=list(raw._reset_grasps);n=raw.num_envs;assert n%particles==0 and len(rows)==n
    assert raw.cfg.domain_profile=='nominal' and raw._diagnostic_finger_sensors
    names=JOINT_NAMES+[j for j in ACTIVE_JOINTS if j not in JOINT_NAMES] if all_joints else JOINT_NAMES
    late_window=3 if cycles_per_trial>=8 else 2
    states=n//particles;ids=[ACTIVE_JOINTS.index(j) for j in names]
    elite_count=max(8,particles//16);broad_count=max(8,particles//16)
    for i,r in enumerate(rows):assert r['cycle_source_id']==rows[i-i%particles]['cycle_source_id']
    base=torch.tensor([rows[i*particles].get('joint_targets',rows[i*particles]['joint_positions']) for i in range(states)],device=raw.device)
    limits=raw._active_limits[::particles]
    lo=limits[:,ids,0];hi=limits[:,ids,1]
    mean=base[:,None,ids].expand(-1,4,-1).clone()
    # Seed a genuine release/recovery loop. CEM may change every selected joint.
    mean[:,0,3]-=.25
    mean[:,1,3]-=.25;mean[:,1,2]+=.2;mean[:,1,1]+=.1
    mean[:,2,2]+=.2;mean[:,2,1]+=.1
    if warm_start:
        prior=json.loads(Path(warm_start).read_text());lookup={w['state_id']:w for w in prior['winners']}
        previous_names=prior.get('joint_order',JOINT_NAMES)
        for si in range(states):
            w=lookup[rows[si*particles]['cycle_source_id']]
            for j,name in enumerate(previous_names):mean[si,:,names.index(name)]=torch.tensor(w['waypoints'],device=raw.device)[:,j]
    mean=mean.clamp(lo[:,None],hi[:,None])
    sigma=torch.full_like(mean,.14);sigma[:,:,:4]=.22
    best=mean.clone();best_score=torch.full((states,),-1e9,device=raw.device)
    archive=[];started=time.monotonic()
    save(output.with_name(output.stem+'_protocol.json'),dict(states=states,candidates_per_state=particles,
        rounds=rounds,cycles_per_trial=cycles_per_trial,baseline_seconds=40*raw.step_dt,
        seconds_per_phase=PHASE_STEPS*raw.step_dt,cycle_seconds=CYCLE_STEPS*raw.step_dt,
        joint_order=names,other_joints='None held fixed' if all_joints else 'Original targets held through unchanged actuator path',
        max_target_increment_rad=.015,search=f'CEM, {elite_count} elites/state; {broad_count} broad candidates per round',
        candidate_gate=f'All strict physical stability after1s, no termination, peak<=18kPa, no pose jumps; last{late_window} cycles net>=20deg each, effective-pad presence>=25% in each late cycle',
        warm_start=str(warm_start) if warm_start else None,
        contact_feasible_first=contact_feasible_first,
        final_task_gate_unchanged=True,no_ppo=True,no_core_changes=True,
        initial_velocities_zero=True,not_a_learned_controller=True))
    for iteration in range(rounds):
        x=mean[:,None]+torch.randn(states,particles,4,len(ids),device=raw.device)*sigma[:,None]
        x[:,:broad_count]=base[:,None,None,ids]+torch.randn(states,broad_count,4,len(ids),device=raw.device)*.35
        x[:,broad_count]=best;x[:,broad_count+1]=mean;x[:,broad_count+2]=base[:,None,ids].expand(-1,4,-1)
        x=x.clamp(lo[:,None,None],hi[:,None,None])
        d,_=run_cycles(env,x.reshape(n,4,len(ids)),cycles=cycles_per_trial,joint_names=names,late_window=late_window)
        late=d['cycle_net_rad'][:,-late_window:]
        score=(30*late.amin(1).clamp(-torch.pi,torch.pi)+10*late.mean(1).clamp(-torch.pi,torch.pi)
            +10*d['cycle_stable_fraction'].mean(1)+2*d['cycle_any_pad_fraction'][:,-late_window:].mean(1)
            -150*(~d['alive']).float()-100*(~d['stable']).float()-75*(~d['valid_pose']).float()
            +50*d['safe'].float()+100*d['passed'].float())
        contact_feasible=d['safe'] & (d['cycle_any_pad_fraction'][:,-late_window:]>=.25).all(1)
        if contact_feasible_first:
            # Bounded within-class scores span <1000: feasible trajectories are
            # always ranked before ones violating the existing contact gate.
            # No task/sensor threshold or PPO reward is changed.
            assert float(score.max()-score.min())<1000
            score=score+1000*contact_feasible.float()
        assert torch.isfinite(score).all()
        score=score.reshape(states,particles);vals,elite_ids=score.topk(elite_count,dim=1)
        if contact_feasible_first:
            feasible_rows=contact_feasible.reshape(states,particles)
            for si in range(states):
                feasible_ids=feasible_rows[si].nonzero().flatten()
                if feasible_ids.numel():
                    ranked=feasible_ids[score[si,feasible_ids].argsort(descending=True)]
                    # If fewer than the elite count are feasible, repeat them;
                    # do not dilute their proposal mean with off-pad paths.
                    chosen=ranked[:elite_count]
                    elite_ids[si]=chosen.repeat((elite_count+len(chosen)-1)//len(chosen))[:elite_count]
                    vals[si]=score[si,elite_ids[si]]
        better=vals[:,0]>best_score
        best[better]=x[torch.arange(states,device=raw.device),elite_ids[:,0]][better]
        best_score=torch.maximum(best_score,vals[:,0])
        elite=torch.gather(x,1,elite_ids[:,:,None,None].expand(-1,-1,4,len(ids)))
        mean=.2*mean+.8*elite.mean(1);sigma=(.2*sigma+.8*elite.std(1)).clamp_min(.025)
        arrays={k:v.cpu().numpy() for k,v in d.items()}
        np.savez_compressed(output.with_name(f'{output.stem}_round{iteration:02d}.npz'),waypoints=x.cpu().numpy(),score=score.cpu().numpy(),**arrays)
        for si in range(states):
            ci=int(elite_ids[si,0]);i=si*particles+ci
            archive.append(dict(state_id=rows[i]['cycle_source_id'],round=iteration,environment=i,
                waypoints=x[si,ci].cpu().tolist(),score=float(score[si,ci]),
                **{k:v[i].tolist() for k,v in arrays.items()}))
        winners=[]
        for si in range(states):
            sid=rows[si*particles]['cycle_source_id']
            w=max((r for r in archive if r['state_id']==sid),key=lambda r:r['score'])
            winners.append(dict(source_state=rows[si*particles],**w))
        save(output,dict(search_completed=iteration+1==rounds,rounds_completed=iteration+1,
            total_trials=(iteration+1)*n,states=states,observed_passes=sum(w['passed'] for w in winners),
            joint_order=names,cycles_per_trial=cycles_per_trial,late_gate_cycles=late_window,
            contact_feasible_first=contact_feasible_first,
            elapsed_seconds=time.monotonic()-started,winners=winners,archive=archive,
            long_replay_complete=False,goal_complete=False))
        safe_candidates=[r for r in archive if r['safe']]
        best_late=max((min(r['cycle_net_rad'][-late_window:])*180/np.pi for r in safe_candidates),default=None)
        print(f'Cycle search {iteration+1}/{rounds}: {sum(w["passed"] for w in winners)}/{states} candidates; best safe minimum late-cycle angle {best_late}deg',flush=True)


@torch.no_grad()
def validate_cycles(env,output,source,cycles=8,time_scale=1.):
    raw=env.unwrapped;output=Path(output);assert not output.exists() and cycles>=3
    assert time_scale in (1.,2.)
    source_data=json.loads(Path(source).read_text());source_rows=source_data['winners']
    selected={w['state_id']:w for w in source_rows}
    rows=list(raw._reset_grasps)
    assert len(rows)==raw.num_envs and all(r['cycle_source_id'] in selected for r in rows)
    waypoints=torch.tensor([selected[r['cycle_source_id']]['waypoints'] for r in rows],device=raw.device)
    phase_steps=round(PHASE_STEPS/time_scale)
    slew=min(.015*time_scale,raw.cfg.maximum_joint_target_delta_rad)
    d,records=run_cycles(env,waypoints,cycles=cycles,capture=True,phase_steps=phase_steps,target_slew=slew,
        joint_names=source_data.get('joint_order',JOINT_NAMES),late_window=source_data.get('late_gate_cycles'))
    arrays={k:np.asarray(v) for k,v in records.items()}
    np.savez_compressed(output.with_suffix('.npz'),**arrays,waypoints=waypoints.cpu().numpy())
    result=[]
    for i,r in enumerate(rows):
        row={k:v[i].cpu().tolist() for k,v in d.items()}
        row.update(state_id=r['cycle_source_id'],after10_net_degrees=float(arrays['heading_delta_rad'][200:,i].sum()*180/np.pi),
            net_degrees_per_cycle=(d['cycle_net_rad'][i]*180/torch.pi).cpu().tolist())
        result.append(row)
    save(output,dict(completed=True,cycles=cycles,total=raw.num_envs,results=result,
        time_scale=time_scale,phase_steps=phase_steps,cycle_seconds=4*phase_steps*raw.step_dt,
        target_slew_rad_per_step=slew,actuator_limits_unchanged=True,
        candidate_passes=sum(r['passed'] for r in result),safe=sum(r['safe'] for r in result),
        goal_complete=False,learned_policy=False,domain_profile=raw.cfg.domain_profile,
        density_range=list(raw.cfg.events.geometry.params['density_range_kg_m3'])))
    print(f'Long cycle replay: {sum(r["passed"] for r in result)}/{len(result)} candidates retained',flush=True)
