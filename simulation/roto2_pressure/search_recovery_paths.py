"""Physical trajectory search outside PPO; immutable actuators, sensors and limits.

Only thumb target waypoints vary. Each trial starts at the supplied real settled
training state; there is no reset or object manipulation during a trial. CEM
scores are search heuristics, never task rewards or acceptance thresholds.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import ACTIVE_JOINTS, quat_inv_rotate
from .reward_geometry import pad_approach_quality
from .calibrate_redundancy import step_physics, save

PER_STATE = 32
STEPS = 200


@torch.no_grad()
def trial(env, endpoints, cycles=1, capture=False, video=None, video_index=0, demonstrations=False):
    raw = env.unwrapped
    env.reset()
    recorder=None
    if video:
        from .recovery_video import RecoveryVideo
        recorder=RecoveryVideo(raw,video,video_index)
    base = raw._q_target.clone()
    joints = [ACTIVE_JOINTS.index('thumb_joint'+str(j)) for j in (0,1,2,3)]
    j2 = joints[2]
    n = raw.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=raw.device)
    stable_all = alive.clone()
    initial_q2 = torch.zeros(n,device=raw.device)
    initial_pad = initial_q2.clone()
    pad = initial_q2.clone(); reserve = initial_q2.clone(); quality = initial_q2.clone()
    pressure = initial_q2.clone(); stable_count = initial_q2.clone()
    q2_final = initial_q2.clone(); whole_free = initial_q2.clone()
    free_streak = initial_q2.clone(); max_free_streak = initial_q2.clone()
    cycle_stats=[]
    keys=('pose','joint','target','pad','stable','failed','peak_pressure','whole_finger_force','displacement')
    traces={k:[] for k in keys} if capture else None
    if demonstrations:
        assert capture
        for k in ('obs_prop','obs_tactile','teacher_action'):
            traces[k]=[]
    # Two-second baseline. Each cycle: open, recover, close/hold, then return.
    # Search uses one cycle without a return; replay includes the return between cycles.
    total=40+160*cycles+40*(cycles-1)
    for t in range(total):
        desired = base.clone()
        phase = (t-40)%200
        active_cycle = t>=40 and phase<160
        if active_cycle:
            desired[:,joints[3]] = base[:,joints[3]]-.25
            if phase>=40:
                desired[:,joints[:3]] = endpoints[:,:3]
            if phase>=80:
                desired[:,joints[3]] = endpoints[:,3]
        desired = desired.clamp(raw._active_limits[...,0],raw._active_limits[...,1])
        target=raw._q_target+(desired-raw._q_target).clamp(-.015,.015)
        if demonstrations:
            # env.reset already produced the initial observation. Read exactly
            # one new sensor sample after each step, as the normal environment does.
            for k in ('prop','tactile'):
                traces['obs_'+k].append(raw.get_observations()['policy'][k].cpu().numpy().copy())
            traces['teacher_action'].append(((target-raw._q_target)/raw.cfg.maximum_joint_target_delta_rad).clamp(-1,1).cpu().numpy().copy())
        d=step_physics(raw,target)
        if demonstrations:raw._get_observations()
        if recorder:recorder.append(t,d)
        alive &= ~d['failed']
        if t>=20:
            stable_all &= d['stable'] & alive
        q=raw._robot.data.joint_pos.torch[:,raw._active_ids]
        pressure=torch.maximum(pressure,d['peak_pressure'])
        if 20<=t<40:
            initial_q2+=q[:,j2]/20
            initial_pad+=d['pad'][:,0].float()/20
        if t>=40:
            stable_count+=d['stable'].float()*alive
        if active_cycle and phase<80:
            free=d['whole_finger_force'][:,0]<.05
            whole_free+=free.float()/80
            free_streak=torch.where(free,free_streak+1,torch.zeros_like(free_streak))
            max_free_streak=torch.maximum(max_free_streak,free_streak)
        if active_cycle and phase>=120:
            pad+=d['pad'][:,0].float()/40
            q2_final+=q[:,j2]/40
            reserve+=((q[:,j2]-initial_q2>=.1)&(q[:,j2]-raw._active_limits[:,j2,0]>=.08)).float()/40
            approach=pad_approach_quality(raw._robot.data.body_pos_w.torch[:,raw._contact_body_ids[:5]],
                raw._wxyz(raw._robot.data.body_quat_w.torch[:,raw._contact_body_ids[:5]]),
                d['pose'][:,:3],raw._wxyz(d['pose'][:,3:]),raw._axes_m)
            quality+=approach[:,0]/40
        if active_cycle and phase==159:
            passed=stable_all & alive & (pad>=.8-1e-6) & (reserve>=.8-1e-6) & (max_free_streak>=10)
            cycle_stats.append(dict(passed=passed.clone(),pad_fraction=pad.clone(),reserve_fraction=reserve.clone(),
                quality=quality.clone(),final_q2=q2_final.clone(),unloaded_fraction=whole_free.clone(),
                maximum_unloaded_streak_steps=max_free_streak.clone()))
            pad.zero_();reserve.zero_();quality.zero_();q2_final.zero_();whole_free.zero_()
            free_streak.zero_();max_free_streak.zero_()
        if capture:
            d['joint']=q;d['target']=raw._q_target
            for k in keys:traces[k].append(d[k].cpu().numpy().copy())
    final=cycle_stats[-1]
    if recorder:recorder.close()
    return dict(**final,alive=alive,stable_all=stable_all,initial_q2=initial_q2,initial_pad=initial_pad,
                peak_pressure=pressure,stable_fraction=stable_count/(total-40),cycles=cycle_stats),traces


@torch.no_grad()
def collect_demonstrations(env,output,source):
    raw=env.unwrapped;output=Path(output);assert not output.exists()
    winners={w['state_id']:w for w in json.loads(Path(source).read_text())['winners'] if w['passed']}
    rows=list(raw._reset_grasps)
    assert all(r['source_state_id'] in winners for r in rows) and len(rows)==raw.num_envs
    endpoints=torch.tensor([winners[r['source_state_id']]['endpoint'] for r in rows],device=raw.device)
    d,traces=trial(env,endpoints,capture=True,demonstrations=True)
    data={k:np.asarray(v) for k,v in traces.items()}
    passed=d['passed'].cpu().numpy() & (d['peak_pressure'].cpu().numpy()<=raw.cfg.maximum_pressure_proxy_pa)
    # Do not teach a timer: ignore initial/static intermediate waypoint waits.
    # Keep terminal holding, including zero commands, so the actor learns to stop.
    moving=np.abs(data['teacher_action']).max(-1)>.04
    times=np.arange(STEPS)[:,None]
    eligible=(moving | (times>=160)) & passed[None] & (times>=40)
    np.savez_compressed(output.with_suffix('.npz'),**data,eligible=eligible,passed=passed,
        endpoints=endpoints.cpu().numpy(),initial_q2=d['initial_q2'].cpu().numpy())
    save(output,dict(completed=True,source=str(source),environments=raw.num_envs,
        successful_trials=int(passed.sum()),labelled_samples=int(eligible.sum()),
        observation_contract={'prop':70,'tactile':432,'frame_stack':4,'pre_action':True},
        no_privileged_actor_inputs=True,no_ppo=True,zero_velocity_resets=True,
        rows=[dict(state_id=r['source_state_id'],split=r['demonstration_split'],passed=bool(passed[i]),
                   labelled_samples=int(eligible[:,i].sum())) for i,r in enumerate(rows)],
        note='Initial and intermediate timed waits excluded, terminal holding retained; complete source states kept in one split; physical replay required after imitation.'))
    print(f'Demonstration collection: {int(passed.sum())}/{raw.num_envs} trials; {int(eligible.sum())} action labels',flush=True)


@torch.no_grad()
def search_paths(env, output, rounds=8):
    raw=env.unwrapped;output=Path(output)
    assert not output.exists()
    assert raw.cfg.domain_profile=='nominal' and raw.num_envs%PER_STATE==0
    rows=list(raw._reset_grasps);n=raw.num_envs;s=n//PER_STATE
    assert len(rows)==n and raw._diagnostic_finger_sensors
    for i,r in enumerate(rows):assert r['source_state_id']==rows[i-i%PER_STATE]['source_state_id']
    joints=[ACTIVE_JOINTS.index('thumb_joint'+str(j)) for j in (0,1,2,3)]
    q=torch.tensor([rows[i*PER_STATE]['joint_positions'] for i in range(s)],device=raw.device)
    lo=raw._active_limits[::PER_STATE,joints,0].clone()
    hi=raw._active_limits[::PER_STATE,joints,1].clone()
    lo[:,2]=torch.maximum(lo[:,2]+.14,q[:,joints[2]]+.12)
    mean=q[:,joints].clone()+q.new_tensor([0.,.2,.25,-.15])
    mean=mean.clamp(lo,hi)
    sigma=torch.full_like(mean,.25)
    best_score=torch.full((s,),-1e9,device=raw.device);best=mean.clone()
    archive=[];started=time.monotonic()
    save(output.with_name(output.stem+'_protocol.json'),dict(states=s,candidates_per_state=PER_STATE,
        rounds=rounds,steps=STEPS,dt=raw.step_dt,thumb_joint_order=[ACTIVE_JOINTS[i] for i in joints],
        optimizer='CEM, four elites per state, 4/32 broad samples per round',
        phases='2s baseline; 2s distal open; 2s recover proximal axes; 4s close/hold',
        gate='Original strict stability after 1s; no physical failure; whole thumb force<0.05N continuously>=0.5s during retreat; last2s thumb pad>=80% and actual joint2 recovered>=0.1rad with reserve>=0.08rad for>=80%',
        no_core_changes=True,no_ppo=True,initial_velocities_zero=True,training_states_only=True))
    for iteration in range(rounds):
        x=mean[:,None]+torch.randn(s,PER_STATE,4,device=raw.device)*sigma[:,None]
        x[:,:4]=lo[:,None]+torch.rand(s,4,4,device=raw.device)*(hi-lo)[:,None]
        x[:,4]=best;x[:,5]=mean
        x=x.clamp(lo[:,None],hi[:,None])
        d,_=trial(env,x.reshape(n,4))
        score=(100*d['passed'].float()+20*d['pad_fraction']+5*d['quality']+5*d['reserve_fraction']
               +10*d['stable_fraction']-30*(~d['alive']).float()-15*(~d['stable_all']).float()).reshape(s,PER_STATE)
        vals,ids=score.topk(4,dim=1)
        improved=vals[:,0]>best_score
        best[improved]=x[torch.arange(s,device=raw.device),ids[:,0]][improved]
        best_score=torch.maximum(best_score,vals[:,0])
        elite=torch.gather(x,1,ids[:,:,None].expand(-1,-1,4))
        mean=.2*mean+.8*elite.mean(1)
        sigma=(.2*sigma+.8*elite.std(1)).clamp_min(.025)
        metrics={k:v.cpu().numpy() for k,v in d.items() if torch.is_tensor(v)}
        np.savez_compressed(output.with_name(f'{output.stem}_round{iteration:02d}.npz'),
                            endpoints=x.cpu().numpy(),score=score.cpu().numpy(),**metrics)
        for si in range(s):
            ci=int(ids[si,0]);i=si*PER_STATE+ci
            archive.append(dict(state_id=rows[i]['source_state_id'],round=iteration,environment=i,
                endpoint=x[si,ci].cpu().tolist(),score=float(score[si,ci]),
                **{k:v[i].item() for k,v in metrics.items()}))
        # Proposals are recorded with their actual observed pass status, not just scores.
        winners=[]
        for si in range(s):
            candidates=[a for a in archive if a['state_id']==rows[si*PER_STATE]['source_state_id']]
            chosen=max(candidates,key=lambda a:a['score'])
            winners.append(dict(source_state=rows[si*PER_STATE],**chosen))
        save(output,dict(search_completed=iteration+1==rounds,rounds_completed=iteration+1,
            total_proposals=(iteration+1)*n,states=s,states_with_observed_pass=sum(w['passed'] for w in winners),
            elapsed_seconds=time.monotonic()-started,winners=winners,archive=archive,
            independent_repeated_validation_complete=False,learned_gait_proven=False))
        print(f'Recovery path search {iteration+1}/{rounds}: {sum(w["passed"] for w in winners)}/{s} states with observed pass',flush=True)


@torch.no_grad()
def validate_paths(env, output, source, video=None, video_index=0):
    raw=env.unwrapped;output=Path(output)
    assert not output.exists()
    selected=[w for w in json.loads(Path(source).read_text())['winners'] if w['passed']]
    assert selected and raw.num_envs==len(selected)
    for row,w in zip(raw._reset_grasps,selected):assert row['source_state_id']==w['state_id']
    endpoints=torch.tensor([w['endpoint'] for w in selected],device=raw.device)
    d,traces=trial(env,endpoints,cycles=3,capture=True,video=video,video_index=video_index)
    np.savez_compressed(output.with_suffix('.npz'),**{k:np.asarray(v) for k,v in traces.items()},
        endpoints=endpoints.cpu().numpy(),robot_root_pose=raw._robot.data.root_pose_w.torch.cpu().numpy())
    rows=[]
    for i,w in enumerate(selected):
        cs=[{k:v[i].item() for k,v in c.items()} for c in d['cycles']]
        rows.append(dict(state_id=w['state_id'],cycles=cs,passed=all(c['passed'] for c in cs),
                         failed=not bool(d['alive'][i]),peak_pressure=float(d['peak_pressure'][i])))
    save(output,dict(validation_completed=True,source=str(source),states=len(rows),cycles=3,
        passed_count=sum(r['passed'] for r in rows),results=rows,learned_gait_proven=False,
        net_rotation_claim=False,initial_velocities_zero=True,domain_profile=raw.cfg.domain_profile,
        density_range_kg_m3=list(raw.cfg.events.geometry.params['density_range_kg_m3']),
        note='Repeated recovery and return, not a learned directional rotation gait.'))
    root=raw._robot.data.root_pose_w.torch
    pose=raw._fruit.data.root_pose_w.torch
    local=quat_inv_rotate(raw._wxyz(root[:,3:]),pose[:,:3]-root[:,:3]).cpu().numpy()
    joint=raw._robot.data.joint_pos.torch[:,raw._active_ids].cpu().numpy()
    target=raw._q_target.cpu().numpy()
    quat=(pose[:,3:] if raw._native_xyzw else pose[:,[4,5,6,3]]).cpu().numpy()
    accepted=[]
    for i,(w,r) in enumerate(zip(selected,rows)):
        if not r['passed']:continue
        accepted.append(dict(id=w['state_id']+'_recovered',hand_posture='palm_up',
            joint_positions=joint[i].tolist(),joint_targets=target[i].tolist(),
            fruit_position_hand=local[i].tolist(),fruit_quaternion_xyzw=quat[i].tolist(),
            proposal_only=False,physical_validation=True,reset_role='recovered_contact_state',
            not_n_minus_one_anchor=True,split='train',source_state_id=w['state_id'],
            validation_scope=f'Three {raw.cfg.domain_profile} recovery-return cycles; not N-1 nor learned directed rotation',
            validation_file=str(output)))
    save(output.with_name(output.stem+'_accepted.json'),accepted)
    print(f'Repeated path validation: {sum(r["passed"] for r in rows)}/{len(rows)} passed all3 cycles',flush=True)
