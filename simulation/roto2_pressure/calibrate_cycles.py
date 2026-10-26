"""Repeated physical release/recontact validation; diagnostic commands, never PPO actions."""
from pathlib import Path
import time
import numpy as np
import torch
from .core import ACTIVE_JOINTS, FINGERS
from .calibrate_redundancy import step_physics, release_directions, save


@torch.no_grad()
def validate_recontact_cycles(env, output, cycles=3):
    raw=env.unwrapped; output=Path(output); n=raw.num_envs; device=raw.device
    if n%6 or cycles<2: raise ValueError('Six conditions per grasp and >=2 cycles required')
    if not raw._diagnostic_finger_sensors: raise RuntimeError('Whole-finger sensors required')
    candidates=list(raw._reset_grasps); per=n//6; group=torch.arange(n,device=device)%6
    results=[]; accepted=[]; start=time.monotonic()
    save(output.with_name(output.stem+'_protocol.json'),dict(cycles=cycles,baseline_steps=40,
         cycle_steps=110,retract_steps=40,unloaded_hold_steps=10,recovery_steps=60,
         dt=raw.step_dt,unchanged_original_n1_safety=True,no_resets_within_trial=True,
         require_initial_four_pads=True,require_released_loaded_pad_return=True,
         require_four_pad_recovery=True,not_a_rotation_or_learned_policy_test=True))
    for offset in range(0,len(candidates),per):
        chunk=candidates[offset:offset+per]; actual=len(chunk)
        padded=chunk+[chunk[-1]]*(per-actual)
        raw._reset_grasps=[c for c in padded for _ in range(6)]
        env.reset();base=raw._q_target.clone(); target=base.clone()
        alive=torch.ones(n,dtype=torch.bool,device=device); was_free=torch.zeros_like(alive)
        traces={k:[] for k in ('pad','stable','failed','whole_finger_force','peak_pressure','displacement','pose','joint_position')}
        for t in range(40+110*cycles):
            phase=(t-40)%110
            if t<40: target=base.clone()
            elif phase<50:
                if phase==0: was_free.zero_()
                q=raw._robot.data.joint_pos.torch[:,raw._active_ids]
                for fi,finger in enumerate(FINGERS):
                    free=d['whole_finger_force'][:,fi]<.05
                    profiles={c.get('release_profiles',{}).get(finger,c.get('release_profile','flexion')) for c in padded}
                    for profile in profiles:
                        selected=torch.tensor([c.get('release_profiles',{}).get(finger,c.get('release_profile','flexion'))==profile for c in padded],device=device).repeat_interleave(6)
                        ids=(group==fi+1)&selected
                        for ji,direction in release_directions(finger,profile):
                            newly_free=ids&free&~was_free
                            retained=direction*torch.maximum(direction*target[newly_free,ji],direction*q[newly_free,ji])
                            target[newly_free,ji]=(retained+direction*.02).clamp(
                                raw._active_limits[newly_free,ji,0]+.002,raw._active_limits[newly_free,ji,1]-.002)
                            moving=ids&~free
                            target[moving,ji]=(target[moving,ji]+direction*.015).clamp(
                                raw._active_limits[moving,ji,0]+.002,raw._active_limits[moving,ji,1]-.002)
                    ids=group==fi+1;was_free[ids]=free[ids]
            else: target+=(base-target).clamp(-.015,.015)
            d=step_physics(raw,target);alive&=~d['failed'];d['stable']&=alive
            d['joint_position']=raw._robot.data.joint_pos.torch[:,raw._active_ids]
            for key in traces:traces[key].append(d[key].cpu().numpy().copy())
            if t>=40 and phase==109:
                print(f'Cycle {1+(t-40)//110}/{cycles}, batch {offset}, alive {int(alive.sum())}/{n}',flush=True)
        data={k:np.asarray(v) for k,v in traces.items()}
        trace=output.with_name(f'{output.stem}_batch{offset//per:03d}.npz')
        np.savez_compressed(trace,**data)
        for ci,row in enumerate(chunk):
            ids=slice(ci*6,(ci+1)*6)
            initial=(data['pad'][20:40,ids].mean(0)>=.8).all(0)
            baseline_ok=bool(initial.sum()>=4 and data['stable'][20:40,ids].all())
            trials=[]
            for condition in range(6):
                ei=ci*6+condition; cs=[]
                for cycle in range(cycles):
                    begin=40+cycle*110
                    released=condition==0 or bool((data['whole_finger_force'][begin+40:begin+50,ei,condition-1]<.05).all())
                    recovered=data['pad'][begin+90:begin+110,ei].mean(0)>=.8
                    selected_recovered=condition==0 or not initial[condition-1] or bool(recovered[condition-1])
                    four_recovered=bool(recovered.sum()>=4)
                    stable=bool(data['stable'][begin:begin+110,ei].all())
                    cs.append(dict(cycle=cycle+1,released=released,selected_loaded_pad_recovered=bool(selected_recovered),
                                   four_pad_recovered=four_recovered,stable=stable,
                                   passed=bool(released and selected_recovered and four_recovered and stable)))
                trials.append(dict(condition='hold' if condition==0 else FINGERS[condition-1],cycles=cs))
            passed=baseline_ok and all(c['passed'] for tr in trials for c in tr['cycles']) and not data['failed'][:,ids].any()
            results.append(dict(candidate_id=row['id'],baseline_ok=baseline_ok,initial_pad_fingers=initial.tolist(),
                                passed=bool(passed),trials=trials,trace_file=str(trace),
                                peak_pressure_pa=float(data['peak_pressure'][:,ids].max())))
            if passed:accepted.append({**row,'recontact_cycles_passed':cycles,'recontact_validation':str(output)})
        save(output,dict(validation_completed=offset+actual==len(candidates),candidates_tested=len(results),
             passed_count=len(accepted),cycles=cycles,results=results,elapsed_seconds=time.monotonic()-start,
             learned_gait_proven=False))
        save(output.with_name(output.stem+'_accepted.json'),accepted)
    return accepted
