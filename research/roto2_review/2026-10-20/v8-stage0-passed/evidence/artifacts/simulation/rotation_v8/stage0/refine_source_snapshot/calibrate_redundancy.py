"""Stage-0 real-physics search and N-1 verification, independent of PPO.

No injected object support, teleport during a trial, altered friction, scripted
training controller, or automatic post-failure resets. Search output is proposals,
not an accepted reset library. Whole-finger sensors verify actual release.
"""
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
from .core import ACTIVE_JOINTS, FINGERS, quat_rotate
from .reward_geometry import pad_approach_quality


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rows_to_grasps(rows, prefix):
    return [dict(id=f'{prefix}_{i}', joint_positions=r[:16], fruit_position_hand=r[16:],
                 hand_posture='palm_up', proposal_only=True) for i, r in enumerate(rows)]


@torch.no_grad()
def step_physics(raw, target):
    """Use the unchanged actuator path, but never auto-reset a failed trial."""
    action = ((target - raw._q_target) / raw.cfg.maximum_joint_target_delta_rad).clamp(-1, 1)
    raw._pre_physics_step(action)
    for _ in range(raw.cfg.decimation):
        raw._apply_action(); raw.scene.write_data_to_sim()
        raw.sim.step(render=False); raw.scene.update(raw.cfg.sim.dt)
    raw.episode_length_buf += 1
    failed, _ = raw._get_dones()
    pose = raw._fruit.data.root_pose_w.torch.clone()
    vel = raw._fruit.data.root_lin_vel_w.torch.clone()
    quat = raw._wxyz(pose[:, 3:])
    tilt = 2 * quat[:, 1:3].square().sum(-1)
    displacement = torch.linalg.vector_norm(pose[:, :3] - raw._center_reference_w, dim=-1)
    speed = torch.linalg.vector_norm(vel, dim=-1)
    finite = torch.isfinite(pose).all(-1) & torch.isfinite(vel).all(-1)
    stable = (finite & ~failed & (displacement < .025) & (speed < .12) & (tilt < .5)
              & (pose[:, 2] > raw._center_reference_w[:, 2] - .015)
              & (raw._dense.amax(-1) <= raw.cfg.maximum_pressure_proxy_pa))
    force = torch.zeros(raw.num_envs, 5, device=raw.device)
    for body, sensor in raw._diagnostic_finger_sensors.items():
        matrix = sensor.data.force_matrix_w
        if matrix is None: raise RuntimeError('Whole-finger contact instrumentation missing')
        magnitude = matrix.torch[:, 0, 0].norm(dim=-1)
        if not torch.isfinite(magnitude).all(): raise RuntimeError('Nonfinite whole-finger contact')
        force[:, FINGERS.index(body.split('_')[0])] += magnitude
        stable &= magnitude <= raw.cfg.maximum_contact_force_n
    return dict(pose=pose, speed=speed, displacement=displacement, tilt=tilt,
                stable=stable, failed=failed | ~finite, pad=raw._pad_forces > .15,
                whole_finger_force=force, peak_pressure=raw._dense.amax(-1).clone())


@torch.no_grad()
def search_redundant_resets(env, output, rounds=32, horizon=80, spread=.12):
    """16 CEM islands, broad grasp/fruit-center exploration; retain diverse proposals."""
    raw = env.unwrapped; n = raw.num_envs; device = raw.device; output = Path(output)
    if n % 16 or n < 256: raise ValueError('Search needs >=256 environments, multiple of 16')
    if horizon < 40 or rounds < 1: raise ValueError('Search horizon >=40 and rounds >=1')
    if not .005 <= spread <= .3: raise ValueError('Search spread must be in [.005,.3]')
    seeds = list(raw._reset_grasps)
    root = Path(__file__).resolve().parents[2]
    historical = root/'simulation/.runtime/setup/palm_up/reset_candidates.json'
    if historical.exists(): seeds.extend(json.loads(historical.read_text()))
    seed_rows = torch.tensor([s['joint_positions']+s['fruit_position_hand'] for s in seeds], device=device)
    means = seed_rows[torch.arange(16, device=device) % len(seed_rows)].clone()
    # Separate islands explore towards palm and ulnar side; no asset change.
    if spread >= .08:
        means[4:8, 16] -= .018; means[4:8, 18] -= .025
        means[8:12, 17] -= .018; means[8:12, 18] -= .015
        means[12:, 17] += .018; means[12:, 18] -= .035
    limits = raw._active_limits[0]
    lo = torch.cat((limits[:, 0]+.002, torch.tensor([.025, -.045, .10], device=device)))
    hi = torch.cat((limits[:, 1]-.002, torch.tensor([.115, .045, .235], device=device)))
    scale = hi-lo
    sigma = scale.expand(16, -1).clone() * spread
    sigma[:, 16:] = spread*.1
    floor = sigma * .15
    best_vectors = means.clone()
    best_scores = torch.full((16,), -float('inf'), device=device)
    per = n//16; elite = max(4, per//8); archive = []; started = time.monotonic()
    for iteration in range(rounds):
        x = means[:, None] + torch.randn(16, per, 19, device=device)*sigma[:, None]
        x[:, 0] = means; x[:, 1] = best_vectors
        x = x.clamp(lo, hi); flat = x.reshape(n, 19)
        rows = flat.cpu().tolist(); raw._reset_grasps = rows_to_grasps(rows, f's{raw.cfg.seed}_r{iteration}')
        env.reset(); target = flat[:, :16].clone()
        alive = torch.ones(n, device=device, dtype=torch.bool)
        count_sum = torch.zeros(n, device=device); four = count_sum.clone(); five = count_sum.clone()
        stable_sum = count_sum.clone(); approach_sum = count_sum.clone(); pad_sum = torch.zeros(n, 5, device=device)
        pressure_max = count_sum.clone(); measured = 0
        for t in range(horizon):
            d = step_physics(raw, target); alive &= ~d['failed']
            if t < 20: continue  # settling cannot earn contact credit
            measured += 1; ok = alive & d['stable']; pads = d['pad']; count = pads.sum(-1)
            pad_sum += pads * ok[:, None]; count_sum += count * ok
            four += (count >= 4) * ok; five += (count >= 5) * ok; stable_sum += ok
            approach = pad_approach_quality(raw._robot.data.body_pos_w.torch[:, raw._contact_body_ids[:5]],
                raw._wxyz(raw._robot.data.body_quat_w.torch[:, raw._contact_body_ids[:5]]),
                d['pose'][:, :3], raw._wxyz(d['pose'][:, 3:]), raw._axes_m)
            approach_sum += approach.mean(-1) * alive
            pressure_max = torch.maximum(pressure_max, d['peak_pressure'])
        # No thumb privilege; four/five contacts receive the dominant credit.
        score = (40*four + 20*five + 3*count_sum + 4*stable_sum + 2*approach_sum)/measured - 20*(~alive)
        scores = score.reshape(16, per); ids = scores.topk(elite, dim=1).indices
        current_best, best_ids = scores.max(1)
        better = current_best > best_scores
        best_vectors[better] = x[torch.arange(16, device=device), best_ids][better]
        best_scores = torch.maximum(best_scores, current_best)
        selected = torch.gather(x, 1, ids[..., None].expand(-1, -1, 19))
        means = .25*means + .75*selected.mean(1)
        sigma = torch.maximum(floor, .3*sigma + .7*selected.std(1))
        # Save up to 256 per round, not just repeated global winners.
        top = scores.topk(min(16, per), dim=1).indices + torch.arange(16, device=device)[:, None]*per
        for i in top.flatten().cpu().tolist():
            row = rows_to_grasps([rows[i]], f's{raw.cfg.seed}_r{iteration}_env{i}')[0]
            row.update(score=float(score[i]), four_pad_fraction=float(four[i]/measured),
                       five_pad_fraction=float(five[i]/measured), stable_fraction=float(stable_sum[i]/measured),
                       per_finger_pad_fraction=(pad_sum[i]/measured).tolist(),
                       failed=not bool(alive[i]), peak_pressure_pa=float(pressure_max[i]))
            archive.append(row)
        archive.sort(key=lambda r:r['score'], reverse=True)
        diverse = []; vectors = np.empty((1024, 19))
        scale_np = scale.cpu().numpy()
        for row in archive:
            v = np.array(row['joint_positions']+row['fruit_position_hand']) / scale_np
            if diverse and (np.linalg.norm(vectors[:len(diverse)]-v, axis=1) < .07).any(): continue
            vectors[len(diverse)] = v; diverse.append(row)
            if len(diverse) == 1024: break
        archive = diverse
        np.savez_compressed(output.with_name(f'{output.stem}_round{iteration:03d}.npz'),
                            proposals=flat.cpu().numpy(), score=score.cpu().numpy(),
                            four_pad_fraction=(four/measured).cpu().numpy(),
                            five_pad_fraction=(five/measured).cpu().numpy(),
                            stable_fraction=(stable_sum/measured).cpu().numpy(),
                            pad_fraction=(pad_sum/measured).cpu().numpy(), alive=alive.cpu().numpy(),
                            mass_kg=raw._fruit.data.body_mass.torch.cpu().numpy())
        save(output, archive)
        status = dict(stage='proposal_search', round=iteration+1, rounds=rounds,
                      candidate_count=len(archive), evaluated_proposals=(iteration+1)*n,
                      best_score=archive[0]['score'], best_four_pad_fraction=archive[0]['four_pad_fraction'],
                      best_per_finger=archive[0]['per_finger_pad_fraction'],
                      four_pad_80pct_candidates=sum(r['four_pad_fraction'] >= .8 and not r['failed'] for r in archive),
                      elapsed_seconds=time.monotonic()-started, training_started=False,
                      stage0_passed=False, search_completed=iteration+1==rounds)
        save(output.with_name(output.stem+'_status.json'), status)
        print(json.dumps(status), flush=True)
    return archive


@torch.no_grad()
def validate_n_minus_one(env, output):
    """Six matched conditions per grasp: hold and independently release each finger.

    All links of the selected finger must lose measured normal contact for the
    full final 0.5s. Passive object stability and restoration are mandatory.
    Noncontact fingers are marked non-load-bearing, not claimed as releases.
    """
    raw = env.unwrapped; n = raw.num_envs; device = raw.device; output = Path(output)
    if n % 6: raise ValueError('N-1 validation requires num_envs divisible by 6')
    if not raw._diagnostic_finger_sensors: raise RuntimeError('Whole-finger sensors required')
    candidates = list(raw._reset_grasps); per = n//6; results = []; accepted = []
    baseline_steps, retract_steps, release_steps, recovery_steps = 40, 40, 10, 60
    protocol = dict(dt=raw.step_dt, baseline_steps=baseline_steps, retract_steps=retract_steps,
        release_steps=release_steps, recovery_steps=recovery_steps, minimum_baseline_pads=4,
        baseline_pad_fraction=.8, release_force_max_n=.05, stable_displacement_m=.025,
        stable_speed_m_s=.12, stable_height_drop_m=.015, tilt_max=.5,
        whole_finger_sensor_bodies=list(raw._diagnostic_finger_sensors),
        matched_density_kg_m3=raw.cfg.events.geometry.params["density_range_kg_m3"],
        pressure_limit_pa=raw.cfg.maximum_pressure_proxy_pa,
        note='Individual tests per starting grasp, no automatic reset, all five fingers tested; candidate-level success is not a population or training result.')
    save(output.with_name(output.stem+'_protocol.json'), protocol)
    group = torch.arange(n, device=device) % 6
    for batch_start in range(0, len(candidates), per):
        chunk = candidates[batch_start:batch_start+per]; actual = len(chunk)
        padded = chunk + [chunk[-1]]*(per-actual)
        raw._reset_grasps = [c for c in padded for _ in range(6)]
        env.reset(); base_target = raw._q_target.clone(); open_target = base_target.clone()
        for finger_i, finger in enumerate(FINGERS):
            # Flexion decreases towards open; abduction/opposition stay fixed.
            flexions = (2, 3) if finger == 'thumb' else (1, 2)
            for j in flexions:
                idx = ACTIVE_JOINTS.index(f'{finger}_joint{j}')
                open_target[group == finger_i+1, idx] = raw._active_limits[group == finger_i+1, idx, 0]+.002
        traces = {k:[] for k in ('pose','speed','displacement','tilt','stable','failed','pad','whole_finger_force','peak_pressure')}
        alive = torch.ones(n, device=device, dtype=torch.bool)
        for t in range(baseline_steps+retract_steps+release_steps+recovery_steps):
            target = open_target if baseline_steps <= t < baseline_steps+retract_steps+release_steps else base_target
            d = step_physics(raw, target); alive &= ~d['failed']; d['stable'] &= alive
            for k in traces: traces[k].append(d[k].cpu().numpy().copy())
        data = {k:np.array(v) for k,v in traces.items()}
        trace_path = output.with_name(f'{output.stem}_batch{batch_start//per:03d}.npz')
        np.savez_compressed(trace_path, **data, target_base=base_target.cpu().numpy(),
                            target_open=open_target.cpu().numpy(), group=group.cpu().numpy())
        for ci in range(actual):
            ids = slice(ci*6, (ci+1)*6)
            pad_base = data['pad'][20:40, ids].mean(0)
            stable_base = data['stable'][20:40, ids].all(0)
            # Require the same initial load-bearing set across all six conditions.
            loaded = (pad_base >= .8).all(0)
            baseline_force = data['whole_finger_force'][20:40, ids].mean(0)
            physically_loaded = (baseline_force >= .15).any(0)
            baseline_ok = loaded.sum() >= 4 and stable_base.all()
            trial_rows = []
            for condition in range(6):
                idx = ci*6+condition
                tested = condition > 0 and bool(physically_loaded[condition-1])
                released = condition == 0 or bool((data['whole_finger_force'][80:90, idx, condition-1] < .05).all())
                stable = bool(data['stable'][40:, idx].all())
                recovered = bool((data['pad'][-20:, idx].mean(0) >= .8).sum() >= 4)
                trial_rows.append(dict(condition='hold' if condition==0 else FINGERS[condition-1],
                    baseline_loaded_finger=tested, actually_released=released,
                    stable_through_release_and_recovery=stable, four_pad_recovered=recovered,
                    physical_failure=bool(data['failed'][:, idx].any()),
                    max_displacement_m=float(data['displacement'][:, idx].max()),
                    max_speed_m_s=float(data['speed'][20:, idx].max()),
                    max_pressure_pa=float(data['peak_pressure'][:, idx].max()),
                    minimum_release_force_n=None if condition==0 else float(data['whole_finger_force'][80:90,idx,condition-1].min()),
                    maximum_release_force_n=None if condition==0 else float(data['whole_finger_force'][80:90,idx,condition-1].max())))
            passed = bool(baseline_ok and all(r['actually_released'] and r['stable_through_release_and_recovery']
                                            and r['four_pad_recovered'] and not r['physical_failure'] for r in trial_rows))
            row = dict(candidate_id=chunk[ci].get('id',batch_start+ci), baseline_pad_fingers=loaded.tolist(),
                       baseline_loaded_fingers=physically_loaded.tolist(),
                       baseline_whole_finger_mean_force_n=baseline_force.tolist(),
                       baseline_pad_fraction=pad_base.tolist(), baseline_ok=bool(baseline_ok),
                       n_minus_one_passed=passed, trials=trial_rows, trace_file=trace_path.name)
            results.append(row)
            if passed:
                accepted.append({**chunk[ci], 'proposal_only':False, 'n_minus_one_passed':True,
                                 'n_minus_one_seed':raw.cfg.seed, 'n_minus_one_result':str(output),
                                 'validation_scope':'single seed; independent reseed validation still required'})
        save(output, dict(seed=raw.cfg.seed, candidates_tested=len(results), accepted_count=len(accepted),
                          validation_completed=batch_start+actual==len(candidates),
                          stage0_passed=False, independent_reseed_required=True, results=results))
        save(output.with_name(output.stem+'_accepted.json'), accepted)
        print(json.dumps(dict(candidates_tested=len(results), accepted_count=len(accepted),
                              baseline_ok=sum(r['baseline_ok'] for r in results))),flush=True)
    return accepted
