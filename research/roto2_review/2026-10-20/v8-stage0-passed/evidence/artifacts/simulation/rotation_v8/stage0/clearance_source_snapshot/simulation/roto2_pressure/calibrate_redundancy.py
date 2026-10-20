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
from .core import ACTIVE_JOINTS, FINGERS, quat_rotate, quat_inv_rotate
from .reward_geometry import pad_approach_quality


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rows_to_grasps(rows, prefix):
    return [dict(id=f'{prefix}_{i}', joint_positions=r[:16], fruit_position_hand=r[16:],
                 hand_posture='palm_up', proposal_only=True) for i, r in enumerate(rows)]

def release_directions(finger, profile):
    flexions = (2,3) if finger == "thumb" else (1,2)
    if profile == "distal":
        return [(ACTIVE_JOINTS.index(f"{finger}_joint{flexions[-1]}"), -1)]
    if profile == "proximal":
        return [(ACTIVE_JOINTS.index(f"{finger}_joint{flexions[0]}"), -1)]
    moves = [(ACTIVE_JOINTS.index(f"{finger}_joint{j}"), -1) for j in flexions]
    if profile in ("spread_positive","spread_negative"):
        moves.append((ACTIVE_JOINTS.index(f"{finger}_joint0"), 1 if profile.endswith("positive") else -1))
    elif profile in ("thumb_roll_positive","thumb_roll_negative") and finger == "thumb":
        moves.append((ACTIVE_JOINTS.index("thumb_joint1"), 1 if profile.endswith("positive") else -1))
    elif profile not in ("flexion","thumb_roll_positive","thumb_roll_negative"):
        raise ValueError(f"Unknown release profile: {profile}")
    return moves


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
        matrix = sensor.data.force_matrix_w_history
        if matrix is None: raise RuntimeError('Whole-finger contact instrumentation missing')
        history = matrix.torch
        if history.shape[1] != raw.cfg.decimation:
            raise RuntimeError('Whole-finger force history must cover every physics substep')
        magnitude = history[:, :, 0, 0].norm(dim=-1).amax(dim=1)
        if not torch.isfinite(magnitude).all(): raise RuntimeError('Nonfinite whole-finger contact')
        force[:, FINGERS.index(body.split('_')[0])] += magnitude
        stable &= magnitude <= raw.cfg.maximum_contact_force_n
    return dict(pose=pose, speed=speed, displacement=displacement, tilt=tilt,
                stable=stable, failed=failed | ~finite, pad=raw._pad_forces > .15,
                whole_finger_force=force, peak_pressure=raw._dense.amax(-1).clone())


@torch.no_grad()
def search_redundant_resets(env, output, rounds=32, horizon=80, spread=.12, region="broad", replicas=1):
    """16 CEM islands, broad grasp/fruit-center exploration; retain diverse proposals."""
    raw = env.unwrapped; n = raw.num_envs; device = raw.device; output = Path(output)
    if n % 16 or n < 256: raise ValueError('Search needs >=256 environments, multiple of 16')
    if replicas < 1 or n % (16*replicas) or n//replicas < 256:
        raise ValueError('Need >=256 unique candidates and num_envs divisible by 16*replicas')
    unique = n//replicas
    if horizon < 40 or rounds < 1: raise ValueError('Search horizon >=40 and rounds >=1')
    if not .005 <= spread <= .3: raise ValueError('Search spread must be in [.005,.3]')
    seeds = list(raw._reset_grasps)
    root = Path(__file__).resolve().parents[2]
    historical = root/'simulation/.runtime/setup/palm_up/reset_candidates.json'
    if historical.exists(): seeds.extend(json.loads(historical.read_text()))
    seed_rows = torch.tensor([s['joint_positions']+s['fruit_position_hand'] for s in seeds], device=device)
    means = seed_rows[torch.arange(16, device=device) % len(seed_rows)].clone()
    # Separate islands explore towards palm and ulnar side; no asset change.
    if spread >= .08 and region == "broad":
        means[4:8, 16] -= .018; means[4:8, 18] -= .025
        means[8:12, 17] -= .018; means[8:12, 18] -= .015
        means[12:, 17] += .018; means[12:, 18] -= .035
    limits = raw._active_limits[0]
    lo = torch.cat((limits[:, 0]+.002, torch.tensor([.025, -.045, .10], device=device)))
    hi = torch.cat((limits[:, 1]-.002, torch.tensor([.115, .045, .235], device=device)))
    if region == "palm":
        lo[16:] = lo.new_tensor([.035,-.035,.095])
        hi[16:] = hi.new_tensor([.060,.035,.175])
    scale = hi-lo
    sigma = scale.expand(16, -1).clone() * spread
    sigma[:, 16:] = spread*.1
    floor = sigma * .15
    best_vectors = means.clone()
    best_scores = torch.full((16,), -float('inf'), device=device)
    per = unique//16; elite = max(4, per//8); archive = []; started = time.monotonic()
    for iteration in range(rounds):
        x = means[:, None] + torch.randn(16, per, 19, device=device)*sigma[:, None]
        x[:, 0] = means; x[:, 1] = best_vectors
        x = x.clamp(lo, hi); flat = x.reshape(unique,19).repeat_interleave(replicas,dim=0)
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
        score_env = (40*four + 20*five + 3*count_sum + 4*stable_sum + 2*approach_sum)/measured - 20*(~alive)
        score = score_env.reshape(unique,replicas).amin(1)
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
        root_pose = raw._robot.data.root_pose_w.torch
        settled_local = quat_inv_rotate(raw._wxyz(root_pose[:,3:]), d['pose'][:,:3]-root_pose[:,:3])
        settled_q = raw._robot.data.joint_pos.torch[:,raw._active_ids].clone()
        settled_quat = d['pose'][:,3:] if raw._native_xyzw else d['pose'][:,[4,5,6,3]]
        for candidate_i in top.flatten().cpu().tolist():
            rs = slice(candidate_i*replicas,(candidate_i+1)*replicas)
            # Archive the worst observed replica's actual settled state; never
            # cherry-pick the easiest mass/environment among the replications.
            i = candidate_i*replicas + int(score_env[rs].argmin())
            row = rows_to_grasps([rows[i]], f's{raw.cfg.seed}_r{iteration}_env{i}')[0]
            row.update(source_initial_joint_positions=rows[i][:16],
                       source_initial_fruit_position_hand=rows[i][16:],
                       joint_positions=settled_q[i].cpu().tolist(),
                       joint_targets=raw._q_target[i].cpu().tolist(),
                       fruit_position_hand=settled_local[i].cpu().tolist(),
                       fruit_quaternion_xyzw=settled_quat[i].cpu().tolist(),
                       reset_schema="settled_state_and_preload_v1",
                       terminal_pad_contact=d['pad'][i].cpu().tolist(),
                       terminal_stable=bool((d['stable'][rs] & alive[rs]).all()))
            row.update(score=float(score[candidate_i]), four_pad_fraction=float(four[rs].min()/measured),
                       five_pad_fraction=float(five[rs].min()/measured), stable_fraction=float(stable_sum[rs].min()/measured),
                       per_finger_pad_fraction=(pad_sum[rs].amin(0)/measured).tolist(),
                       failed=not bool(alive[rs].all()), peak_pressure_pa=float(pressure_max[rs].max()),
                       calibration_replicas=replicas,
                       replica_four_pad_fractions=(four[rs]/measured).cpu().tolist())
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
                            proposals=flat.cpu().numpy(), score=score_env.cpu().numpy(),
                            worst_replica_score=score.cpu().numpy(), replicas=replicas,
                            four_pad_fraction=(four/measured).cpu().numpy(),
                            five_pad_fraction=(five/measured).cpu().numpy(),
                            stable_fraction=(stable_sum/measured).cpu().numpy(),
                            pad_fraction=(pad_sum/measured).cpu().numpy(), alive=alive.cpu().numpy(),
                            settled_joint_positions=settled_q.cpu().numpy(),
                            settled_joint_targets=raw._q_target.cpu().numpy(),
                            settled_fruit_position_hand=settled_local.cpu().numpy(),
                            settled_fruit_quaternion_xyzw=settled_quat.cpu().numpy(),
                            terminal_pad_contact=d['pad'].cpu().numpy(),
                            terminal_stable=d['stable'].cpu().numpy(),
                            mass_kg=raw._fruit.data.body_mass.torch.cpu().numpy())
        save(output, archive)
        status = dict(stage='proposal_search', round=iteration+1, rounds=rounds,
                      search_region=region,
                      candidate_count=len(archive), evaluated_proposals=(iteration+1)*n,
                      unique_candidate_proposals=(iteration+1)*unique, replicas=replicas,
                      best_score=archive[0]['score'], best_four_pad_fraction=archive[0]['four_pad_fraction'],
                      best_per_finger=archive[0]['per_finger_pad_fraction'],
                      four_pad_80pct_candidates=sum(r['four_pad_fraction'] >= .8 and not r['failed'] for r in archive),
                      elapsed_seconds=time.monotonic()-started, training_started=False,
                      stage0_passed=False, search_completed=iteration+1==rounds)
        save(output.with_name(output.stem+'_status.json'), status)
        print(json.dumps(status), flush=True)
    return archive


@torch.no_grad()
def validate_n_minus_one(env, output, release_mode="full_open", video=None, video_env_index=0):
    """Six matched conditions per grasp: hold and independently release each finger.

    All links of the selected finger must lose measured normal contact for the
    full final 0.5s. Passive object stability through recovery is mandatory.
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
        release_mode=release_mode, adaptive_increment_rad=.015,
        release_controller="monotonic_clearance_v1",
        default_release_clearance_rad=.02,
        release_measurement="sum over links of max force magnitude over all 6 physics substeps; 120Hz sensor updates",
        gate_version="n1_release_stability_v2",
        gate_definition="Initial >=4 pads; each finger measurably unloaded for 0.5s; stability and safety through release/recovery. Fixed-target four-pad recovery reported separately, not user-required stage0 gate.",
        note='Individual tests per starting grasp, no automatic reset, all five fingers tested; candidate-level success is not a population or training result.')
    save(output.with_name(output.stem+'_protocol.json'), protocol)
    group = torch.arange(n, device=device) % 6
    for batch_start in range(0, len(candidates), per):
        chunk = candidates[batch_start:batch_start+per]; actual = len(chunk)
        padded = chunk + [chunk[-1]]*(per-actual)
        raw._reset_grasps = [c for c in padded for _ in range(6)]
        env.reset(); base_target = raw._q_target.clone(); open_target = base_target.clone()
        initial_center = raw._center_reference_w.clone()
        for finger_i, finger in enumerate(FINGERS):
            # Flexion decreases towards open; abduction/opposition stay fixed.
            flexions = (2, 3) if finger == 'thumb' else (1, 2)
            for j in flexions:
                idx = ACTIVE_JOINTS.index(f'{finger}_joint{j}')
                open_target[group == finger_i+1, idx] = raw._active_limits[group == finger_i+1, idx, 0]+.002
        traces = {k:[] for k in ('pose','speed','displacement','tilt','stable','failed','pad','whole_finger_force','peak_pressure',
                                'joint_position', 'joint_target')}
        alive = torch.ones(n, device=device, dtype=torch.bool)
        adaptive_target = base_target.clone()
        was_free = torch.zeros(n, device=device, dtype=torch.bool)
        recorder = None
        if video and batch_start == 0:
            from .calibration_video import CalibrationVideo
            recorder = CalibrationVideo(raw, video, video_env_index, release_mode)
        for t in range(baseline_steps+retract_steps+release_steps+recovery_steps):
            target = open_target if baseline_steps <= t < baseline_steps+retract_steps+release_steps else base_target
            if release_mode == "adaptive":
                q = raw._robot.data.joint_pos.torch[:, raw._active_ids]
                if baseline_steps <= t < baseline_steps+retract_steps+release_steps:
                    for fi, finger in enumerate(FINGERS):
                        free = d['whole_finger_force'][:, fi] < .05
                        for profile in {c.get("release_profiles",{}).get(finger,c.get("release_profile","flexion")) for c in padded}:
                            selected = torch.as_tensor([
                                c.get("release_profiles",{}).get(finger,c.get("release_profile","flexion"))==profile
                                for c in padded],device=device).repeat_interleave(6)
                            ids = (group==fi+1) & selected
                            stepsize = torch.as_tensor([c.get("release_increment_rad",.015) for c in padded],
                                                       device=device).repeat_interleave(6)
                            clearance = torch.as_tensor([c.get("release_clearance_rad",.02) for c in padded],
                                                        device=device).repeat_interleave(6)
                            for ji,direction in release_directions(finger,profile):
                                # A PD-controlled joint can lag its command. Replacing
                                # an open command with the measured position on the
                                # first zero-force frame can close it again. Retain
                                # the farther-open position and add a small clearance.
                                newly_free = ids & free & ~was_free
                                retained = direction * torch.maximum(
                                    direction*adaptive_target[newly_free,ji],
                                    direction*q[newly_free,ji])
                                adaptive_target[newly_free,ji] = (
                                    retained + direction*clearance[newly_free]).clamp(
                                        min=raw._active_limits[newly_free,ji,0]+.002,
                                        max=raw._active_limits[newly_free,ji,1]-.002)
                                moved=adaptive_target[ids & ~free,ji]+direction*stepsize[ids & ~free]
                                adaptive_target[ids & ~free,ji]=moved.clamp(
                                    min=raw._active_limits[ids & ~free,ji,0]+.002,
                                    max=raw._active_limits[ids & ~free,ji,1]-.002)
                        ids = group==fi+1
                        was_free[ids] = free[ids]
                elif t >= baseline_steps+retract_steps+release_steps:
                    adaptive_target += (base_target-adaptive_target).clamp(-.015,.015)
                target = adaptive_target
            d = step_physics(raw, target); alive &= ~d['failed']; d['stable'] &= alive
            d['joint_position'] = raw._robot.data.joint_pos.torch[:,raw._active_ids].clone()
            d['joint_target'] = raw._q_target.clone()
            if recorder:
                recorder.append(t, d)
            for k in traces: traces[k].append(d[k].cpu().numpy().copy())
        if recorder:
            recorder.close()
        data = {k:np.array(v) for k,v in traces.items()}
        trace_path = output.with_name(f'{output.stem}_batch{batch_start//per:03d}.npz')
        np.savez_compressed(trace_path, **data, target_base=base_target.cpu().numpy(),
                            target_open=open_target.cpu().numpy(), group=group.cpu().numpy(),
                            initial_center=initial_center.cpu().numpy())
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
                                            and not r['physical_failure'] for r in trial_rows))
            same_target_recovery = passed and all(r['four_pad_recovered'] for r in trial_rows)
            row = dict(candidate_id=chunk[ci].get('id',batch_start+ci), baseline_pad_fingers=loaded.tolist(),
                       source_grasp_id=chunk[ci].get("source_grasp_id",chunk[ci].get("id",batch_start+ci)),
                       baseline_loaded_fingers=physically_loaded.tolist(),
                       baseline_whole_finger_mean_force_n=baseline_force.tolist(),
                       baseline_pad_fraction=pad_base.tolist(), baseline_ok=bool(baseline_ok),
                       n_minus_one_passed=passed, same_target_four_pad_recovery_passed=same_target_recovery,
                       release_profile=chunk[ci].get("release_profile","flexion"),
                       release_profiles=chunk[ci].get("release_profiles",{}),
                       trials=trial_rows, trace_file=trace_path.name)
            results.append(row)
            if passed:
                accepted.append({**chunk[ci], 'proposal_only':False, 'n_minus_one_passed':True,
                                 'n_minus_one_seed':raw.cfg.seed, 'n_minus_one_result':str(output),
                                 'validation_scope':'single seed; independent reseed validation still required'})
        save(output, dict(seed=raw.cfg.seed, candidates_tested=len(results), accepted_count=len(accepted),
                          release_mode=release_mode,
                          gate_version="n1_release_stability_v2",
                          validation_completed=batch_start+actual==len(candidates),
                          stage0_passed=False, independent_reseed_required=True, results=results))
        save(output.with_name(output.stem+'_accepted.json'), accepted)
        print(json.dumps(dict(candidates_tested=len(results), accepted_count=len(accepted),
                              baseline_ok=sum(r['baseline_ok'] for r in results))),flush=True)
    return accepted
