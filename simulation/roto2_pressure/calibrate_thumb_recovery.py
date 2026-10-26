"""Bounded physical thumb-recovery diagnostic, never a training controller.

Input states are repeated once per path. All paths share nominal physics and
the original actuator delta/lead limits. No state reset occurs inside a trial.
Failure of these paths is not a proof that no feasible trajectory exists.
"""
from pathlib import Path
import time
import numpy as np
import torch
from .core import ACTIVE_JOINTS
from .calibrate_redundancy import save, step_physics

PATHS = ('hold', 'joint2_inward_return', 'roll_inward_return',
         'distal_open_return', 'combined_retreat_return',
         'open_recover_close', 'direct_recover_hold')


@torch.no_grad()
def diagnose_thumb_recovery(env, output):
    raw = env.unwrapped
    output = Path(output)
    rows = list(raw._reset_grasps)
    n = raw.num_envs
    assert len(rows) == n and n % len(PATHS) == 0
    assert raw._diagnostic_finger_sensors and raw.cfg.domain_profile == 'nominal'
    assert not output.exists(), 'Preserve earlier diagnostic results'
    for i, row in enumerate(rows):
        assert row['diagnostic_path'] == PATHS[i % len(PATHS)]
        assert row['source_state_id'] == rows[i-i % len(PATHS)]['source_state_id']
    env.reset()
    base = raw._q_target.clone()
    initial_q = raw._robot.data.joint_pos.torch[:, raw._active_ids].clone()
    group = torch.arange(n, device=raw.device) % len(PATHS)
    j2, roll, distal = [ACTIVE_JOINTS.index('thumb_joint'+str(j)) for j in (2, 1, 3)]
    protocol = dict(paths=list(PATHS), states=n//len(PATHS), environments=n,
        dt=raw.step_dt, steps=240, baseline_steps=40, phase_steps=40,
        inward_delta_rad=.2, target_slew_rad_per_step=.015,
        phases=['baseline 0-2s', 'retreat/open 2-4s', 'hold/recover 4-6s',
                'return/close 6-8s', 'final hold 8-12s'],
        whole_finger_unloaded_threshold_n=.05, inward_recovery_threshold_rad=.1,
        minimum_joint2_reserve_rad=.08, effective_pad_threshold_n=.15,
        baseline_and_final_fraction=.8,
        unchanged_actuator_and_task_physics=True, no_resets_within_trial=True,
        initial_velocities_zero=True, live_policy_state_replay=False,
        learned_gait_proven=False,
        limitations='Finite nominal open-loop paths from settled training states, not a dynamic policy or impossibility proof.')
    save(output.with_name(output.stem+'_protocol.json'), protocol)
    traces = {k: [] for k in ('pose', 'pad', 'stable', 'failed', 'whole_finger_force',
        'peak_pressure', 'displacement', 'palm_force_n', 'joint', 'target')}
    started = time.monotonic()
    for t in range(240):
        desired = base.clone()
        if t >= 40:
            if t < 120:
                desired[group == 1, j2] += .2
                desired[group == 2, roll] += .2
                desired[group == 3, distal] -= .2
                for ji, delta in ((j2, .2), (roll, .2), (distal, -.2)):
                    desired[group == 4, ji] += delta
            if t < 120:
                desired[group == 5, distal] -= .2
            if t >= 80:
                desired[group == 5, j2] += .2
                desired[group == 5, roll] += .2
            desired[group == 6, j2] += .2
            desired[group == 6, roll] += .2
        desired = desired.clamp(raw._active_limits[..., 0], raw._active_limits[..., 1])
        target = raw._q_target + (desired - raw._q_target).clamp(-.015, .015)
        d = step_physics(raw, target)
        d['joint'] = raw._robot.data.joint_pos.torch[:, raw._active_ids]
        d['target'] = raw._q_target
        for k in traces:
            traces[k].append(d[k].cpu().numpy().copy())
        if (t+1) % 40 == 0:
            print(f'Thumb diagnostic {t+1}/240 steps; currently stable {int(d["stable"].sum())}/{n}', flush=True)
    data = {k: np.asarray(v) for k, v in traces.items()}
    limits = raw._active_limits.cpu().numpy()
    np.savez_compressed(output.with_suffix('.npz'), **data,
                        limits=limits, initial_joint=initial_q.cpu().numpy())
    results = []
    for i, row in enumerate(rows):
        initial_loaded = bool(data['pad'][20:40, i, 0].mean() >= .8)
        baseline_ok = bool(data['stable'][20:40, i].all() and not data['failed'][:40, i].any())
        stage_stable = [bool(data['stable'][a:b, i].all() and not data['failed'][:b, i].any())
                        for a, b in ((40,80), (80,120), (120,160), (160,240))]
        # Require a continuous 0.5s of actual whole-thumb unload, not just pad OFF.
        free = data['whole_finger_force'][40:120, i, 0] < .05
        unloaded = bool(np.convolve(free.astype(int), np.ones(10, dtype=int), 'valid').max() == 10)
        final_pad = bool(data['pad'][200:240, i, 0].mean() >= .8)
        initial_j2 = float(data['joint'][20:40, i, j2].mean())
        final_j2 = float(data['joint'][200:240, i, j2].mean())
        recovered_reserve = bool(final_j2-initial_j2 >= .1 and final_j2-limits[i,j2,0] >= .08)
        retained = bool(baseline_ok and all(stage_stable))
        results.append(dict(state_id=row['source_state_id'], path=row['diagnostic_path'],
            baseline_ok=baseline_ok, initially_thumb_pad_loaded=initial_loaded,
            stable_by_phase=stage_stable, stable_entire_transition=retained,
            failed=bool(data['failed'][:,i].any()), thumb_unloaded=unloaded,
            final_thumb_pad=final_pad, joint2_initial_rad=initial_j2, joint2_final_rad=final_j2,
            joint2_recovered_reserve=recovered_reserve,
            final_other_effective_pads_mean=float(data['pad'][200:240,i,1:].sum(-1).mean()),
            final_palm_force_mean_n=float(data['palm_force_n'][200:240,i].mean()),
            peak_pressure_pa=float(data['peak_pressure'][:,i].max()),
            maximum_displacement_m=float(data['displacement'][:,i].max()),
            release_recontact_pass=bool(retained and initial_loaded and unloaded and final_pad),
            useful_recovery_pass=bool(retained and final_pad and recovered_reserve)))
    aggregate = {}
    for path in PATHS:
        subset = [r for r in results if r['path'] == path]
        aggregate[path] = {'total': len(subset)}
        for key in ('baseline_ok', 'initially_thumb_pad_loaded', 'stable_entire_transition',
                    'failed', 'thumb_unloaded', 'final_thumb_pad', 'joint2_recovered_reserve',
                    'release_recontact_pass', 'useful_recovery_pass'):
            aggregate[path][key] = sum(r[key] for r in subset)
    save(output, dict(validation_completed=True, protocol=protocol, aggregate=aggregate,
                      results=results, elapsed_seconds=time.monotonic()-started))
    print(aggregate, flush=True)
